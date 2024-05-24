import os
import yaml
from functools import partial

import wandb
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, BackwardPrefetch, CPUOffload, wrap

from galore_torch import GaLoreAdamW
from transformers import get_cosine_schedule_with_warmup

import dataloader
from dataloader import SigLIPDataLoader
from models import SigmoidLoss, SigLIPModel
from models.encoders import ImageEncoder, TextEncoder, Connector


def save_model(model, rank, save_directory):
    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(states, save_directory)


def run_epoch(
    model,
    criterion,
    data_loader,
    rank,
    world_size,
    epoch,
    optimizer=None,
    scheduler=None,
    train_mode=True,
):
    model.train() if train_mode else model.eval()
    ddp_loss = torch.tensor(0.0).to(rank)
    data_loader.set_epoch(epoch)

    if rank == 0:
        inner_pbar = tqdm(range(len(data_loader)), desc="Training epoch")

    with torch.set_grad_enabled(train_mode):
        for images, all_texts in data_loader:
            if train_mode:
                optimizer.zero_grad()

            images = images.to(rank)
            for texts in all_texts:
                for key in texts:
                    texts[key] = texts[key].to(rank)
            batch_size = len(images) * world_size

            img_emb, txt_emb = model(images, all_texts[0])
            loss = criterion(img_emb, txt_emb, positive=True) / batch_size
            ddp_loss += loss.item()
            if train_mode:
                loss.backward()

            for texts in all_texts[1:]:
                img_emb, txt_emb = model(images, texts)
                loss = criterion(img_emb, txt_emb, positive=False) / batch_size
                ddp_loss += loss.item()
                if train_mode:
                    loss.backward()

            if train_mode:
                optimizer.step()
                scheduler.step()

            if rank == 0:
                inner_pbar.update()

    ddp_loss /= len(data_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        inner_pbar.close()
        mode = "Train" if train_mode else "Test"
        # wandb.log({f"{mode} loss": ddp_loss.item(), "Epoch": epoch})
        print(f"Epoch {epoch}\t{mode} loss: {ddp_loss.item():.6f}")


def fsdp_main(rank, world_size, train_dataset, test_dataset, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    train_loader = SigLIPDataLoader(train_dataset, rank=rank, world_size=world_size, **args["Dataloader parameters"])
    test_loader = SigLIPDataLoader(test_dataset, rank=rank, world_size=world_size, **args["Dataloader parameters"])

    torch.cuda.set_device(rank)

    model = SigLIPModel(**args["Model parameters"]).to(rank)

    siglip_auto_wrap_policy = partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={
            ImageEncoder, TextEncoder, Connector
        },
    )

    model = FullyShardedDataParallel(model,
                 cpu_offload=CPUOffload(offload_params=True),
                 backward_prefetch=BackwardPrefetch.BACKWARD_POST,
                 auto_wrap_policy=siglip_auto_wrap_policy)

    if rank == 0:
        pass
        # wandb.login()
        # wandb.init(project="RuSigLIP", config=args, sync_tensorboard=True)
        # wandb.watch(model, log="all", log_freq=10)

    criterion = SigmoidLoss(**args["Loss parameters"])

    param_groups = []
    galore_rank = args["GaLore parameters"]["rank"]
    for param in model.parameters():
        if len(param.shape) == 2 and param.shape[0] > galore_rank and param.shape[1] > galore_rank:
            param_groups.append({'params': param, **args["GaLore parameters"]})
        else:
            param_groups.append({'params': param})
    optimizer = GaLoreAdamW(param_groups, **args["Optimizer parameters"])

    scheduler = get_cosine_schedule_with_warmup(optimizer, **args["Scheduler parameters"])

    epochs, saving_mode, save_frequency, save_directory = args["Train parameters"].values()
    for epoch in range(1, epochs + 1):
        run_epoch(
            model,
            criterion,
            train_loader,
            rank,
            world_size,
            epoch,
            optimizer,
            scheduler,
            train_mode=True,
        )
        run_epoch(
            model, criterion, test_loader, rank, world_size, epoch, train_mode=False
        )
        if saving_mode and epoch % save_frequency == 0:
            save_model(model, rank, save_directory)

    if saving_mode:
        save_model(model, rank, save_directory)

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(42)

    WORLD_SIZE = torch.cuda.device_count()

    with open("config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    train_dataset = getattr(dataloader, args["Dataset name"])(
        train=True, **args["Dataset parameters"]
    )
    test_dataset = getattr(dataloader, args["Dataset name"])(
        train=False, **args["Dataset parameters"]
    )

    torch.multiprocessing.spawn(
        fsdp_main, args=(WORLD_SIZE, train_dataset, test_dataset, args), nprocs=WORLD_SIZE, join=True
    )
