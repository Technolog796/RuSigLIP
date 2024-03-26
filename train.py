import os
import yaml

from tqdm import tqdm
import wandb

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import dataloader
from dataloader import SigLIPDataLoader
from models import SigmoidLoss, SigLIPModel


def run_epoch(
    model,
    criterion,
    data_loader,
    rank,
    world_size,
    epoch,
    optimizer=None,
    train_mode=True,
):
    model.train() if train_mode else model.eval()
    ddp_loss = torch.tensor(0.0).to(rank)
    data_loader.set_epoch(epoch)

    with torch.set_grad_enabled(train_mode):
        for images, texts in tqdm(data_loader):
            optimizer.zero_grad()

            img_emb, txt_emb = model(images, texts[0])
            loss = criterion(img_emb, txt_emb, positive=True)

            for i in range(1, world_size):
                img_emb, txt_emb = model(images, texts[i])
                loss += criterion(img_emb, txt_emb, positive=False)

            loss /= len(img_emb)

            if train_mode:
                loss.backward()
                optimizer.step()

            ddp_loss += loss.item()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        mode = "Train" if train_mode else "Test"
        wandb.log({f"{mode} loss": ddp_loss.item(), "Epoch": epoch})
        print(f"Epoch {epoch}\t{mode} loss: {ddp_loss.item():.6f}")


def fsdp_main(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    train_dataset = getattr(dataloader, args["Train dataset name"])(
        **args["Train dataset parameters"]
    )
    train_loader = SigLIPDataLoader(train_dataset, **args["Dataloader parameters"])

    test_dataset = getattr(dataloader, args["Test dataset name"])(
        **args["Test dataset parameters"]
    )
    test_loader = SigLIPDataLoader(test_dataset, **args["Dataloader parameters"])

    torch.cuda.set_device(rank)

    model = SigLIPModel(**args["Model parameters"]).to(rank)
    model = FSDP(model, use_orig_params=True)

    wandb.watch(model, log="all", log_freq=10)

    criterion = SigmoidLoss(**args["Loss parameters"])
    optimizer = Adam(model.parameters(), **args["Optimizer parameters"])
    scheduler = StepLR(optimizer, **args["Scheduler parameters"])

    for epoch in range(1, args["Train parameters"]["epochs"] + 1):
        run_epoch(
            model,
            criterion,
            train_loader,
            rank,
            world_size,
            epoch,
            optimizer,
            train_mode=True,
        )
        run_epoch(
            model, criterion, test_loader, rank, world_size, epoch, train_mode=False
        )
        scheduler.step()

    if args["Train parameters"]["save_model"]:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, args["Train parameters"]["save_directory"])

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(42)

    WORLD_SIZE = torch.cuda.device_count()

    with open("config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    wandb.login()
    wandb.init(project="RuSigLIP", config=args, sync_tensorboard=True)

    torch.multiprocessing.spawn(
        fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True
    )
