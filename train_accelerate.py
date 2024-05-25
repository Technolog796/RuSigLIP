import os
import yaml

import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from galore_torch import GaLoreAdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration

import dataloader
from dataloader.collate_function import get_collate_fn
from models import SigmoidLoss, SigLIPModel


def get_galore_optimizer(model, args):
    param_groups = []
    galore_rank = args["GaLore parameters"]["rank"]
    for param in model.parameters():
        if len(param.shape) == 2 and param.shape[0] >= param.shape[1] > galore_rank:
            param_groups.append({'params': param, **args["GaLore parameters"]})
        else:
            param_groups.append({'params': param})
    return GaLoreAdamW(param_groups, no_deprecation_warning=True, **args["Optimizer parameters"])



def eval_epoch(accelerator, model, eval_dataset):
    model.eval()

    with torch.no_grad():
        ...


def train_epoch(accelerator, model, train_dataset):
    model.train()
    ...






def run_epoch(
    accelerator,
    model,
    criterion,
    data_loader,
    epoch,
    optimizer=None, # Тут не очень коректно так делать - ruff не в восторге
    scheduler=None, # Как и тут
    train_mode=True,
):
    model.train() if train_mode else model.eval()
    ddp_loss = 0

    if accelerator.is_main_process:
        inner_pbar = tqdm(range(len(data_loader)), desc="Epoch")

    import time
    for images, all_texts in data_loader:
        batch_size = len(images) * len(all_texts)
        
        begin = time.time()

        if train_mode:
            optimizer.zero_grad()

        img_emb, txt_emb = model(images, all_texts[0])
        loss = criterion(img_emb, txt_emb, positive=True) / batch_size

        if train_mode:
            accelerator.backward(loss)

        for texts in all_texts[1:]:
            img_emb, txt_emb = model(images, texts)
            loss = criterion(img_emb, txt_emb, positive=False) / batch_size
            ddp_loss += loss.item()
            if train_mode:
                accelerator.backward(loss)

        if train_mode:
            optimizer.step()
            scheduler.step()

        if accelerator.is_main_process:
            inner_pbar.update()

            print("Train" if train_mode else "Test", "time:", time.time() - begin)
        ddp_loss += loss.item()

    ddp_loss /= len(data_loader)

    if accelerator.is_main_process:
        mode = "Train" if train_mode else "Test"
        wandb.log({f"{mode} loss": ddp_loss.item(), "Epoch": epoch})
        print(f"Epoch {epoch}\t{mode} loss: {ddp_loss:.6f}")


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dataloader_config = DataLoaderConfiguration(split_batches=True, dispatch_batches=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],
                              dataloader_config=dataloader_config)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = \
            args["Train dataloader parameters"]["batch_size"] // world_size

    train_dataset = getattr(dataloader, args["Dataset name"])(rank=rank,
        train=True, **args["Dataset parameters"]
    )
    test_dataset = getattr(dataloader, args["Dataset name"])(rank=rank,
        train=False, **args["Dataset parameters"]
    )


    #train_loader = SigLIPDataLoader(train_dataset, rank=rank, world_size=world_size, **args["Train dataloader parameters"])
    train_loader = DataLoader(   train_dataset,   collate_fn=get_collate_fn(world_size, **args["Language parameters"]),   **args["Train dataloader parameters"])
    #test_loader = SigLIPDataLoader(test_dataset, rank=rank, world_size=world_size, **args["Test dataloader parameters"])
    test_loader = DataLoader(    test_dataset,    collate_fn=get_collate_fn(world_size, **args["Language parameters"]),    **args["Test dataloader parameters"])

    criterion = SigmoidLoss(**args["Loss parameters"])

    model = SigLIPModel(**args["Model parameters"])

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        optimizer = get_galore_optimizer(model, args)
        scheduler = get_cosine_schedule_with_warmup(optimizer, **args["Scheduler parameters"])
        model, train_loader, test_loader, optimizer, scheduler = accelerator.prepare(
            model, train_loader, test_loader, optimizer, scheduler)

    else:
        model, train_loader, test_loader = accelerator.prepare(
            model, train_loader, test_loader)
        optimizer = get_galore_optimizer(model, args)
        scheduler = get_cosine_schedule_with_warmup(optimizer, **args["Scheduler parameters"])
        optimizer, scheduler = accelerator.prepare(optimizer,  scheduler)

    if accelerator.is_main_process:

        wandb.login()
        wandb.init(project="RuSigLIP", config=args, sync_tensorboard=True)
        wandb.watch(model, log="all", log_freq=10)

    epochs, saving_mode, save_frequency, save_directory = args["Train parameters"].values()
    for epoch in range(1, epochs + 1):
        run_epoch(
            accelerator,
            model,
            criterion,
            train_loader,
            epoch,
            optimizer,
            scheduler,
            train_mode=True,
        )
        run_epoch(
            accelerator,
            model,
            criterion,
            test_loader,
            epoch,
            train_mode=False
        )

        if saving_mode and epoch % save_frequency == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, save_directory)

    if saving_mode:
        accelerator.wait_for_everyone()
        accelerator.save_model(model, save_directory)


if __name__ == "__main__":
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open("train_config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    main(args)
