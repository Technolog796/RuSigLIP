import os
import yaml

from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import dataloader
from dataloader import SigLIPDataLoader
from models import SigmoidLoss, SigLIPModel
from models.encoders import ImageEncoder, TextEncoder, Connector


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_epoch(
    model, criterion, data_loader, rank, world_size, epoch, optimizer=None, train_mode=True
):
    model.train() if train_mode else model.eval()
    loss_sum = torch.tensor(0.0).to(rank)
    data_loader.set_epoch(epoch)

    if rank == 0:
        inner_pbar = tqdm(range(len(data_loader)), desc="Training epoch")

    with torch.set_grad_enabled(train_mode):
        for images, texts in data_loader:
            if train_mode:
                optimizer.zero_grad()

            output = model(images, texts[rank])
            loss = criterion(output, positive=(rank % 3 == 0))

            loss /= len(images)

            if train_mode:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()

            if rank == 0:
                inner_pbar.update()

    loss_avg = loss_sum / len(data_loader)
    dist.all_reduce(loss_avg, op=dist.ReduceOp.SUM)
    if rank == 0:
        inner_pbar.close()
        mode = "Train" if train_mode else "Test"
        print(f"Epoch {epoch}\t{mode} loss: {loss_avg.item():.6f}")


def main(rank, world_size, args):
    setup(rank, world_size)

    train_dataset = getattr(dataloader, args["Train dataset name"])(
        **args["Train dataset parameters"]
    )
    train_loader = SigLIPDataLoader(
        train_dataset, rank=rank, world_size=world_size, **args["Dataloader parameters"]
    )

    test_dataset = getattr(dataloader, args["Test dataset name"])(
        **args["Test dataset parameters"]
    )
    test_loader = SigLIPDataLoader(
        test_dataset, rank=rank, world_size=world_size, **args["Dataloader parameters"]
    )

    torch.cuda.set_device(rank)

    model = SigLIPModel(**args["Model parameters"]).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = SigmoidLoss(**args["Loss parameters"])
    optimizer = Adam(model.parameters(), **args["Optimizer parameters"])
    scheduler = StepLR(optimizer, **args["Scheduler parameters"])

    for epoch in range(1, args["Train parameters"]["epochs"] + 1):
        run_epoch(
            ddp_model,
            criterion,
            train_loader,
            rank,
            world_size,
            epoch,
            optimizer,
            train_mode=True,
        )
        run_epoch(
            ddp_model,
            criterion,
            test_loader,
            rank,
            world_size,
            epoch,
            train_mode=False,
        )
        scheduler.step()

    if args["Train parameters"]["save_model"]:
        dist.barrier()
        if rank == 0:
            torch.save(model.state_dict(), args["Train parameters"]["save_directory"])

    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)

    with open("config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
