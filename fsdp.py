import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from loss import sig_loss
# from models import Model


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.tensor(0).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (images, texts) in enumerate(train_loader):
        images, texts = images.to(rank), texts.to(rank)
        optimizer.zero_grad()
        img_emb, txt_emb = model(images, texts)
        loss = sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"]) / len(img_emb)

        if world_size > 1:
            # TODO: experimental feature

            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size

            for shift in range(world_size - 1):
                dist.barrier()
                dist.send(txt_emb, next_rank)
                dist.barrier()
                dist.recv(txt_emb, prev_rank)
                loss += (
                    sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"])
                    / args["train_batch_size"]
                )

            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        loss.backward()
        optimizer.step()
        if rank == 0:
            ddp_loss += loss.item()

    ddp_loss /= len(train_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Epoch: {} \t Train Loss: {:.6f}".format(epoch, ddp_loss.item()))


def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.tensor(0).to(rank)
    with torch.no_grad():
        for images, texts in test_loader:
            images, texts = images.to(rank), texts.to(rank)
            img_emb, txt_emb = model(images, texts)
            loss = sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"]) / len(
                img_emb
            )

            if world_size > 1:
                # TODO: experimental feature

                next_rank = (rank + 1) % world_size
                prev_rank = (rank - 1) % world_size

                for shift in range(world_size - 1):
                    dist.barrier()
                    dist.send(txt_emb, next_rank)
                    dist.barrier()
                    dist.recv(txt_emb, prev_rank)
                    loss += (
                        sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"])
                        / args["train_batch_size"]
                    )

                dist.barrier()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if rank == 0:
                ddp_loss += loss.item()

    ddp_loss /= len(test_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print("Test Loss: {:.6f}".format(ddp_loss.item()))


def fsdp_main(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # TODO: add datasets
    train_dataset = ...
    test_dataset = ...

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": args["train_batch_size"], "sampler": train_sampler}
    test_kwargs = {"batch_size": args["test_batch_size"], "sampler": test_sampler}
    cuda_kwargs = {
        "num_workers": args["num_workers"],
        "pin_memory": True,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    torch.cuda.set_device(rank)

    model = Model().to(rank)
    model = FSDP(model)

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["gamma"])

    for epoch in range(1, args["epochs"] + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=train_sampler,
        )
        test(model, rank, world_size, test_loader)
        scheduler.step()

    if args["save_model"]:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "model.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    args = {
        "train_batch_size": 1000,
        "test_batch_size": 1000,
        "epochs": 1,
        "lr": 0.1,
        "gamma": 0.5,
        "seed": 42,
        "num_workers": 0,
        "t_prime": torch.log(torch.tensor(10.0)),
        "bias": torch.tensor(-10),
        "save_model": False,
    }

    torch.manual_seed(args["seed"])

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
