import os
import yaml

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from models.loss import SigmoidLoss
from models.main_model import SigLIPModel
from dataloader.dataloader import SigLIPDataLoader
from dataloader.dataset import RuSigLIPDataset

from transformers import AutoTokenizer


def train(args, model, criterion, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.tensor(0.0).to(rank)
    if sampler:
        train_loader.set_epoch(epoch)
    for images, texts in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()

        img_emb, txt_emb = model(images, texts[0])
        loss = criterion(img_emb, txt_emb, positive=True)
        
        for i in range(1, world_size):
            img_emb, txt_emb = model(images, texts[i])
            loss += criterion(img_emb, txt_emb, positive=False)

        loss /= args["Train parameters"]["batch_size"]

        loss.backward()
        optimizer.step()

        ddp_loss += loss.item()

    ddp_loss /= len(train_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Epoch: {} \t Train Loss: {:.6f}".format(epoch, ddp_loss.item()))


def test(args, model, criterion, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.tensor(0.0).to(rank)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {key: value.to(rank) for key, value in batch.items()
                     if key in ["image", "input_ids", "attention_mask"]}
            img_emb, txt_emb = model(batch)
            loss = criterion(img_emb, txt_emb, positive=True)

            ddp_loss += loss.item()

    ddp_loss /= len(test_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print("Test Loss: {:.6f}".format(ddp_loss.item()))


def fsdp_main(rank, world_size, train_dataset, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # TODO
    #test_dataset = train_dataset
    train_loader = SigLIPDataLoader(dataset=train_dataset,
                                    batch_size=args["Train parameters"]["batch_size"],
                                    rank=rank,
                                    world_size=world_size,
                                    seed=args["Train parameters"]["seed"])
    #test_loader = DataLoader(test_dataset)

    torch.cuda.set_device(rank)

    model = SigLIPModel(args["Image Encoder"]["embedding_size"], args["Text Encoder"]["embedding_size"],
                        args["Connector"]["connector_size"], args["Connector"]["dropout"]).to(rank)
    model = FSDP(model, use_orig_params=True)

    criterion = SigmoidLoss(temperature=args["Train parameters"]["temperature"],
                            bias=args["Train parameters"]["bias"])

    optimizer = optim.Adam(model.parameters(), lr=args["Train parameters"]["learning_rate"])
    scheduler = StepLR(optimizer, step_size=10, gamma=args["Train parameters"]["gamma"])

    for epoch in range(1, args["Train parameters"]["epochs"] + 1):
        train(args, model, criterion, rank, world_size, train_loader, optimizer, epoch)
        # test(args, model, criterion, rank, world_size, test_loader)
        scheduler.step()

    if args["Train parameters"]["save_model"]:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "model.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()

    with open("config.yml") as file:
        args = yaml.load(file, yaml.Loader)
        assert args["Train parameters"]["batch_size"] % WORLD_SIZE == 0

    torch.manual_seed(args["Train parameters"]["seed"])

    train_dataset = RuSigLIPDataset(data_file="datasets/laion_coco.json",
                                    tokenizer=AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased"),
                                    max_len=args["Text Encoder"]["max_length"])

    mp.spawn(fsdp_main, args=(WORLD_SIZE, train_dataset, args), nprocs=WORLD_SIZE, join=True)
