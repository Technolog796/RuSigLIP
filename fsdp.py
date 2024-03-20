import os
import yaml
import random

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from models.loss import positive_sig_loss, negative_sig_loss
from models.main_model import SigLIPModel
from dataloader.wiki_dataset import RuSigLIPDataset

from transformers import AutoTokenizer


class SigLIPSampler:
    def __init__(self, dataset, batch_size, seed=42):
        self.dataset = dataset
        self.batch_number = len(dataset) // batch_size
        self.seed = seed
        self.epoch = 0

    def get_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        return indices

    def get_langs(self):
        random.seed(self.seed + self.epoch)
        langs = [random.choice([0, 1]) for _ in range(self.batch_number)]
        return langs

    def set_epoch(self, epoch):
        self.epoch = epoch


class SigLIPDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, sampler, rank, world_size):
        super().__init__(dataset, batch_size, sampler=sampler)
        self.rank = rank
        self.world_size = world_size
        self.chunk_size = batch_size // world_size
        self.size = len(self.dataset) // self.batch_size

    def __len__(self):
        return self.size

    def __iter__(self):
        indices = self.sampler.get_indices()
        langs = self.sampler.get_langs()
        for i in range(self.size):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_indices = indices[start:end]
            batch_lang = langs[i]
            batch_indices = batch_indices[self.rank:] + batch_indices[:self.rank]

            images = torch.stack([self.dataset.get_image(idx) for idx in batch_indices[:self.chunk_size]])
            input_ids = []
            attention_mask = []
            for i in range(self.world_size):
                input_ids.append(
                    torch.stack([self.dataset.get_input_ids(idx)[batch_lang] for idx in batch_indices[self.chunk_size * i:self.chunk_size * (i + 1)]]))
                attention_mask.append(
                    torch.stack([self.dataset.get_attention_mask(idx)[batch_lang] for idx in batch_indices[self.chunk_size * i:self.chunk_size * (i + 1)]]))

            batch = (images, input_ids, attention_mask)
            yield batch


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.tensor(0.0).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, all_input_ids, all_attention_mask = batch
        images = images.to(rank)

        optimizer.zero_grad()

        img_emb, txt_emb = model(images, all_input_ids[0], all_attention_mask[0])
        loss = positive_sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"])
        
        for i in range(1, world_size):
            img_emb, txt_emb = model(images, all_input_ids[i], all_attention_mask[i])
            loss += negative_sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"])

        loss /= args["batch_size"]

        loss.backward()
        optimizer.step()

        ddp_loss += loss.item()

    ddp_loss /= len(train_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Epoch: {} \t Train Loss: {:.6f}".format(epoch, ddp_loss.item()))


def test(args, model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.tensor(0.0).to(rank)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {key: value.to(rank) for key, value in batch.items()
                     if key in ["image", "input_ids", "attention_mask"]}
            img_emb, txt_emb = model(batch)
            loss = positive_sig_loss(img_emb, txt_emb, args["t_prime"], args["bias"]) / len(img_emb)

            ddp_loss += loss.item()

    ddp_loss /= len(test_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print("Test Loss: {:.6f}".format(ddp_loss.item()))


def fsdp_main(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "0"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # TODO
    train_dataset = RuSigLIPDataset(data_file="datasets/wiki_all_en_ru.json",
                                    tokenizer=AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased"))
    #test_dataset = train_dataset

    train_sampler = SigLIPSampler(dataset=train_dataset, batch_size=args["batch_size"], seed=args["seed"])
    #test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    train_loader = SigLIPDataLoader(dataset=train_dataset,
                                    batch_size=args["batch_size"],
                                    sampler=train_sampler,
                                    rank=rank,
                                    world_size=world_size)
    #test_loader = DataLoader(test_dataset)

    torch.cuda.set_device(rank)

    model = SigLIPModel(2048, 768, 256, 0.1).to(rank)
    model = FSDP(model, use_orig_params=True)

    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=5, gamma=args["gamma"])

    for epoch in range(1, args["epochs"] + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=train_sampler)
        # test(args, model, rank, world_size, test_loader)
        scheduler.step()

    if args["save_model"]:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "model.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()

    with open("config.yaml") as file:
        args = yaml.load(file, yaml.Loader)["Train parameters"]
        assert args["batch_size"] % WORLD_SIZE == 0

    torch.manual_seed(args["seed"])

    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
