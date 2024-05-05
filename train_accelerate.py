import os
import yaml

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

import accelerate
from accelerate import Accelerator

import dataloader
from dataloader.dataloader_accelerate import SigLIPDataLoader
from models import SigmoidLoss, SigLIPModel
from models.encoders import ImageEncoder, TextEncoder, Connector


def run_epoch(
    accelerator,
    model,
    criterion,
    data_loader,
    optimizer=None,
    train_mode=True,
):
    model.train() if train_mode else model.eval()
    epoch_loss = 0

    for batch in tqdm(data_loader):
        images = batch["image"]
        texts = {"input_ids": batch["input_ids_en"], "attention_mask": batch["attention_mask_en"]}

        if train_mode:
            optimizer.zero_grad()

        img_emb, txt_emb = model(images, texts)
        loss = criterion(img_emb, txt_emb)

        if train_mode:
            accelerator.backward(loss)
            optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(data_loader.dataset)

    return epoch_loss.item()


def main(args):
    accelerator = Accelerator()
    accelerate.DataLoaderConfiguration(split_batches=True)

    train_dataset = getattr(dataloader, args["Train dataset name"])(
        **args["Train dataset parameters"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["Dataloader parameters"]["batch_size"],
        shuffle=True,
        num_workers=100,
        prefetch_factor=1,
        pin_memory=True,
        pin_memory_device="cuda"
    )

    test_dataset = getattr(dataloader, args["Test dataset name"])(
        **args["Test dataset parameters"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["Dataloader parameters"]["batch_size"],
        num_workers=10,
        prefetch_factor=1,
        pin_memory=True,
        pin_memory_device="cuda"
    )

    criterion = SigmoidLoss(**args["Loss parameters"])

    model = SigLIPModel(**args["Model parameters"])
    optimizer = Adam(model.parameters(), **args["Optimizer parameters"])
    scheduler = StepLR(optimizer, **args["Scheduler parameters"])

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler)

    # wandb.login()
    # wandb.init(project="RuSigLIP", config=args, sync_tensorboard=True)
    # wandb.watch(model, log="all", log_freq=10)

    for epoch in range(1, args["Train parameters"]["epochs"] + 1):
        train_loss = run_epoch(
            accelerator,
            model,
            criterion,
            train_loader,
            optimizer,
            train_mode=True,
        )
        test_loss = run_epoch(
            accelerator,
            model,
            criterion,
            test_loader,
            train_mode=False
        )

        # wandb.log({
        #     "Train loss": train_loss,
        #     "Test loss": test_loss,
        #     "Epoch": epoch
        # })

        scheduler.step()

    if args["Train parameters"]["save_model"]:
        accelerator.wait_for_everyone()
        torch.save(model.state_dict(), args["Train parameters"]["save_directory"])

    accelerator.wait_for_everyone()
    accelerator.cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)

    with open("config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    main(args)
