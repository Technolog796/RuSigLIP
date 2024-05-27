import os
import yaml

import wandb
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from safetensors import safe_open
from composer.optim import DecoupledAdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration

import dataloader
from dataloader import DummyDataset
from dataloader.collate_function import get_train_collate_fn, get_test_collate_fn
from models import SigmoidLoss, SigLIPModel


def update_topk_accuracy(labels: Tensor, img_emb: Tensor, txt_emb: Tensor, 
                         accuracy: dict[str, Tensor], topk: list[int], batch_size: int):
    logits = img_emb @ txt_emb.T
    top_indices = logits.argsort(dim=-1, descending=True)

    for i, indices in enumerate(top_indices):
        predicted_labels = labels[indices]
        unique_indices = np.unique(predicted_labels.cpu(), return_index=True)[1]
        unique_indices = np.sort(unique_indices)
        predicted_labels = predicted_labels[unique_indices]
        for k in topk:
            if torch.any(labels[i] == predicted_labels[:k]):
                accuracy[f"Accuracy@{k}"] += 1 / batch_size
    return accuracy


@torch.no_grad
def eval_epoch(accelerator,
               model,
               criterion,
               loader,
               topk=[1, 5]):
    model.eval()

    rank = accelerator.process_index
    accuracy_en = {f"Accuracy@{k}": torch.tensor([0.0], device=rank) for k in topk}
    accuracy_ru = {f"Accuracy@{k}": torch.tensor([0.0], device=rank) for k in topk}
    loss_en = torch.tensor([0.0], device=rank)
    loss_ru = torch.tensor([0.0], device=rank)

    if accelerator.is_main_process:
        inner_pbar = tqdm(range(len(loader)), desc="Test epoch")

    for batch in loader:
        images, labels_en, labels_ru, texts_en, texts_ru = batch
        batch_size = len(images) * accelerator.num_processes

        img_emb, txt_emb_en = model.predict(images, texts_en)
        img_emb, txt_emb_ru = model.predict(images, texts_ru)

        loss_en += criterion(img_emb, txt_emb_en) / batch_size
        loss_ru += criterion(img_emb, txt_emb_ru) / batch_size

        update_topk_accuracy(labels_en, img_emb, txt_emb_en, accuracy_en, topk, batch_size)
        update_topk_accuracy(labels_ru, img_emb, txt_emb_ru, accuracy_ru, topk, batch_size)

        if accelerator.is_main_process:
            inner_pbar.update()

    log = {"Test loss (en)": loss_en / len(loader), "Test loss (ru)": loss_ru / len(loader)}
    log.update({key + "(en)": value / len(loader) for key, value in accuracy_en.items()})
    log.update({key + "(ru)": value / len(loader) for key, value in accuracy_ru.items()})
    for key in log:
        log[key] = accelerator.gather(log[key]).sum().item()
    
    if accelerator.is_main_process:
        #wandb.log(log)
        for key, value in log.items():
            print(f"\t{key}: {value:.6f}")


def train_epoch(accelerator,
                model,
                criterion,
                loader,
                epoch,
                optimizer,
                scheduler):
    model.train()
    ddp_loss = torch.tensor([0.0], device=accelerator.process_index)

    if accelerator.is_main_process:
        inner_pbar = tqdm(range(len(loader)), desc=f"Train epoch {epoch}")

    for images, all_texts in loader:
        batch_size = len(images) * accelerator.num_processes
        optimizer.zero_grad()
        
        img_emb, txt_emb = model(images, all_texts[0])
        loss = criterion(img_emb, txt_emb, positive=True) / batch_size
        ddp_loss += loss.item()
        accelerator.backward(loss)

        for texts in all_texts[1:]:
            img_emb, txt_emb = model(images, texts)
            loss = criterion(img_emb, txt_emb, positive=False) / batch_size
            ddp_loss += loss.item()
            accelerator.backward(loss)

        optimizer.step()
        scheduler.step()

        if accelerator.is_main_process:
            inner_pbar.update()

    ddp_loss /= len(loader)
    ddp_loss = accelerator.gather(ddp_loss).sum().item()

    if accelerator.is_main_process:
        #wandb.log({"Train loss": ddp_loss, "Epoch": epoch})
        print(f"Epoch {epoch}\n\tTrain loss: {ddp_loss:.6f}")


def main(args):
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], 
                              dataloader_config=DataLoaderConfiguration(split_batches=True, dispatch_batches=True))

    if accelerator.is_main_process:
        train_dataset = getattr(dataloader, args["Train dataset name"])(**args["Train dataset parameters"])
        test_dataset = getattr(dataloader, args["Test dataset name"])(**args["Test dataset parameters"])
    else:
        train_dataset = DummyDataset(**args["Train dataset parameters"])
        test_dataset = DummyDataset(**args["Test dataset parameters"])

    train_loader = DataLoader(train_dataset, 
                              collate_fn=get_train_collate_fn(accelerator.num_processes, **args["Language parameters"]), 
                              **args["Train dataloader parameters"])
    test_loader = DataLoader(test_dataset, 
                             collate_fn=get_test_collate_fn(),
                             **args["Test dataloader parameters"])

    model = SigLIPModel(**args["Model parameters"])
    if args["Train parameters"]["load_model"]:
        weights = {}
        with safe_open(args["Train parameters"]["load_file"], 
                       framework="pt", 
                       device="cpu") as file:
            for k in file.keys():
                weights[k] = file.get_tensor(k)
        model.load_state_dict(weights)

    criterion = SigmoidLoss(**args["Loss parameters"])
    optimizer = DecoupledAdamW(model.parameters(), **args["Optimizer parameters"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, **args["Scheduler parameters"])

    model, train_loader, test_loader, optimizer, scheduler = accelerator.prepare(
            model, train_loader, test_loader, optimizer, scheduler)

    if accelerator.is_main_process:
        pass
        #wandb.login()
        #wandb.init(project="RuSigLIP", config=args, sync_tensorboard=True)
        #wandb.watch(model, log="all", log_freq=10)

    save_model = args["Train parameters"]["save_model"]
    if save_model:
        save_frequency = args["Train parameters"]["save_frequency"]
        save_directory = args["Train parameters"]["save_directory"]
    
    for epoch in range(1, args["Train parameters"]["epochs"] + 1):
        train_epoch(
            accelerator,
            model,
            criterion,
            train_loader,
            epoch,
            optimizer,
            scheduler,
        )
        eval_epoch(
            accelerator,
            model,
            criterion,
            test_loader
        )

        if save_model and epoch % save_frequency == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, save_directory)

    if save_model:
        accelerator.wait_for_everyone()
        accelerator.save_model(model, save_directory)


if __name__ == "__main__":
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open("train_config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    main(args)
