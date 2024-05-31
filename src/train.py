import yaml
from typing import Any
import dataset

import wandb
import torch
from tqdm import tqdm
from safetensors import safe_open
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from composer.optim import DecoupledAdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DataLoaderConfiguration

from dataset import DummyDataset
from model import SigLIPModel
from src.loss import SigmoidLoss
from src.train_utils import (
    get_train_collate_fn,
    get_test_collate_fn,
    update_topk_accuracy,
)


@torch.no_grad()
def eval_epoch(
    accelerator: Accelerator,
    model: SigLIPModel,
    criterion: SigmoidLoss,
    loader: DataLoader,
    topk: tuple[int] = (1, 5),
) -> dict[str, float]:
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

        update_topk_accuracy(
            labels_en, img_emb, txt_emb_en, accuracy_en, topk, batch_size
        )
        update_topk_accuracy(
            labels_ru, img_emb, txt_emb_ru, accuracy_ru, topk, batch_size
        )

        if accelerator.is_main_process:
            inner_pbar.update()

    test_log = {
        "Test loss (en)": loss_en / len(loader),
        "Test loss (ru)": loss_ru / len(loader),
    }
    test_log.update(
        {key + "(en)": value / len(loader) for key, value in accuracy_en.items()}
    )
    test_log.update(
        {key + "(ru)": value / len(loader) for key, value in accuracy_ru.items()}
    )

    for key in test_log:
        test_log[key] = accelerator.gather(test_log[key]).sum().item()
    return test_log


def train_epoch(
    accelerator: Accelerator,
    model: SigLIPModel,
    criterion: SigmoidLoss,
    loaders: list[DataLoader],
    optimizer: DecoupledAdamW,
    scheduler: LambdaLR,
) -> dict[str, float]:
    model.train()
    ddp_loss = torch.tensor([0.0], device=accelerator.process_index)
    steps_number = sum(len(loader) for loader in loaders)

    if accelerator.is_main_process:
        inner_pbar = tqdm(range(steps_number), desc="Train epoch")

    for loader in loaders:
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

    ddp_loss /= steps_number
    ddp_loss = accelerator.gather(ddp_loss).sum().item()

    return {"Train loss": ddp_loss}


def main(params: dict[str, Any]) -> None:
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            split_batches=True, dispatch_batches=True
        )
    )

    model = SigLIPModel(**params["Model parameters"])
    if params["Train parameters"]["load_model"]:
        weights = {}
        with safe_open(
            params["Train parameters"]["load_file"], framework="pt", device="cpu"
        ) as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)
        model.load_state_dict(weights)

    criterion = SigmoidLoss(**params["Loss parameters"])
    optimizer = DecoupledAdamW(model.parameters(), **params["Optimizer parameters"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, **params["Scheduler parameters"]
    )

    if accelerator.is_main_process:
        train_datasets = []
        for dataset_name, dataset_directory in zip(
            params["Train dataset names"], params["Train dataset directories"]
        ):
            train_datasets.append(
                getattr(dataset, dataset_name)(
                    dataset_directory, **params["Dataset parameters"]
                )
            )

        test_dataset = getattr(dataset, params["Test dataset name"])(
            params["Test dataset directory"], **params["Dataset parameters"]
        )

    else:
        train_datasets = [
            DummyDataset(directory) for directory in params["Train dataset directories"]
        ]
        test_dataset = DummyDataset(params["Test dataset directory"])

    train_loaders = [
        DataLoader(
            train_dataset,
            collate_fn=get_train_collate_fn(
                accelerator.num_processes, **params["Language parameters"]
            ),
            **params["Train dataloader parameters"],
        )
        for train_dataset in train_datasets
    ]
    test_loader = DataLoader(
        test_dataset,
        collate_fn=get_test_collate_fn(),
        **params["Test dataloader parameters"],
    )

    model, optimizer, scheduler, test_loader = accelerator.prepare(
        model, optimizer, scheduler, test_loader
    )
    if len(train_loaders) > 1:
        train_loaders = accelerator.prepare(*train_loaders)
    else:
        train_loaders = [accelerator.prepare(train_loaders[0])]

    if accelerator.is_main_process:
        wandb.login()
        wandb.init(project="RuSigLIP", config=params, sync_tensorboard=True)
        wandb.watch(model, log="all", log_freq=10)

    save_model = params["Train parameters"]["save_model"]
    if save_model:
        save_frequency = params["Train parameters"]["save_frequency"]
        save_directory = params["Train parameters"]["save_directory"]

    for epoch in range(1, params["Train parameters"]["epochs"] + 1):
        train_log = train_epoch(
            accelerator,
            model,
            criterion,
            train_loaders,
            optimizer,
            scheduler,
        )
        test_log = eval_epoch(accelerator, model, criterion, test_loader)

        if accelerator.is_main_process:
            log = {"Epoch": epoch, **train_log, **test_log}
            wandb.log(log)
            for key, value in log.items():
                print(f"{key}: {value:.6f}")

        if save_model and epoch % save_frequency == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, save_directory)

    if save_model:
        accelerator.wait_for_everyone()
        accelerator.save_model(model, save_directory)


if __name__ == "__main__":
    torch.manual_seed(42)

    with open("train_config.yml") as file:
        args = yaml.load(file, yaml.Loader)

    main(args)
