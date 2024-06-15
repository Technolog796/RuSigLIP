from random import random
from typing import Callable
from typing import Any, Dict, Union


import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from transformers import get_cosine_schedule_with_warmup
from composer.optim import DecoupledAdamW


def _extract_row_from_batch(batch: list[dict], key: str) -> Tensor:
    return torch.stack([item[key] for item in batch])


def get_train_collate_fn(
    world_size: int, language: str | None = None, ru_probability: float | None = None
) -> Callable[[list[dict]], tuple[Tensor, dict[str, Tensor]]]:
    if language is None and ru_probability is None:
        raise ValueError(
            "One of the parameters `language` or `ru_probability` must not be None."
        )

    def collate_fn(batch: list[dict]) -> tuple[Tensor, dict[str, Tensor]]:
        if language is None:
            lang = "ru" if random() < ru_probability else "en"
        else:
            lang = language

        batch_size = len(batch)
        chunk_size = batch_size // world_size

        images = _extract_row_from_batch(batch, "image")
        all_texts = []
        for i in range(world_size):
            shifted_batch = batch[chunk_size * i :] + batch[: chunk_size * i]
            all_texts.append(
                {
                    "input_ids": _extract_row_from_batch(
                        shifted_batch, "input_ids_" + lang
                    ),
                    "attention_mask": _extract_row_from_batch(
                        shifted_batch, "attention_mask_" + lang
                    ),
                }
            )
        return images, all_texts

    return collate_fn


def get_test_collate_fn() -> Callable[[list[dict]], tuple]:
    def collate_fn(batch: list[dict]) -> tuple:
        images = _extract_row_from_batch(batch, "image")
        index_labels_en = torch.tensor([item["index_label_en"] for item in batch])
        index_labels_ru = torch.tensor([item["index_label_ru"] for item in batch])
        texts_en = {
            "input_ids": _extract_row_from_batch(batch, "input_ids_en"),
            "attention_mask": _extract_row_from_batch(batch, "attention_mask_en"),
        }
        texts_ru = {
            "input_ids": _extract_row_from_batch(batch, "input_ids_ru"),
            "attention_mask": _extract_row_from_batch(batch, "attention_mask_ru"),
        }
        return images, index_labels_en, index_labels_ru, texts_en, texts_ru

    return collate_fn


def update_topk_accuracy(
    labels: Tensor,
    img_emb: Tensor,
    txt_emb: Tensor,
    accuracy: dict[str, Tensor],
    topk: list[int],
    batch_size: int,
) -> dict[str, Tensor]:
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


def configure_optimizer_and_scheduler(
    model_parameters: Any,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any]
) -> Union[Optimizer, _LRScheduler]:
    
    optimizer_name = optimizer_config["name"]
    optimizer_params = optimizer_config["params"]

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model_parameters, **optimizer_params)
    elif optimizer_name == "Lamb":
        from bitsandbytes.optim import LAMB
        optimizer = LAMB(model_parameters, **optimizer_params)
    elif optimizer_name == "DecoupledAdamW":
        optimizer = DecoupledAdamW(model_parameters, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
    

    scheduler_name = scheduler_config["name"]
    scheduler_params = scheduler_config["params"]

    if scheduler_name == "Cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_params)
    elif scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

    return optimizer, scheduler