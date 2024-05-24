import torch
from random import random


def get_collate_fn(world_size: int, language: str | None = None, ru_probability: float | None = None):
    if language is None and ru_probability is None:
        raise ValueError("One of the parameters `language` or `ru_probability` must not be None.")

    def collate_fn(batch):
        if language is None:
            lang = "ru" if random() > ru_probability else "en"
        else:
            lang = language

        batch_size = len(batch)
        chunk_size = batch_size // world_size

        images = torch.stack([item["image"] for item in batch])
        all_texts = []
        for i in range(world_size):
            shifted_batch = batch[chunk_size * i:] + batch[:chunk_size * i]
            all_texts.append({"input_ids": torch.stack([item["input_ids_" + lang] for item in shifted_batch]),
                              "attention_mask": torch.stack([item["attention_mask_" + lang] for item in shifted_batch])})
        return images, all_texts

    return collate_fn
