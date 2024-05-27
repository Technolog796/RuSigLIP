import torch
from random import random


def _extract_row_from_batch(batch, key):
    return torch.stack([item[key] for item in batch])


def get_train_collate_fn(world_size: int, language: str | None = None, ru_probability: float | None = None):
    if language is None and ru_probability is None:
        raise ValueError("One of the parameters `language` or `ru_probability` must not be None.")

    def collate_fn(batch):
        if language is None:
            lang = "ru" if random() < ru_probability else "en"
        else:
            lang = language

        batch_size = len(batch)
        chunk_size = batch_size // world_size

        images = _extract_row_from_batch(batch, "image")
        all_texts = []
        for i in range(world_size):
            shifted_batch = batch[chunk_size * i:] + batch[:chunk_size * i]
            all_texts.append({"input_ids": _extract_row_from_batch(shifted_batch, "input_ids_" + lang),
                              "attention_mask": _extract_row_from_batch(shifted_batch, "attention_mask_" + lang)})
        return images, all_texts

    return collate_fn


def get_test_collate_fn():

    def collate_fn(batch):
        images = _extract_row_from_batch(batch, "image")
        index_labels_en = torch.tensor([item["index_label_en"] for item in batch])
        index_labels_ru = torch.tensor([item["index_label_ru"] for item in batch])
        texts_en = {
            "input_ids": _extract_row_from_batch(batch, "input_ids_en"),
            "attention_mask": _extract_row_from_batch(batch, "attention_mask_en")
        }
        texts_ru = {
            "input_ids": _extract_row_from_batch(batch, "input_ids_ru"),
            "attention_mask": _extract_row_from_batch(batch, "attention_mask_ru")
        }
        return images, index_labels_en, index_labels_ru, texts_en, texts_ru

    return collate_fn
