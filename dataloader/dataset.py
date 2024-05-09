import os
import json
import warnings

import PIL
from PIL import Image
from transformers import AutoTokenizer

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RuSigLIPDataset(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        tokenizer_name: str = None,
        load_tokenized_files: bool = True,
        save_tokenized_files: bool = True,
        max_sequence_length: int = 512,
        train: bool = True,
        only_main_process: bool = False,
        rank: int = 0,
    ):
        super().__init__()

        self.only_main_process = only_main_process
        self.rank = rank

        if train:
            self.dataset_directory = os.path.join(dataset_directory, "train")
        else:
            self.dataset_directory = os.path.join(dataset_directory, "test")

        self.transforms = lambda image: image

        self.images_ids, self.labels_en, self.labels_ru = self._get_images_and_labels(
            os.path.join(self.dataset_directory, "data.json")
        )

        if load_tokenized_files:
            self.tokenized_labels_en = torch.load(
                os.path.join(self.dataset_directory, "tokenized_labels_en.pt")
            )
            self.tokenized_labels_ru = torch.load(
                os.path.join(self.dataset_directory, "tokenized_labels_ru.pt")
            )

        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            self.tokenized_labels_en = tokenizer(
                self.labels_en,
                max_length=max_sequence_length,
                return_tensors="pt",
                return_token_type_ids=False,
                padding=True,
                truncation=True,
            )
            self.tokenized_labels_ru = tokenizer(
                self.labels_ru,
                max_length=max_sequence_length,
                return_tensors="pt",
                return_token_type_ids=False,
                padding=True,
                truncation=True,
            )

            if save_tokenized_files:
                torch.save(
                    self.tokenized_labels_en,
                    os.path.join(self.dataset_directory, "tokenized_labels_en.pt"),
                )
                torch.save(
                    self.tokenized_labels_ru,
                    os.path.join(self.dataset_directory, "tokenized_labels_ru.pt"),
                )

    def __len__(self) -> int:
        return len(self.images_ids)

    def __getitem__(self, idx: int) -> dict:
        if self.only_main_process and self.rank != 0:
            return {}
        return {
            "image": self.get_image(idx),
            "label_en": self.labels_en[idx],
            "label_ru": self.labels_ru[idx],
            "input_ids_en": self.tokenized_labels_en["input_ids"][idx],
            "input_ids_ru": self.tokenized_labels_ru["input_ids"][idx],
            "attention_mask_en": self.tokenized_labels_en["attention_mask"][idx],
            "attention_mask_ru": self.tokenized_labels_ru["attention_mask"][idx],
        }

    def get_image(self, idx: int) -> Tensor:
        image_path = os.path.join(self.dataset_directory, "images", self.images_ids[idx] + ".jpg")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = Image.open(image_path)
                image = image.convert("RGB")
        except PIL.UnidentifiedImageError:
            image = Image.fromarray(np.zeros([256, 256, 3], dtype=np.uint8))
        image = self.transforms(image)
        return image

    def get_texts(self, indices: Tensor, language: str) -> dict[str, Tensor]:
        if language == "en":
            return {"input_ids": self.tokenized_labels_en["input_ids"][indices],
                    "attention_mask": self.tokenized_labels_en["attention_mask"][indices]}
        elif language == "ru":
            return {"input_ids": self.tokenized_labels_ru["input_ids"][indices],
                    "attention_mask": self.tokenized_labels_ru["attention_mask"][indices]}

    @staticmethod
    def _get_images_and_labels(filename: str) -> tuple[list[str], list[str], list[str]]:
        images_ids = []
        labels_en = []
        labels_ru = []

        with open(filename, "r") as file:
            data = json.load(file)

        for idx, i in enumerate(data):
            images_ids.append(i["image_id"])
            labels_en.append(i["caption_eng"])
            labels_ru.append(i["caption_rus"])

        return images_ids, labels_en, labels_ru
