import os
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from transformers import AutoTokenizer

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
        target_image_size: int = 224,
        max_sequence_length: int = 32,
        train: bool = True,
        only_main_process: bool = False,
        rank: int = 0,
        load_in_ram: bool = False
    ):
        super().__init__()

        self.only_main_process = only_main_process
        self.rank = rank
        self.image_size = [target_image_size, target_image_size]
        self.transforms = lambda image: image

        if train:
            self.dataset_directory = os.path.join(dataset_directory, "train")
        else:
            self.dataset_directory = os.path.join(dataset_directory, "test")


        self.images_ids, self.labels_en, self.labels_ru = self._get_images_and_labels(
            os.path.join(self.dataset_directory, "data.json")
        )

        self.images = None
        self.tokenized_labels_en = None
        self.tokenized_labels_ru = None
        if not only_main_process or only_main_process and rank == 0:
            if load_in_ram:
                with ThreadPoolExecutor() as executor:
                    self.images = list(executor.map(self.load_image, self.images_ids))

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
    
    def load_image(self, image_id: int) -> Tensor:
        image_path = os.path.join(self.dataset_directory, "images", image_id + ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros([*self.image_size, 3], dtype=np.uint8)
        else:
            image = cv2.resize(image, self.image_size)
        return image

    def get_image(self, idx: int) -> Tensor:
        if self.images:
            image = self.images[idx]
        else:
            image = self.load_image(self.images_ids[idx])
        image = self.transforms(image=image)["image"]
        return torch.FloatTensor(image).permute(2, 0, 1)

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

        for i in data:
            images_ids.append(i["image_id"])
            labels_en.append(i["caption_eng"])
            labels_ru.append(i["caption_rus"])

        return images_ids, labels_en, labels_ru
