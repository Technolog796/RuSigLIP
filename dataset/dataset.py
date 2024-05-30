import os
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RuSigLIPDataset(Dataset):
    def __init__(self,
                 dataset_directory: str,
                 tokenizer_name: str,
                 target_image_size: int = 224,
                 max_sequence_length: int = 32,
                 load_tokenized_files: bool = False,
                 save_tokenized_files: bool = False,
                 preload_images: bool = False,
                 compress_images: bool = False) -> None:
        super().__init__()

        self.image_size = target_image_size
        self.transforms = lambda image: image
        self.compress_images = compress_images

        self.dataset_directory = dataset_directory

        self.image_ids, self.labels_en, self.labels_ru = self._get_images_and_labels(
            os.path.join(self.dataset_directory, "data.json")
        )

        self.index_labels_en = np.unique(self.labels_en, return_inverse=True)[1]
        self.index_labels_ru = np.unique(self.labels_ru, return_inverse=True)[1]

        self.images = None
        if preload_images:
            with ThreadPoolExecutor() as executor:
                self.images = list(executor.map(self.preload_image, self.image_ids))

        if load_tokenized_files:
            self.tokenized_labels_en = torch.load(
                os.path.join(self.dataset_directory, "tokenized_labels_en.pt")
            )
            self.tokenized_labels_ru = torch.load(
                os.path.join(self.dataset_directory, "tokenized_labels_ru.pt")
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            tokenizer_params = {"max_length": max_sequence_length,
                                "return_tensors": "pt",
                                "return_token_type_ids": False,
                                "padding": True,
                                "truncation": True}

            self.tokenized_labels_en = tokenizer(self.labels_en, **tokenizer_params)
            self.tokenized_labels_ru = tokenizer(self.labels_ru, **tokenizer_params)

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
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.get_image(idx),
            "label_en": self.labels_en[idx],
            "label_ru": self.labels_ru[idx],
            "index_label_en": self.index_labels_en[idx],
            "index_label_ru": self.index_labels_ru[idx],
            "input_ids_en": self.tokenized_labels_en["input_ids"][idx],
            "input_ids_ru": self.tokenized_labels_ru["input_ids"][idx],
            "attention_mask_en": self.tokenized_labels_en["attention_mask"][idx],
            "attention_mask_ru": self.tokenized_labels_ru["attention_mask"][idx],
        }

    def load_image(self, image_id: str) -> np.ndarray:
        image_path = os.path.join(self.dataset_directory, "images", image_id + ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros([self.image_size, self.image_size, 3], dtype=np.uint8)
        return image

    def preload_image(self, image_id: str) -> bytes | np.ndarray:
        image = self.load_image(image_id)
        image = cv2.resize(image, [self.image_size, self.image_size])
        if self.compress_images:
            _, buffer = cv2.imencode('.jpg', image)
            return buffer.tobytes()
        else:
            return image

    def get_image(self, idx: int) -> Tensor:
        if self.images:
            image = self.images[idx]
            if self.compress_images:
                image = self.decompress_image(image)
        else:
            image = self.load_image(self.image_ids[idx])
        image = self.transforms(image=image)["image"]
        return torch.FloatTensor(image).permute(2, 0, 1)

    def get_texts(self, indices: Tensor, language: str) -> dict[str, Tensor]:
        if language == "en":
            return {
                "input_ids": self.tokenized_labels_en["input_ids"][indices],
                "attention_mask": self.tokenized_labels_en["attention_mask"][indices],
            }
        else:
            return {
                "input_ids": self.tokenized_labels_ru["input_ids"][indices],
                "attention_mask": self.tokenized_labels_ru["attention_mask"][indices],
            }

    @staticmethod
    def decompress_image(buffer: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def _get_images_and_labels(filename: str) -> tuple[list[str], list[str], list[str]]:
        image_ids = []
        labels_en = []
        labels_ru = []

        with open(filename, "r") as file:
            data = json.load(file)

        for i in data:
            image_ids.append(i["image_id"])
            labels_en.append(i["caption_eng"])
            labels_ru.append(i["caption_rus"])

        return image_ids, labels_en, labels_ru


class DummyDataset(Dataset):
    def __init__(self, dataset_directory: str, *args, **kwargs) -> None:
        self.size = self._get_size(os.path.join(dataset_directory, "data.json"))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {}

    @staticmethod
    def _get_size(filename: str) -> int:
        with open(filename, "r") as file:
            return len(json.load(file))
