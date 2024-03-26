import json

from PIL import Image
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
        max_sequence_length: int = 512,
    ):
        super().__init__()

        self.dataset_directory = dataset_directory
        self.transforms = lambda image: {"image": image}

        self.images_ids, self.labels_en, self.labels_ru = self._get_images_and_labels(
            dataset_directory + "data.json"
        )

        if load_tokenized_files:
            self.tokenized_labels_en = torch.load(
                dataset_directory + "tokenized_labels_en.pt"
            )
            self.tokenized_labels_ru = torch.load(
                dataset_directory + "tokenized_labels_ru.pt"
            )

        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            tokenized_labels_en = tokenizer(
                self.labels_en,
                max_length=max_sequence_length,
                return_token_type_ids=False,
                padding=True,
                truncation=True,
            )
            tokenized_labels_ru = tokenizer(
                self.labels_ru,
                max_length=max_sequence_length,
                return_token_type_ids=False,
                padding=True,
                truncation=True,
            )

            self.tokenized_labels_en = {
                key: torch.tensor(value) for key, value in tokenized_labels_en.items()
            }
            self.tokenized_labels_ru = {
                key: torch.tensor(value) for key, value in tokenized_labels_ru.items()
            }

            if save_tokenized_files:
                torch.save(
                    self.tokenized_labels_en,
                    dataset_directory + "tokenized_labels_en.pt",
                )
                torch.save(
                    self.tokenized_labels_ru,
                    dataset_directory + "tokenized_labels_ru.pt",
                )

    def __len__(self) -> int:
        return len(self.images_ids)

    def __getitem__(self, idx: int) -> dict:
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
        image_path = self.dataset_directory + "images/" + self.images_ids[idx]
        image = Image.open(image_path)
        image = self.transforms(image=image)["image"]
        return torch.tensor(image).permute(2, 0, 1).float()

    def get_texts(self, indices: Tensor) -> dict[str, dict[str, Tensor]]:
        return {
            "en": {
                "input_ids": self.tokenized_labels_en["input_ids"][indices],
                "attention_mask": self.tokenized_labels_en["attention_mask"][indices],
            },
            "ru": {
                "input_ids": self.tokenized_labels_ru["input_ids"][indices],
                "attention_mask": self.tokenized_labels_ru["attention_mask"][indices],
            },
        }

    @staticmethod
    def _get_images_and_labels(filename: str) -> tuple[list[str], list[str], list[str]]:
        images_ids = []
        labels_en = []
        labels_ru = []

        with open(filename, "r") as file:
            data = json.load(file)

        for idx, i in enumerate(data):
            images_ids.append(i["image_id"])
            labels_en.append(i["caption_description_en"])
            labels_ru.append(i["caption_description_ru"])

        return images_ids, labels_en, labels_ru
