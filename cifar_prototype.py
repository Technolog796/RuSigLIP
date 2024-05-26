import json

import torchvision
from torchvision.datasets import CIFAR10, CIFAR100

import numpy as np
from transformers import AutoTokenizer

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Union

from models.main_model import SigLIPModel

from torch.nn.functional import normalize

from dataloader import RuSigLIPDataset

import os


class RuSigLIPDatasetEvaluate(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        dataset_name: str = "cifar100",
        tokenizer_name: str = None,
        load_tokenized_files: bool = True,
        save_tokenized_files: bool = True,
        max_sequence_length: int = 512,
    ):
        super().__init__()

        self.dataset_directory = dataset_directory
        self.dataset_name = dataset_name
        self.transforms = lambda image: {"image": image}

        (
            self.dataset,
            self.labels_en,
            self.labels_ru,
            self.classes_en,
            self.classes_ru,
        ) = self._get_images_and_labels(dataset_name)

        if load_tokenized_files:
            self.tokenized_labels_en = torch.load(
                dataset_directory + "tokenized_labels_en.pt"
            )
            self.tokenized_labels_ru = torch.load(
                dataset_directory + "tokenized_labels_ru.pt"
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
                    dataset_directory + "tokenized_labels_en.pt",
                )
                torch.save(
                    self.tokenized_labels_ru,
                    dataset_directory + "tokenized_labels_ru.pt",
                )

    def __len__(self) -> int:
        return len(self.dataset)

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
        if self.dataset_name == "cifar10":  # https://huggingface.co/datasets/cifar10
            dataset = CIFAR100(root=".", train=False, download=True)
            image = np.asarray(dataset[idx][0])
        elif self.dataset_name == "cifar100":
            dataset = CIFAR100(
                root=".", train=False, download=True
            )  # https://huggingface.co/datasets/cifar100
            image = np.asarray(dataset[idx][0])
        else:
            image = None
        if image is None:
            image = np.zeros((256, 256, 3))
        image = self.transforms(image=image)["image"]
        return torch.tensor(image).permute(2, 0, 1).float()

    def get_texts(self, indices: Tensor, language: str) -> dict[str, Tensor]:
        if language == "en":
            return {
                "input_ids": self.tokenized_labels_en["input_ids"][indices],
                "attention_mask": self.tokenized_labels_en["attention_mask"][indices],
            }
        else:
            return {
                "input_ids": self.tokenized_labels_en["input_ids"][indices],
                "attention_mask": self.tokenized_labels_en["attention_mask"][indices],
            }

    @staticmethod
    def _get_images_and_labels(
        dataset_name: str,
    ) -> tuple[
        Union[torchvision.datasets.cifar.CIFAR10, torchvision.datasets.cifar.CIFAR100],
        list[str],
        list[str],
        list[str],
        list[str],
    ]:
        if dataset_name == "cifar10":
            dataset = CIFAR10(root=".", download=True)
        elif dataset_name == "cifar100":
            dataset = CIFAR100(root=".", download=True)
        else:
            return None, None, None, None, None

        classes_en = dataset.classes
        with open("benchmark/labels-ru/" + dataset_name + "-labels-ru.json") as f:
            classes_ru = json.load(f)

        labels_en = []
        labels_ru = []
        for image, i in dataset:
            labels_en.append(classes_en[i])
            labels_ru.append(classes_ru[i])

        return dataset, labels_en, labels_ru, classes_en, classes_ru


def validate_cifar(
    model_name: str,
    dataset_directory: str,
    dataset_name: str,
    language: str = "en",
    batch_size: int = 16,
    tokenizer_name: str = None,
    load_tokenized_files: bool = True,
    save_tokenized_files: bool = True,
    max_sequence_length: int = 512,
) -> Union[float, None]:
    test_data = RuSigLIPDatasetEvaluate(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        load_tokenized_files=load_tokenized_files,
        save_tokenized_files=save_tokenized_files,
        max_sequence_length=max_sequence_length,
    )
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SigLIPModel()  # Create default initialization
    model.load_state_dict(torch.load(model_name))
    model = model.to(device).eval()

    true = 0
    with torch.no_grad():
        for data in loader:
            image = data["image"]
            if language == "en":
                texts = {
                    "input_ids": data["input_ids_en"],
                    "attention_mask": data["attention_mask_en"],
                }
                labels = data["label_en"]
            elif language == "ru":
                texts = {
                    "input_ids": data["input_ids_ru"],
                    "attention_mask": data["attention_mask_ru"],
                }
                labels = data["label_ru"]
            else:
                return None
            image_embeddings, text_embeddings = model.predict(image, texts)
            z_img = normalize(image_embeddings)
            z_txt = normalize(text_embeddings)

            logits = z_img @ z_txt.T * 10 - 10
            probs = torch.sigmoid(logits)
            predict = [labels[i] for _, i in probs.topk(k=1, dim=-1)]
            for i in range(len(labels)):
                true += predict[i] == labels[i]
    return true / len(test_data)


def validate(
    model_name: str,
    language: str = "en",
    batch_size: int = 32,
    dataset_directory: str = "cifar100/test/",
) -> float | None:
    if not os.path.exists(dataset_directory):
        print(f"Dataset directory {dataset_directory} is not exists")
        return None
    if language not in ["en", "ru"]:
        language = "en"
    test_data = RuSigLIPDataset(dataset_directory=dataset_directory)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SigLIPModel()  # Create default initialization
    if not os.path.isfile(model_name):
        print(f"Model {model_name} is not exists")
        return None
    model.load_state_dict(torch.load(model_name))
    model = model.to(device).eval()

    true = 0
    with torch.no_grad():
        for data in loader:
            image = data["image"]
            if language == "en":
                texts = {
                    "input_ids": data["input_ids_en"],
                    "attention_mask": data["attention_mask_en"],
                }
                labels = data["label_en"]
            elif language == "ru":
                texts = {
                    "input_ids": data["input_ids_ru"],
                    "attention_mask": data["attention_mask_ru"],
                }
                labels = data["label_ru"]
            else:
                return None
            image_embeddings, text_embeddings = model.predict(image, texts)
            z_img = normalize(image_embeddings)
            z_txt = normalize(text_embeddings)

            logits = z_img @ z_txt.T * 10 - 10
            probs = torch.sigmoid(logits)
            predict = [labels[i] for _, i in probs.topk(k=1, dim=-1)]
            for i in range(len(labels)):
                true += predict[i] == labels[i]
    return true / len(test_data)
