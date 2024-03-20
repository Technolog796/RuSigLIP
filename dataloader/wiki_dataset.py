import random

from torch.utils.data import Dataset

import json
import albumentations as A
import cv2
import torch
from PIL import Image
from skimage import io
import numpy as np


def load_img_from_url(image_url):
    array = io.imread(image_url).astype(np.uint8)
    return array


def get_images_and_labels(data_file):
    images_path = []
    labels_en = []
    labels_ru = []
    with open(data_file, "r") as f:
        data = json.load(f)
    for i in data:
        if i["image_url"][-3:] == "svg":
            continue
        images_path.append(i["image_url"])
        labels_en.append(i["caption_description_en"])
        labels_ru.append(i["caption_description_ru"])
    return images_path[:30], labels_en[:30], labels_ru[:30]


class RuSigLIPDataset(Dataset):
    def __init__(
        self, data_file=None, tokenizer=None, target_size=256, max_len=512
    ) -> None:
        super().__init__()
        self.data = data_file
        self.target_size = target_size
        self.max_len = max_len

        self.images_path, self.labels_en, self.labels_ru = get_images_and_labels(data_file)

        self.tokenized_labels_en = tokenizer(
            self.labels_en, padding=True, truncation=True, max_length=self.max_len
        )
        self.tokenized_labels_ru = tokenizer(
            self.labels_ru, padding=True, truncation=True, max_length=self.max_len
        )

        self.transforms = A.Compose(
            [
                A.Resize(self.target_size, self.target_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
            ]
        )

    def read_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((self.target_size, self.target_size))
        return image

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.tokenized_labels_en.items()
        }

        image = load_img_from_url(self.images_path[idx])

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 2:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = self.transforms(image=image)["image"]

        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["label"] = self.labels_en[idx]

        return item

    def get_image(self, idx):
        image = load_img_from_url(self.images_path[idx])

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 2:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = self.transforms(image=image)["image"]

        return torch.tensor(image).permute(2, 0, 1).float()

    def get_input_ids(self, idx):
        return torch.tensor(self.tokenized_labels_en["input_ids"][idx]), \
               torch.tensor(self.tokenized_labels_ru["input_ids"][idx])

    def get_attention_mask(self, idx):
        return torch.tensor(self.tokenized_labels_en["attention_mask"][idx]), \
               torch.tensor(self.tokenized_labels_ru["attention_mask"][idx])
