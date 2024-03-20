from torch.utils.data import Dataset

import json
import albumentations as A
import cv2
import torch
from PIL import Image


def get_images_and_labels(data_file):
    images_path = []
    labels = []
    with open(data_file, "r") as f:
        data = json.load(f)
    for i in data:
        images_path.append(i["image"])
        labels.append(i["label"])
    return images_path, labels


class RuSigLIPDataset(Dataset):
    def __init__(
        self, data_file=None, tokenizer=None, target_size=256, max_len=512
    ) -> None:
        super().__init__()
        self.data = data_file
        self.target_size = target_size
        self.max_len = max_len

        self.images_path, self.labels = get_images_and_labels(data_file)

        self.tokenized_labels = tokenizer(
            self.labels, padding=True, truncation=True, max_length=self.max_len
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
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.tokenized_labels.items()
        }

        image = cv2.imread(self.images_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]

        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["label"] = self.labels[idx]

        return item