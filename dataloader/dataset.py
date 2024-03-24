from torch.utils.data import Dataset
from torch import Tensor

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


def get_images_and_labels(data_file, transforms):
    images_path = []
    labels_en = []
    labels_ru = []
    with open(data_file, "r") as f:
        data = json.load(f)
    j = 0
    k = 100
    skip_list = {411, 258, 515, 6, 262, 8, 520, 266, 521, 269, 14, 15, 16, 271, 18, 525, 20, 21, 276, 278, 527, 529, 27, 284, 30, 292, 38, 298, 43, 45, 301, 47, 308, 314, 60, 62, 319, 320, 321, 67, 324, 71, 73, 74, 329, 330, 335, 338, 83, 340, 341, 86, 343, 90, 94, 98, 100, 101, 357, 103, 106, 366, 111, 112, 367, 372, 117, 376, 377, 378, 126, 384, 131, 387, 133, 134, 389, 137, 140, 141, 144, 400, 404, 405, 407, 155, 157, 415, 161, 417, 422, 168, 429, 434, 179, 180, 181, 182, 438, 437, 440, 187, 443, 444, 190, 191, 449, 452, 453, 458, 459, 204, 460, 466, 215, 216, 217, 218, 472, 474, 483, 229, 232, 234, 235, 236, 490, 492, 239, 240, 242, 501, 503, 510}
    for idx, i in enumerate(data):
        if idx in skip_list:
            j += 1
            continue
        print(idx)
        #print(skip_list)
        if idx - j == k:
            break
        if i["image_url"][-3:] == "svg":
            skip_list |= {idx}
            j += 1
            continue
        try:
            image = load_img_from_url(i["image_url"])

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 2:
                image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            image = transforms(image=image)["image"]

            image = torch.tensor(image).permute(2, 0, 1).float()
        except Exception as e:
            skip_list |= {idx}
            j += 1
            continue
        if not isinstance(i["caption_description_ru"], str):
            skip_list |= {idx}
            j += 1
            continue
        images_path.append(image)
        # images_path.append(i["image_url"])
        labels_en.append(i["caption_description_en"])
        labels_ru.append(i["caption_description_ru"])
    print(f"Skipped: {j}")
    print(skip_list)
    return images_path, labels_en, labels_ru


class RuSigLIPDataset(Dataset):
    def __init__(
        self, data_file=None, tokenizer=None, target_size=256, max_len=64
    ) -> None:
        super().__init__()
        self.data = data_file
        self.target_size = target_size
        self.max_len = max_len

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

        self.images_path, self.labels_en, self.labels_ru = get_images_and_labels(data_file, self.transforms)

        tokenized_labels_en = tokenizer(
            self.labels_en, padding=True, truncation=True, max_length=self.max_len, return_token_type_ids=False
        )

        tokenized_labels_ru = tokenizer(
            self.labels_ru, padding=True, truncation=True, max_length=self.max_len, return_token_type_ids=False
        )

        self.tokenized_labels_en = {}
        self.tokenized_labels_ru = {}

        for key, value in tokenized_labels_en.items():
            self.tokenized_labels_en[key] = torch.tensor(value)
        for key, value in tokenized_labels_ru.items():
            self.tokenized_labels_ru[key] = torch.tensor(value)

        torch.save(self.tokenized_labels_en, "datasets/tokenized_labels_en.pt")
        torch.save(self.tokenized_labels_ru, "datasets/tokenized_labels_ru.pt")

        # self.tokenized_labels_en = torch.load("datasets/tokenized_labels_en.pt")
        # self.tokenized_labels_ru = torch.load("datasets/tokenized_labels_ru.pt")

    def read_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((self.target_size, self.target_size))
        return image

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, idx) -> dict:
        return {"image": self.get_image(idx),
                "label_en": self.labels_en[idx],
                "label_ru": self.labels_ru[idx],
                "input_ids_en": self.tokenized_labels_en["input_ids"][idx],
                "input_ids_ru": self.tokenized_labels_ru["input_ids"][idx],
                "attention_mask_en": self.tokenized_labels_en["attention_mask"][idx],
                "attention_mask_ru": self.tokenized_labels_ru["attention_mask"][idx]}

    def get_image(self, idx) -> Tensor:
        return self.images_path[idx]
        # image = load_img_from_url(self.images_path[idx])
        #
        # if len(image.shape) == 2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # elif image.shape[2] == 2:
        #     image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        # elif image.shape[2] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # elif image.shape[2] == 4:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        #
        # image = self.transforms(image=image)["image"]
        #
        # return torch.tensor(image).permute(2, 0, 1).float()

    def get_texts(self, indices) -> dict[str, dict[str, Tensor]]:
        text_data = {"en": {"input_ids": self.tokenized_labels_en["input_ids"][indices],
                            "attention_mask": self.tokenized_labels_en["attention_mask"][indices]},
                     "ru": {"input_ids": self.tokenized_labels_ru["input_ids"][indices],
                            "attention_mask": self.tokenized_labels_ru["attention_mask"][indices]}}
        return text_data

