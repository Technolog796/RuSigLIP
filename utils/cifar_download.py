import os
import json
from typing import List, Dict, Any
from torchvision.datasets import CIFAR10, CIFAR100
import argparse


def create_directories() -> None:
    """
    Create directories if they do not exist.
    """
    for name in [
        "cifar100/train/images",
        "cifar100/test/images",
        "cifar10/train/images",
        "cifar10/test/images",
    ]:
        if not os.path.exists(name):
            os.makedirs(name)


def load_labels(dataset_name: str) -> List[str]:
    """
    Load labels from a JSON file.

    :param dataset_name: Name of the dataset (cifar10 or cifar100)
    :return: List of labels in Russian
    """
    with open(f"labels-ru/{dataset_name}-labels-ru.json") as f:
        classes_ru = json.load(f)
    return [label.replace("_", " ") for label in classes_ru]


def save_images_and_data(
    dataset: Any,
    dataset_name: str,
    train: bool,
    classes_en: List[str],
    classes_ru: List[str],
) -> None:
    """
    Save images and metadata from the dataset.

    :param dataset: The dataset (CIFAR10 or CIFAR100)
    :param dataset_name: Name of the dataset (cifar10 or cifar100)
    :param train: Boolean indicating if it's training data
    :param classes_en: List of labels in English
    :param classes_ru: List of labels in Russian
    """
    subset = "train" if train else "test"
    data: List[Dict[str, Any]] = []

    for i, (img, num) in enumerate(dataset):
        img.save(f"{dataset_name}/{subset}/images/{i}.jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_rus": classes_ru[num],
            }
        )

    with open(f"{dataset_name}/{subset}/data.json", "w") as f:
        json.dump(data, f, indent=4)


def main(train: bool) -> None:
    """
    Main function to process CIFAR10 and CIFAR100 datasets.

    :param train: Boolean indicating if it's training data
    """
    create_directories()

    dataset100 = CIFAR100(root=".", train=train, download=True)
    classes_en100 = [label.replace("_", " ") for label in dataset100.classes]
    classes_ru100 = load_labels("cifar100")
    save_images_and_data(dataset100, "cifar100", train, classes_en100, classes_ru100)

    dataset10 = CIFAR10(root=".", train=train, download=True)
    classes_en10 = [label.replace("_", " ") for label in dataset10.classes]
    classes_ru10 = load_labels("cifar10")
    save_images_and_data(dataset10, "cifar10", train, classes_en10, classes_ru10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CIFAR10 and CIFAR100 datasets."
    )
    parser.add_argument("--train", action="store_true", help="Process training data")
    args = parser.parse_args()
    main(args.train)
