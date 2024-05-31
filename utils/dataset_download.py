import os
import sys
import json
from typing import List, Dict, Any
from torchvision import datasets
from torchvision.datasets import VisionDataset


def main(dataset_name: str, dataset_dir: str) -> None:
    """
    Main function to download a dataset, save images and metadata.

    Args:
        dataset_name (str): The name of the dataset to download from torchvision.
        dataset_dir (str): The directory to save images and metadata.
    """
    images_dir: str = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    dataset: VisionDataset = getattr(datasets, dataset_name)(root=".", download=True)
    classes_en: List[str] = [cls.replace("_", " ") for cls in dataset.classes]

    data: List[Dict[str, Any]] = []

    for i, (img, num) in enumerate(dataset):
        img_path: str = os.path.join(images_dir, f"{i}.jpg")
        img.save(img_path)

        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_rus": "",
            }
        )

    json_path: str = os.path.join(dataset_dir, "data.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    """
    Entry point for the script.
    """
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dataset_name> <dataset_dir>")
        sys.exit(1)

    dataset_name: str = sys.argv[1]
    dataset_dir: str = sys.argv[2]

    main(dataset_name, dataset_dir)
