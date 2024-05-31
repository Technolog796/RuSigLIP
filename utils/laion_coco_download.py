import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import json
import requests
import pandas as pd
import os
from pandarallel import pandarallel
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Optional

# Initialize pandarallel for parallel processing of data
pandarallel.initialize(progress_bar=True)


def load_dataset_df(split: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset split.

    Args:
    split (str): The dataset split to load ('train' or 'test').

    Returns:
    pd.DataFrame: The preprocessed dataset.
    """
    raw_dataset = pd.DataFrame(load_dataset("visheratin/laion-coco-nllb", split=split))
    raw_dataset.drop(columns=["url", "score"], inplace=True)
    raw_dataset["rus_captions"] = raw_dataset["captions"].parallel_apply(
        lambda captions: [
            caption[1] for caption in captions if caption[0] == "rus_Cyrl"
        ]
    )
    dataset = raw_dataset[
        raw_dataset["rus_captions"].parallel_apply(lambda x: len(x) != 0)
    ].copy()
    del raw_dataset
    dataset.loc[:, "rus_captions"] = dataset.rus_captions.parallel_apply(lambda x: x[0])
    dataset.drop(columns=["captions"], inplace=True)
    return dataset


def format_link(id: str) -> str:
    """
    Format the image link based on the provided ID.

    Args:
    id (str): The ID of the image.

    Returns:
    str: The formatted URL of the image.
    """
    return f"https://nllb-data.com/{id}.jpg"


def download_image(
    data: Tuple[str, str, str], folder: str
) -> Tuple[Optional[Dict[str, str]], Optional[Exception]]:
    """
    Download an image and save it to the specified folder.

    Args:
    data (Tuple[str, str, str]): A tuple containing image ID, English caption, and Russian caption.
    folder (str): The folder to save the downloaded image.

    Returns:
    Tuple[Optional[Dict[str, str]], Optional[Exception]]: The image data and any exception occurred.
    """
    id, en_cap, rus_cap = data
    try:
        img = requests.get(format_link(id), timeout=10).content
        with open(f"{folder}/{id}.jpg", "wb") as f:
            f.write(img)
        return {"image_id": id, "caption_eng": en_cap, "caption_rus": rus_cap}, None
    except Exception as e:
        return None, e


def download_images(
    dataset: pd.DataFrame, folder: str
) -> Tuple[List[Dict[str, str]], int]:
    """
    Download images from the dataset and save them to the specified folder.

    Args:
    dataset (pd.DataFrame): The dataset containing image IDs and captions.
    folder (str): The folder to save the downloaded images.

    Returns:
    Tuple[List[Dict[str, str]], int]: List of valid image data and the count of skipped images.
    """
    skipped_images = 0
    valid_json = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(download_image, data, folder): data
            for data in zip(
                dataset["id"], dataset["eng_caption"], dataset["rus_captions"]
            )
        }

        # Use tqdm to track progress
        progress = tqdm(
            as_completed(futures), total=len(futures), desc="Downloading images"
        )
        for future in progress:
            result, error = future.result()
            if error:
                skipped_images += 1
            else:
                valid_json.append(result)

    return valid_json, skipped_images


def main(train_folder: str, test_folder: str) -> None:
    """
    Main function to download and save images and their captions from the dataset.

    Args:
    train_folder (str): The folder to save training images.
    test_folder (str): The folder to save testing images.
    """
    # Check and create directories if not exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    train_dataset = load_dataset_df("train")
    val_dataset = load_dataset_df("test")

    valid_json_train, skip_images_train = download_images(train_dataset, train_folder)
    print(f"Skipped {skip_images_train} images")

    valid_json_val, skip_images_val = download_images(val_dataset, test_folder)
    print(f"Skipped {skip_images_val} images")

    # Save valid image data to JSON files
    with open(os.path.join(train_folder, "train.json"), "w") as f:
        json.dump(valid_json_train, f, indent=4)

    with open(os.path.join(test_folder, "test.json"), "w") as f:
        json.dump(valid_json_val, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess dataset images."
    )
    parser.add_argument(
        "--train_folder", type=str, required=True, help="Folder to save training images"
    )
    parser.add_argument(
        "--test_folder", type=str, required=True, help="Folder to save testing images"
    )
    args = parser.parse_args()
    main(args.train_folder, args.test_folder)
