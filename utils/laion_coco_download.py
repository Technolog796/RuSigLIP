from datasets import load_dataset
from tqdm.contrib import tzip
import json
import requests
import os

train_dataset = load_dataset("visheratin/laion-coco-nllb", split="train")
val_dataset = load_dataset("visheratin/laion-coco-nllb", split="test")


def format_link(id: str) -> str:
    return f"https://nllb-data.com/{id}.jpg"


def main():
    skip_images_train = 0
    skip_images_val = 0

    valid_json_train = []
    valid_json_val = []

    if not os.path.exists("laion-coco-nllb/train_images"):
        os.makedirs("laion-coco-nllb/train_images")

    if not os.path.exists("laion-coco-nllb/test_images"):
        os.makedirs("laion-coco-nllb/test_images")

    for id, en_cap, rus_cap in tzip(
        train_dataset["id"],
        train_dataset["caption_en"],
        train_dataset["captions"]["rus_Cyrl"],
    ):
        try:
            img = requests.get(format_link(id)).content
            with open(f"laion-coco-nllb/train_images/{id}.jpg", "wb") as f:
                f.write(img)
            valid_json_train.append(
                {"image_id": id, "caption": en_cap, "caption_rus": rus_cap}
            )

        except Exception as e:
            print(f"Error: {e}")
            skip_images_train += 1

    print(f"Skipped {skip_images_train} images")

    for id, en_cap, rus_cap in tzip(
        val_dataset["id"],
        val_dataset["caption_en"],
        val_dataset["captions"]["rus_Cyrl"],
    ):
        try:
            img = requests.get(format_link(id)).content
            with open(f"laion-coco-nllb/test_images/{id}.jpg", "wb") as f:
                f.write(img)
            valid_json_val.append(
                {"image_id": id, "caption": en_cap, "caption_rus": rus_cap}
            )

        except Exception as e:
            print(f"Error: {e}")
            skip_images_val += 1

    print(f"Skipped {skip_images_val} images")

    with open("laion-coco-nllb/train.json", "w") as f:
        json.dump(valid_json_train, f, indent=4)

    with open("laion-coco-nllb/test.json", "w") as f:
        json.dump(valid_json_val, f, indent=4)


if __name__ == "__main__":
    main()
