from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import json
import requests
import pandas as pd
import os
from pandarallel import pandarallel
from tqdm.auto import tqdm

pandarallel.initialize(progress_bar=True)

def load_dataset_df(split):
    raw_dataset = pd.DataFrame(load_dataset("visheratin/laion-coco-nllb", split=split))
    raw_dataset.drop(columns=["url", "score"], inplace=True)
    raw_dataset["rus_captions"] = raw_dataset["captions"].parallel_apply(
        lambda captions: [caption[1] for caption in captions if caption[0] == "rus_Cyrl"]
    )
    dataset = raw_dataset[raw_dataset["rus_captions"].parallel_apply(lambda x: len(x) != 0)].copy()
    del raw_dataset
    dataset.loc[:, "rus_captions"] = dataset.rus_captions.parallel_apply(lambda x: x[0])
    dataset.drop(columns=["captions"], inplace=True)
    return dataset

train_dataset = load_dataset_df("train")
val_dataset = load_dataset_df("test")

def format_link(id: str) -> str:
    return f"https://nllb-data.com/{id}.jpg"

def download_image(data, folder):
    id, en_cap, rus_cap = data
    try:
        img = requests.get(format_link(id), timeout=10).content
        with open(f"{folder}/{id}.jpg", "wb") as f:
            f.write(img)
        return {"image_id": id, "caption_eng": en_cap, "caption_rus": rus_cap}, None
    except Exception as e:
        return None, e

def download_images(dataset, folder):
    skipped_images = 0
    valid_json = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_image, data, folder): data for data in zip(
            dataset["id"], dataset["eng_caption"], dataset["rus_captions"])}

        # Set up tqdm progress bar
        progress = tqdm(as_completed(futures), total=len(futures), desc="Downloading images")
        for future in progress:
            result, error = future.result()
            if error:
                skipped_images += 1
            else:
                valid_json.append(result)

    return valid_json, skipped_images

def main():
    if not os.path.exists("laion-coco-nllb/train_images"):
        os.makedirs("laion-coco-nllb/train_images")
    if not os.path.exists("laion-coco-nllb/test_images"):
        os.makedirs("laion-coco-nllb/test_images")

    valid_json_train, skip_images_train = download_images(train_dataset, "laion-coco-nllb/train_images")
    print(f"Skipped {skip_images_train} images")

    valid_json_val, skip_images_val = download_images(val_dataset, "laion-coco-nllb/test_images")
    print(f"Skipped {skip_images_val} images")

    with open("laion-coco-nllb/train.json", "w") as f:
        json.dump(valid_json_train, f, indent=4)

    with open("laion-coco-nllb/test.json", "w") as f:
        json.dump(valid_json_val, f, indent=4)

if __name__ == "__main__":
    main()