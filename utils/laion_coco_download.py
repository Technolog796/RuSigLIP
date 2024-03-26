from datasets import load_dataset
from tqdm.contrib import tzip
import json
import requests
import pandas as pd
import os
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


raw_train_dataset = pd.DataFrame(load_dataset("visheratin/laion-coco-nllb", split="train"))
raw_train_dataset.drop(columns=['url', 'score'], inplace=True)
raw_train_dataset["rus_captions"] = raw_train_dataset["captions"].parallel_apply(lambda x: [x[i][1] for i in range(len(x)) if x[i][0] == 'rus_Cyrl'])
train_dataset = raw_train_dataset[raw_train_dataset["rus_captions"].parallel_apply(lambda x: len(x) != 0)].copy()
del raw_train_dataset
train_dataset.loc[:, "rus_captions"] = train_dataset.rus_captions.parallel_apply(lambda x : x[0])
train_dataset.drop(columns=["captions"], inplace=True)



raw_val_dataset = pd.DataFrame(load_dataset("visheratin/laion-coco-nllb", split="test"))
raw_val_dataset.drop(columns=['url', 'score'], inplace=True)
raw_val_dataset["rus_captions"] = raw_val_dataset["captions"].parallel_apply(lambda x: [x[i][1] for i in range(len(x)) if x[i][0] == 'rus_Cyrl'])
val_dataset = raw_val_dataset[raw_val_dataset["rus_captions"].parallel_apply(lambda x: len(x) != 0)].copy()
del raw_val_dataset
val_dataset.loc[:, "rus_captions"] = val_dataset.rus_captions.parallel_apply(lambda x : x[0])
val_dataset.drop(columns=["captions"], inplace=True)



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
        train_dataset["eng_caption"],
        train_dataset["rus_captions"],
    ):
        try:
            img = requests.get(format_link(id)).content
            with open(f"laion-coco-nllb/train_images/{id}.jpg", "wb") as f:
                f.write(img)
            valid_json_train.append(
                {"image_id": id, "caption_eng": en_cap, "caption_rus": rus_cap}
            )

        except Exception as e:
            print(f"Error: {e}")
            skip_images_train += 1

    print(f"Skipped {skip_images_train} images")

    for id, en_cap, rus_cap in tzip(
        val_dataset["id"],
        val_dataset["eng_caption"],
        val_dataset["rus_captions"],
    ):
        try:
            img = requests.get(format_link(id)).content
            with open(f"laion-coco-nllb/test_images/{id}.jpg", "wb") as f:
                f.write(img)
            valid_json_val.append(
                {"image_id": id, "caption_eng": en_cap, "caption_rus": rus_cap}
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
