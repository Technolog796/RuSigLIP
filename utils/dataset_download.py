import os
import sys
from torchvision import datasets
import json


def main(dataset_name, dataset_dir):
    if not os.path.exists(dataset_dir + "/train/images"):
        os.makedirs(dataset_dir + "/train/images")

    dataset = getattr(datasets, dataset_name)(root=".", download=True)
    classes_en = dataset.classes
    classes_en = [i.replace("_", " ") for i in classes_en]
    #with open("labels-ru/" + "cifar100" + "-labels-ru.json") as f:
    #    classes_ru = json.load(f)
    #    classes_ru = [i.replace("_", " ") for i in classes_ru]

    data = []
    for i, (img, num) in enumerate(dataset):
        img.save(dataset_dir + "/train/images/" + str(i) + ".jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_rus": "",
            }
        )
    with open(dataset_dir + "/train/data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dataset_name> <dataset_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
    