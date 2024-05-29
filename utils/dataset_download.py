import os
import sys
import json

from torchvision import datasets


def main(dataset_name: str, dataset_dir: str) -> None:
    if not os.path.exists(dataset_dir + "/images"):
        os.makedirs(dataset_dir + "/images")

    dataset = getattr(datasets, dataset_name)(root=".", download=True)
    classes_en = dataset.classes
    classes_en = [i.replace("_", " ") for i in classes_en]

    data = []
    for i, (img, num) in enumerate(dataset):
        img.save(dataset_dir + "/images/" + str(i) + ".jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_rus": "",
            }
        )
    with open(dataset_dir + "/data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dataset_name> <dataset_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
    