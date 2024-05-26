import os
from torchvision.datasets import CIFAR10, CIFAR100
import json


def main():
    for name in [
        "cifar100/train/images",
        "cifar100/test/images",
        "cifar10/train/images",
        "cifar10/test/images",
    ]:
        if not os.path.exists(name):
            os.makedirs(name)

    # cifar100
    dataset100 = CIFAR100(root=".", train=True, download=True)
    classes_en = dataset100.classes
    classes_en = [i.replace("_", " ") for i in classes_en]
    with open("labels-ru/" + "cifar100" + "-labels-ru.json") as f:
        classes_ru = json.load(f)
        classes_ru = [i.replace("_", " ") for i in classes_ru]

    data = []
    for i, (img, num) in enumerate(dataset100):
        img.save("cifar100/train/images/" + str(i) + ".jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_ru": classes_ru[num],
            }
        )
    with open("cifar100/train/data.json", "w") as f:
        json.dump(data, f, indent=4)
    data = []
    dataset100 = CIFAR100(root=".", train=False, download=True)
    for i, (img, num) in enumerate(dataset100):
        img.save("cifar100/test/images/" + str(i) + ".jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_ru": classes_ru[num],
            }
        )
    with open("cifar100/test/data.json", "w") as f:
        json.dump(data, f, indent=4)

    # cifar10
    dataset10 = CIFAR10(root=".", train=True, download=True)
    classes_en = dataset10.classes
    classes_en = [i.replace("_", " ") for i in classes_en]
    with open("labels-ru/" + "cifar10" + "-labels-ru.json") as f:
        classes_ru = json.load(f)
        classes_ru = [i.replace("_", " ") for i in classes_ru]

    data = []
    for i, (img, num) in enumerate(dataset10):
        img.save("cifar10/train/images/" + str(i) + ".jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_ru": classes_ru[num],
            }
        )
    with open("cifar10/train/data.json", "w") as f:
        json.dump(data, f, indent=4)
    data = []
    dataset10 = CIFAR10(root=".", train=False, download=True)
    for i, (img, num) in enumerate(dataset10):
        img.save("cifar10/test/images/" + str(i) + ".jpg")
        data.append(
            {
                "image_id": str(i),
                "caption_eng": classes_en[num],
                "caption_ru": classes_ru[num],
            }
        )
    with open("cifar10/test/data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
