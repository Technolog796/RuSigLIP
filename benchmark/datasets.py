from torchvision.datasets import CIFAR10, CIFAR100, DTD, Food101, OxfordIIITPet
import json

all_datasets = ["cifar10", "cifar100", "dtd", "food101", "oxfordiiitpet"]


def get_dataset(dataset_name, split="test", language="en"):
    if dataset_name == "cifar10":
        dataset = CIFAR10(root='.', download=True)
    elif dataset_name == "cifar100":
        dataset = CIFAR100(root='.', download=True)
    elif dataset_name == "dtd":
        dataset = DTD(root='.', split=split, download=True)
    elif dataset_name == "food101":
        dataset = Food101(root='.', split=split, download=True)
    elif dataset_name == "oxfordiiitpet":
        dataset = OxfordIIITPet(root='.', split=split, download=True)
    if language == "ru":
        with open('./labels-ru/' + dataset_name + "-labels-ru.json") as f:
            labels = json.load(f)
    else:
        labels = dataset.classes
    return dataset, labels
