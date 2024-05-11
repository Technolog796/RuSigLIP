from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    Food101,
    OxfordIIITPet,
    GTSRB,
    MNIST,
    SUN397,
    Country211,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
)
import torchvision
import json
from typing import Tuple, List, Union

all_datasets = [
    "cifar10",
    "cifar100",
    "dtd",
    "food101",
    "oxfordiiitpet",
    "mnist",
    "country211",
    "fgvcircraft",
    "flowers102",
]


def get_dataset(
    dataset_name: str, split: str = "test", language: str = "en"
) -> Tuple[
    Union[
        torchvision.datasets.cifar.CIFAR10,
        torchvision.datasets.cifar.CIFAR100,
        torchvision.datasets.dtd.DTD,
        torchvision.datasets.food101.Food101,
        torchvision.datasets.oxford_iiit_pet.OxfordIIITPet,
        torchvision.datasets.mnist.MNIST,
        torchvision.datasets.country211.Country211,
        torchvision.datasets.fgvc_aircraft.FGVCAircraft,
        torchvision.datasets.flowers102.Flowers102,
    ],
    List[str],
]:
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=".", download=True)
    elif dataset_name == "cifar100":
        dataset = CIFAR100(root=".", download=True)
    elif dataset_name == "dtd":
        dataset = DTD(root=".", split=split, download=True)
    elif dataset_name == "food101":
        dataset = Food101(root=".", split=split, download=True)
    elif dataset_name == "oxfordiiitpet":
        dataset = OxfordIIITPet(root=".", split=split, download=True)
    elif dataset_name == "mnist":
        dataset = MNIST(root=".", download=True)
    elif dataset_name == "country211":
        dataset = Country211(root=".", split=split, download=True)
    elif dataset_name == "eurosat":
        dataset = EuroSAT(root=".", download=True)
    elif dataset_name == "fgvcircraft":
        dataset = FGVCAircraft(root=".", split=split, download=True)
    elif dataset_name == "flowers102":
        dataset = Flowers102(root=".", split=split, download=True)

    if language == "ru":
        with open("./labels-ru/" + dataset_name + "-labels-ru.json") as f:
            labels = json.load(f)
    else:
        if dataset_name in ["flowers102", "country211"]:
            with open("./labels-en/" + dataset_name + "-labels-en.json") as f:
                labels = json.load(f)
        else:
            labels = dataset.classes
    return dataset, labels
