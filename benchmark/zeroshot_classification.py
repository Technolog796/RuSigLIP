import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import numpy as np
from datasets_get import get_dataset
from datasets_get import all_datasets
from transformers import AutoProcessor, AutoModel
from typing import Dict, List, Tuple, Union
import torchvision


def accuracy(true: List[int], probs: List[torch.Tensor], k: int = 1) -> float:
    right = 0
    for name, p in zip(true, probs):
        _, top = p.topk(k, dim=-1)
        right += int(name in top.numpy())
    return right / len(true)


def precision_macro(names: List[int], probs: List[torch.Tensor], k: int = 1) -> float:
    num_of_classes = len(np.unique(names))
    res = 0
    for c in np.unique(names).tolist():
        true = [1 if x == c else 0 for x in names]
        pred = [1 if c in (p.topk(k=k, dim=-1)[1]).numpy() else 0 for p in probs]
        tp = sum([1 if t == 1 and p == 1 else 0 for t, p in zip(true, pred)])
        fp = sum([1 if t == 0 and p == 1 else 0 for t, p in zip(true, pred)])
        res += tp / (tp + fp) if (tp + fp) != 0 else 0
    return res / num_of_classes


def recall_macro(names: List[int], probs: List[torch.Tensor], k: int = 1) -> float:
    num_of_classes = len(np.unique(names))
    res = 0
    for c in np.unique(names).tolist():
        true = [1 if x == c else 0 for x in names]
        pred = [1 if c in (p.topk(k=k, dim=-1)[1]).numpy() else 0 for p in probs]
        tp = sum([1 if t == 1 and p == 1 else 0 for t, p in zip(true, pred)])
        fn = sum([1 if t == 1 and p == 0 else 0 for t, p in zip(true, pred)])
        res += tp / (tp + fn) if (tp + fn) != 0 else 0
    return res / num_of_classes


def f1_macro(names: List[int], probs: List[torch.Tensor], k: int = 1) -> float:
    p = precision_macro(names, probs, k)
    r = recall_macro(names, probs, k)
    return 0 if p + r == 0 else (2 * p * r) / (p + r)


def classification_siglip(
    dataset: Union[
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
    labels: List[str],
    size: int = -1,
) -> Tuple[List[int], List[torch.Tensor]]:
    if size == -1 or size > len(dataset):
        size = len(dataset)
    probs = []
    true = []

    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    for i in range(size):
        inputs = processor(
            text=labels, images=dataset[i][0], padding="max_length", return_tensors="pt"
        )
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs.append(torch.sigmoid(logits_per_image))
        true.append(dataset[i][1])
    return true, probs


def classification_clip(
    dataset: Union[
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
    labels: List[str],
    size: int = -1,
) -> Tuple[List[int], List[torch.Tensor]]:
    if size == -1 or size > len(dataset):
        size = len(dataset)
    probs = []
    true = []

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
    for i in range(size):
        inputs = processor(
            text=labels,
            images=dataset[i][0],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs.append(logits_per_image.softmax(dim=1))
        true.append(dataset[i][1])
    return true, probs


def evaluate(
    model: str,
    dataset_name: str = "all",
    split: str = "test",
    size: int = -1,
    language: str = "en",
    k: int = 1,
) -> Dict[str, Dict[str, float]]:
    result = {}
    if dataset_name == "all":
        for name in all_datasets:
            dataset, labels = get_dataset(name, split=split, language=language)
            true, probs = classification_siglip(dataset, labels, size=size)
            result[name] = {}
            result[name]["accuracy"] = accuracy(true, probs, k)
            p = precision_macro(true, probs, k)
            result[name]["precision_macro"] = p
            r = recall_macro(true, probs, k)
            result[name]["recall_macro"] = r
            result[name]["f1_macro"] = 0 if p + r == 0 else (2 * p * r) / (p + r)
    else:
        dataset, labels = get_dataset(dataset_name, split=split, language=language)
        true, probs = classification_siglip(dataset, labels, size=size)
        result[dataset_name] = {}
        result[dataset_name]["accuracy"] = accuracy(true, probs, k)
        p = precision_macro(true, probs, k)
        result[dataset_name]["precision_macro"] = p
        r = recall_macro(true, probs, k)
        result[dataset_name]["recall_macro"] = r
        result[dataset_name]["f1_macro"] = 0 if p + r == 0 else (2 * p * r) / (p + r)
    return result
