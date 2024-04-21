import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from datasets import get_dataset
from datasets import all_datasets
from transformers import AutoProcessor, AutoModel


def accuracy(true, probs, k=1):
    right = 0
    for name, p in zip(true, probs):
        _, top = p.topk(k, dim=-1)
        right += int(name in top.numpy())
    return right / len(true)


def classification(X, y, labels, model_path="google/siglip-base-patch16-224", size=-1):
    if size == -1:
        size = X.shape[0]
    probs = []
    true = []

    model = AutoModel.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    for i in range(size):
        inputs = processor(
            text=labels, images=X[i], padding="max_length", return_tensors="pt"
        )
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs.append(torch.sigmoid(logits_per_image))
        true.append(y[i])
    return true, probs


def evaluate(model, dataset_name="all", k=1):
    result = {}
    if dataset_name == "all":
        for name in all_datasets:
            X, y, labels = get_dataset(name)
            true, probs = classification(X, y, labels, model, size=10)
            result[name] = accuracy(true, probs, k)
    else:
        X, y, labels = get_dataset(dataset_name)
        true, probs = classification(X, y, labels, model, size=10)
        result[dataset_name] = accuracy(true, probs, k)
    return result
