from torchvision.datasets import CIFAR10, CIFAR100, DTD, Food101, OxfordIIITPet
import json

all_datasets = ["cifar10", "cifar100", "dtd", "food101", "oxfordiiitpet"]


def get_dataset(dataset_name, split="test", language="en"):
    if dataset_name == "cifar10":
        try:
            dataset = CIFAR10(root='.', download=False)
        except:
            dataset = CIFAR10(root='.', download=True)
    elif dataset_name == "cifar100":
        try:
            dataset = CIFAR100(root='.', download=False)
        except:
            dataset = CIFAR100(root='.', download=True)
    elif dataset_name == "dtd":
        try:
            dataset = DTD(root='.', split=split, download=False)
        except:
            dataset = DTD(root='.', split=split, download=True)
    elif dataset_name == "food101":
        try:
            dataset = Food101(root='.', split=split, download=False)
        except:
            dataset = Food101(root='.', split=split, download=True)
    elif dataset_name == "oxfordiiitpet":
        try:
            dataset = OxfordIIITPet(root='.', split=split, download=False)
        except:
            dataset = OxfordIIITPet(root='.', split=split, download=True)
    if language == "ru":
        with open(dataset_name + "-labels-ru.json") as f:
            labels = json.load(f)
    else:
        labels = dataset.classes
    '''X = []
    y = []
    for i in range(len(dataset)):
        X.append(cv2.rotate(dataset[i][0].numpy().T, cv2.ROTATE_90_CLOCKWISE))
        # X.append(dataset[i][0].numpy().T)
        y.append(dataset[i][1]) '''
    return dataset, labels
