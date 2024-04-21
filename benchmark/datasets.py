from tensorflow.keras.datasets import cifar10, cifar100
from torchvision.datasets import DTD
from torchvision import transforms
import json

all_datasets = ["cifar10", "cifar100", "dtd"]


def get_dataset(dataset_name, split="test", language="en"):
    if dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        if language == "ru":
            labels = ["самолет", "автомобиль", "птица", "кот", "олень", "собака", "лягушка", "лошадь", "корабль", "грузовик"]
        else:
            labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if split == "train":
            return X_train, y_train, labels
        else:
            return X_test, y_test, labels
    elif dataset_name == "cifar100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        if language == "ru":
            with open("cifar-100-labels-ru.json") as f:
                labels = json.load(f)
        else:
            with open("cifar-100-labels.json") as f:
                labels = json.load(f)
        if split == "train":
            return X_train, y_train, labels
        else:
            return X_test, y_test, labels
    elif dataset_name == "dtd":
        dataset = DTD(root='.', split=split, download=True, transform=transforms.ToTensor())
        lables = dataset.classes #TODO translate to ru
        X = []
        y = []
        for i in range(len(dataset)):
            X.append(dataset[i][0].numpy().T)
            y.append(dataset[i][1])
        return X, y, lables
