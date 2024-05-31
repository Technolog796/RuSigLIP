import os

from predict import predict
from zeroshot_classification import accuracy
from zeroshot_classification import precision_macro
from zeroshot_classification import recall_macro
from zeroshot_classification import f1_macro


# tests for predict
def test_predict_asserts(capfd):
    pred = predict()
    out, err = capfd.readouterr()
    assert pred is None
    assert out == "Len of list of labels must be > 0\n"

    pred = predict(path="does_not_exist.jpg", labels=["cat", "dog"])
    out, err = capfd.readouterr()
    assert pred is None
    assert out == "File not found\n"

    pred = predict(path="predict.py", labels=["cat", "dog"])
    out, err = capfd.readouterr()
    assert pred is None
    assert out == "Image cannot be opened and identified\n"


def test_predict_result():
    image_path = "image.jpg"
    labels = ["cat", "dog"]
    if os.path.exists(image_path):
        pred = predict(path=image_path, labels=labels)
        assert isinstance(pred, int)
        assert pred in list(range(len(labels)))


# tests for zeroshot_classification
def test_accuracy():
    true = [0, 0, 1, 2, 4]
    probs = [
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
    ]
    acc = accuracy(true=true, probs=probs, k=1)
    assert acc == 0.2
    acc = accuracy(true=true, probs=probs, k=2)
    assert acc == 0.4
    acc = accuracy(true=true, probs=probs, k=3)
    assert acc == 0.4
    acc = accuracy(true=true, probs=probs, k=4)
    assert acc == 0.6
    acc = accuracy(true=true, probs=probs, k=5)
    assert acc == 1


# tests for precision_macro
def test_precision_macro():
    true = [0, 0, 1, 2, 4]
    probs = [
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
    ]
    precision = precision_macro(names=true, probs=probs, k=1)
    assert precision == (0 + 0 + 0 + 1.0 / 5) / 4
    precision = precision_macro(names=true, probs=probs, k=2)
    assert precision == (0 + 0 + 1.0 / 5 + 1.0 / 5) / 4
    precision = precision_macro(names=true, probs=probs, k=3)
    assert precision == (0 + 0 + 1.0 / 5 + 1.0 / 5) / 4
    precision = precision_macro(names=true, probs=probs, k=4)
    assert precision == (0 + 1.0 / 5 + 1.0 / 5 + 1.0 / 5) / 4
    precision = precision_macro(names=true, probs=probs, k=5)
    assert precision == (2.0 / 5 + 1.0 / 5 + 1.0 / 5 + 1.0 / 5) / 4


# tests for precision_macro
def test_recall_macro():
    true = [0, 0, 1, 2, 4]
    probs = [
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
    ]
    recall = recall_macro(names=true, probs=probs, k=1)
    assert recall == (0 + 0 + 0 + 1.0 / 1) / 4
    recall = recall_macro(names=true, probs=probs, k=2)
    assert recall == (0 + 0 + 1.0 / 1 + 1.0 / 1) / 4
    recall = recall_macro(names=true, probs=probs, k=3)
    assert recall == (0 + 0 + 1.0 / 1 + 1.0 / 1) / 4
    recall = recall_macro(names=true, probs=probs, k=4)
    assert recall == (0 + 1.0 / 1 + 1.0 / 1 + 1.0 / 1) / 4
    recall = recall_macro(names=true, probs=probs, k=5)
    assert recall == (2.0 / 2 + 1.0 / 1 + 1.0 / 1 + 1.0 / 1) / 4


# tests for f1_macro
def test_f1_macro():
    true = [0, 0, 1, 2, 4]
    probs = [
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
        [0.1, 0.15, 0.2, 0.2, 0.4],
    ]
    for k in range(1, 6):
        precision = precision_macro(names=true, probs=probs, k=k)
        recall = recall_macro(names=true, probs=probs, k=k)
        f1 = f1_macro(names=true, probs=probs, k=k)
        assert f1 == (2 * precision * recall) / (precision + recall)
