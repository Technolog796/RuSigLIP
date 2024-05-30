import os

import PIL
from PIL import Image
import torch
import argparse

from transformers import AutoProcessor, AutoModel

from typing import List, Union

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def predict(
    path: str = "image.jpg",
    labels: List[str] = [],
    model: str = "google/siglip-base-patch16-224",
) -> Union[None, int]:
    if len(labels) == 0:
        print("Len of list of labels must be > 0")
        return None
    try:
        image = Image.open(path)
    except PermissionError:
        print("Permission denied")
        return None
    except FileNotFoundError:
        print("File not found")
        return None
    except PIL.UnidentifiedImageError:
        print("Image cannot be opened and identified")
        return None
    except ValueError:
        print(
            "The mode in Image.open() is not “r”, or if a StringIO instance is used for fp"
        )
        return None
    except TypeError:
        print("Formats in Image.open() is not None, a list or a tuple")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    if model == "google/siglip-base-patch16-224":
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    else:
        return None

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)
    return torch.argmax(probs[0]).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction of model")
    parser.add_argument("--image", type=str, default="", help="Path to image")
    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Name of model",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=[], help="List of labels"
    )
    args = parser.parse_args()

    pred = predict(path=args.image, labels=args.labels, model=args.model)
    print(
        "Prediction is impossible"
        if pred is None or pred >= len(args.labels) or pred < 0
        else args.labels[pred]
    )
