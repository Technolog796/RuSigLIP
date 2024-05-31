import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import PIL
from PIL import Image
import torch
import argparse
import numpy as np

# from transformers import AutoProcessor, AutoModel

from typing import List, Union

from utils.inference_utils import load_model_and_tokenizer, preprocess, get_probs


def predict(
    path: str = "image.jpg",
    labels: List[str] = [],
    model_weights: str = "trained_models/model12/model.safetensors",
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
    except:
        print("Unexpected error")
        return None

    model, tokenizer = load_model_and_tokenizer(
        model_weights=model_weights,
        tokenizer_name="/home/jovyan/clip-research/models/ru-e5-base/",
    )
    images, texts = preprocess([np.asarray(image)], labels, tokenizer)
    img_emb, txt_emb = model(images, texts)
    probs = get_probs(img_emb, txt_emb)[0].tolist()
    return np.array(probs).argmax()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction of model")
    parser.add_argument("--image", type=str, default="", help="Path to image")
    parser.add_argument(
        "--model_weights",
        type=str,
        default="trained_models/model12/model.safetensors",
        help="Path to model weights",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=[], help="List of labels"
    )
    args = parser.parse_args()

    pred = predict(
        path=args.image, labels=args.labels, model_weights=args.model_weights
    )
    print(
        "Prediction is impossible"
        if pred is None or pred >= len(args.labels) or pred < 0
        else args.labels[pred]
    )
