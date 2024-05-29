import numpy as np
import albumentations as A
import torch
from torch import Tensor
from torch.nn.functional import softmax

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from safetensors import safe_open

from model import SigLIPModel


def load_model_and_tokenizer(model_params: dict[Tensor] | None = None,
                             model_weights: str | None = None,
                             tokenizer_name: str = "../../models/encoders/ru-e5-base") \
        -> tuple[SigLIPModel, PreTrainedTokenizerFast]:
    if model_params is None:
        model_params = {}
    model = SigLIPModel(**model_params)
    if model_weights is not None:
        weights = {}
        with safe_open(model_weights, framework="pt", device="cpu") as file:
            for k in file.keys():
                weights[k] = file.get_tensor(k)
        model.load_state_dict(weights)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return model, tokenizer


def preprocess(images: list[np.ndarray], labels: list[str],
               tokenizer: PreTrainedTokenizerFast, transforms: A.BaseCompose = None,
               img_size: int = 224, seq_length: int = 32) -> tuple[Tensor, dict[str, Tensor]]:
    if transforms is None:
        transforms = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    max_pixel_value=255.0,
                    always_apply=True,
                )
            ]
        )
    images = np.array([transforms(image=image)["image"] for image in images])
    images = torch.FloatTensor(images).permute(0, 3, 1, 2)

    tokenizer_params = {"max_length": seq_length,
                        "return_tensors": "pt",
                        "return_token_type_ids": False,
                        "padding": True,
                        "truncation": True}
    texts = tokenizer(list(labels), **tokenizer_params)
    return images, texts


def get_probs(img_emb: Tensor, txt_emb: Tensor,
              temperature: float = 10.0, bias: float = -10.0) -> Tensor:
    logits = img_emb @ txt_emb.T * temperature + bias
    probs = softmax(logits, dim=-1)
    return probs
