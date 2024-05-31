import numpy as np
import torch
from PIL import Image

from utils.inference_utils import load_model_and_tokenizer, preprocess, get_probs


def test_consistency():
    model, tokenizer = load_model_and_tokenizer()

    image = np.array(Image.open("images/Sample.jpg"))
    labels = ["a diagram", "a girl", "a cat"]
    images, texts = preprocess([image], labels, tokenizer)

    with torch.no_grad():
        img_emb, txt_emb = model(images, texts)
        probs = get_probs(img_emb, txt_emb).squeeze()

    assert np.allclose(probs.sum(), 1.0, atol=0.01, rtol=0.1)
