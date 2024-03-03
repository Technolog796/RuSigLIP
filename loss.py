import torch
import torch.nn.functional as F
from torch import Tensor


def sig_loss(img_emb: Tensor, txt_emb: Tensor, t_prime: float | Tensor, b: float | Tensor) -> Tensor:
    if img_emb.shape != txt_emb.shape:
        raise TypeError("Input image and text embeddings must be the same size.")

    n = len(img_emb)
    t = torch.Tensor(t_prime).exp()

    zimg = img_emb / img_emb.norm()
    ztxt = txt_emb / txt_emb.norm()

    logits = zimg @ ztxt.T * t + b
    labels = 2 * torch.eye(n) - torch.ones(n)

    return -F.logsigmoid(labels * logits).sum()
