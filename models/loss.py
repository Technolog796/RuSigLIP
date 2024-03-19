import torch
import torch.nn.functional as F
from torch import Tensor


def positive_sig_loss(img_emb: Tensor, txt_emb: Tensor, temperature: float, bias: float) -> Tensor:
    if img_emb.shape != txt_emb.shape:
        raise TypeError("Input image and text embeddings must be the same size.")

    n = len(img_emb)

    zimg = img_emb / img_emb.norm()
    ztxt = txt_emb / txt_emb.norm()

    logits = zimg @ ztxt.T * temperature + bias
    device = torch.get_device(logits)
    labels = 2 * torch.eye(n, device=device) - torch.ones(n, device=device)

    return -F.logsigmoid(labels * logits).sum()

def negative_sig_loss(img_emb: Tensor, txt_emb: Tensor, temperature: float, bias: float) -> Tensor:
    if img_emb.shape != txt_emb.shape:
        raise TypeError("Input image and text embeddings must be the same size.")

    n = len(img_emb)

    zimg = img_emb / img_emb.norm()
    ztxt = txt_emb / txt_emb.norm()

    logits = zimg @ ztxt.T * temperature + bias
    device = torch.get_device(logits)
    labels = -torch.ones(n, device=device)

    return -F.logsigmoid(labels * logits).sum()


