import torch
from torch import Tensor
from torch.nn.functional import logsigmoid


class SigmoidLoss:
    def __init__(self, temperature: float = 10, bias: float = -10) -> None:
        self.temperature = temperature
        self.bias = bias

    def __call__(
        self, img_emb: Tensor, txt_emb: Tensor, positive: bool = True
    ) -> Tensor:
        if img_emb.shape != txt_emb.shape:
            raise TypeError(
                f"Input image and text embeddings must have the same shape. "
                f"But {img_emb.shape} != {txt_emb.shape}."
            )

        n = len(img_emb)
        device = torch.get_device(img_emb)

        logits = img_emb @ txt_emb.T * self.temperature + self.bias

        labels = -torch.ones((n, n), device=device)
        if positive:
            labels += 2 * torch.eye(n, device=device)

        return -logsigmoid(labels * logits).sum()
