from transformers import AutoModel
from torch import nn, Tensor
import torch.nn.functional as F


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TextEncoder(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool = False, trainable: bool = True
    ):
        super().__init__()

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(model_name)

        for name, param in self.model.named_parameters():
            param.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Нормализация
        return embeddings
