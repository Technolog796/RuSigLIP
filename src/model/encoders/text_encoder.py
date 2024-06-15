from transformers import AutoModel
from torch import nn, Tensor
import torch


def mean_pooling(model_output: tuple[Tensor, ...], attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).to(torch.bfloat16) # For bf16 training support


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "hivaze/ru-e5-base",
        freeze: bool = False,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        for name, param in self.model.named_parameters():
            param.requires_grad = not freeze

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask, return_dict=True)
        embeddings = mean_pooling(model_output=outputs, attention_mask=attention_mask)
        return embeddings
