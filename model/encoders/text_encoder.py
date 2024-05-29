from transformers import AutoModel
from torch import nn, Tensor


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "pretrained_models/ru-e5-base",
                 pretrained: bool = True,
                 freeze: bool = False):
        super().__init__()

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            # TODO
            self.model = AutoModel.from_config(model_name)

        for name, param in self.model.named_parameters():
            param.requires_grad = not freeze

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        return embeddings
