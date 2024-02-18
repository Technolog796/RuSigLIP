from transformers import AutoModel
from torch import nn


class TextEncoder(nn.Module):
    def __init__(
        self, model_name: str, freeze: bool = False, target_token_idx: int = 0
    ):
        super().__init__()
        self.target_token_idx = target_token_idx

        self.model = AutoModel.from_pretrained(model_name, torch_dtype="auto")

        for name, param in self.model.named_parameters():
            param.requires_grad = not freeze

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]
