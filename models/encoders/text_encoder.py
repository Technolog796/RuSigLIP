from transformers import AutoModel
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False, trainable: bool = True, target_token_idx: int = 0):
        super().__init__()

        self.target_token_idx = target_token_idx

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(model_name)

        for name, param in self.model.named_parameters():
            param.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]
