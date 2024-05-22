from torch import nn
from transformers import ViTImageProcessor, ViTModel


class ImageEncoder(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool = False, freeze: bool = True
    ):
        super().__init__()

        self.model = ViTModel.from_pretrained(model_name)
        
        for name, param in self.model.named_parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        return self.model(x).last_hidden_state.mean(axis=1)
