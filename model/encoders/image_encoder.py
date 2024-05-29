from torch import nn, Tensor
from transformers import ViTModel


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k",
                 freeze: bool = False) -> None:
        super().__init__()

        self.model = ViTModel.from_pretrained(model_name)
        
        for name, param in self.model.named_parameters():
            param.requires_grad = not freeze

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).last_hidden_state.mean(axis=1)
