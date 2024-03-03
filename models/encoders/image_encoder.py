from torch import nn
import timm  # Implement hugging face models


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained=False, freeze: bool = False):
        super().__init__()

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        for name, param in self.model.named_parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        return self.model(x)
