import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from .encoders import ImageEncoder
from .encoders import TextEncoder
from .encoders import ModularConnector


class SigLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder_params: dict | None = None,
        text_encoder_params: dict | None = None,
        connector_params: dict | None = None,
        image_embedding_size: int = 768,
        text_embedding_size: int = 768,
    ):
        super().__init__()

        if image_encoder_params is None:
            image_encoder_params = {}
        if text_encoder_params is None:
            text_encoder_params = {}
        if connector_params is None:
            connector_params = {}

        self.image_encoder = ImageEncoder(**image_encoder_params)
        self.text_encoder = TextEncoder(**text_encoder_params)

        self.image_connector = ModularConnector(
            input_size=image_embedding_size, **connector_params
        )
        self.text_connector = ModularConnector(
            input_size=text_embedding_size, **connector_params
        )

    def forward(self, images: Tensor, texts: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        image_embeddings = normalize(self.image_connector(self.image_encoder(images)))
        text_embeddings = normalize(self.text_connector(
            self.text_encoder(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])))
        return image_embeddings, text_embeddings

    @torch.no_grad()
    def predict(self, images: Tensor, texts: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        image_embeddings, text_embeddings = self.forward(images, texts)
        return image_embeddings, text_embeddings
