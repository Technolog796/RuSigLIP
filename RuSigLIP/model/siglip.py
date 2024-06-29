import torch
from torch import nn, Tensor
from torch.nn.functional import normalize
from typing import Optional, Dict, Tuple

from .encoders import ImageEncoder, TextEncoder, ModularConnector


class SigLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder_params: Optional[Dict] = None,
        text_encoder_params: Optional[Dict] = None,
        connector_params: Optional[Dict] = None,
        image_embedding_size: int = 768,
        text_embedding_size: int = 768,
    ):
        super().__init__()

        image_encoder_params = image_encoder_params or {}
        text_encoder_params = text_encoder_params or {}
        connector_params = connector_params or {}

        self.image_encoder = ImageEncoder(**image_encoder_params)
        self.text_encoder = TextEncoder(**text_encoder_params)

        self.image_connector = ModularConnector(
            input_size=image_embedding_size, **connector_params
        )
        self.text_connector = ModularConnector(
            input_size=text_embedding_size, **connector_params
        )

    def forward(
        self, images: Tensor, texts: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        image_embeddings = normalize(self.image_connector(self.image_encoder(images)))
        text_embeddings = normalize(
            self.text_connector(
                self.text_encoder(
                    input_ids=texts["input_ids"], attention_mask=texts["attention_mask"]
                )
            )
        )
        return image_embeddings, text_embeddings

    @torch.no_grad()
    def predict(
        self, images: Tensor, texts: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        image_embeddings, text_embeddings = self.forward(images, texts)
        return image_embeddings, text_embeddings
