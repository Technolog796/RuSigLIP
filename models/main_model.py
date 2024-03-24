from torch import nn
from .encoders import ImageEncoder
from .encoders import TextEncoder
from .encoders import Connector


class SigLIPModel(nn.Module):
    def __init__(
            self,
            image_encoder_params: dict,
            text_encoder_params: dict,
            connector_params: dict,
            image_embeddings_dim: int,
            text_embeddings_dim: int,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(**image_encoder_params)
        self.text_encoder = TextEncoder(**text_encoder_params)

        self.image_connector = Connector(embedding_dim=image_embeddings_dim, **connector_params)
        self.text_connector = Connector(embedding_dim=text_embeddings_dim, **connector_params)

    def forward(self, images, texts):
        image_embeddings = self.image_connector(self.image_encoder(images))
        text_embeddings = self.text_connector(
            self.text_encoder(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])
        )
        return image_embeddings, text_embeddings
