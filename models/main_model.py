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
        image_embedding_size: int,
        text_embedding_size: int,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(**image_encoder_params)
        self.text_encoder = TextEncoder(**text_encoder_params)

        self.image_connector = Connector(
            connector_size=image_embedding_size, **connector_params
        )
        self.text_connector = Connector(
            connector_size=text_embedding_size, **connector_params
        )

    def forward(self, images, texts):
        image_embeddings = self.image_connector(self.image_encoder(images))
        text_embeddings = self.text_connector(
            self.text_encoder(
                input_ids=texts["input_ids"], attention_mask=texts["attention_mask"]
            )
        )
        return image_embeddings, text_embeddings

    def predict(self, images, texts):
        image_embeddings, text_embeddings = self.forward(images, texts)
        return image_embeddings, text_embeddings
