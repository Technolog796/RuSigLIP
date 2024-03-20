from torch import nn
from .encoders import ImageEncoder
from .encoders import TextEncoder
from .encoders import Connector


class SigLIPModel(nn.Module):
    def __init__(
        self,
        image_embeddings_dim: int,
        text_embeddings_dim: int,
        connector_dim: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            model_name="resnet50", pretrained=True, freeze=True
        )
        self.text_encoder = TextEncoder(
            model_name="google-bert/bert-base-multilingual-cased",
            pretrained=True,
            freeze=True,
        )

        self.image_connector = Connector(
            embedding_dim=image_embeddings_dim,
            projection_dim=connector_dim,
            dropout_rate=dropout_rate,
        )

        self.text_connector = Connector(
            embedding_dim=text_embeddings_dim,
            projection_dim=connector_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.image_connector(self.image_encoder(images))
        text_embeddings = self.text_connector(
            self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        )
        return image_embeddings, text_embeddings
