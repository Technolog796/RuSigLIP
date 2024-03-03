from torch import nn
from encoders.image_encoder import ImageEncoder
from encoders.text_encoder import TextEncoder
from encoders.connector import Connector
from loss import Sig_Loss


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

    def forward(self, sample):
        image_embeddings = self.image_connector(self.image_encoder(sample["image"]))
        text_embeddings = self.text_connector(
            self.text_encoder(
                input_ids=sample["input_ids"], attention_mask=sample["attention_mask"]
            )
        )

        return Sig_Loss(image_embeddings, text_embeddings, t_prime=0.1, b=-10)
