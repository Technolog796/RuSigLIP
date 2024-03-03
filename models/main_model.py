from torch import nn
from encoders.image_encoder import ImageEncoder
from encoders.text_encoder import TextEncoder
from encoders.connector import Connector


class SigLIPModel(nn.Module):
    def __init__(self, image_embedding_size, text_image_embedding_size):
        super(SigLIPModel, self).__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()  # Тут указать модель конфиге

        self.image_connector = Connector(image_embedding_size, 256)

        self.text_connector = Connector(text_image_embedding_size, 256)

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        # Los implementation here
