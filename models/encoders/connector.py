import torch.nn as nn


class Connector(nn.Module):
    def __init__(self, connector_size: int, projection_size: int, dropout_rate: float) -> None:
        super().__init__()

        self.projection = nn.Linear(connector_size, projection_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_size, projection_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(projection_size)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)
