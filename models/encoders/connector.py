import torch.nn as nn


class Connector(nn.Module):
    def __init__(
        self, connector_size: int, projection_size: int, dropout_rate: float
    ) -> None:
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


# Test
class ConnectorBlock(nn.Module):
    def __init__(self, input_size: int, out_size: int, dropout_rate: float) -> None:
        super().__init__()

        self.projection = nn.Linear(input_size, out_size)
        self.gelu = nn.GELU()  # Test https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html for pretrain
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.projection(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class ModularConnector(nn.Module):
    def __init__(self, connector_shapes: list[int], dropout_rate: float) -> None:
        super().__init__()

        self.connector_blocks = nn.ModuleList(
            [
                ConnectorBlock(
                    connector_shapes[i], connector_shapes[i + 1], dropout_rate
                )
                for i in range(len(connector_shapes) - 1)
            ]
        )
        self.layer_norm_blocks = nn.ModuleList(
            [nn.LayerNorm(connector_shapes[i]) for i in range(1, len(connector_shapes))]
        )

    def forward(self, x):
        for block, layer_norm in zip(
            self.connector_blocks, self.layer_norm_blocks
        ):  # Вот тут обдумать ещё раз
            projection = block(x)
            x += projection
            x = layer_norm(x)
        return x
