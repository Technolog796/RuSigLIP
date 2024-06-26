from torch import nn, Tensor


class ConnectorBlock(nn.Module):
    def __init__(self, input_size: int, out_size: int, dropout_rate: float) -> None:
        super().__init__()

        self.projection = nn.Linear(input_size, out_size)
        self.fc = nn.Linear(out_size, out_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(out_size)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        x = self.layer_norm(x)
        return x


class ModularConnector(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] = (256,),
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        connector_shapes = [input_size] + list(output_sizes)

        self.connector_blocks = nn.ModuleList(
            [
                ConnectorBlock(
                    connector_shapes[i], connector_shapes[i + 1], dropout_rate
                )
                for i in range(len(connector_shapes) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.connector_blocks:
            x = block(x)

        return x
