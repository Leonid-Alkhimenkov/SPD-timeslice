import torch
import gin
from torch import nn


class FFN(nn.Module):
    """FFN layer following Transformers and MLP mixer architectures"""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        dropout: float = 0.2
    ):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, model_dim: int, sublayer: nn.Module):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor):
        "Apply residual connection to any sublayer with the same size."
        return x + self.sublayer(self.norm(x))


@gin.configurable
class TrackEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,  # 3D координаты хитов и вершин
        n_blocks: int = 2,
        model_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        dropout: float = 0.2
    ):
        super(TrackEmbedder, self).__init__()

        self.input_projection = nn.Linear(input_dim, model_dim)

        self.blocks = nn.Sequential(
            *[SublayerConnection(
                model_dim, FFN(model_dim, hidden_dim, dropout))
              for _ in range(n_blocks)]
        )

        self.layer_norm = nn.LayerNorm(model_dim)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, combined_input: torch.Tensor):
        """
        Forward pass для обработки объединенного входа.
        Args:
            combined_input: Тензор с объединенными хитами и вершинами (batch_size, max_num_hits + 1, 3).
        Returns:
            embeddings: Tensor с эмбеддингами (batch_size, max_num_hits + 1, output_dim).
        """
        x = self.input_projection(combined_input)  # (batch_size, max_num_hits + 1, model_dim)

        x = self.blocks(x)
        x = self.layer_norm(x)

        x = self.output_layer(x)

        embeddings, _ = x.max(dim=1)

        return embeddings