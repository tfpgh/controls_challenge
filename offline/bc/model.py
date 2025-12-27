import torch
import torch.nn as nn

from offline.config import BCConfig


class BCModel(nn.Module):
    """
    MLP for BC.

    LayerNorm -> (Linear + GELU) x N -> Linear -> Tanh x 2.0
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()

        self.config = config

        layers: list[nn.Module] = []

        # Input norm
        layers.append(nn.LayerNorm(config.input_size))

        # Hidden layers
        prev_size = config.input_size
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.GELU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, 247] input features

        Returns:
            actions: [batch] in range [-2, 2]
        """
        out = self.net(x).squeeze(-1)
        return torch.tanh(out) * 2.0

    def count_parameters(self) -> int:
        """Count trainable params"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
