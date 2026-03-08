"""
models/mlp.py

Defines the MLP (Multi-Layer Perceptron) model for MNIST classification.
Supports configurable depth, width, activation functions, batch normalization,
and dropout via ModelConfig dataclass.
"""

import torch
import torch.nn as nn
from torch.nn import ModuleList, Sequential
from torch.nn.modules.flatten import Flatten
from typing import List

from parameters import ModelConfig


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

def get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of 'relu' or 'gelu'.

    Returns:
        nn.Module: Corresponding activation module.

    Raises:
        ValueError: If an unsupported activation name is given.

    Example::

        >>> act = get_activation("gelu")
        >>> act
        GELU()
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {list(activations.keys())}.")
    return activations[name]


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST image classification.

    Architecture per hidden layer (when all options enabled):
        Linear → BatchNorm1d → Activation → Dropout

    The input (28x28 grayscale image) is first flattened to a 784-dim vector.
    The output is a 10-dim logit vector (one per digit class).

    Args:
        config: ModelConfig dataclass containing all architectural hyperparameters.

    Attributes:
        flatten: Flatten layer converting (B, 1, 28, 28) → (B, 784).
        hidden_layers: ModuleList of Sequential blocks, one per hidden layer.
        output_layer: Final Linear(last_hidden_size → 10).

    Example::

        >>> from parameters import ModelConfig
        >>> cfg = ModelConfig(hidden_sizes=[256, 128], activation="relu",
        ...                   dropout_rate=0.3, use_batch_norm=True)
        >>> model = MLP(cfg)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 10])
    """

    INPUT_SIZE: int = 784   # 28 * 28
    NUM_CLASSES: int = 10   # digits 0-9

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config
        self.flatten = Flatten()

        # Build hidden layers as a ModuleList of Sequential blocks
        self.hidden_layers: ModuleList = self._build_hidden_layers(config.hidden_sizes)

        # Output layer — no activation, raw logits for CrossEntropyLoss
        last_size = config.hidden_sizes[-1] if config.hidden_sizes else self.INPUT_SIZE
        self.output_layer = nn.Linear(last_size, self.NUM_CLASSES)

        # Weight initialisation
        self._init_weights()

    def _build_hidden_layers(self, hidden_sizes: List[int]) -> ModuleList:
        """Construct one Sequential block per hidden layer and wrap in ModuleList.

        Each block follows the order:
            Linear → [BatchNorm1d] → Activation → [Dropout]

        Args:
            hidden_sizes: Ordered list of hidden layer widths.

        Returns:
            ModuleList of Sequential blocks.
        """
        layers: ModuleList = ModuleList()
        in_size = self.INPUT_SIZE

        for out_size in hidden_sizes:
            block: List[nn.Module] = []

            # 1. Fully connected
            block.append(nn.Linear(in_size, out_size))

            # 2. BatchNorm BEFORE activation (stabilises input distribution to activation)
            if self.config.use_batch_norm:
                block.append(nn.BatchNorm1d(out_size))

            # 3. Activation
            block.append(get_activation(self.config.activation))

            # 4. Dropout AFTER activation
            if self.config.dropout_rate > 0.0:
                block.append(nn.Dropout(p=self.config.dropout_rate))

            layers.append(Sequential(*block))
            in_size = out_size

        return layers

    def _init_weights(self) -> None:
        """Initialise Linear layers with Kaiming (He) uniform init.

        Kaiming init is designed for ReLU-family activations and keeps
        gradient variance stable across layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through flatten → hidden layers → output layer.

        Args:
            x: Input tensor of shape (B, 1, 28, 28) or (B, 784).

        Returns:
            torch.Tensor: Raw logits of shape (B, 10).
        """
        x = self.flatten(x)                     # (B, 784)
        for layer in self.hidden_layers:        # (B, hidden_size)
            x = layer(x)
        x = self.output_layer(x)               # (B, 10)
        return x

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            int: Sum of all trainable parameter elements.

        Example::

            >>> model.count_parameters()
            269322
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)