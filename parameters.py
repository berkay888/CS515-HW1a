"""
parameters.py

Defines all hyperparameters and configuration dataclasses for the MLP MNIST project.
Uses Python dataclasses for structured parameter passing and argparse for CLI support.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the MLP model architecture.

    Attributes:
        hidden_sizes: List of hidden layer widths, e.g. [256, 128].
        activation: Activation function to use ('relu' or 'gelu').
        dropout_rate: Dropout probability applied after each hidden layer (0.0 = disabled).
        use_batch_norm: Whether to insert BatchNorm1d before each activation.
    """
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout_rate: float = 0.0
    use_batch_norm: bool = False


@dataclass
class TrainConfig:
    """Configuration for the training loop.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per mini-batch.
        learning_rate: Initial learning rate for the optimizer.
        optimizer: Optimizer type ('adam' or 'sgd').
        scheduler: LR scheduler type ('step', 'cosine', or 'none').
        step_size: Step size for StepLR scheduler (used when scheduler='step').
        gamma: Multiplicative decay factor for StepLR (used when scheduler='step').
        early_stop_patience: Number of epochs with no val improvement before stopping (0 = disabled).
        l1_lambda: L1 regularization coefficient (0.0 = disabled).
        l2_lambda: L2 weight decay coefficient passed to the optimizer (0.0 = disabled).
        val_split: Fraction of training data to use for validation.
        seed: Random seed for reproducibility.
    """
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    early_stop_patience: int = 5
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    val_split: float = 0.1
    seed: int = 42


@dataclass
class PathConfig:
    """File-system paths used across the project.

    Attributes:
        data_dir: Directory where MNIST data will be downloaded / cached.
        checkpoint_dir: Directory where model checkpoints are saved.
        output_dir: Directory where plots and results are saved.
        experiment_name: Unique name for the current run (used in file names).
    """
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    experiment_name: str = "baseline"


# ---------------------------------------------------------------------------
# Argparse — CLI entry point
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments and return an argparse Namespace.

    Returns:
        argparse.Namespace: Parsed arguments covering model, training, and path configs.

    Example::

        $ python main.py --hidden_sizes 512 256 --activation gelu --dropout_rate 0.3
    """
    parser = argparse.ArgumentParser(
        description="Train an MLP on MNIST with configurable hyperparameters."
    )

    # ---- Model ----
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--hidden_sizes",
        nargs="+",
        type=int,
        default=[256, 128],
        help="Hidden layer widths, e.g. --hidden_sizes 512 256 128",
    )
    model_group.add_argument(
        "--activation",
        type=str,
        choices=["relu", "gelu"],
        default="relu",
        help="Activation function (default: relu)",
    )
    model_group.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout probability [0, 1) (default: 0.0 = disabled)",
    )
    model_group.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Enable BatchNorm1d before each activation",
    )

    # ---- Training ----
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=30)
    train_group.add_argument("--batch_size", type=int, default=64)
    train_group.add_argument("--learning_rate", type=float, default=1e-3)
    train_group.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
    )
    train_group.add_argument(
        "--scheduler",
        type=str,
        choices=["step", "cosine", "none"],
        default="none",
    )
    train_group.add_argument("--step_size", type=int, default=10)
    train_group.add_argument("--gamma", type=float, default=0.1)
    train_group.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Epochs to wait before early stopping (0 = disabled)",
    )
    train_group.add_argument("--l1_lambda", type=float, default=0.0)
    train_group.add_argument("--l2_lambda", type=float, default=0.0)
    train_group.add_argument("--val_split", type=float, default=0.1)
    train_group.add_argument("--seed", type=int, default=42)

    # ---- Paths ----
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--data_dir", type=str, default="./data")
    path_group.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    path_group.add_argument("--output_dir", type=str, default="./outputs")
    path_group.add_argument("--experiment_name", type=str, default="baseline")

    # ---- Mode ----
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the full ablation study instead of a single experiment",
    )

    return parser.parse_args()


def args_to_configs(args: argparse.Namespace) -> tuple[ModelConfig, TrainConfig, PathConfig]:
    """Convert a parsed argparse Namespace into typed dataclass instances.

    Args:
        args: Parsed CLI arguments from :func:`get_args`.

    Returns:
        Tuple of (ModelConfig, TrainConfig, PathConfig).
    """
    model_cfg = ModelConfig(
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        use_batch_norm=args.use_batch_norm,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        early_stop_patience=args.early_stop_patience,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        val_split=args.val_split,
        seed=args.seed,
    )
    path_cfg = PathConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    return model_cfg, train_cfg, path_cfg