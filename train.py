"""
train.py

Contains the training loop, validation logic, early stopping,
LR scheduling, and L1/L2 regularization for the MLP MNIST project.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, List

from parameters import TrainConfig, PathConfig
from models.mlp import MLP


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(
    train_cfg: TrainConfig,
    path_cfg: PathConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download MNIST and return train, validation, and test DataLoaders.

    The training set (60 000 samples) is split into train/val according to
    ``train_cfg.val_split``. The test set (10 000 samples) is kept separate.

    Args:
        train_cfg: Training configuration (batch size, val split, seed).
        path_cfg: Path configuration (data directory).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean / std
    ])

    full_train = datasets.MNIST(
        root=path_cfg.data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=path_cfg.data_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train) * train_cfg.val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(train_cfg.seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=train_cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Optimizer & Scheduler factories
# ---------------------------------------------------------------------------

def build_optimizer(model: MLP, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    """Instantiate the optimizer specified in train_cfg.

    L2 regularization is applied via the ``weight_decay`` parameter of the
    optimizer (standard practice — avoids applying decay to bias terms when
    using Adam).

    Args:
        model: The MLP model whose parameters will be optimized.
        train_cfg: Training configuration.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If an unsupported optimizer name is given.
    """
    if train_cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.l2_lambda,
        )
    elif train_cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.learning_rate,
            momentum=0.9,
            weight_decay=train_cfg.l2_lambda,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{train_cfg.optimizer}'. Choose 'adam' or 'sgd'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Instantiate the LR scheduler specified in train_cfg.

    Args:
        optimizer: The optimizer to attach the scheduler to.
        train_cfg: Training configuration.

    Returns:
        LRScheduler instance, or None if scheduler='none'.
    """
    if train_cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_cfg.step_size, gamma=train_cfg.gamma
        )
    elif train_cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.epochs
        )
    else:
        return None


# ---------------------------------------------------------------------------
# L1 Regularization
# ---------------------------------------------------------------------------

def l1_penalty(model: MLP, l1_lambda: float) -> torch.Tensor:
    """Compute the L1 regularization term over all Linear layer weights.

    Note: L1 is not natively supported by most optimizers (unlike L2 via
    weight_decay), so we add it manually to the loss.

    Args:
        model: The MLP model.
        l1_lambda: L1 regularization coefficient.

    Returns:
        Scalar tensor representing the L1 penalty.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    return l1_lambda * l1_norm


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor validation loss and stop training when it stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        checkpoint_path: Path to save the best model weights.

    Attributes:
        best_loss: Best validation loss observed so far.
        counter: Number of epochs since last improvement.
        should_stop: Flag indicating whether training should halt.

    Example::

        >>> early_stop = EarlyStopping(patience=5, checkpoint_path="best.pth")
        >>> early_stop(val_loss, model)
        >>> if early_stop.should_stop:
        ...     break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        checkpoint_path: str = "best_model.pth",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float, model: MLP) -> None:
        """Update state given the current validation loss.

        Saves model weights whenever a new best is found.

        Args:
            val_loss: Current epoch validation loss.
            model: Model whose weights to checkpoint.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    l1_lambda: float = 0.0,
) -> Tuple[float, float]:
    """Run one full pass over the training DataLoader.

    Args:
        model: The MLP model (set to train mode inside).
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function (CrossEntropyLoss).
        device: Compute device (cpu / cuda).
        l1_lambda: L1 regularization coefficient (0.0 = disabled).

    Returns:
        Tuple of (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        if l1_lambda > 0.0:
            loss = loss + l1_penalty(model, l1_lambda)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader without gradient computation.

    Args:
        model: The MLP model (set to eval mode inside).
        loader: Validation or test DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        device: Compute device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    path_cfg: PathConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Full training loop with early stopping and LR scheduling.

    Saves the best model checkpoint to ``path_cfg.checkpoint_dir``.
    Returns a history dict for plotting.

    Args:
        model: Initialised MLP model (moved to device before calling).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        train_cfg: Training configuration.
        path_cfg: Path configuration.
        device: Compute device.

    Returns:
        Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc',
        each mapping to a list of per-epoch float values.
    """
    os.makedirs(path_cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        path_cfg.checkpoint_dir, f"{path_cfg.experiment_name}_best.pth"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    early_stopping = EarlyStopping(
        patience=train_cfg.early_stop_patience,
        checkpoint_path=checkpoint_path,
    ) if train_cfg.early_stop_patience > 0 else None

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
    }

    print(f"\n{'='*60}")
    print(f"  Experiment : {path_cfg.experiment_name}")
    print(f"  Device     : {device}")
    print(f"  Parameters : {model.count_parameters():,}")
    print(f"{'='*60}\n")

    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, train_cfg.l1_lambda
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:>3}/{train_cfg.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # Load best weights back into model
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nBest model loaded from {checkpoint_path}")

    return history