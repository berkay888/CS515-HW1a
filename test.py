"""
test.py

Evaluation, confusion matrix, t-SNE embedding visualization,
and per-class accuracy reporting for the trained MLP model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Tuple, Dict, List

from parameters import PathConfig
from models.mlp import MLP


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(
    model: MLP,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and collect all predictions and labels.

    Args:
        model: Trained MLP model (set to eval mode inside).
        loader: DataLoader to iterate over.
        device: Compute device.

    Returns:
        Tuple of (all_preds, all_labels) as numpy arrays of shape (N,).
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute overall classification accuracy.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).

    Returns:
        float: Accuracy in [0, 1].
    """
    return float((preds == labels).mean())


def per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 10,
) -> Dict[int, float]:
    """Compute per-class accuracy for each digit class.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).
        num_classes: Total number of classes (10 for MNIST).

    Returns:
        Dict mapping class index → accuracy float.
    """
    acc_per_class: Dict[int, float] = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            acc_per_class[c] = 0.0
        else:
            acc_per_class[c] = float((preds[mask] == labels[mask]).mean())
    return acc_per_class


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    experiment_name: str,
    num_classes: int = 10,
) -> None:
    """Compute and save a normalised confusion matrix heatmap.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).
        output_dir: Directory to save the figure.
        experiment_name: Used in the output filename.
        num_classes: Number of classes.
    """
    # Build matrix
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        matrix[t][p] += 1

    # Normalise row-wise
    matrix_norm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {experiment_name}", fontsize=13)

    # Annotate cells
    thresh = matrix_norm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, f"{matrix_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if matrix_norm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# t-SNE visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: MLP,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings for t-SNE visualisation.

    Hooks into the output of the last hidden layer (before the output Linear).

    Args:
        model: Trained MLP model.
        loader: DataLoader to sample from.
        device: Compute device.
        max_samples: Maximum number of samples to embed (t-SNE is O(n²)).

    Returns:
        Tuple of (embeddings, labels) as numpy arrays.
    """
    model.eval()
    embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    collected = 0

    # Hook to capture last hidden layer output
    activation_store: Dict[str, torch.Tensor] = {}

    def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        activation_store["embedding"] = output.detach().cpu()

    # Register hook on the last hidden layer Sequential block
    handle = model.hidden_layers[-1].register_forward_hook(hook_fn)

    for images, labels in loader:
        if collected >= max_samples:
            break
        images = images.to(device)
        model(images)  # forward triggers the hook
        emb = activation_store["embedding"].numpy()
        embeddings.append(emb)
        all_labels.append(labels.numpy())
        collected += images.size(0)

    handle.remove()

    embeddings_arr = np.concatenate(embeddings)[:max_samples]
    labels_arr = np.concatenate(all_labels)[:max_samples]
    return embeddings_arr, labels_arr


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    experiment_name: str,
) -> None:
    """Run t-SNE on embeddings and save a 2-D scatter plot.

    Args:
        embeddings: Array of shape (N, D) — penultimate layer activations.
        labels: Array of shape (N,) — ground-truth digit labels.
        output_dir: Directory to save the figure.
        experiment_name: Used in the output filename and plot title.
    """
    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        mask = labels == c
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            s=5, color=cmap(c), label=str(c), alpha=0.7,
        )

    ax.legend(title="Digit", markerscale=3, fontsize=9)
    ax.set_title(f"t-SNE Embedding — {experiment_name}", fontsize=13)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_tsne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved → {path}")


# ---------------------------------------------------------------------------
# Loss / accuracy curve plotter (called from main after training)
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str,
    experiment_name: str,
) -> None:
    """Plot and save training & validation loss and accuracy curves.

    Args:
        history: Dict returned by ``train.train()`` with keys
                 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        output_dir: Directory to save the figure.
        experiment_name: Used in the output filename and plot title.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(experiment_name, fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


# ---------------------------------------------------------------------------
# Full test pipeline
# ---------------------------------------------------------------------------

def run_test(
    model: MLP,
    test_loader: DataLoader,
    path_cfg: PathConfig,
    device: torch.device,
    plot_tsne_flag: bool = True,
) -> float:
    """Run the full evaluation pipeline on the test set.

    Steps:
        1. Compute overall and per-class accuracy.
        2. Plot confusion matrix.
        3. (Optionally) extract embeddings and plot t-SNE.

    Args:
        model: Trained MLP model with best weights loaded.
        test_loader: Test DataLoader.
        path_cfg: Path configuration (output directory, experiment name).
        device: Compute device.
        plot_tsne_flag: Whether to run t-SNE (can be slow on CPU).

    Returns:
        float: Overall test accuracy.
    """
    preds, labels = get_predictions(model, test_loader, device)

    # Overall accuracy
    acc = compute_accuracy(preds, labels)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # Per-class accuracy
    per_class = per_class_accuracy(preds, labels)
    print("\nPer-class accuracy:")
    for cls, cls_acc in per_class.items():
        print(f"  Digit {cls}: {cls_acc * 100:.2f}%")

    # Confusion matrix
    plot_confusion_matrix(preds, labels, path_cfg.output_dir, path_cfg.experiment_name)

    # t-SNE
    if plot_tsne_flag:
        embeddings, emb_labels = extract_embeddings(model, test_loader, device)
        plot_tsne(embeddings, emb_labels, path_cfg.output_dir, path_cfg.experiment_name)

    return acc