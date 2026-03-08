"""
main.py

Entry point for the MLP MNIST project.
Supports single-run mode (via CLI args) and ablation study mode
(runs a predefined grid of configurations automatically).
"""

import os
import json
import torch
import random
import numpy as np
from typing import List, Dict, Any

from parameters import ModelConfig, TrainConfig, PathConfig, get_args, args_to_configs
from models.mlp import MLP
from train import get_dataloaders, train
from test import run_test, plot_training_curves


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across torch, numpy, and python.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    path_cfg: PathConfig,
    device: torch.device,
    plot_tsne_flag: bool = True,
) -> Dict[str, Any]:
    """Run a single training + evaluation experiment.

    Args:
        model_cfg: Model architecture configuration.
        train_cfg: Training hyperparameter configuration.
        path_cfg: Path configuration (checkpoint / output dirs).
        device: Compute device.
        plot_tsne_flag: Whether to generate t-SNE plots after evaluation.

    Returns:
        Dict summarising the experiment results with keys:
        'experiment_name', 'test_acc', 'best_val_loss', 'num_params'.
    """
    set_seed(train_cfg.seed)

    # Build dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, path_cfg)

    # Build model
    model = MLP(model_cfg).to(device)

    # Train
    history = train(model, train_loader, val_loader, train_cfg, path_cfg, device)

    # Plot training curves
    plot_training_curves(history, path_cfg.output_dir, path_cfg.experiment_name)

    # Evaluate on test set
    test_acc = run_test(model, test_loader, path_cfg, device, plot_tsne_flag=plot_tsne_flag)

    result = {
        "experiment_name": path_cfg.experiment_name,
        "test_acc": round(test_acc * 100, 4),
        "best_val_loss": round(min(history["val_loss"]), 6),
        "num_params": model.count_parameters(),
        "hidden_sizes": model_cfg.hidden_sizes,
        "activation": model_cfg.activation,
        "dropout_rate": model_cfg.dropout_rate,
        "use_batch_norm": model_cfg.use_batch_norm,
        "optimizer": train_cfg.optimizer,
        "scheduler": train_cfg.scheduler,
        "l1_lambda": train_cfg.l1_lambda,
        "l2_lambda": train_cfg.l2_lambda,
        "epochs_run": len(history["train_loss"]),
    }

    print(f"\n[{path_cfg.experiment_name}] Test Acc: {result['test_acc']:.2f}%\n")
    return result


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

def get_ablation_configs() -> List[Dict[str, Any]]:
    """Return the list of ablation study configurations.

    Each entry is a dict with keys: 'name', 'model', 'train'.
    Add or remove entries here to customize your ablation study.

    Returns:
        List of experiment configuration dicts.
    """
    configs = [
        # ── Baseline ────────────────────────────────────────────────────────
        {
            "name": "baseline",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu"),
            "train": TrainConfig(),
        },

        # ── Depth ablation ──────────────────────────────────────────────────
        {
            "name": "depth_1layer",
            "model": ModelConfig(hidden_sizes=[256], activation="relu"),
            "train": TrainConfig(),
        },
        {
            "name": "depth_3layer",
            "model": ModelConfig(hidden_sizes=[512, 256, 128], activation="relu"),
            "train": TrainConfig(),
        },
        {
            "name": "depth_4layer",
            "model": ModelConfig(hidden_sizes=[512, 256, 128, 64], activation="relu"),
            "train": TrainConfig(),
        },

        # ── Width ablation ──────────────────────────────────────────────────
        {
            "name": "width_small",
            "model": ModelConfig(hidden_sizes=[64, 32], activation="relu"),
            "train": TrainConfig(),
        },
        {
            "name": "width_large",
            "model": ModelConfig(hidden_sizes=[512, 512], activation="relu"),
            "train": TrainConfig(),
        },

        # ── Activation ablation ─────────────────────────────────────────────
        {
            "name": "activation_gelu",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="gelu"),
            "train": TrainConfig(),
        },

        # ── Dropout ablation ────────────────────────────────────────────────
        {
            "name": "dropout_02",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu", dropout_rate=0.2),
            "train": TrainConfig(),
        },
        {
            "name": "dropout_05",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu", dropout_rate=0.5),
            "train": TrainConfig(),
        },

        # ── Batch normalisation ablation ────────────────────────────────────
        {
            "name": "bn_only",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu", use_batch_norm=True),
            "train": TrainConfig(),
        },
        {
            "name": "bn_and_dropout",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu",
                                 dropout_rate=0.3, use_batch_norm=True),
            "train": TrainConfig(),
        },

        # ── Regularization ablation ─────────────────────────────────────────
        {
            "name": "l2_1e3",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu"),
            "train": TrainConfig(l2_lambda=1e-3),
        },
        {
            "name": "l1_1e4",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu"),
            "train": TrainConfig(l1_lambda=1e-4),
        },

        # ── LR Scheduler ablation ───────────────────────────────────────────
        {
            "name": "scheduler_step",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu"),
            "train": TrainConfig(scheduler="step", step_size=10, gamma=0.1),
        },
        {
            "name": "scheduler_cosine",
            "model": ModelConfig(hidden_sizes=[256, 128], activation="relu"),
            "train": TrainConfig(scheduler="cosine"),
        },
    ]
    return configs


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(device: torch.device) -> None:
    """Run the full ablation study sequentially.

    Results from each experiment are accumulated and saved to a JSON summary
    file in the outputs directory.

    Args:
        device: Compute device to run all experiments on.
    """
    configs = get_ablation_configs()
    all_results: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  Starting ablation study: {len(configs)} experiments")
    print(f"{'='*60}\n")

    for cfg in configs:
        path_cfg = PathConfig(experiment_name=cfg["name"])
        # Only plot t-SNE for baseline to save time
        plot_tsne = cfg["name"] == "baseline"

        result = run_experiment(
            model_cfg=cfg["model"],
            train_cfg=cfg["train"],
            path_cfg=path_cfg,
            device=device,
            plot_tsne_flag=plot_tsne,
        )
        all_results.append(result)

    # Save summary JSON
    os.makedirs("./outputs", exist_ok=True)
    summary_path = "./outputs/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print leaderboard
    all_results.sort(key=lambda x: x["test_acc"], reverse=True)
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS (sorted by test accuracy)")
    print(f"{'='*60}")
    print(f"  {'Experiment':<25} {'Test Acc':>10} {'Val Loss':>12} {'#Params':>10}")
    print(f"  {'-'*57}")
    for r in all_results:
        print(
            f"  {r['experiment_name']:<25} "
            f"{r['test_acc']:>9.2f}% "
            f"{r['best_val_loss']:>12.6f} "
            f"{r['num_params']:>10,}"
        )
    print(f"\nFull results saved → {summary_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run either single-experiment or ablation mode.

    Run modes:

    Single experiment (default)::

        python main.py --hidden_sizes 256 128 --activation relu --epochs 20

    Ablation study::

        python main.py --ablation

    """
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation:
        run_ablation(device)
    else:
        model_cfg, train_cfg, path_cfg = args_to_configs(args)
        run_experiment(model_cfg, train_cfg, path_cfg, device)


if __name__ == "__main__":
    main()