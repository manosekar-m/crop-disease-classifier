"""
utils.py
────────
Shared helpers:
  • Logger setup
  • Checkpoint save / load
  • EarlyStopping callback
  • Training curve plotting
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: str, level=logging.INFO) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("CropClassifier")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    path: Path,
):
    torch.save({
        "epoch"       : epoch,
        "val_f1"      : val_f1,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training if the monitored metric does not improve for `patience`
    consecutive epochs.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter    = 0
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_csv: str, output_path: str):
    """
    Reads the CSV written by train.py and saves a 2-panel figure:
      Left  → train / val loss
      Right → train / val macro-F1
    """
    df = pd.read_csv(log_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Curves — Crop Disease Classifier",
                 fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss",
                 color="#2980b9", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",
                 color="#e74c3c", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    # F1
    axes[1].plot(df["epoch"], df["train_f1"], label="Train Macro-F1",
                 color="#27ae60", linewidth=2)
    axes[1].plot(df["epoch"], df["val_f1"],   label="Val Macro-F1",
                 color="#f39c12", linewidth=2, linestyle="--")
    axes[1].axhline(df["val_f1"].max(), color="grey", linestyle=":",
                    linewidth=1.2, label=f"Best val F1 = {df['val_f1'].max():.4f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Macro-F1"); axes[1].legend()
    axes[1].set_ylim(0, 1.0); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {output_path}")
