"""
evaluate.py
───────────
Loads the best checkpoint and evaluates on the test set.

Outputs:
  • Macro-F1, Per-class F1, Precision, Recall
  • Confusion matrix (saved as PNG)
  • Full classification report (saved as TXT)

Usage:
    python src/evaluate.py --config configs/config.yaml \
                           --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import build_dataloaders, get_val_transforms
from model import build_model


def evaluate(cfg_path: str, ckpt_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["classes"]
    report_dir = Path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val F1 = {ckpt.get('val_f1', 0):.4f})")

    # ── Test loader ───────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(cfg)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        digits=4,
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    report_txt = report_dir / "classification_report.txt"
    report_txt.write_text(report)

    # ── Per-class F1 JSON ─────────────────────────────────────────────────────
    report_dict = classification_report(
        all_labels, all_preds, target_names=classes,
        digits=4, output_dict=True
    )
    (report_dir / "metrics.json").write_text(
        json.dumps(report_dict, indent=2)
    )
    print(f"\nMacro F1: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Reports saved to: {report_dir}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    cm_path = report_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    # ── Per-class F1 bar chart ────────────────────────────────────────────────
    f1_scores = [report_dict[c]["f1-score"] for c in classes]
    colors_bar = ["#2ecc71" if f >= 0.9 else "#f39c12" if f >= 0.75 else "#e74c3c"
                  for f in f1_scores]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(classes, f1_scores, color=colors_bar, edgecolor="white", linewidth=0.8)
    ax.axhline(y=np.mean(f1_scores), color="navy", linestyle="--",
               linewidth=1.5, label=f"Macro avg F1 = {np.mean(f1_scores):.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Disease Class", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
    ax.legend()
    plt.xticks(rotation=40, ha="right")

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    f1_path = report_dir / "per_class_f1.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class F1 chart saved: {f1_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
