"""
train.py
────────
Full training loop featuring:
  • Mixed-precision (torch.amp) for faster GPU training
  • Warmup + Cosine Annealing LR scheduler
  • Gradient clipping
  • Early stopping on val macro-F1
  • Staged fine-tuning: head-only for warmup epochs, then unfreeze backbone
  • Checkpoint saving (top-k by macro-F1)
  • TensorBoard-compatible CSV logging

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm
import yaml

from dataset import build_dataloaders
from model import build_model, build_criterion
from utils import save_checkpoint, EarlyStopping, setup_logger


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler builder
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_name = cfg["training"]["scheduler"]
    epochs     = cfg["training"]["epochs"]
    warmup     = cfg["training"]["warmup_epochs"]

    if sched_name == "cosine":
        # Linear warmup → cosine decay
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress))).item()
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif sched_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, optimizer,
    scaler, device, num_classes, is_train=True
) -> Dict[str, float]:

    print(f"Starting run_epoch, is_train={is_train}")
    model.train() if is_train else model.eval()
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for i, (images, labels) in enumerate(loader):
            print(f"Batch {i} loaded.")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            print("Running forward pass...")
            # with autocast(enabled=scaler is not None):
            logits = model(images)
            loss   = criterion(logits, labels)

            print(f"Forward pass done, loss={loss.item()}")
            if is_train:
                print("Running backward pass...")
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                print("Backward pass done.")

            total_loss += loss.item()
            n_batches  += 1

            preds = logits.argmax(dim=1)
            if labels.dim() > 1:
                hard_labels = labels.argmax(dim=1)
            else:
                hard_labels = labels
            f1_metric.update(preds, hard_labels)
            print(f"Batch {i} done.")

    return {
        "loss"     : total_loss / max(n_batches, 1),
        "macro_f1" : f1_metric.compute().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(cfg["paths"]["log_file"])
    logger.info(f"Device: {device}")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(cfg)
    num_classes = cfg["data"]["num_classes"]

    # ── Model / Loss / Optimizer ─────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    early_stop = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])

    # Staged fine-tuning: freeze backbone first
    model.freeze_backbone()
    logger.info(f"Trainable params (head only): {model.trainable_params():,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler() if cfg["training"]["mixed_precision"] and device.type == "cuda" else None

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = Path(cfg["paths"]["report_dir"]) / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_f1",
                         "val_loss", "val_f1", "lr"])

    best_val_f1 = 0.0
    warmup_done = False

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        # ── Unfreeze backbone after warmup ────────────────────────────────────
        if epoch == cfg["training"]["warmup_epochs"] + 1 and not warmup_done:
            model.unfreeze_backbone()
            # Rebuild optimizer so all params are tracked
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["learning_rate"] * 0.1,   # lower LR for backbone
                weight_decay=cfg["training"]["weight_decay"],
            )
            scheduler = build_scheduler(optimizer, cfg, len(train_loader))
            logger.info(f"Backbone unfrozen. Trainable: {model.trainable_params():,}")
            warmup_done = True

        # ── Train & validate ──────────────────────────────────────────────────
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, num_classes, is_train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, None,
            None, device, num_classes, is_train=False
        )

        # ── LR step ───────────────────────────────────────────────────────────
        if cfg["training"]["scheduler"] == "plateau":
            scheduler.step(val_metrics["macro_f1"])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        logger.info(
            f"Epoch [{epoch:02d}/{cfg['training']['epochs']}]  "
            f"Train Loss: {train_metrics['loss']:.4f}  F1: {train_metrics['macro_f1']:.4f}  |  "
            f"Val   Loss: {val_metrics['loss']:.4f}  F1: {val_metrics['macro_f1']:.4f}  |  "
            f"LR: {current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # ── CSV ───────────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(train_metrics["loss"], 6),
                round(train_metrics["macro_f1"], 6),
                round(val_metrics["loss"], 6),
                round(val_metrics["macro_f1"], 6),
                round(current_lr, 8),
            ])

        # ── Checkpoint ────────────────────────────────────────────────────────
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, val_f1,
                            ckpt_dir / "best_model.pth")
            logger.info(f"  * New best val macro-F1: {best_val_f1:.4f}")

        save_checkpoint(model, optimizer, epoch, val_f1,
                        ckpt_dir / f"epoch_{epoch:03d}.pth")

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stop(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info(f"Training complete. Best val macro-F1: {best_val_f1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    
    # Resolve default config path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_cfg = os.path.join(project_root, "configs", "config.yaml")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_cfg)
    args = parser.parse_args()
    import traceback
    try:
        train(args.config)
    except Exception as e:
        print("ERROR CAUGHT:")
        traceback.print_exc()
        raise
