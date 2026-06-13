"""
dataset.py
──────────
Custom PyTorch Dataset for multi-label crop-disease classification.
Uses Albumentations for rich augmentation and supports MixUp on the fly.

Directory layout expected:
    data/processed/
        train/
            Healthy/          <- one folder per class
            Blast_Fungus/
            ...
        val/
        test/
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 380) -> A.Compose:
    """Heavy augmentation for training — simulates real-field photography."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size),
                            scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20,
                             sat_shift_limit=30,
                             val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 380) -> A.Compose:
    """Minimal pipeline for validation / test — only resize + normalize."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    """
    Loads images from a class-folder tree.

    Args:
        root_dir   : Path to split folder (e.g., 'data/processed/train').
        classes    : Ordered list of class names (defines label indices).
        transform  : Albumentations Compose pipeline.
        mixup_alpha: Beta distribution alpha for MixUp. 0.0 disables it.
    """

    def __init__(
        self,
        root_dir: str,
        classes: List[str],
        transform: Optional[A.Compose] = None,
        mixup_alpha: float = 0.0,
    ):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.mixup_alpha = mixup_alpha

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"[WARN] Class folder not found: {cls_dir}")
                continue
            for fp in cls_dir.iterdir():
                if fp.suffix.lower() in extensions:
                    self.samples.append((fp, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(
                f"No images found under '{self.root_dir}'. "
                "Check your directory layout and class names in config.yaml."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _apply_mixup(
        self,
        img: torch.Tensor,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blends current sample with a random sample from the dataset."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        rand_idx = random.randint(0, len(self.samples) - 1)
        rand_path, rand_label = self.samples[rand_idx]
        rand_img = self._load_image(rand_path)
        if self.transform:
            rand_img = self.transform(image=rand_img)["image"]
        mixed_img = lam * img + (1 - lam) * rand_img

        # Soft label vector
        n = len(self.classes)
        soft_label = torch.zeros(n)
        soft_label[label] += lam
        soft_label[rand_label] += (1 - lam)
        return mixed_img, soft_label

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = self._load_image(path)

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.mixup_alpha > 0 and self.training_mode:
            image, label_tensor = self._apply_mixup(image, label)
            return image, label_tensor

        return image, torch.tensor(label, dtype=torch.long)

    # Helper so Dataset knows whether we're training (for MixUp)
    training_mode: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / val / test DataLoaders from config dict.
    Applies class-balanced WeightedRandomSampler to handle imbalanced datasets.
    """
    classes    = cfg["classes"]
    root       = cfg["data"]["root_dir"]
    img_size   = cfg["data"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    workers    = cfg["data"]["num_workers"]
    mixup_a    = cfg["model"].get("mixup_alpha",
                  cfg.get("augmentation", {}).get("mixup_alpha", 0.0))

    train_ds = CropDiseaseDataset(
        root_dir=f"{root}/train",
        classes=classes,
        transform=get_train_transforms(img_size),
        mixup_alpha=mixup_a,
    )
    train_ds.training_mode = True

    val_ds = CropDiseaseDataset(
        root_dir=f"{root}/val",
        classes=classes,
        transform=get_val_transforms(img_size),
    )
    val_ds.training_mode = False

    test_ds = CropDiseaseDataset(
        root_dir=f"{root}/test",
        classes=classes,
        transform=get_val_transforms(img_size),
    )
    test_ds.training_mode = False

    # ── Weighted sampler to handle class imbalance ───────────────────────────
    class_counts = [0] * len(classes)
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    class_weights = [1.0 / (c + 1e-6) for c in class_counts]
    sample_weights = [class_weights[lbl] for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
