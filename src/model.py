"""
model.py
────────
EfficientNet-B4 classifier with:
  - Custom multi-head classification head (dropout → BN → FC)
  - Configurable label smoothing
  - Grad-CAM compatible (exposes feature extractor)
  - Freeze / unfreeze utilities for staged fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Label Smoothing Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing. Works with both hard int labels
    and soft float label vectors (produced by MixUp).
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            # Hard labels — convert to one-hot then smooth
            with torch.no_grad():
                soft = torch.zeros_like(log_probs)
                soft.fill_(self.smoothing / (n_classes - 1))
                soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        else:
            # Already soft labels from MixUp
            soft = targets.clamp(0, 1)

        loss = -(soft * log_probs).sum(dim=-1)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class CropDiseaseClassifier(nn.Module):
    """
    EfficientNet-B4 backbone + custom classification head.

    Architecture of the head:
        AdaptiveAvgPool  →  Dropout(p)  →  BatchNorm1d  →  Linear(num_classes)

    Args:
        num_classes  : Number of disease categories.
        pretrained   : Load ImageNet weights.
        dropout      : Dropout probability in the head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # Remove built-in classifier
            global_pool="",         # We handle pooling ourselves
        )
        # Spatial feature map dimension for EfficientNet-B4
        in_features = self.backbone.num_features   # 1792

        # ── Custom head ──────────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, num_classes),
        )

        # ── Weight init for the head ─────────────────────────────────────────
        nn.init.kaiming_normal_(self.head[2].weight, mode="fan_out")
        nn.init.zeros_(self.head[2].bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                # (B, 1792, H', W')
        pooled   = self.global_pool(features)      # (B, 1792, 1, 1)
        flat     = pooled.flatten(1)               # (B, 1792)
        logits   = self.head(flat)                 # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw spatial feature map — used by Grad-CAM."""
        return self.backbone(x)

    # ── Staged fine-tuning helpers ────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone parameters (train head only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, num_blocks: int = -1):
        """
        Unfreeze backbone.
        num_blocks=-1  → unfreeze everything
        num_blocks=N   → unfreeze the last N blocks only (gradual unfreezing)
        """
        if num_blocks == -1:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            blocks = list(self.backbone.blocks)
            for block in blocks[-num_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> CropDiseaseClassifier:
    model = CropDiseaseClassifier(
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    )
    return model


def build_criterion(cfg: dict) -> LabelSmoothingCrossEntropy:
    return LabelSmoothingCrossEntropy(
        smoothing=cfg["model"].get("label_smoothing", 0.1)
    )


if __name__ == "__main__":
    # Quick sanity check
    model = CropDiseaseClassifier(num_classes=10)
    x = torch.randn(2, 3, 380, 380)
    out = model(x)
    print(f"Output shape : {out.shape}")          # (2, 10)
    print(f"Total params : {model.total_params():,}")
    print(f"Trainable    : {model.trainable_params():,}")
