"""
predict.py
──────────
Single-image (or folder-batch) inference with:
  • Top-5 class probabilities
  • Grad-CAM heatmap overlay
  • JSON output for API integration

Usage:
    # Single image
    python src/predict.py --image path/to/leaf.jpg \
                          --config configs/config.yaml \
                          --checkpoint outputs/checkpoints/best_model.pth

    # Folder batch
    python src/predict.py --image_dir data/sample/ \
                          --config configs/config.yaml \
                          --checkpoint outputs/checkpoints/best_model.pth \
                          --output_dir outputs/gradcam/
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from dataset import get_val_transforms
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM heatmap from the last convolutional block.
    Works with any model that exposes `get_features()` returning a
    (B, C, H, W) spatial feature map.
    """

    def __init__(self, model: torch.nn.Module):
        self.model    = model
        self.gradients = None
        self.features  = None
        self._register_hooks()

    def _register_hooks(self):
        # Target: last block of EfficientNet-B4 backbone
        target_layer = self.model.backbone.blocks[-1]

        def forward_hook(module, inp, output):
            self.features = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,  # (1, 3, H, W)
        class_idx: int,
    ) -> np.ndarray:
        self.model.eval()
        image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pooled gradients
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])      # (C,)
        features     = self.features[0]                         # (C, H', W')

        for i, g in enumerate(pooled_grads):
            features[i] *= g

        heatmap = features.mean(dim=0).cpu().detach().numpy()  # (H', W')
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    @staticmethod
    def overlay(
        orig_img_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        h, w = orig_img_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        colored         = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlaid        = cv2.addWeighted(orig_img_bgr, 1 - alpha,
                                          colored, alpha, 0)
        return overlaid


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    image_path: str,
    model: torch.nn.Module,
    transform,
    classes: List[str],
    device: torch.device,
    gradcam: GradCAM,
    output_dir: Path,
) -> dict:
    """Run inference + Grad-CAM on one image, save visual, return result dict."""

    # Load & preprocess
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top5_idx  = probs.argsort()[::-1][:5]
    top5      = [{"class": classes[i], "probability": float(probs[i])} for i in top5_idx]
    pred_idx  = int(top5_idx[0])
    pred_name = classes[pred_idx]
    conf      = float(probs[pred_idx])

    # Grad-CAM
    tensor_grad = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    heatmap     = gradcam.generate(tensor_grad, pred_idx)
    overlay_img = GradCAM.overlay(img_bgr, heatmap)

    # Save visualisation
    stem     = Path(image_path).stem
    vis_path = output_dir / f"{stem}_gradcam.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_rgb);               axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet");   axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Prediction: {pred_name}\n({conf:.1%})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=120, bbox_inches="tight")
    plt.close()

    result = {
        "image"     : str(image_path),
        "prediction": pred_name,
        "confidence": round(conf, 4),
        "top5"      : top5,
        "gradcam"   : str(vis_path),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--image",      default=None, help="Path to a single image")
    parser.add_argument("--image_dir",  default=None, help="Path to folder of images")
    parser.add_argument("--output_dir", default="outputs/gradcam")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["classes"]
    img_size = cfg["data"]["image_size"]

    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = get_val_transforms(img_size)
    gradcam   = GradCAM(model)
    out_dir   = Path(args.output_dir)

    # Collect images
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [
            str(p) for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in exts
        ]
    else:
        raise ValueError("Provide --image or --image_dir")

    results = []
    for img_path in image_paths:
        res = predict_single(img_path, model, transform, classes,
                             device, gradcam, out_dir)
        print(f"[{res['prediction']:20s}] {res['confidence']:.1%}  →  {img_path}")
        results.append(res)

    # Save JSON
    json_out = out_dir / "predictions.json"
    json_out.write_text(json.dumps(results, indent=2))
    print(f"\nAll predictions saved: {json_out}")


if __name__ == "__main__":
    main()
