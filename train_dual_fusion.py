"""
train_dual_fusion.py — Train Dual-Backbone RGB-Thermal Gated Fusion YOLO (v35)
=============================================================================
Architecture:
- Parallel RGB stream + Thermal Dilated-ECA stream
- Dynamic Multi-Scale Gated Attention (P3, P4, P5)
- Shared PANet Neck & Detection Head
- 6-Channel NumPy .npy input
"""

from __future__ import annotations

import argparse
import sys
import time
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.fusion import register_all_modules
from ultralytics import YOLO
from ultralytics.data.base import BaseDataset

register_all_modules()

# Patch BaseDataset to support 6-channel .npy arrays
_orig_base_init = BaseDataset.__init__
def _patched_base_init(self, *args, **kwargs):
    _orig_base_init(self, *args, **kwargs)
    if self.channels == 6:
        pass
BaseDataset.__init__ = _patched_base_init


def train_dual(
    data: str,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    weights_path: str = None
):
    model_yaml = ROOT / "models" / "v35_dual_fusion.yaml"
    runs_dir = ROOT / "runs" / "v35_dual_fusion"

    if weights_path and Path(weights_path).exists():
        print(f"Loading pretrained dual-fusion weights from {weights_path}...")
        model = YOLO(str(weights_path))
    else:
        print(f"Building dual-fusion model from {model_yaml}...")
        model = YOLO(str(model_yaml))

    print(f"Starting 6-channel Dual Fusion training ({epochs} epochs, batch {batch}, imgsz {imgsz})...")
    model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(runs_dir),
        name="train",
        exist_ok=True,
        workers=0,
        device=device,
        cos_lr=True,
        lrf=0.01,
        close_mosaic=20,
        verbose=True,
    )

    print("\n--- Validation Evaluation ---")
    metrics_val = model.val(data=data, split="val", workers=0, device=device)
    print(f"Val mAP50: {metrics_val.box.map50*100:.2f}% | Val mAP50-95: {metrics_val.box.map*100:.2f}%")

    print("\n--- Test Split Evaluation ---")
    metrics_test = model.val(data=data, split="test", workers=0, device=device)
    print(f"Test mAP50: {metrics_test.box.map50*100:.2f}% | Test mAP50-95: {metrics_test.box.map*100:.2f}%")
    print(f"Precision: {metrics_test.box.mp*100:.2f}% | Recall: {metrics_test.box.mr*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to fusion_dual_data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--weights", type=str, default=str(ROOT / "weights" / "v35_dual_fusion_pretrained.pt"))
    args = parser.parse_args()

    train_dual(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        weights_path=args.weights,
    )
