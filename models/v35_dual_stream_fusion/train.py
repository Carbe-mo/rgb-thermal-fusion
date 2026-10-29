"""
models/v35_dual_stream_fusion/train.py
=======================================
Train the Dual-Stream RGB-Thermal Fusion YOLO model.

Uses a custom PyTorch training loop (not standard Ultralytics trainer)
because the model requires dual-input (RGB + Thermal) per sample.

Data loading:
  - Each sample loads a thermal image and its paired RGB image
  - Thermal: FLIR{N}.jpg (even IDs) from data/thermal/images/
  - RGB: FLIR{N+1}.jpg (odd IDs) from D:\\Yolo\\Data\\project_flir\\100_FLIR\\
  - Labels: YOLO format .txt from data/thermal/labels/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import generate_splits, CLASS_NAMES, NC
from models.v35_dual_stream_fusion.dual_stream_model import build_model


# ---------------------------------------------------------------------------
# Paired RGB-Thermal Dataset
# ---------------------------------------------------------------------------
class PairedRGBThermalDataset(Dataset):
    """
    Loads paired RGB + Thermal images and YOLO-format labels.
    """
    THERMAL_DIR = PROJECT_ROOT / "data" / "thermal" / "images"
    LABEL_DIR = PROJECT_ROOT / "data" / "thermal" / "labels"
    RGB_DIR = Path(r"D:\Yolo\Data\project_flir\100_FLIR")

    def __init__(self, stems: list[str], img_size: int = 640, augment: bool = False):
        self.stems = stems
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        flir_id = int(stem.replace("FLIR", ""))

        # Load thermal image
        thm_path = self.THERMAL_DIR / f"{stem}.jpg"
        thm_img = cv2.imread(str(thm_path))
        if thm_img is None:
            thm_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Load RGB pair (odd ID = flir_id + 1)
        rgb_path = self.RGB_DIR / f"FLIR{flir_id + 1}.jpg"
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            rgb_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Resize both to target size
        thm_img = cv2.resize(thm_img, (self.img_size, self.img_size))
        rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size))

        # Augmentation (basic)
        if self.augment:
            if np.random.random() > 0.5:
                thm_img = cv2.flip(thm_img, 1)
                rgb_img = cv2.flip(rgb_img, 1)

        # Normalize and convert to tensors [C, H, W]
        thm_tensor = torch.from_numpy(thm_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1).astype(np.float32) / 255.0)

        # Load labels (YOLO format: cls cx cy w h)
        lbl_path = self.LABEL_DIR / f"{stem}.txt"
        targets = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        targets.append([cls, cx, cy, w, h])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))

        return rgb_tensor, thm_tensor, targets, stem


def collate_fn(batch):
    """Custom collate: stack images, keep targets as list."""
    rgb_imgs, thm_imgs, targets_list, stems = zip(*batch)
    rgb_batch = torch.stack(rgb_imgs, 0)
    thm_batch = torch.stack(thm_imgs, 0)
    return rgb_batch, thm_batch, list(targets_list), list(stems)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(
    epochs: int = 100,
    batch_size: int = 8,
    img_size: int = 640,
    lr: float = 0.001,
    device: str = "0",
):
    print("=" * 65)
    print("  v35 DUAL-STREAM RGB-THERMAL FUSION YOLO — TRAINING")
    print("=" * 65)

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load splits
    splits = generate_splits()
    train_stems = splits["train"]
    val_stems = splits["val"]
    test_stems = splits["test"]

    # Datasets
    train_ds = PairedRGBThermalDataset(train_stems, img_size=img_size, augment=True)
    val_ds = PairedRGBThermalDataset(val_stems, img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=collate_fn, pin_memory=True)

    # Build model
    model = build_model(nc=NC, scale="n").to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {total_params:.2f}M")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Simple detection loss placeholder
    # NOTE: For full training, integrate Ultralytics v8 detection loss (TaskAlignedAssigner + DFL)
    # This is a simplified training skeleton demonstrating the dual-stream forward pass
    print("\n  ⚠ NOTE: This uses a simplified loss for demonstration.")
    print("  For full training, integrate Ultralytics TaskAlignedAssigner + DFL loss.\n")

    save_dir = PROJECT_ROOT / "runs" / "v35_dual_stream_fusion"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for rgb_batch, thm_batch, targets, stems in train_loader:
            rgb_batch = rgb_batch.to(device)
            thm_batch = thm_batch.to(device)

            optimizer.zero_grad()

            # Forward pass through dual-stream model
            preds = model(rgb_batch, thm_batch)

            # Simplified placeholder loss (L2 on raw predictions)
            # Replace with proper detection loss for real training
            if isinstance(preds, (list, tuple)):
                loss = sum(p.mean() ** 2 for p in preds)
            else:
                loss = preds.mean() ** 2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.6f} | Time: {elapsed:.1f}min")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

    # Save final
    torch.save(model.state_dict(), save_dir / "last.pt")
    total_time = (time.time() - t0) / 60
    print(f"\n  Training complete! Total time: {total_time:.1f} min")
    print(f"  Best model saved to: {save_dir / 'best.pt'}")
    print(f"  Final model saved to: {save_dir / 'last.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v35 Dual-Stream Fusion YOLO")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz, lr=args.lr, device=args.device)
