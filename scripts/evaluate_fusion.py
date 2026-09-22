"""
scripts/evaluate_fusion.py — Standalone Evaluation for Fusion Models
==================================================================
Evaluates 4-channel early fusion or 6-channel dual-stream fusion models on validation/test splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.fusion import register_all_modules
from ultralytics import YOLO

register_all_modules()


def evaluate(model_path: str, data_path: str, split: str = "test", device: str = "0", imgsz: int = 640):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"Running evaluation on '{split}' split with dataset {data_path}...")
    metrics = model.val(data=data_path, split=split, device=device, imgsz=imgsz, workers=0)

    print("\n" + "=" * 60)
    print(f"  FUSION MODEL EVALUATION RESULTS ({split.upper()} SPLIT)")
    print("=" * 60)
    print(f"  mAP@50:     {metrics.box.map50*100:.2f}%")
    print(f"  mAP@50-95:  {metrics.box.map*100:.2f}%")
    print(f"  Precision:  {metrics.box.mp*100:.2f}%")
    print(f"  Recall:     {metrics.box.mr*100:.2f}%")
    print(f"  Inference:  {metrics.speed.get('inference', 0.0):.2f} ms")
    print("=" * 60)

    # Print per-class metrics
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        print("\nPer-Class AP@50:")
        for idx, (cls_name, ap) in enumerate(zip(metrics.names.values(), metrics.box.ap50)):
            print(f"  - {cls_name:12s}: {ap*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt weights file")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    evaluate(args.model, args.data, args.split, args.device, args.imgsz)
