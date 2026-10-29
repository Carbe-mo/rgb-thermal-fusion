"""
models/v38_eca_neck_fusion/train.py
===================================
Train v38 Receptive-Field-Aligned Dual-Stream RGB+Thermal Fusion Detector with PANetNeckECA.
Uses stock Ultralytics DetectionTrainer with cache='ram' for fast epoch throughput.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402

from common.results_logger import log_run  # noqa: E402
from models.v38_eca_neck_fusion.model import V38DualStreamModel  # noqa: E402

VARIANTS = {
    "ca_dilated":  dict(data="fusion6_data.yaml", rgb_mode="ca_dilated",  copy_paste=0.0),
    "eca_dilated": dict(data="fusion6_data.yaml", rgb_mode="eca_dilated", copy_paste=0.0),
    "ca_cp15":     dict(data="fusion6_data.yaml", rgb_mode="ca_dilated",  copy_paste=0.15),
}


def make_trainer_cls(rgb_mode: str):
    class V38Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = V38DualStreamModel(nc=self.data["nc"], rgb_mode=rgb_mode,
                                       neck_eca=True, verbose=verbose)
            ok, tot = model.load_yolov8n(str(PROJECT_ROOT / "yolov8n.pt"))
            print(f"[v38] transferred {ok}/{tot} tensors from yolov8n.pt")
            return model
    return V38Trainer


def evaluate(weights: Path, data_yaml: Path, split: str, project: Path, rect: bool = True, iou: float = 0.65):
    model, _ = load_checkpoint(str(weights))
    model = model.float()
    tag = f"eval_{split}_{'rect' if rect else 'sq'}"
    v = DetectionValidator(args=dict(data=str(data_yaml), split=split, imgsz=640, batch=16,
                                     rect=rect, iou=iou, workers=0, device=0,
                                     project=str(project), name=tag, exist_ok=True, plots=False))
    v(model=model)
    m = v.metrics
    out = dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
               precision=float(m.box.mp), recall=float(m.box.mr),
               inference_ms=float(m.speed.get("inference", 0.0)))
    per_class = {v.names[i]: (round(float(m.box.ap50[j]) * 100, 2), round(float(m.box.ap[j]) * 100, 2))
                 for j, i in enumerate(m.box.ap_class_index)}
    return out, per_class


def main(variant: str, epochs: int, seed: int, batch: int):
    cfg = VARIANTS[variant]
    data_yaml = PROJECT_ROOT / "data" / cfg["data"]
    name = f"v38_{variant}" + (f"_s{seed}" if seed != 0 else "")
    runs = PROJECT_ROOT / "runs" / "v38_eca_neck_fusion"

    Trainer = make_trainer_cls(cfg["rgb_mode"])
    trainer = Trainer(overrides=dict(
        model="v38_dual_stream", data=str(data_yaml), epochs=epochs, batch=batch, imgsz=640,
        project=str(runs), name=name, exist_ok=True, workers=0, device=0, cache="ram",
        cos_lr=True, lrf=0.01, close_mosaic=20, copy_paste=cfg["copy_paste"],
        seed=seed, deterministic=True, verbose=True, plots=True,
    ))
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60

    best = runs / name / "weights" / "best.pt"
    val_rect, val_cls = evaluate(best, data_yaml, "val", runs / name, rect=True)
    test_rect, test_cls = evaluate(best, data_yaml, "test", runs / name, rect=True)
    val_sq, _ = evaluate(best, data_yaml, "val", runs / name, rect=False)
    test_sq, test_sq_cls = evaluate(best, data_yaml, "test", runs / name, rect=False)

    model, _ = load_checkpoint(str(best))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    note = (f"variant={variant} rgb_mode={cfg['rgb_mode']} neck_eca=True "
            f"val_rect={val_rect['mAP50']*100:.2f}% test_rect={test_rect['mAP50']*100:.2f}% "
            f"test_sq={test_sq['mAP50']*100:.2f}%")
    for split, met in (("val", val_rect), ("test", test_rect)):
        log_run(
            model_id=name if split == "val" else f"{name}_eval",
            model_name=f"{name} (RF-Aligned Dual-Stream + NeckECA) [{split}]",
            folder="models/v38_eca_neck_fusion",
            params_M=round(n_params, 2), gflops=11.6, epochs=epochs,
            mAP50=round(met["mAP50"], 4), mAP50_95=round(met["mAP50_95"], 4),
            precision=round(met["precision"], 4), recall=round(met["recall"], 4),
            inference_ms=round(met["inference_ms"], 2), train_time_min=round(train_min, 1),
            notes=note + (" | official test split (108 images)" if split == "test" else ""),
        )
    print(f"\n[v38] {name} VAL (rect=True) : {val_rect} | per-cls: {val_cls}")
    print(f"[v38] {name} TEST (rect=True): {test_rect} | per-cls: {test_cls}")
    print(f"[v38] {name} VAL (rect=False): {val_sq}")
    print(f"[v38] {name} TEST (rect=False): {test_sq} | per-cls: {test_sq_cls}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ca_dilated", choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    main(a.variant, a.epochs, a.seed, a.batch)
