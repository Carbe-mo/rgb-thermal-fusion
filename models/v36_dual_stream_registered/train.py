"""
models/v36_dual_stream_registered/train.py
==========================================
Train the dual-stream RGB+thermal fusion detector with the stock Ultralytics
DetectionTrainer (same loss, assigner, mosaic, validator and hyper-parameters as v30).

Variants:
  --variant main        CA+ECA RGB stream + ECA/Dilated thermal stream + gated fusion,
                        registered RGB (data/fusion6)
  --variant unreg       same model, UNregistered RGB (data/fusion6_unreg)   [control]
  --variant thermal     thermal-only stream inside the identical 6-ch pipeline [control]
  --variant eca_rgb     RGB stream without SE-style CA (C2f_ECA only)        [ablation]

Hyper-parameters copied from models/v30_v27_cosine_100ep/train.py:
  epochs=100, batch=16, imgsz=640, cos_lr=True, lrf=0.01, close_mosaic=20, init yolov8n.pt
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
from models.v36_dual_stream_registered.model import DualStreamDetectionModel  # noqa: E402

VARIANTS = {
    "main":    dict(data="fusion6_data.yaml",       rgb_attn="ca_eca", thermal_only=False),
    "unreg":   dict(data="fusion6_unreg_data.yaml", rgb_attn="ca_eca", thermal_only=False),
    "thermal": dict(data="fusion6_data.yaml",       rgb_attn="ca_eca", thermal_only=True),
    "eca_rgb": dict(data="fusion6_data.yaml",       rgb_attn="eca",    thermal_only=False),
}


def make_trainer_cls(rgb_attn: str, thermal_only: bool):
    class DualTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = DualStreamDetectionModel(nc=self.data["nc"], rgb_attn=rgb_attn,
                                             thermal_only=thermal_only, verbose=verbose)
            ok, tot = model.load_yolov8n("yolov8n.pt")
            print(f"[v36] transferred {ok}/{tot} tensors from yolov8n.pt")
            return model
    return DualTrainer


def evaluate(weights: Path, data_yaml: Path, split: str, project: Path):
    model, _ = load_checkpoint(str(weights))
    model = model.float()
    v = DetectionValidator(args=dict(data=str(data_yaml), split=split, imgsz=640, batch=16,
                                     workers=0, device=0, project=str(project),
                                     name=f"eval_{split}", exist_ok=True, plots=True))
    v(model=model)
    m = v.metrics
    out = dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
               precision=float(m.box.mp), recall=float(m.box.mr),
               inference_ms=float(m.speed.get("inference", 0.0)))
    per_class = {v.names[i]: (float(m.box.ap50[j]), float(m.box.ap[j]))
                 for j, i in enumerate(m.box.ap_class_index)}
    return out, per_class


def main(variant: str, epochs: int, seed: int, batch: int):
    cfg = VARIANTS[variant]
    data_yaml = PROJECT_ROOT / "data" / cfg["data"]
    name = f"v36_{variant}" + (f"_s{seed}" if seed != 0 else "")
    runs = PROJECT_ROOT / "runs" / "v36_dual_stream_registered"

    Trainer = make_trainer_cls(cfg["rgb_attn"], cfg["thermal_only"])
    trainer = Trainer(overrides=dict(
        model="v36_dual_stream", data=str(data_yaml), epochs=epochs, batch=batch, imgsz=640,
        project=str(runs), name=name, exist_ok=True, workers=0, device=0,
        cos_lr=True, lrf=0.01, close_mosaic=20, seed=seed, deterministic=True,
        verbose=True, plots=True,
    ))
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60

    best = runs / name / "weights" / "best.pt"
    val, _ = evaluate(best, data_yaml, "val", runs / name)
    test, per_class = evaluate(best, data_yaml, "test", runs / name)
    model, _ = load_checkpoint(str(best))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    note = (f"variant={variant} rgb_attn={cfg['rgb_attn']} thermal_only={cfg['thermal_only']} "
            f"data={cfg['data']} seed={seed}")
    for split, met in (("val", val), ("test", test)):
        log_run(
            model_id=name if split == "val" else f"{name}_eval",
            model_name=f"{name} (dual-stream registered RGB+thermal) [{split}]",
            folder="models/v36_dual_stream_registered",
            params_M=round(n_params, 2), gflops=0.0, epochs=epochs,
            mAP50=round(met["mAP50"], 4), mAP50_95=round(met["mAP50_95"], 4),
            precision=round(met["precision"], 4), recall=round(met["recall"], 4),
            inference_ms=round(met["inference_ms"], 2), train_time_min=round(train_min, 1),
            notes=note + (" | official test split (108 images)" if split == "test" else ""),
        )
    print(f"\n[v36] {name} VAL : {val}\n[v36] {name} TEST: {test}\n[v36] per-class test (AP50, AP50-95): {per_class}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    main(a.variant, a.epochs, a.seed, a.batch)
