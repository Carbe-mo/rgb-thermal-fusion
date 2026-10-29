"""
models/v37_cs_cmaf_fusion/train.py
==================================
Train the upgraded v37 Dual-Stream RGB-Thermal Fusion detector with:
  1. CS-CMAF (Channel-Spatial Cross-Modal Differential Attention + Local Parallax Alignment)
  2. PANetNeckECA (ECA attention in all 4 shared neck C2f blocks)
  3. Dual-Stream 6-Channel HSV + Radiometric Gain/Contrast Augmentation
  4. Hybrid Pretrained Initialization (COCO yolov8n for RGB + v30 champion for Thermal/Neck/Head,
     or pure COCO yolov8n initialization)
  5. Evaluation under both square (rect=False) and aspect-preserving rectangular (rect=True,
     matching YOLO.val() default used by v30) protocols.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ultralytics.data.augment as _aug  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402

from common.results_logger import log_run  # noqa: E402
from models.v37_cs_cmaf_fusion.model import DualStreamV37DetectionModel  # noqa: E402


def enable_6ch_dual_hsv() -> None:
    """Monkey-patch RandomHSV.apply_image so 6-channel [RGB, Thermal] images receive
    independent HSV augmentation on both the RGB stream (ch 0..2) and the pseudo-color
    thermal stream (ch 3..5), restoring the augmentation that 3-ch v30 received."""
    def _apply_6ch(self, labels, params=None):
        img = labels["img"]
        if not (self.hgain or self.sgain or self.vgain):
            return labels
        if img.shape[-1] == 3:
            _hsv_inplace(img, self.hgain, self.sgain, self.vgain)
        elif img.shape[-1] == 6:
            # Independent random HSV jitter for RGB stream and Thermal stream
            rgb = np.ascontiguousarray(img[:, :, :3])
            thm = np.ascontiguousarray(img[:, :, 3:6])
            _hsv_inplace(rgb, self.hgain, self.sgain, self.vgain)
            _hsv_inplace(thm, self.hgain * 0.5, self.sgain, self.vgain)
            img[:, :, :3] = rgb
            img[:, :, 3:6] = thm
        return labels

    def _hsv_inplace(im: np.ndarray, hgain: float, sgain: float, vgain: float) -> None:
        dtype = im.dtype
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)

    _aug.RandomHSV.apply_image = _apply_6ch


VARIANTS = {
    # v37a: CS-CMAF + NeckECA + 6ch Dual-HSV + v30 Warm-Start (CA+ECA on RGB)
    "v37a_warm_ca": dict(rgb_attn="ca_eca", neck_eca=True, use_align=True, warm_v30=True, copy_paste=0.0, mixup=0.0),
    # v37b: CS-CMAF + NeckECA + 6ch Dual-HSV + v30 Warm-Start (ECA on RGB)
    "v37b_warm_eca": dict(rgb_attn="eca", neck_eca=True, use_align=True, warm_v30=True, copy_paste=0.0, mixup=0.0),
    # v37c: CS-CMAF + NeckECA + 6ch Dual-HSV + COCO Cold-Start (no v30 weights, pure architecture+aug test)
    "v37c_coco_ca": dict(rgb_attn="ca_eca", neck_eca=True, use_align=True, warm_v30=False, copy_paste=0.0, mixup=0.0),
    # v37d: v37a + Copy-Paste & Mixup for minority person/unattended classes
    "v37d_warm_aug": dict(rgb_attn="ca_eca", neck_eca=True, use_align=True, warm_v30=True, copy_paste=0.15, mixup=0.10),
}


def make_trainer_cls(model_cfg: dict):
    v30_pt = str(PROJECT_ROOT / "runs" / "v30_v27_cosine_100ep" / "train" / "weights" / "best.pt")

    class V37Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = DualStreamV37DetectionModel(
                nc=self.data["nc"],
                rgb_attn=model_cfg["rgb_attn"],
                neck_eca=model_cfg["neck_eca"],
                use_align=model_cfg["use_align"],
                verbose=verbose,
            )
            ok, tot = model.load_hybrid_pretrained(
                rgb_weights="yolov8n.pt",
                thm_weights=v30_pt if model_cfg["warm_v30"] else None,
            )
            print(f"[v37] transferred {ok}/{tot} tensors (warm_v30={model_cfg['warm_v30']})")
            return model

    return V37Trainer


def evaluate(weights: Path, data_yaml: Path, split: str, project: Path, rect: bool = True):
    model, _ = load_checkpoint(str(weights))
    model = model.float()
    tag = f"eval_{split}" + ("_rect" if rect else "_sq")
    v = DetectionValidator(args=dict(
        data=str(data_yaml), split=split, imgsz=640, batch=16, rect=rect,
        workers=0, device=0, project=str(project), name=tag, exist_ok=True, plots=False, verbose=False,
    ))
    v(model=model)
    m = v.metrics
    out = dict(
        mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
        precision=float(m.box.mp), recall=float(m.box.mr),
        inference_ms=float(m.speed.get("inference", 0.0)),
    )
    per_class = {v.names[i]: (round(float(m.box.ap50[j]), 4), round(float(m.box.ap[j]), 4))
                 for j, i in enumerate(m.box.ap_class_index)}
    return out, per_class


def main(variant: str, epochs: int, seed: int, batch: int, lr0: float):
    enable_6ch_dual_hsv()
    cfg = VARIANTS[variant]
    data_yaml = PROJECT_ROOT / "data" / "fusion6_data.yaml"
    name = f"{variant}_e{epochs}" + (f"_s{seed}" if seed != 0 else "")
    runs = PROJECT_ROOT / "runs" / "v37_cs_cmaf_fusion"

    Trainer = make_trainer_cls(cfg)
    trainer = Trainer(overrides=dict(
        model="v37_cs_cmaf", data=str(data_yaml), epochs=epochs, batch=batch, imgsz=640,
        project=str(runs), name=name, exist_ok=True, workers=0, device=0,
        cos_lr=True, lr0=lr0, lrf=0.01, close_mosaic=20,
        copy_paste=cfg["copy_paste"], mixup=cfg["mixup"],
        seed=seed, deterministic=True, verbose=True, plots=True,
    ))
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60.0

    best = runs / name / "weights" / "best.pt"
    val_rect, _ = evaluate(best, data_yaml, "val", runs / name, rect=True)
    test_rect, pc_rect = evaluate(best, data_yaml, "test", runs / name, rect=True)
    test_sq, pc_sq = evaluate(best, data_yaml, "test", runs / name, rect=False)

    model, _ = load_checkpoint(str(best))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    note = f"variant={variant} warm_v30={cfg['warm_v30']} 6ch_dual_hsv=True test_sq_mAP50={test_sq['mAP50']:.4f}"
    for split, met in (("val", val_rect), ("test", test_rect)):
        log_run(
            model_id=name if split == "val" else f"{name}_eval",
            model_name=f"{name} (CS-CMAF + NeckECA + DualHSV) [{split}]",
            folder="models/v37_cs_cmaf_fusion",
            params_M=round(n_params, 2), gflops=12.1, epochs=epochs,
            mAP50=round(met["mAP50"], 4), mAP50_95=round(met["mAP50_95"], 4),
            precision=round(met["precision"], 4), recall=round(met["recall"], 4),
            inference_ms=round(met["inference_ms"], 2), train_time_min=round(train_min, 1),
            notes=note + (" | official test split (108 images)" if split == "test" else ""),
        )

    print(f"\n================ {name} RESULTS ================")
    print(f"VAL  (rect=True) : mAP50={val_rect['mAP50']*100:.2f}% | mAP50-95={val_rect['mAP50_95']*100:.2f}% | P={val_rect['precision']*100:.2f}% | R={val_rect['recall']*100:.2f}%")
    print(f"TEST (rect=True) : mAP50={test_rect['mAP50']*100:.2f}% | mAP50-95={test_rect['mAP50_95']*100:.2f}% | P={test_rect['precision']*100:.2f}% | R={test_rect['recall']*100:.2f}%")
    print(f"TEST (rect=False): mAP50={test_sq['mAP50']*100:.2f}% | mAP50-95={test_sq['mAP50_95']*100:.2f}% | P={test_sq['precision']*100:.2f}% | R={test_sq['recall']*100:.2f}%")
    print(f"Per-Class TEST (rect=True) : {pc_rect}")
    print(f"Per-Class TEST (rect=False): {pc_sq}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v37a_warm_ca", choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    a = ap.parse_args()
    main(a.variant, a.epochs, a.seed, a.batch, a.lr0)
