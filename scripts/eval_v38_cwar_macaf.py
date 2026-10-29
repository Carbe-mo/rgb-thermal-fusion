"""
Evaluate the impact of adding v38_ca_dilated (High-Recall RF-Aligned Dual-Stream + PANetNeckECA)
to our 2D-Grid Anchor-Aligned Confidence-Weighted Regression (CWAR) and Modality-Aware
Class-Specific Fusion (MACAF) ensemble on both Val (213) and Test (108) splits.
Also logs the official v38_cwar_macaf results to results/results.csv.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
import models.v36_dual_stream_registered.model  # noqa: F401
import models.v37_cs_cmaf_fusion.model  # noqa: F401
import models.v38_eca_neck_fusion.model  # noqa: F401
from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
from scripts.optimize_macaf_val_to_test import (
    CachingModelWrapper, fuse_cached_batches, eval_fast_replay, DATA
)

register_c2f_eca()
register_c2f_eca_dilated()


def main():
    ckpt_specs = {
        "v36_main": (ROOT / "runs/v36_dual_stream_registered/v36_main/weights/best.pt", False),
        "v36_eca":  (ROOT / "runs/v36_dual_stream_registered/v36_eca_rgb/weights/best.pt", False),
        "v38_ca":   (ROOT / "runs/v38_eca_neck_fusion/v38_ca_dilated/weights/best.pt", False),
        "v30":      (ROOT / "runs/v30_v27_cosine_100ep/train/weights/best.pt", True),
        "v27":      (ROOT / "runs/v27_v21_cosine/train/weights/best.pt", True),
        "v24_cp":   (ROOT / "runs/v24_copy_paste_aug/train/weights/best.pt", True),
        "v36_thm":  (ROOT / "runs/v36_dual_stream_registered/v36_thermal/weights/best.pt", False),
    }

    models_loaded = {}
    for k, (p, is3) in ckpt_specs.items():
        models_loaded[k] = (load_checkpoint(str(p))[0].float().cuda().eval(), is3)

    base_meta = models_loaded["v36_main"][0]
    mnames = list(ckpt_specs.keys())

    all_out = {}
    for rect in (True, False):
        print(f"\n==================== CACHING WITH v38_ca (rect={rect}) ====================", flush=True)
        cache = {"val": {"orig": {}, "flip": {}}, "test": {"orig": {}, "flip": {}}}
        for split in ("val", "test"):
            for k, (m, is3) in models_loaded.items():
                for mode_key, dflip in (("orig", False), ("flip", True)):
                    cw = CachingModelWrapper(m, is3, do_flip=dflip).cuda().eval()
                    v = DetectionValidator(args=dict(
                        data=str(DATA), split=split, imgsz=640, batch=16, rect=rect,
                        conf=0.001, iou=0.65, workers=0, device=0,
                        project=str(ROOT / "runs" / "phase1"), name="cache_v38",
                        exist_ok=True, plots=False, verbose=False,
                    ))
                    v(model=cw)
                    cache[split][mode_key][k] = cw.recorded

        # Order: [v36_main, v36_eca, v38_ca, v30, v27, v24_cp, v36_thm]
        configs = {
            "1_Dual3_NoThm (v36m+v36e+v38)": (
                np.array([
                    [0.42, 0.42, 0.42, 0.44],  # v36_main
                    [0.38, 0.38, 0.42, 0.36],  # v36_eca
                    [0.20, 0.20, 0.16, 0.20],  # v38_ca
                    [0.00, 0.00, 0.00, 0.00],  # v30
                    [0.00, 0.00, 0.00, 0.00],  # v27
                    [0.00, 0.00, 0.00, 0.00],  # v24_cp
                    [0.00, 0.00, 0.00, 0.00],  # v36_thm
                ]),
                np.array([0.42, 0.38, 0.20, 0.0, 0.0, 0.0, 0.0]),
            ),
            "2_Core4 (v36m+v36e+v38+v30)_Uniform": (
                np.array([
                    [0.36, 0.36, 0.36, 0.36],  # v36_main
                    [0.32, 0.32, 0.32, 0.32],  # v36_eca
                    [0.14, 0.14, 0.14, 0.14],  # v38_ca
                    [0.18, 0.18, 0.18, 0.18],  # v30
                    [0.00, 0.00, 0.00, 0.00],  # v27
                    [0.00, 0.00, 0.00, 0.00],  # v24_cp
                    [0.00, 0.00, 0.00, 0.00],  # v36_thm
                ]),
                np.array([0.36, 0.32, 0.14, 0.18, 0.0, 0.0, 0.0]),
            ),
            "3_Core4_MACAF (v36m+v36e+v38+v30)": (
                np.array([
                    [0.36, 0.35, 0.36, 0.42],  # v36_main
                    [0.32, 0.33, 0.42, 0.30],  # v36_eca
                    [0.14, 0.16, 0.10, 0.14],  # v38_ca
                    [0.18, 0.16, 0.12, 0.14],  # v30
                    [0.00, 0.00, 0.00, 0.00],  # v27
                    [0.00, 0.00, 0.00, 0.00],  # v24_cp
                    [0.00, 0.00, 0.00, 0.00],  # v36_thm
                ]),
                np.array([0.38, 0.34, 0.14, 0.14, 0.0, 0.0, 0.0]),
            ),
            "4_Full7_MACAF": (
                np.array([
                    [0.30, 0.30, 0.34, 0.36],  # v36_main
                    [0.26, 0.28, 0.38, 0.26],  # v36_eca
                    [0.12, 0.14, 0.10, 0.12],  # v38_ca
                    [0.12, 0.12, 0.08, 0.10],  # v30
                    [0.06, 0.06, 0.04, 0.06],  # v27
                    [0.08, 0.05, 0.04, 0.05],  # v24_cp
                    [0.06, 0.05, 0.02, 0.05],  # v36_thm
                ]),
                np.array([0.32, 0.28, 0.12, 0.12, 0.06, 0.06, 0.04]),
            ),
        }

        for cname, (cw, bw) in configs.items():
            for hcf, fw, iou in ((False, 0.0, 0.65), (False, 0.45, 0.65), (True, 0.85, 0.60)):
                mode_str = "noFlip" if fw == 0 else ("hcfFlip" if hcf else "gridFlip")
                tag = f"{cname}|{mode_str}|rect={rect}"
                fb_val = fuse_cached_batches(cache["val"]["orig"], cache["val"]["flip"], mnames, cw, bw, fw, 1.25, hcf)
                fb_test = fuse_cached_batches(cache["test"]["orig"], cache["test"]["flip"], mnames, cw, bw, fw, 1.25, hcf)
                rv = eval_fast_replay(base_meta, fb_val, "val", rect, iou=iou)
                rt = eval_fast_replay(base_meta, fb_test, "test", rect, iou=iou)
                all_out[tag] = {"val": rv, "test": rt}
                print(f"{tag:52s} -> VAL={rv['mAP50']:6.2f}% ({rv['mAP50_95']:5.2f}%) | TEST={rt['mAP50']:6.2f}% ({rt['mAP50_95']:5.2f}%) P={rt['P']:5.2f}% R={rt['R']:5.2f}% {rt['per_cls']}", flush=True)

    out_path = ROOT / "results" / "v38_cwar_macaf_eval.json"
    out_path.write_text(json.dumps(all_out, indent=2))
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
