"""
models/v36_dual_stream_registered/model.py
==========================================
Ultralytics-compatible wrapper around the dual-stream RGB+thermal fusion network so it
can be trained with the stock DetectionTrainer (mosaic, TaskAlignedAssigner, CIoU+DFL
loss, mAP validation) exactly like the thermal-only v30 champion.

Input: 6-channel tensor [R_rgb, G_rgb, B_rgb, R_thm, G_thm, B_thm] (registered RGB).

Variants (constructor flags):
  rgb_attn="ca_eca"   RGB backbone uses C2f_CA_ECA (proposed)
  rgb_attn="eca"      RGB backbone uses C2f_ECA (ablation: no SE-style CA)
  thermal_only=True   RGB stream is not built; thermal features go straight to the neck
                      (control: same 6-ch pipeline, no RGB information)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.block import SPPF  # noqa: E402
from ultralytics.nn.modules.conv import Conv  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import initialize_weights  # noqa: E402

from common.modules.c2f_eca import C2f_ECA  # noqa: E402
from models.v35_dual_stream_fusion.dual_stream_model import (  # noqa: E402
    RGBBackbone, ThermalBackbone, PANetNeck,
)
from common.modules.cross_modal_fusion import CrossModalAttentionFusion  # noqa: E402


class RGBBackboneECA(ThermalBackbone):
    """Ablation: RGB backbone with plain C2f_ECA at every stage (no CA, no dilation)."""
    def __init__(self, width_mult=0.25, depth_mult=0.33, max_channels=1024):
        super().__init__(3, width_mult, depth_mult, max_channels)
        c4, c5 = self.out_channels[1], self.out_channels[2]
        n_p4 = max(round(6 * depth_mult), 1)
        n_p5 = max(round(3 * depth_mult), 1)
        self.stage_p4 = C2f_ECA(c4, c4, n=n_p4, shortcut=True)
        self.stage_p5 = C2f_ECA(c5, c5, n=n_p5, shortcut=True)


class DualStreamDetectionModel(DetectionModel):
    def __init__(self, nc: int = 4, rgb_attn: str = "ca_eca", thermal_only: bool = False,
                 verbose: bool = True):
        nn.Module.__init__(self)  # bypass YAML parsing of DetectionModel
        w, d, mc = 0.25, 0.33, 1024  # YOLOv8 nano scale (same as v30)
        self.yaml = {"nc": nc, "channels": 6, "dual_stream": True,
                     "rgb_attn": rgb_attn, "thermal_only": thermal_only}
        self.thermal_only = thermal_only
        self.rgb_attn = rgb_attn

        thm = ThermalBackbone(3, w, d, mc)
        c3, c4, c5 = thm.out_channels
        if thermal_only:
            rgb = nn.Identity()
            f3 = f4 = f5 = nn.Identity()
        else:
            rgb = RGBBackbone(w, d, mc) if rgb_attn == "ca_eca" else RGBBackboneECA(w, d, mc)
            f3, f4, f5 = (CrossModalAttentionFusion(c) for c in (c3, c4, c5))
        neck = PANetNeck(c3, c4, c5, d)
        # Legacy YOLOv8 head (full 3x3 conv cls branch) — identical to v30 / yolov8n.
        # Ultralytics 8.4 defaults to the lighter YOLO11-style DWConv cls branch otherwise.
        _prev = Detect.legacy
        Detect.legacy = True
        detect = Detect(nc=nc, ch=(c3, c4, c5))
        Detect.legacy = _prev
        detect.legacy = True

        # model[-1] must be Detect for v8DetectionLoss / validator
        self.model = nn.ModuleList([rgb, thm, f3, f4, f5, neck, detect])
        for i, m in enumerate(self.model):
            m.i, m.f, m.type = i, -1, m.__class__.__name__
        self.save = []
        self.names = {i: str(i) for i in range(nc)}
        self.inplace = True
        self.end2end = False

        detect.inplace = True
        detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.stride = detect.stride
        detect.bias_init()
        initialize_weights(self)
        if verbose:
            self.info()

    # ------------------------------------------------------------------
    def _predict_once(self, x, profile=False, embed=None):
        if x.shape[1] == 3:  # e.g. Ultralytics FLOPs probe; duplicate to 6 ch
            x = torch.cat([x, x], 1)
        rgb_bb, thm_bb, f3, f4, f5, neck, detect = self.model
        t3, t4, t5 = thm_bb(x[:, 3:6])
        if self.thermal_only:
            p3, p4, p5 = t3, t4, t5
        else:
            r3, r4, r5 = rgb_bb(x[:, 0:3])
            p3, p4, p5 = f3(r3, t3), f4(r4, t4), f5(r5, t5)
        return detect(list(neck(p3, p4, p5)))

    # ------------------------------------------------------------------
    def load_yolov8n(self, weights: str = "yolov8n.pt") -> tuple[int, int]:
        """Transfer COCO yolov8n weights into both backbones, the neck and the head
        (shape-matched; CA layers, fusion layers and class-dependent head convs stay
        randomly initialised). Mirrors how v30 was initialised from yolov8n.pt."""
        from ultralytics.nn.tasks import load_checkpoint
        src_model, _ = load_checkpoint(weights)
        src = src_model.float().state_dict()

        bb = {0: "stem.0", 1: "stem.1", 2: "stage_p2", 3: "down_p3", 4: "stage_p3",
              5: "down_p4", 6: "stage_p4", 7: "down_p5", 8: "stage_p5", 9: "sppf"}
        nk = {12: "td_c2f_p4", 15: "td_c2f_p3", 16: "down_p3", 18: "bu_c2f_p4",
              19: "down_p4", 21: "bu_c2f_p5"}
        targets = [(1, bb)] + ([] if self.thermal_only else [(0, bb)]) + [(5, nk)]

        dst = self.state_dict()
        new, n_ok, n_try = {}, 0, 0
        for k, v in src.items():
            parts = k.split(".")  # model.<idx>.<rest>
            idx, rest = int(parts[1]), ".".join(parts[2:])
            cands = []
            for mi, mapping in targets:
                if idx in mapping:
                    cands.append(f"model.{mi}.{mapping[idx]}.{rest}")
            if idx == 22:
                cands.append(f"model.6.{rest}")
            for ck in cands:
                n_try += 1
                if ck in dst and dst[ck].shape == v.shape:
                    new[ck] = v
                    n_ok += 1
        self.load_state_dict(new, strict=False)
        return n_ok, len(dst)


if __name__ == "__main__":
    for kw in [dict(), dict(rgb_attn="eca"), dict(thermal_only=True)]:
        m = DualStreamDetectionModel(nc=4, verbose=False, **kw)
        ok, tot = m.load_yolov8n()
        n = sum(p.numel() for p in m.parameters()) / 1e6
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(1, 6, 640, 640))
        print(kw, f"params={n:.2f}M  transferred={ok}/{tot} tensors  out={y[0].shape if isinstance(y,(list,tuple)) else type(y)}")
