"""
models/v37_cs_cmaf_fusion/model.py
==================================
Upgraded Dual-Stream RGB-Thermal Fusion Detector (v37):
1. CS-CMAF (Channel-Spatial Cross-Modal Differential Attention Fusion + Local Parallax Alignment)
   at P3, P4, P5.
2. Attention-Enhanced Shared Neck (PANetNeckECA using C2f_ECA instead of plain C2f).
3. Hybrid Warm-Start Weight Initialization:
   - Thermal Backbone + Neck + Head initialized from our trained v30 thermal champion
     (or COCO yolov8n.pt via flag)
   - RGB Backbone initialized from COCO yolov8n.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.conv import Conv  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, load_checkpoint  # noqa: E402
from ultralytics.utils.torch_utils import initialize_weights  # noqa: E402

from common.modules.c2f_eca import C2f_ECA  # noqa: E402
from common.modules.cs_cmaf import CSCMAF  # noqa: E402
from models.v35_dual_stream_fusion.dual_stream_model import (  # noqa: E402
    RGBBackbone, ThermalBackbone, PANetNeck,
)
from models.v36_dual_stream_registered.model import RGBBackboneECA  # noqa: E402


class PANetNeckECA(nn.Module):
    """Shared PANet neck upgraded with C2f_ECA attention blocks at all 4 fusion stages."""
    def __init__(self, c3: int, c4: int, c5: int, depth_mult: float = 0.33):
        super().__init__()
        n = max(round(3 * depth_mult), 1)

        self.up_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c2f_p4 = C2f_ECA(c4 + c5, c4, n=n, shortcut=False)

        self.up_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c2f_p3 = C2f_ECA(c3 + c4, c3, n=n, shortcut=False)

        self.down_p3 = Conv(c3, c3, 3, 2)
        self.bu_c2f_p4 = C2f_ECA(c3 + c4, c4, n=n, shortcut=False)

        self.down_p4 = Conv(c4, c4, 3, 2)
        self.bu_c2f_p5 = C2f_ECA(c4 + c5, c5, n=n, shortcut=False)

        self.out_channels = (c3, c4, c5)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        td_p4 = self.td_c2f_p4(torch.cat([self.up_p5(p5), p4], dim=1))
        td_p3 = self.td_c2f_p3(torch.cat([self.up_p4(td_p4), p3], dim=1))
        bu_p4 = self.bu_c2f_p4(torch.cat([self.down_p3(td_p3), td_p4], dim=1))
        bu_p5 = self.bu_c2f_p5(torch.cat([self.down_p4(bu_p4), p5], dim=1))
        return td_p3, bu_p4, bu_p5


class DualStreamV37DetectionModel(DetectionModel):
    def __init__(
        self,
        nc: int = 4,
        rgb_attn: str = "ca_eca",
        neck_eca: bool = True,
        use_align: bool = True,
        verbose: bool = True,
    ):
        nn.Module.__init__(self)
        w, d, mc = 0.25, 0.33, 1024
        self.yaml = {
            "nc": nc, "channels": 6, "dual_stream_v37": True,
            "rgb_attn": rgb_attn, "neck_eca": neck_eca, "use_align": use_align,
        }
        self.rgb_attn = rgb_attn

        thm = ThermalBackbone(3, w, d, mc)
        c3, c4, c5 = thm.out_channels
        rgb = RGBBackbone(w, d, mc) if rgb_attn == "ca_eca" else RGBBackboneECA(w, d, mc)
        f3, f4, f5 = (CSCMAF(c, use_align=use_align) for c in (c3, c4, c5))
        neck = PANetNeckECA(c3, c4, c5, d) if neck_eca else PANetNeck(c3, c4, c5, d)

        _prev = Detect.legacy
        Detect.legacy = True
        detect = Detect(nc=nc, ch=(c3, c4, c5))
        Detect.legacy = _prev
        detect.legacy = True

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

        # Re-zero the parallax alignment last BN weight so alignment starts as identity
        for f in (f3, f4, f5):
            if getattr(f, "align", None) is not None:
                nn.init.zeros_(f.align.offset_extract[-1].weight)

        if verbose:
            self.info()

    def _predict_once(self, x, profile=False, embed=None):
        if x.shape[1] == 3:
            x = torch.cat([x, x], 1)
        rgb_bb, thm_bb, f3, f4, f5, neck, detect = self.model
        t3, t4, t5 = thm_bb(x[:, 3:6])
        r3, r4, r5 = rgb_bb(x[:, 0:3])
        p3, p4, p5 = f3(r3, t3), f4(r4, t4), f5(r5, t5)
        return detect(list(neck(p3, p4, p5)))

    def load_hybrid_pretrained(
        self,
        rgb_weights: str = "yolov8n.pt",
        thm_weights: str | None = None,
    ) -> tuple[int, int]:
        """Load COCO yolov8n into RGB backbone, and either v30 thermal champion or yolov8n
        into Thermal backbone + Neck + Detect head."""
        bb = {0: "stem.0", 1: "stem.1", 2: "stage_p2", 3: "down_p3", 4: "stage_p3",
              5: "down_p4", 6: "stage_p4", 7: "down_p5", 8: "stage_p5", 9: "sppf"}
        nk = {12: "td_c2f_p4", 15: "td_c2f_p3", 16: "down_p3", 18: "bu_c2f_p4",
              19: "down_p4", 21: "bu_c2f_p5"}

        dst = self.state_dict()
        new = {}

        # 1. Load RGB backbone from COCO yolov8n.pt
        rgb_m, _ = load_checkpoint(rgb_weights)
        rgb_sd = rgb_m.float().state_dict()
        for k, v in rgb_sd.items():
            parts = k.split(".")
            idx, rest = int(parts[1]), ".".join(parts[2:])
            if idx in bb:
                ck = f"model.0.{bb[idx]}.{rest}"
                if ck in dst and dst[ck].shape == v.shape:
                    new[ck] = v

        # 2. Load Thermal backbone + Neck + Head from thm_weights (or rgb_weights if None)
        thm_source = thm_weights if thm_weights else rgb_weights
        thm_m, _ = load_checkpoint(thm_source)
        thm_sd = thm_m.float().state_dict()
        for k, v in thm_sd.items():
            parts = k.split(".")
            idx, rest = int(parts[1]), ".".join(parts[2:])
            ck = None
            if idx in bb:
                ck = f"model.1.{bb[idx]}.{rest}"
            elif idx in nk:
                ck = f"model.5.{nk[idx]}.{rest}"
            elif idx == 22:
                ck = f"model.6.{rest}"
            if ck and ck in dst and dst[ck].shape == v.shape:
                new[ck] = v

        self.load_state_dict(new, strict=False)
        return len(new), len(dst)


if __name__ == "__main__":
    v30_pt = str(PROJECT_ROOT / "runs" / "v30_v27_cosine_100ep" / "train" / "weights" / "best.pt")
    for kw in [dict(rgb_attn="ca_eca"), dict(rgb_attn="eca")]:
        m = DualStreamV37DetectionModel(nc=4, verbose=False, **kw)
        ok, tot = m.load_hybrid_pretrained("yolov8n.pt", v30_pt)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(1, 6, 640, 640))
        print(kw, f"params={n:.2f}M  transferred={ok}/{tot}  out={y[0].shape}")
