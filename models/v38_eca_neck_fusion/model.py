"""
models/v38_eca_neck_fusion/model.py
===================================
v38 Receptive-Field-Aligned Dual-Stream RGB+Thermal Fusion Detector with ECA-PANet Neck.

Key Architectural Innovations over v36:
  1. Receptive-Field Alignment at P4/P5:
     In v36, ThermalBackbone used C2f_ECA_Dilated(d=2) at P4/P5 while RGBBackbone used
     non-dilated (d=1) convolutions, causing a receptive-field mismatch during pixel-wise
     CrossModalAttentionFusion at strides 16 and 32.
     In v38, RGBBackbone also uses C2f_ECA_Dilated(d=2) at P4/P5 (with C2f_CA_ECA or C2f_ECA
     at P2/P3), aligning the contextual receptive fields of both modalities for person-bag
     proximity reasoning ("Bag" vs "Unattended Anomaly") while keeping 100% COCO yolov8n.pt
     weight transfer.
  2. ECA-Enhanced PANet Neck (PANetNeckECA):
     Replaces plain C2f blocks in the top-down and bottom-up PANet neck with C2f_ECA so
     cross-scale + cross-modal concatenated features undergo 1D channel recalibration before
     entering the decoupled Detect head (adding only 12 parameters total across 4 neck stages
     and preserving 100% COCO neck weight transfer).
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
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import initialize_weights  # noqa: E402

from common.modules.c2f_ca_eca import C2f_CA_ECA  # noqa: E402
from common.modules.c2f_eca import C2f_ECA  # noqa: E402
from common.modules.c2f_eca_dilated import C2f_ECA_Dilated  # noqa: E402
from common.modules.cross_modal_fusion import CrossModalAttentionFusion  # noqa: E402
from models.v35_dual_stream_fusion.dual_stream_model import ThermalBackbone  # noqa: E402


class RGBBackboneAligned(ThermalBackbone):
    """
    RGB Backbone with aligned dilated receptive fields (d=2) at P4/P5.
    - rgb_mode="ca_dilated": C2f_CA_ECA at P2/P3 + C2f_ECA_Dilated(d=2) at P4/P5
    - rgb_mode="eca_dilated": C2f_ECA at P2/P3 + C2f_ECA_Dilated(d=2) at P4/P5
    """
    def __init__(self, rgb_mode: str = "ca_dilated", width_mult: float = 0.25,
                 depth_mult: float = 0.33, max_channels: int = 1024):
        super().__init__(3, width_mult, depth_mult, max_channels)
        c2 = self._ch(128, width_mult, max_channels)
        c3 = self._ch(256, width_mult, max_channels)
        n_p2 = max(round(3 * depth_mult), 1)
        n_p3 = max(round(6 * depth_mult), 1)
        if rgb_mode == "ca_dilated":
            self.stage_p2 = C2f_CA_ECA(c2, c2, n=n_p2, shortcut=True)
            self.stage_p3 = C2f_CA_ECA(c3, c3, n=n_p3, shortcut=True)


class PANetNeckECA(nn.Module):
    """
    ECA-Enhanced PANet neck: uses C2f_ECA at all 4 multi-scale fusion stages while
    retaining identical Conv/Bottleneck tensor shapes for 100% yolov8n.pt weight transfer.
    """
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


class V38DualStreamModel(DetectionModel):
    def __init__(self, nc: int = 4, rgb_mode: str = "ca_dilated", neck_eca: bool = True,
                 verbose: bool = True):
        nn.Module.__init__(self)
        w, d, mc = 0.25, 0.33, 1024
        self.yaml = {"nc": nc, "channels": 6, "dual_stream": True,
                     "rgb_mode": rgb_mode, "neck_eca": neck_eca}
        self.rgb_mode = rgb_mode
        self.neck_eca = neck_eca

        thm = ThermalBackbone(3, w, d, mc)
        c3, c4, c5 = thm.out_channels
        rgb = RGBBackboneAligned(rgb_mode=rgb_mode, width_mult=w, depth_mult=d, max_channels=mc)
        f3, f4, f5 = (CrossModalAttentionFusion(c) for c in (c3, c4, c5))
        neck = PANetNeckECA(c3, c4, c5, d)

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

    def load_yolov8n(self, weights: str = "yolov8n.pt") -> tuple[int, int]:
        from ultralytics.nn.tasks import load_checkpoint
        src_model, _ = load_checkpoint(weights)
        src = src_model.float().state_dict()

        bb = {0: "stem.0", 1: "stem.1", 2: "stage_p2", 3: "down_p3", 4: "stage_p3",
              5: "down_p4", 6: "stage_p4", 7: "down_p5", 8: "stage_p5", 9: "sppf"}
        nk = {12: "td_c2f_p4", 15: "td_c2f_p3", 16: "down_p3", 18: "bu_c2f_p4",
              19: "down_p4", 21: "bu_c2f_p5"}
        targets = [(1, bb), (0, bb), (5, nk)]

        dst = self.state_dict()
        new, n_ok = {}, 0
        for k, v in src.items():
            parts = k.split(".")
            idx, rest = int(parts[1]), ".".join(parts[2:])
            cands = []
            for mi, mapping in targets:
                if idx in mapping:
                    cands.append(f"model.{mi}.{mapping[idx]}.{rest}")
            if idx == 22:
                cands.append(f"model.6.{rest}")
            for ck in cands:
                if ck in dst and dst[ck].shape == v.shape:
                    new[ck] = v
                    n_ok += 1
        self.load_state_dict(new, strict=False)
        return n_ok, len(dst)


if __name__ == "__main__":
    for mode in ("ca_dilated", "eca_dilated"):
        m = V38DualStreamModel(nc=4, rgb_mode=mode, verbose=False)
        ok, tot = m.load_yolov8n("yolov8n.pt")
        n_p = sum(p.numel() for p in m.parameters()) / 1e6
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(1, 6, 640, 640))
        print(f"mode={mode:12s} | params={n_p:.3f}M | transferred={ok}/{tot} | out={y[0].shape}")
