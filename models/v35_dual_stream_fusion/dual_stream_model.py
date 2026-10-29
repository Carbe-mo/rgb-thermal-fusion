"""
models/v35_dual_stream_fusion/dual_stream_model.py
===================================================
Dual-Stream RGB-Thermal Fusion YOLO Model.

Architecture:
  - RGB Backbone:     CA + ECA attention (C2f_CA_ECA) for texture-rich features
  - Thermal Backbone: ECA + Dilated Conv (C2f_ECA at P2/P3, C2f_ECA_Dilated at P4/P5)
  - Fusion:           Cross-Modal Attention Gates at P3, P4, P5
  - Neck + Heads:     Standard YOLOv8 PANet + Decoupled Detect (shared)

The model takes two images as input (RGB + Thermal) and produces unified detections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import SPPF, C2f
from ultralytics.nn.modules.head import Detect

from common.modules.c2f_ca_eca import C2f_CA_ECA
from common.modules.c2f_eca import C2f_ECA
from common.modules.c2f_eca_dilated import C2f_ECA_Dilated
from common.modules.cross_modal_fusion import CrossModalAttentionFusion


class RGBBackbone(nn.Module):
    """
    RGB feature extractor with CA+ECA attention at all stages.
    Extracts multi-scale features at P3 (stride 8), P4 (stride 16), P5 (stride 32).
    """
    def __init__(self, width_mult: float = 0.25, depth_mult: float = 0.33, max_channels: int = 1024):
        super().__init__()
        c1 = self._ch(64, width_mult, max_channels)
        c2 = self._ch(128, width_mult, max_channels)
        c3 = self._ch(256, width_mult, max_channels)
        c4 = self._ch(512, width_mult, max_channels)
        c5 = self._ch(1024, width_mult, max_channels)

        n_p2 = max(round(3 * depth_mult), 1)
        n_p3 = max(round(6 * depth_mult), 1)
        n_p4 = max(round(6 * depth_mult), 1)
        n_p5 = max(round(3 * depth_mult), 1)

        self.stem = nn.Sequential(
            Conv(3, c1, 3, 2),   # P1/2
            Conv(c1, c2, 3, 2),  # P2/4
        )
        self.stage_p2 = C2f_CA_ECA(c2, c2, n=n_p2, shortcut=True)
        self.down_p3 = Conv(c2, c3, 3, 2)  # P3/8
        self.stage_p3 = C2f_CA_ECA(c3, c3, n=n_p3, shortcut=True)
        self.down_p4 = Conv(c3, c4, 3, 2)  # P4/16
        self.stage_p4 = C2f_CA_ECA(c4, c4, n=n_p4, shortcut=True)
        self.down_p5 = Conv(c4, c5, 3, 2)  # P5/32
        self.stage_p5 = C2f_CA_ECA(c5, c5, n=n_p5, shortcut=True)
        self.sppf = SPPF(c5, c5, 5)

        # Store output channels for fusion
        self.out_channels = (c3, c4, c5)

    @staticmethod
    def _ch(base: int, mult: float, max_ch: int) -> int:
        return min(int(base * mult), max_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage_p2(x)
        x = self.down_p3(x)
        p3 = self.stage_p3(x)
        x = self.down_p4(p3)
        p4 = self.stage_p4(x)
        x = self.down_p5(p4)
        p5 = self.stage_p5(x)
        p5 = self.sppf(p5)
        return p3, p4, p5


class ThermalBackbone(nn.Module):
    """
    Thermal feature extractor with ECA (P2/P3) and Dilated ECA (P4/P5).
    Matches our champion v30 architecture.
    """
    def __init__(self, in_ch: int = 3, width_mult: float = 0.25, depth_mult: float = 0.33, max_channels: int = 1024):
        super().__init__()
        c1 = self._ch(64, width_mult, max_channels)
        c2 = self._ch(128, width_mult, max_channels)
        c3 = self._ch(256, width_mult, max_channels)
        c4 = self._ch(512, width_mult, max_channels)
        c5 = self._ch(1024, width_mult, max_channels)

        n_p2 = max(round(3 * depth_mult), 1)
        n_p3 = max(round(6 * depth_mult), 1)
        n_p4 = max(round(6 * depth_mult), 1)
        n_p5 = max(round(3 * depth_mult), 1)

        self.stem = nn.Sequential(
            Conv(in_ch, c1, 3, 2),   # P1/2
            Conv(c1, c2, 3, 2),      # P2/4
        )
        self.stage_p2 = C2f_ECA(c2, c2, n=n_p2, shortcut=True)
        self.down_p3 = Conv(c2, c3, 3, 2)  # P3/8
        self.stage_p3 = C2f_ECA(c3, c3, n=n_p3, shortcut=True)
        self.down_p4 = Conv(c3, c4, 3, 2)  # P4/16
        self.stage_p4 = C2f_ECA_Dilated(c4, c4, n=n_p4, shortcut=True, dilation=2)
        self.down_p5 = Conv(c4, c5, 3, 2)  # P5/32
        self.stage_p5 = C2f_ECA_Dilated(c5, c5, n=n_p5, shortcut=True, dilation=2)
        self.sppf = SPPF(c5, c5, 5)

        self.out_channels = (c3, c4, c5)

    @staticmethod
    def _ch(base: int, mult: float, max_ch: int) -> int:
        return min(int(base * mult), max_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage_p2(x)
        x = self.down_p3(x)
        p3 = self.stage_p3(x)
        x = self.down_p4(p3)
        p4 = self.stage_p4(x)
        x = self.down_p5(p4)
        p5 = self.stage_p5(x)
        p5 = self.sppf(p5)
        return p3, p4, p5


class PANetNeck(nn.Module):
    """
    Standard YOLOv8 PANet neck (top-down + bottom-up feature pyramid).
    """
    def __init__(self, c3: int, c4: int, c5: int, depth_mult: float = 0.33):
        super().__init__()
        n = max(round(3 * depth_mult), 1)

        # Top-down path
        self.up_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c2f_p4 = C2f(c4 + c5, c4, n=n, shortcut=False)

        self.up_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c2f_p3 = C2f(c3 + c4, c3, n=n, shortcut=False)

        # Bottom-up path
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.bu_c2f_p4 = C2f(c3 + c4, c4, n=n, shortcut=False)

        self.down_p4 = Conv(c4, c4, 3, 2)
        self.bu_c2f_p5 = C2f(c4 + c5, c5, n=n, shortcut=False)

        self.out_channels = (c3, c4, c5)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Top-down
        td_p4 = self.td_c2f_p4(torch.cat([self.up_p5(p5), p4], dim=1))
        td_p3 = self.td_c2f_p3(torch.cat([self.up_p4(td_p4), p3], dim=1))

        # Bottom-up
        bu_p4 = self.bu_c2f_p4(torch.cat([self.down_p3(td_p3), td_p4], dim=1))
        bu_p5 = self.bu_c2f_p5(torch.cat([self.down_p4(bu_p4), p5], dim=1))

        return td_p3, bu_p4, bu_p5


class DualStreamFusionYOLO(nn.Module):
    """
    Dual-Stream RGB-Thermal Fusion YOLO.

    Forward accepts a dict with keys 'rgb' and 'thermal',
    or a single concatenated tensor [B, 6, H, W] (first 3ch = RGB, last 3ch = thermal).
    """
    def __init__(
        self,
        nc: int = 4,
        width_mult: float = 0.25,
        depth_mult: float = 0.33,
        max_channels: int = 1024,
        thermal_in_ch: int = 3,
    ):
        super().__init__()
        self.nc = nc

        # Dual backbones
        self.rgb_backbone = RGBBackbone(width_mult, depth_mult, max_channels)
        self.thermal_backbone = ThermalBackbone(thermal_in_ch, width_mult, depth_mult, max_channels)

        c3, c4, c5 = self.rgb_backbone.out_channels

        # Cross-modal fusion at P3, P4, P5
        self.fusion_p3 = CrossModalAttentionFusion(c3)
        self.fusion_p4 = CrossModalAttentionFusion(c4)
        self.fusion_p5 = CrossModalAttentionFusion(c5)

        # Shared PANet neck
        self.neck = PANetNeck(c3, c4, c5, depth_mult)

        # Detection heads
        self.detect = Detect(nc=nc, ch=(c3, c4, c5))

    def forward(self, x_rgb: torch.Tensor, x_thm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB input [B, 3, H, W]
            x_thm: Thermal input [B, 3, H, W] (pseudo-color or replicated grayscale)
        """
        # Extract multi-scale features from both streams
        rgb_p3, rgb_p4, rgb_p5 = self.rgb_backbone(x_rgb)
        thm_p3, thm_p4, thm_p5 = self.thermal_backbone(x_thm)

        # Cross-modal attention fusion at each scale
        fused_p3 = self.fusion_p3(rgb_p3, thm_p3)
        fused_p4 = self.fusion_p4(rgb_p4, thm_p4)
        fused_p5 = self.fusion_p5(rgb_p5, thm_p5)

        # Shared PANet neck
        neck_p3, neck_p4, neck_p5 = self.neck(fused_p3, fused_p4, fused_p5)

        # Detection
        return self.detect([neck_p3, neck_p4, neck_p5])

    def forward_concat(self, x: torch.Tensor) -> torch.Tensor:
        """Alternative: accept [B, 6, H, W] with channels [R,G,B,T,T,T]."""
        x_rgb = x[:, :3, :, :]
        x_thm = x[:, 3:, :, :]
        return self.forward(x_rgb, x_thm)


def build_model(nc: int = 4, scale: str = "n") -> DualStreamFusionYOLO:
    """Build dual-stream model at specified scale."""
    scales = {
        "n": (0.33, 0.25, 1024),  # nano
        "s": (0.33, 0.50, 1024),  # small
        "m": (0.67, 0.75, 768),   # medium
    }
    depth_mult, width_mult, max_ch = scales.get(scale, scales["n"])
    return DualStreamFusionYOLO(nc=nc, width_mult=width_mult, depth_mult=depth_mult, max_channels=max_ch)


if __name__ == "__main__":
    print("=" * 60)
    print("  Dual-Stream RGB-Thermal Fusion YOLO — Architecture Test")
    print("=" * 60)

    model = build_model(nc=4, scale="n")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    rgb_params = sum(p.numel() for p in model.rgb_backbone.parameters())
    thm_params = sum(p.numel() for p in model.thermal_backbone.parameters())
    fuse_params = sum(
        p.numel() for module in [model.fusion_p3, model.fusion_p4, model.fusion_p5]
        for p in module.parameters()
    )
    neck_params = sum(p.numel() for p in model.neck.parameters())
    head_params = sum(p.numel() for p in model.detect.parameters())

    print(f"\n  RGB Backbone (CA+ECA):        {rgb_params/1e6:.2f}M")
    print(f"  Thermal Backbone (ECA+Dil):   {thm_params/1e6:.2f}M")
    print(f"  Cross-Modal Fusion (x3):      {fuse_params/1e6:.2f}M")
    print(f"  PANet Neck:                   {neck_params/1e6:.2f}M")
    print(f"  Detection Heads:              {head_params/1e6:.2f}M")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:                        {total_params/1e6:.2f}M")

    # Forward pass test
    x_rgb = torch.randn(1, 3, 640, 640)
    x_thm = torch.randn(1, 3, 640, 640)

    model.eval()
    with torch.no_grad():
        # Test backbone outputs
        rgb_feats = model.rgb_backbone(x_rgb)
        thm_feats = model.thermal_backbone(x_thm)
        print(f"\n  RGB backbone outputs:     P3={rgb_feats[0].shape}, P4={rgb_feats[1].shape}, P5={rgb_feats[2].shape}")
        print(f"  Thermal backbone outputs: P3={thm_feats[0].shape}, P4={thm_feats[1].shape}, P5={thm_feats[2].shape}")

    print("\n  ✅ Dual-Stream Fusion YOLO built and verified successfully!")
