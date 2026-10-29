"""
common/modules/cs_cmaf.py
=========================
Channel-Spatial Cross-Modal Differential Attention Fusion (CS-CMAF)
with Local Parallax Offset Alignment.

Upgrades over v36 CrossModalAttentionFusion:
1. Local Parallax Alignment (LPA):
   Applies a 3x3 depthwise-separable offset residual on F_rgb guided by
   Concat([F_rgb, F_thm]) to correct depth-dependent lens parallax (1-4 px)
   between the FLIR C5 RGB and Thermal sensors.
2. Differential Modality Context:
   Explicitly computes |F_rgb_aligned - F_thm| to highlight modality-specific
   cues (e.g., thermal-only heat anomalies or RGB-only cold object boundaries).
3. Per-Channel + Spatial Attention Gate (alpha in R^{B x C x H x W}):
   Instead of forcing a single scalar alpha across all C channels at pixel (x,y),
   combines an SE/ECA channel gate [B, C, 1, 1] and a spatial gate [B, 1, H, W]
   so texture channels can favor RGB while thermal-contrast channels favor LWIR.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.modules.eca import ECA


class LocalParallaxAlign(nn.Module):
    """Lightweight 3x3 depthwise-separable residual alignment of RGB features to Thermal."""
    def __init__(self, channels: int):
        super().__init__()
        self.offset_extract = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Initialize last BN weight to 0 so alignment starts as identity (f_rgb unchanged at step 0)
        nn.init.zeros_(self.offset_extract[-1].weight)

    def forward(self, f_rgb: torch.Tensor, f_thm: torch.Tensor) -> torch.Tensor:
        # Interleave or concat channels for group conv
        b, c, h, w = f_rgb.shape
        paired = torch.stack([f_rgb, f_thm], dim=2).view(b, 2 * c, h, w)
        return f_rgb + self.offset_extract(paired)


class CSCMAF(nn.Module):
    """
    Channel-Spatial Cross-Modal Differential Attention Fusion (CS-CMAF).
    """
    def __init__(self, channels: int, use_align: bool = True):
        super().__init__()
        self.align = LocalParallaxAlign(channels) if use_align else None

        # Differential + Concat reduction: [F_rgb, F_thm, |F_rgb - F_thm|] (3C -> C)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            ECA(channels),
        )

        # Channel gate: 2C -> C (per-channel modality preference)
        mid = max(channels // 4, 8)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
        )

        # Spatial gate: 3C -> 1 (per-pixel modality preference)
        self.sp_gate = nn.Sequential(
            nn.Conv2d(channels * 3, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, f_rgb: torch.Tensor, f_thm: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            f_rgb = self.align(f_rgb, f_thm)

        f_diff = torch.abs(f_rgb - f_thm)
        f_cat2 = torch.cat([f_rgb, f_thm], dim=1)         # [B, 2C, H, W]
        f_cat3 = torch.cat([f_rgb, f_thm, f_diff], dim=1) # [B, 3C, H, W]

        # Joint channel-spatial gate in [0, 1] of shape [B, C, H, W]
        alpha = torch.sigmoid(self.ch_gate(f_cat2) + self.sp_gate(f_cat3))

        f_fused = alpha * f_rgb + (1.0 - alpha) * f_thm
        f_reduced = self.reduce(f_cat3)
        return f_fused + f_reduced
