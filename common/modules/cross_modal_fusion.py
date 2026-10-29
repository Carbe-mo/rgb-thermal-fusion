"""
common/modules/cross_modal_fusion.py — Cross-Modal Attention Fusion
====================================================================

Implements cross-modal feature fusion between RGB and Thermal backbone streams
at each spatial scale (P3, P4, P5).

Fusion strategy:
  1. Concatenate RGB + Thermal features along channel dim → [B, 2C, H, W]
  2. 1×1 Conv to reduce channels back to C → [B, C, H, W]
  3. Cross-modal spatial attention gate:
     α = Sigmoid(Conv1x1(Concat([F_rgb, F_thm])))   → [B, 1, H, W]
     F_fused = α ⊗ F_rgb + (1 - α) ⊗ F_thm
  4. Residual: F_out = F_fused + F_reduced
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class CrossModalAttentionFusion(nn.Module):
    """
    Fuses features from RGB and Thermal streams at a single scale level
    using a learned spatial attention gate.
    
    The gate α learns which spatial regions should trust RGB (texture-rich)
    vs Thermal (illumination-invariant), enabling adaptive modality weighting.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Channel reduction: 2C → C
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        # Spatial attention gate: 2C → 1
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb: torch.Tensor, f_thm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_rgb: RGB features [B, C, H, W]
            f_thm: Thermal features [B, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        # Concatenate along channels
        f_cat = torch.cat([f_rgb, f_thm], dim=1)  # [B, 2C, H, W]

        # Compute channel-reduced features
        f_reduced = self.reduce(f_cat)  # [B, C, H, W]

        # Compute spatial attention gate
        alpha = self.gate(f_cat)  # [B, 1, H, W]

        # Weighted fusion: trust RGB where α≈1, thermal where α≈0
        f_fused = alpha * f_rgb + (1 - alpha) * f_thm  # [B, C, H, W]

        # Residual enhancement
        return f_fused + f_reduced


class MultiScaleFusion(nn.Module):
    """
    Applies CrossModalAttentionFusion at three scale levels (P3, P4, P5).
    """
    def __init__(self, c3: int = 64, c4: int = 128, c5: int = 256):
        super().__init__()
        self.fuse_p3 = CrossModalAttentionFusion(c3)
        self.fuse_p4 = CrossModalAttentionFusion(c4)
        self.fuse_p5 = CrossModalAttentionFusion(c5)

    def forward(
        self,
        rgb_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        thm_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_feats: (P3_rgb, P4_rgb, P5_rgb)
            thm_feats: (P3_thm, P4_thm, P5_thm)
        Returns:
            (P3_fused, P4_fused, P5_fused)
        """
        p3 = self.fuse_p3(rgb_feats[0], thm_feats[0])
        p4 = self.fuse_p4(rgb_feats[1], thm_feats[1])
        p5 = self.fuse_p5(rgb_feats[2], thm_feats[2])
        return p3, p4, p5


if __name__ == "__main__":
    print("=== Testing CrossModalAttentionFusion ===")
    fuse = CrossModalAttentionFusion(channels=64)
    f_rgb = torch.randn(2, 64, 80, 80)
    f_thm = torch.randn(2, 64, 80, 80)
    out = fuse(f_rgb, f_thm)
    print(f"RGB: {f_rgb.shape} + Thermal: {f_thm.shape} → Fused: {out.shape}")
    n_params = sum(p.numel() for p in fuse.parameters())
    print(f"Fusion parameters: {n_params:,}")
    assert out.shape == (2, 64, 80, 80)

    print("\n=== Testing MultiScaleFusion ===")
    ms_fuse = MultiScaleFusion(c3=64, c4=128, c5=256)
    rgb = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20))
    thm = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20))
    p3, p4, p5 = ms_fuse(rgb, thm)
    print(f"P3: {p3.shape}, P4: {p4.shape}, P5: {p5.shape}")
    total = sum(p.numel() for p in ms_fuse.parameters())
    print(f"Total MultiScaleFusion parameters: {total:,}")
    print("All fusion tests passed!")
