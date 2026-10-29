"""
common/modules/c2f_ca_eca.py — C2f with Channel Attention (SE-style) + ECA Bottlenecks
========================================================================================

Implements `CAECABottleneck` and `C2f_CA_ECA` for the RGB backbone stream.
The bottleneck applies two sequential attention mechanisms:
  1. CA (Channel Attention): SE-style MLP bottleneck (FC → ReLU → FC → Sigmoid)
     to model complex inter-channel dependencies in texture-rich RGB features.
  2. ECA (Efficient Channel Attention): 1D convolution without reduction for
     lightweight local cross-channel refinement.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from common.modules.eca import ECA


class ChannelAttention(nn.Module):
    """
    SE-style Channel Attention with MLP bottleneck.
    GAP → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        w = self.fc(x)  # [B, C]
        return x * w.unsqueeze(-1).unsqueeze(-1)


class CAECABottleneck(nn.Module):
    """
    Bottleneck with sequential Channel Attention (SE) + ECA after the second conv.
    Forward: cv1 → cv2 → CA → ECA → residual add
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple = (3, 3),
        e: float = 0.5,
        ca_reduction: int = 16,
        eca_k: int = 3,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.ca = ChannelAttention(c2, reduction=ca_reduction)
        self.eca = ECA(channels=c2, k_size=eca_k)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        y = self.ca(y)    # SE-style channel attention
        y = self.eca(y)   # 1D ECA refinement
        return x + y if self.add else y


class C2f_CA_ECA(nn.Module):
    """
    C2f module with CA+ECA bottlenecks for the RGB stream.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        ca_reduction: int = 16,
        eca_k: int = 3,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            CAECABottleneck(
                self.c, self.c,
                shortcut=shortcut, g=g,
                k=(3, 3), e=1.0,
                ca_reduction=ca_reduction, eca_k=eca_k,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    print("=== Testing C2f_CA_ECA ===")
    m = C2f_CA_ECA(c1=64, c2=64, n=2, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = m(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Parameters: {n_params:,}")
    assert out.shape == (2, 64, 40, 40)
    print("C2f_CA_ECA test passed!")
