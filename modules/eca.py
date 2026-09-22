"""
modules/eca.py — Efficient Channel Attention (ECA) & ECABottleneck
==================================================================
Implements Efficient Channel Attention (Wang et al., CVPR 2020) for local cross-channel interaction without dimensionality reduction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Bottleneck


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Captures local cross-channel interaction using a fast 1D convolution without dimensionality reduction.
    """
    def __init__(self, channels: int = 0, k_size: int = 3, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        if channels > 0 and k_size <= 0:
            t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
            k = t if t % 2 != 0 else t + 1
        else:
            k = max(3, k_size if k_size % 2 != 0 else k_size + 1)

        self.k = k
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)
        omega = self.sigmoid(self.conv1d(y))
        omega = omega.transpose(-1, -2).unsqueeze(-1)
        return x * omega


class ECABottleneck(Bottleneck):
    """
    Standard Bottleneck with ECA attention placed inside the residual branch before addition.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        eca_k: int = 3,
    ) -> None:
        super().__init__(c1, c2, shortcut=shortcut, g=g, k=k, e=e)
        self.eca = ECA(c2, k_size=eca_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cv2(self.cv1(x))
        res = self.eca(res)
        return x + res if self.add else res
