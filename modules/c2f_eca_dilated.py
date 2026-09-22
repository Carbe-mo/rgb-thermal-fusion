"""
modules/c2f_eca_dilated.py — Dilated Receptive Field Bottleneck with ECA Attention
==================================================================================
Implements DilatedECABottleneck and C2f_ECA_Dilated.
Expands effective receptive field from 3x3 to 5x5 using dilation rate d=2 on the second convolution,
followed by non-reductive ECA (k=3) channel calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from modules.eca import ECA


class DilatedECABottleneck(nn.Module):
    """
    Standard Bottleneck with dilated second conv (d=2, p=2) for expanded context and ECA attention.
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
        dilation: int = 2,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, p=dilation, d=dilation)
        self.eca = ECA(c2, k_size=eca_k)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        y = self.eca(y)
        return x + y if self.add else y


class C2f_ECA_Dilated(nn.Module):
    """
    C2f module equipped with DilatedECABottlenecks.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        eca_k: int = 3,
        dilation: int = 2,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DilatedECABottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
                eca_k=eca_k,
                dilation=dilation,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
