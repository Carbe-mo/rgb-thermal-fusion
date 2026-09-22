"""
modules/c2f_eca.py — C2f with ECA Attention Bottlenecks
======================================================
Implements C2f_ECA module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from modules.eca import ECABottleneck


class C2f_ECA(nn.Module):
    """
    CSP Bottleneck with 2 convolutions and ECA attention inside each bottleneck.
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
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            ECABottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, eca_k=eca_k)
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
