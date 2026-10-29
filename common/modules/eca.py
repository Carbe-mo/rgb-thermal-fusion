"""
common/modules/eca.py — Efficient Channel Attention (ECA) & ECABottleneck
==========================================================================

Implements Efficient Channel Attention (Wang et al., CVPR 2020):
1. ECA Module:
   - Performs Global Average Pooling: y = AvgPool(x) [B, C, 1, 1]
   - Applies a 1D convolution of kernel size k (adaptive or fixed k=3/5) across
     the channel dimension without dimensionality reduction:
       k = |log2(C)/gamma + b/gamma|_odd  (or explicit k_size)
   - Applies Sigmoid to produce channel attention weights: omega = Sigmoid(1DConv(y))
   - Modulates input: out = x * omega

2. ECABottleneck:
   - Subclass of Ultralytics `Bottleneck`.
   - Inserts ECA attention step right after the second conv (`self.cv2`),
     inside the residual path, before the shortcut addition:
       res = self.cv2(self.cv1(x))
       res = self.eca(res)
       return x + res if self.add else res
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
        # If channels > 0 and k_size not explicitly fixed, compute adaptive kernel size
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
        # Global average pooling -> [B, C, 1, 1]
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        # Reshape for 1D convolution: [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D conv across channels + sigmoid -> [B, 1, C]
        omega = self.sigmoid(self.conv1d(y))
        # Reshape back to [B, C, 1, 1]
        omega = omega.transpose(-1, -2).unsqueeze(-1)
        return x * omega


class ECABottleneck(Bottleneck):
    """
    Standard Bottleneck with ECA inserted after the second conv, inside the residual path.
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
    ):
        super().__init__(c1, c2, shortcut=shortcut, g=g, k=k, e=e)
        self.eca = ECA(channels=c2, k_size=eca_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path: cv1 -> cv2 -> eca -> add shortcut
        res = self.cv2(self.cv1(x))
        res = self.eca(res)
        return x + res if self.add else res


if __name__ == "__main__":
    # Test ECA
    eca = ECA(channels=64, k_size=3)
    x = torch.randn(2, 64, 40, 40)
    out = eca(x)
    print(f"ECA output shape: {out.shape}")
    assert out.shape == (2, 64, 40, 40)

    # Test ECABottleneck
    bottleneck = ECABottleneck(c1=64, c2=64, shortcut=True, eca_k=3)
    out_b = bottleneck(x)
    print(f"ECABottleneck output shape: {out_b.shape}")
    assert out_b.shape == (2, 64, 40, 40)
    print("ECA & ECABottleneck unit tests passed successfully!")
