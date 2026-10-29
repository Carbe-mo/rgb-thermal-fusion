"""
common/modules/c2f_eca.py — C2f with ECA Attention Bottlenecks
==============================================================

Implements `C2f_ECA`, an exact drop-in replacement for Ultralytics `C2f`
that uses `ECABottleneck` (ECA attention inside the residual branch)
instead of standard `Bottleneck`.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.conv import Conv
from common.modules.eca import ECABottleneck, ECA


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


_REGISTERED = False

def register_c2f_eca() -> None:
    """
    Register C2f_ECA and ECABottleneck into Ultralytics nn modules and parse_model
    so that model.yaml can reference 'C2f_ECA' and 'ECABottleneck' directly.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_ECA"] = C2f_ECA
    tasks.__dict__["ECABottleneck"] = ECABottleneck
    tasks.__dict__["ECA"] = ECA
    modules.__dict__["C2f_ECA"] = C2f_ECA
    modules.__dict__["ECABottleneck"] = ECABottleneck
    modules.__dict__["ECA"] = ECA

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_ECA":
                        custom_layers[(section, i)] = ("C2f_ECA", n, args)
                        # Replace temporarily with stock C2f so parse_model computes channel propagation and scaling
                        d_mod[section][i] = [f, n, "C2f", args]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        # Replace C2f placeholders with actual C2f_ECA instances
        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]
            
            c1 = model[idx].cv1.conv.in_channels
            c2 = model[idx].cv2.conv.out_channels
            n = len(model[idx].m)
            shortcut = args[1] if len(args) > 1 else False
            g = args[2] if len(args) > 2 else 1
            e = args[3] if len(args) > 3 else 0.5
            eca_k = args[4] if len(args) > 4 else 3

            eca_layer = C2f_ECA(c1, c2, n=n, shortcut=shortcut, g=g, e=e, eca_k=eca_k)
            eca_layer.i = idx
            eca_layer.f = f
            eca_layer.type = "common.modules.c2f_eca.C2f_ECA"
            eca_layer.np = sum(p.numel() for p in eca_layer.parameters())

            model[idx] = eca_layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_c2f_eca()
    c2f_eca = C2f_ECA(c1=64, c2=64, n=2, shortcut=True, eca_k=3)
    x = torch.randn(2, 64, 40, 40)
    out = c2f_eca(x)
    print(f"C2f_ECA input:  {x.shape}")
    print(f"C2f_ECA output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    n_params = sum(p.numel() for p in c2f_eca.parameters())
    print(f"C2f_ECA parameters: {n_params}")
    print("C2f_ECA unit test passed successfully!")
