"""
modules/fusion.py — RGB-Thermal Multi-Modal Fusion Layers & Parser Integration
=============================================================================
Defines multi-modal fusion primitives and patches Ultralytics parse_model to support:
- SliceChannels: Extracts channel subsets (e.g. RGB 0:3, Thermal 3:6)
- AddFusion: Element-wise feature map summation
- GatedFusion: Dynamic spatial-channel gating map g: F_fused = g * F_rgb + (1 - g) * F_thermal
- C2f_ECA & C2f_ECA_Dilated
"""

import copy
import sys
import types
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, Index
from modules.c2f_eca import C2f_ECA
from modules.c2f_eca_dilated import C2f_ECA_Dilated
import modules.eca as eca_mod
import modules.c2f_eca as c2f_eca_mod
import modules.c2f_eca_dilated as c2f_eca_dilated_mod

# Alias 'common' and 'common.modules' so pickled checkpoints load seamlessly
if "common" not in sys.modules or not hasattr(sys.modules["common"], "modules"):
    _common = types.ModuleType("common")
    _common_modules = types.ModuleType("common.modules")
    _common.modules = _common_modules
    sys.modules["common"] = _common
    sys.modules["common.modules"] = _common_modules
    sys.modules["common.modules.eca"] = eca_mod
    sys.modules["common.modules.c2f_eca"] = c2f_eca_mod
    sys.modules["common.modules.c2f_eca_dilated"] = c2f_eca_dilated_mod
    sys.modules["common.modules.fusion"] = sys.modules[__name__]



class SliceChannels(nn.Module):
    """
    Extracts a slice of channels from the input tensor [B, C, H, W].
    """
    def __init__(self, start_ch: int, end_ch: int):
        super().__init__()
        self.start_ch = start_ch
        self.end_ch = end_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start_ch:self.end_ch, :, :]


class AddFusion(nn.Module):
    """
    Element-wise addition of two feature maps.
    """
    def __init__(self, c: int = 0):
        super().__init__()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return x[0] + x[1]


class GatedFusion(nn.Module):
    """
    Dynamic Gated Fusion (MBNet / GMFNet style).
    Learns spatial-channel gating map g in [0, 1] to dynamically balance modalities:
        F_fused = g * F_rgb + (1 - g) * F_thermal
    """
    def __init__(self, c: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        f_rgb, f_th = x[0], x[1]
        cat = torch.cat([f_rgb, f_th], dim=1)
        g = self.gate_conv(cat)
        return g * f_rgb + (1.0 - g) * f_th


_REGISTERED = False


def register_all_modules() -> None:
    """
    Register all custom modules with the Ultralytics model parser.
    Safe to call multiple times (idempotent).
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    custom_classes = [SliceChannels, AddFusion, GatedFusion, C2f_ECA, C2f_ECA_Dilated]
    for cls in custom_classes:
        tasks.__dict__[cls.__name__] = cls
        modules.__dict__[cls.__name__] = cls

    orig_parse = tasks.parse_model

    def custom_parse(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)
        extra_save = []

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "SliceChannels":
                        custom_layers[(section, i)] = ("SliceChannels", f, n, args)
                        out_c = args[1] - args[0]
                        d_mod[section][i] = [f, 1, "Index", [out_c]]
                    elif m_name in ("GatedFusion", "AddFusion"):
                        custom_layers[(section, i)] = (m_name, f, n, args)
                        f1, f2 = f[0], f[1]
                        extra_save.append(f2)
                        d_mod[section][i] = [f1, 1, "nn.Identity", []]
                    elif m_name == "C2f_ECA":
                        custom_layers[(section, i)] = ("C2f_ECA", f, n, args)
                        d_mod[section][i] = [f, n, "C2f", args]
                    elif m_name == "C2f_ECA_Dilated":
                        custom_layers[(section, i)] = ("C2f_ECA_Dilated", f, n, args)
                        d_mod[section][i] = [f, n, "C2f", args]

        model, save = orig_parse(d_mod, ch, verbose=verbose)

        for s_idx in extra_save:
            if s_idx not in save:
                save.append(s_idx)
        save.sort()

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_name, f, n, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i

            if m_name == "SliceChannels":
                layer = SliceChannels(args[0], args[1])
                layer.i = idx
                layer.f = f
                layer.type = "modules.fusion.SliceChannels"
                layer.np = 0
                model[idx] = layer

            elif m_name == "GatedFusion":
                c = model[f[0]].conv.out_channels if hasattr(model[f[0]], "conv") else (
                    model[f[0]].cv2.conv.out_channels if hasattr(model[f[0]], "cv2") else args[0]
                )
                layer = GatedFusion(c)
                layer.i = idx
                layer.f = f
                layer.type = "modules.fusion.GatedFusion"
                layer.np = sum(p.numel() for p in layer.parameters())
                model[idx] = layer

            elif m_name == "AddFusion":
                c = args[0]
                layer = AddFusion(c)
                layer.i = idx
                layer.f = f
                layer.type = "modules.fusion.AddFusion"
                layer.np = 0
                model[idx] = layer

            elif m_name == "C2f_ECA":
                c1 = model[idx].cv1.conv.in_channels
                c2 = model[idx].cv2.conv.out_channels
                n_rep = len(model[idx].m)
                shortcut = args[1] if len(args) > 1 else False
                g = args[2] if len(args) > 2 else 1
                e = args[3] if len(args) > 3 else 0.5
                eca_k = args[4] if len(args) > 4 else 3
                layer = C2f_ECA(c1, c2, n=n_rep, shortcut=shortcut, g=g, e=e, eca_k=eca_k)
                layer.i = idx
                layer.f = f
                layer.type = "modules.c2f_eca.C2f_ECA"
                layer.np = sum(p.numel() for p in layer.parameters())
                model[idx] = layer

            elif m_name == "C2f_ECA_Dilated":
                c1 = model[idx].cv1.conv.in_channels
                c2 = model[idx].cv2.conv.out_channels
                n_rep = len(model[idx].m)
                shortcut = args[1] if len(args) > 1 else False
                g = args[2] if len(args) > 2 else 1
                e = args[3] if len(args) > 3 else 0.5
                eca_k = args[4] if len(args) > 4 else 3
                dilation = args[5] if len(args) > 5 else 2
                layer = C2f_ECA_Dilated(c1, c2, n=n_rep, shortcut=shortcut, g=g, e=e, eca_k=eca_k, dilation=dilation)
                layer.i = idx
                layer.f = f
                layer.type = "modules.c2f_eca_dilated.C2f_ECA_Dilated"
                layer.np = sum(p.numel() for p in layer.parameters())
                model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse
    _REGISTERED = True
