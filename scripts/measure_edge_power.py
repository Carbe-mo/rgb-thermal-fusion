"""
scripts/measure_edge_power.py
=============================
Lightweight synchronized inference runner for Joulescope edge power profiling.
Runs on the edge board (Rock Pi 4C+ / Raspberry Pi 4B).

Protocol:
1. Loads weights into memory.
2. Warms up 2 inferences.
3. Prints: [SYNC_READY] (flushed to stdout).
4. Waits for line from stdin (sent by host when Joulescope starts recording).
5. Runs exactly N inferences in a tight loop.
6. Prints: [SYNC_DONE duration_s=<float> avg_ms=<float>] (flushed to stdout).
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def register_all():
    try:
        from common.modules.c2f_eca import register_c2f_eca
        register_c2f_eca()
    except Exception:
        pass
    try:
        from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
        register_c2f_eca_dilated()
    except Exception:
        pass
    try:
        from common.modules.fusion import register_fusion_modules
        register_fusion_modules()
    except Exception:
        pass
    try:
        import models.v36_dual_stream_registered.model
        import models.v37_cs_cmaf_fusion.model
        import models.v38_eca_neck_fusion.model
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    register_all()
    from ultralytics.nn.tasks import load_checkpoint

    device = torch.device(args.device)
    dummy_input = torch.zeros(1, args.channels, args.imgsz, args.imgsz, dtype=torch.float32, device=device)

    print(f"Loading weights: {args.weights}...", flush=True)
    if args.weights == "CWAR_TRIAD":
        m1 = load_checkpoint(str(REPO_ROOT / "weights/v36_main_best.pt"))[0].float().to(device).eval()
        m2 = load_checkpoint(str(REPO_ROOT / "weights/v36_eca_rgb_best.pt"))[0].float().to(device).eval()
        m3 = load_checkpoint(str(REPO_ROOT / "weights/v38_ca_dilated_best.pt"))[0].float().to(device).eval()
        w_cls = torch.tensor([0.42, 0.38, 0.20], dtype=torch.float32, device=device).view(3, 1, 1, 1)
        w_box = torch.tensor([0.42, 0.38, 0.20], dtype=torch.float32, device=device).view(3, 1, 1, 1)
        def run_step(x):
            ys = torch.stack([m1(x)[0], m2(x)[0], m3(x)[0]], dim=0)
            boxes, probs = ys[:, :, :4, :], ys[:, :, 4:, :]
            fused_probs = (probs * w_cls).sum(dim=0)
            conf_peak = probs.max(dim=2, keepdim=True)[0].clamp(min=1e-6)
            psi = w_box * (conf_peak ** 1.25)
            psi_norm = psi / psi.sum(dim=0, keepdim=True).clamp(min=1e-6)
            fused_boxes = (boxes * psi_norm).sum(dim=0)
            return torch.cat([fused_boxes, fused_probs], dim=1)
    else:
        m = load_checkpoint(args.weights)[0].float().to(device).eval()
        def run_step(x):
            return m(x)

    # Warmup 2 iterations
    print("Warming up...", flush=True)
    with torch.no_grad():
        for _ in range(2):
            _ = run_step(dummy_input)

    # Signal host that board is ready to start N inferences
    print("[SYNC_READY]", flush=True)

    # Wait for trigger from host
    _ = sys.stdin.readline()

    # Precise timing for the requested inferences
    latencies = []
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = run_step(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    t_end = time.perf_counter()

    total_s = t_end - t_start
    avg_ms = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"[SYNC_DONE duration_s={total_s:.4f} avg_ms={avg_ms:.2f} fps={fps:.2f}]", flush=True)

if __name__ == "__main__":
    main()
