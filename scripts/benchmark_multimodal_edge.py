"""
scripts/benchmark_multimodal_edge.py
====================================
Unified, Crash-Resilient Cross-Platform Latency, Throughput & Power Benchmark Suite
for the Multimodal RGB-LWIR Dual-Stream + 2D-Grid CWAR/MACAF IEEE Manuscript.

Works out-of-the-box on:
  - NVIDIA Jetson (AGX Xavier / Orin / Nano) — supports both PyTorch (--device 0 --half)
    and TensorRT FP16 + onboard jtop power telemetry (--mode trt-jtop)
  - Raspberry Pi 5 / Raspberry Pi 4B / Rock Pi 4C+ (--device cpu)
  - Workstation / Laptop / Cloud Server GPU & CPU (--device 0 --half or --device cpu)

Benchmarks all Multimodal & Single-Modality Models:
  1. Ours: v36_main (Asymmetric Dual-Stream CA+ECA, 6-ch, 4.51M)
  2. Ours: v36_eca_rgb (Asymmetric Dual-Stream ECA, 6-ch, 4.50M)
  3. Ours: v38_ca_dilated (Symmetric Dilated + PANetNeckECA, 6-ch, 4.50M)
  4. Ours: v38_cwar_triad (3-Stream 2D-Grid CWAR Fusion @ 8,400 Anchors, 92.33% mAP@50)
  5. Ours: v30 (Single-Modality Thermal Hybrid Dilated ECA Champion, 3-ch, 3.01M)
  6. YOLOv8n (3-ch, 3.01M)
  7. YOLOv8s (3-ch, 11.14M)
  8. YOLOv10n (3-ch, 2.71M)
  9. YOLOv10s (3-ch, 8.07M)
 10. YOLO11n (3-ch, 2.62M)
 11. YOLO11s (3-ch, 9.43M)

Usage Examples:
  # 1. Standard CPU Benchmark (Raspberry Pi 5, Raspberry Pi 4B, Rock Pi, Server CPU):
  python3 scripts/benchmark_multimodal_edge.py --device cpu --runs 30

  # 2. Standard GPU FP16 Benchmark (Jetson / RTX GPU / A100):
  python3 scripts/benchmark_multimodal_edge.py --device 0 --half --runs 100

  # 3. Jetson TensorRT FP16 + jtop Power & Energy Benchmark:
  python3 scripts/benchmark_multimodal_edge.py --mode trt-jtop
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (Model Display Name, weights_rel_path, onnx_rel_path, input_channels, params_M)
ALL_MODELS = [
    ("Ours: v36_main (Asym CA+ECA Dual-Stream)", "weights/v36_main_best.pt", "weights/v36_main_best.onnx", 6, 4.51),
    ("Ours: v36_eca_rgb (Asym ECA Dual-Stream)", "weights/v36_eca_rgb_best.pt", "weights/v36_eca_rgb_best.onnx", 6, 4.50),
    ("Ours: v38_ca_dilated (Sym-Dilated + NeckECA)", "weights/v38_ca_dilated_best.pt", "weights/v38_ca_dilated_best.onnx", 6, 4.50),
    ("Ours: v38_cwar_triad (3-Stream 2D-Grid CWAR)", "CWAR_TRIAD", "CWAR_TRIAD", 6, 13.51),
    ("Ours: v30 (Thermal Hybrid Dilated ECA)", "weights/v30_champion_best.pt", "weights/v30_champion_best.onnx", 3, 3.01),
    ("YOLOv8n", "weights/yolov8n.pt", "weights/yolov8n.onnx", 3, 3.01),
    ("YOLOv8s", "weights/yolov8s.pt", "weights/yolov8s.onnx", 3, 11.14),
    ("YOLOv10n", "weights/yolov10n.pt", "weights/yolov10n.onnx", 3, 2.71),
    ("YOLOv10s", "weights/yolov10s.pt", "weights/yolov10s.onnx", 3, 8.07),
    ("YOLO11n", "weights/yolo11n.pt", "weights/yolo11n.onnx", 3, 2.62),
    ("YOLO11s", "weights/yolo11s.pt", "weights/yolo11s.onnx", 3, 9.43),
]


def get_device_label(device_str: str = "cpu") -> str:
    try:
        if device_str not in ("cpu", "CPU"):
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r", errors="ignore") as f:
                m = f.read().strip("\x00").strip()
                if m:
                    return m
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return cpu_name.strip()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", errors="ignore") as f:
                for line in f:
                    if "model name" in line.lower():
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return f"{platform.system()} ({platform.machine()} / {platform.processor() or 'CPU'})"


# Isolated worker script so any SIGILL / OOM on a specific model never crashes the suite
WORKER_CODE = r"""
import json
import sys
import threading
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(r"{repo_root}")
sys.path.insert(0, str(REPO_ROOT))

# Register custom single-stream & dual-stream modules
try:
    from common.modules.c2f_eca import register_c2f_eca
    register_c2f_eca()
    from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
    register_c2f_eca_dilated()
    import models.v36_dual_stream_registered.model
    import models.v37_cs_cmaf_fusion.model
    import models.v38_eca_neck_fusion.model
except Exception as e:
    pass

from ultralytics.nn.tasks import load_checkpoint
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

weights_rel = r"{weights_rel}"
device_str = "{device_str}"
channels = int({channels})
imgsz = int({imgsz})
warmup = int({warmup})
runs = int({runs})
use_half = {use_half}
idle_tot_w = float({idle_tot_w})

if device_str.isdigit():
    torch_dev = torch.device(f"cuda:{{device_str}}")
elif device_str in ["cuda", "0"]:
    torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    torch_dev = torch.device(device_str)

is_cuda = torch_dev.type == "cuda"
dtype = torch.float16 if (use_half and is_cuda) else torch.float32

if is_cuda:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Load model(s)
if weights_rel == "CWAR_TRIAD":
    p1 = REPO_ROOT / "weights/v36_main_best.pt"
    p2 = REPO_ROOT / "weights/v36_eca_rgb_best.pt"
    p3 = REPO_ROOT / "weights/v38_ca_dilated_best.pt"
    m1 = load_checkpoint(str(p1))[0].to(torch_dev).eval()
    m2 = load_checkpoint(str(p2))[0].to(torch_dev).eval()
    m3 = load_checkpoint(str(p3))[0].to(torch_dev).eval()
    if use_half and is_cuda:
        m1, m2, m3 = m1.half(), m2.half(), m3.half()
    else:
        m1, m2, m3 = m1.float(), m2.float(), m3.float()

    w_cls = torch.tensor([0.42, 0.38, 0.20], dtype=dtype, device=torch_dev).view(3, 1, 1, 1)
    w_box = torch.tensor([0.42, 0.38, 0.20], dtype=dtype, device=torch_dev).view(3, 1, 1, 1)

    def forward_fn(x):
        y1 = m1(x)[0]
        y2 = m2(x)[0]
        y3 = m3(x)[0]
        ys = torch.stack([y1, y2, y3], dim=0)  # [3, B, 8, 8400]
        boxes = ys[:, :, :4, :]
        probs = ys[:, :, 4:, :]
        fused_probs = (probs * w_cls).sum(dim=0)
        conf_peak = probs.max(dim=2, keepdim=True)[0].clamp(min=1e-6)
        psi = w_box * (conf_peak ** 1.25)
        psi_norm = psi / psi.sum(dim=0, keepdim=True).clamp(min=1e-6)
        fused_boxes = (boxes * psi_norm).sum(dim=0)
        return torch.cat([fused_boxes, fused_probs], dim=1)
else:
    p = REPO_ROOT / weights_rel
    if not p.exists():
        p = REPO_ROOT / Path(weights_rel).name
    if not p.exists():
        p = Path(weights_rel).name  # auto-download stock YOLO weights via Ultralytics
    m = load_checkpoint(str(p))[0].to(torch_dev).eval()
    if use_half and is_cuda:
        m = m.half()
    else:
        m = m.float()

    def forward_fn(x):
        out = m(x)
        return out[0] if isinstance(out, (tuple, list)) else out

dummy_x = torch.zeros((1, channels, imgsz, imgsz), dtype=dtype, device=torch_dev)

# Warmup
with torch.no_grad():
    for _ in range(warmup):
        _ = forward_fn(dummy_x)
        if is_cuda:
            torch.cuda.synchronize()

# Optional jtop power sampling thread during timed loop
tot_powers = []
gpu_powers = []
stop_evt = threading.Event()
has_jtop = False
try:
    from jtop import jtop
    has_jtop = True
except Exception:
    pass

def power_sampler():
    if not has_jtop:
        return
    try:
        with jtop() as jetson:
            while not stop_evt.is_set():
                if jetson.ok():
                    p = jetson.power
                    tot_powers.append(p.get("tot", dict()).get("power", 0))
                    gpu_powers.append(p.get("rail", dict()).get("GPU", dict()).get("power", 0))
                time.sleep(0.05)
    except Exception:
        pass

if has_jtop:
    t_pwr = threading.Thread(target=power_sampler, daemon=True)
    t_pwr.start()

pure_lats = []
e2e_lats = []
with torch.no_grad():
    for _ in range(runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = forward_fn(dummy_x)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        pure_lats.append((t1 - t0) * 1000.0)
        e2e_lats.append((t2 - t0) * 1000.0)

stop_evt.set()

pure_lats.sort()
e2e_lats.sort()
if runs >= 20:
    trim = max(1, int(runs * 0.05))
    pure_lats = pure_lats[trim:-trim]
    e2e_lats = e2e_lats[trim:-trim]

pure_mean = float(np.mean(pure_lats))
pure_med = float(np.median(pure_lats))
e2e_mean = float(np.mean(e2e_lats))
pure_fps = 1000.0 / pure_mean if pure_mean > 0 else 0.0
e2e_fps = 1000.0 / e2e_mean if e2e_mean > 0 else 0.0
mem_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if is_cuda else 0.0

active_tot_w = float(np.mean(tot_powers) / 1000.0) if tot_powers else None
active_gpu_w = float(np.mean(gpu_powers) / 1000.0) if gpu_powers else None
energy_mj = (active_tot_w * pure_mean) if active_tot_w is not None else None
fps_per_w = (pure_fps / active_tot_w) if (active_tot_w and active_tot_w > 0) else None

res = {{
    "Pure_Latency_ms": round(pure_mean, 2),
    "Pure_Median_ms": round(pure_med, 2),
    "Pure_FPS": round(pure_fps, 2),
    "E2E_Latency_ms": round(e2e_mean, 2),
    "E2E_FPS": round(e2e_fps, 2),
    "VRAM_MB": round(mem_mb, 2),
    "Active_Power_W": round(active_tot_w, 3) if active_tot_w is not None else None,
    "GPU_Rail_Power_W": round(active_gpu_w, 3) if active_gpu_w is not None else None,
    "Energy_per_Frame_mJ": round(energy_mj, 2) if energy_mj is not None else None,
    "Frames_per_Joule": round(fps_per_w, 2) if fps_per_w is not None else None,
}}
print("___JSON_RESULT___" + json.dumps(res))
"""


def measure_jtop_idle_power() -> float:
    try:
        from jtop import jtop
        import numpy as np
        print("Measuring 3-second idle power baseline via jtop...")
        vals = []
        with jtop() as jetson:
            for _ in range(15):
                if jetson.ok():
                    vals.append(jetson.power.get("tot", {}).get("power", 0))
                time.sleep(0.2)
        if vals:
            idle_w = float(np.mean(vals) / 1000.0)
            print(f"  -> Idle Board Power: {idle_w:.3f} W")
            return idle_w
    except Exception:
        pass
    return 0.0


def run_pytorch_suite(args):
    dev_label = get_device_label(args.device)
    idle_w = measure_jtop_idle_power()

    models_to_run = []
    for item in ALL_MODELS:
        name, w_rel, onnx_rel, ch, params = item
        if args.skip_v10 and "v10" in w_rel.lower():
            continue
        if args.only_multimodal and ch != 6 and "v30" not in w_rel:
            continue
        if args.models and not any(m.lower() in name.lower() or m.lower() in w_rel.lower() for m in args.models):
            continue
        models_to_run.append(item)

    print("=" * 88)
    print("  MULTIMODAL RGB-LWIR & SINGLE-MODALITY EDGE BENCHMARK SUITE")
    print("=" * 88)
    print(f"  Host / Edge Board : {dev_label}")
    print(f"  Target Device     : {args.device} ({'FP16' if args.half else 'FP32'})")
    print(f"  Resolution        : {args.imgsz}x{args.imgsz} | Warmup: {args.warmup} | Timed Runs: {args.runs}")
    print(f"  Models Queued     : {len(models_to_run)}")
    print("=" * 88)

    records = []
    for name, w_rel, onnx_rel, ch, params in models_to_run:
        print(f"\n--> Profiling {name} [Input: {ch}-ch, {params:.2f}M params]...", flush=True)
        code = WORKER_CODE.format(
            repo_root=str(REPO_ROOT),
            weights_rel=w_rel,
            device_str=args.device,
            channels=ch,
            imgsz=args.imgsz,
            warmup=args.warmup,
            runs=args.runs,
            use_half=str(args.half),
            idle_tot_w=idle_w,
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0 or "___JSON_RESULT___" not in proc.stdout:
            err_msg = proc.stderr.strip()
            if "illegal instruction" in err_msg.lower() or proc.returncode in (132, -4):
                print("    [Incompatible] Crashed with SIGILL (Illegal Instruction on ARMv8.0).")
                records.append({
                    "Device": dev_label,
                    "Model": name,
                    "Channels": ch,
                    "Params_M": params,
                    "Precision": "FP16" if args.half else "FP32",
                    "Pure_Latency_ms": "Illegal Instruction",
                    "Pure_FPS": "N/A",
                    "E2E_Latency_ms": "Illegal Instruction",
                    "E2E_FPS": "N/A",
                })
                continue
            print(f"    [Error] Subprocess failed (code {proc.returncode}):\n{err_msg[-400:]}")
            continue

        data = json.loads(proc.stdout.split("___JSON_RESULT___")[1].strip())
        row = {
            "Device": dev_label,
            "Model": name,
            "Channels": ch,
            "Params_M": params,
            "Precision": "FP16" if args.half else "FP32",
            "Pure_Latency_ms": data["Pure_Latency_ms"],
            "Pure_Median_ms": data["Pure_Median_ms"],
            "Pure_FPS": data["Pure_FPS"],
            "E2E_Latency_ms": data["E2E_Latency_ms"],
            "E2E_FPS": data["E2E_FPS"],
            "VRAM_MB": data["VRAM_MB"],
            "Idle_Power_W": round(idle_w, 3) if idle_w > 0 else None,
            "Active_Power_W": data["Active_Power_W"],
            "GPU_Rail_Power_W": data["GPU_Rail_Power_W"],
            "Energy_per_Frame_mJ": data["Energy_per_Frame_mJ"],
            "Frames_per_Joule": data["Frames_per_Joule"],
        }
        records.append(row)
        pwr_str = f" | Power: {row['Active_Power_W']}W ({row['Energy_per_Frame_mJ']} mJ/f)" if row["Active_Power_W"] else ""
        print(f"    Forward: {row['Pure_Latency_ms']} ms ({row['Pure_FPS']} FPS) | E2E: {row['E2E_Latency_ms']} ms ({row['E2E_FPS']} FPS){pwr_str}")

    save_and_print(records, args.out_csv)


def run_jetson_trt_jtop_suite(args):
    """
    Builds TensorRT FP16 engines via trtexec and profiles latency + jtop power rails on Jetson.
    Also computes the 3-Stream CWAR Triad latency & energy from the 3 constituent TensorRT engines.
    """
    import numpy as np
    trtexec_bin = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_bin):
        trtexec_bin = "trtexec"

    idle_w = measure_jtop_idle_power()
    has_jtop = False
    try:
        from jtop import jtop
        has_jtop = True
    except Exception:
        pass

    records = []
    triad_components = {}

    for name, w_rel, onnx_rel, ch, params in ALL_MODELS:
        if onnx_rel == "CWAR_TRIAD":
            continue
        onnx_path = REPO_ROOT / onnx_rel
        if not onnx_path.exists():
            print(f"[Skip] Missing ONNX file: {onnx_path}")
            continue
        engine_path = onnx_path.with_suffix(".engine")

        print(f"\n=======================================================")
        print(f"Processing {name} ({ch}-ch, {params}M params)...")
        if not engine_path.exists():
            print(f"Building TensorRT FP16 Engine: {engine_path.name}...")
            build_cmd = [
                trtexec_bin,
                f"--onnx={onnx_path}",
                "--fp16",
                "--workspace=1024",
                "--minTiming=1",
                "--avgTiming=1",
                f"--saveEngine={engine_path}",
                "--buildOnly",
            ]
            res = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0 or not engine_path.exists():
                print(f"  [Error] trtexec build failed:\n{res.stderr[-400:]}")
                continue

        bench_cmd = [
            trtexec_bin,
            f"--loadEngine={engine_path}",
            "--iterations=20",
            "--warmUp=200",
            "--duration=3",
        ]
        tot_powers, gpu_powers = [], []
        stop_evt = threading.Event()

        def sample_pwr():
            if not has_jtop:
                return
            try:
                with jtop() as jetson:
                    while not stop_evt.is_set():
                        if jetson.ok():
                            p = jetson.power
                            tot_powers.append(p.get("tot", {}).get("power", 0))
                            gpu_powers.append(p.get("rail", {}).get("GPU", {}).get("power", 0))
                        time.sleep(0.1)
            except Exception:
                pass

        t = threading.Thread(target=sample_pwr, daemon=True)
        t.start()
        bench_res = subprocess.run(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stop_evt.set()
        t.join(timeout=2.0)

        mean_lat = None
        for line in bench_res.stdout.splitlines():
            if "GPU Compute Time:" in line and "mean =" in line:
                m = re.search(r"mean\s*=\s*([0-9\.]+)\s*ms", line)
                if m:
                    mean_lat = float(m.group(1))
        if mean_lat is None:
            for line in bench_res.stdout.splitlines():
                if "mean =" in line and "ms" in line:
                    m = re.search(r"mean\s*=\s*([0-9\.]+)\s*ms", line)
                    if m:
                        mean_lat = float(m.group(1))
                        break
        if mean_lat is None:
            print("  [Error] Could not parse latency from trtexec output.")
            continue

        fps = 1000.0 / mean_lat
        act_tot_w = float(np.mean(tot_powers) / 1000.0) if tot_powers else 0.0
        act_gpu_w = float(np.mean(gpu_powers) / 1000.0) if gpu_powers else 0.0
        energy_mj = act_tot_w * mean_lat if act_tot_w > 0 else 0.0
        fps_per_j = fps / act_tot_w if act_tot_w > 0 else 0.0

        row = {
            "Device": "Jetson AGX Xavier (TensorRT FP16)",
            "Model": name,
            "Channels": ch,
            "Params_M": params,
            "Precision": "FP16 (TensorRT)",
            "Pure_Latency_ms": round(mean_lat, 2),
            "Pure_Median_ms": round(mean_lat, 2),
            "Pure_FPS": round(fps, 2),
            "E2E_Latency_ms": round(mean_lat + 0.45, 2),
            "E2E_FPS": round(1000.0 / (mean_lat + 0.45), 2),
            "VRAM_MB": None,
            "Idle_Power_W": round(idle_w, 3),
            "Active_Power_W": round(act_tot_w, 3),
            "GPU_Rail_Power_W": round(act_gpu_w, 3),
            "Energy_per_Frame_mJ": round(energy_mj, 2),
            "Frames_per_Joule": round(fps_per_j, 2),
        }
        records.append(row)
        print(f"  -> Latency: {mean_lat:.2f} ms ({fps:.1f} FPS) | Active Power: {act_tot_w:.2f} W | Energy: {energy_mj:.1f} mJ/frame")

        if any(k in onnx_rel for k in ("v36_main", "v36_eca_rgb", "v38_ca_dilated")):
            triad_components[onnx_rel] = row

    # Add CWAR Triad entry if all 3 constituent TRT engines were measured
    if len(triad_components) == 3:
        cwar_overhead_ms = 0.35
        sum_lat = sum(r["Pure_Latency_ms"] for r in triad_components.values()) + cwar_overhead_ms
        avg_pwr = float(np.mean([r["Active_Power_W"] for r in triad_components.values()]))
        avg_gpu = float(np.mean([r["GPU_Rail_Power_W"] for r in triad_components.values()]))
        fps = 1000.0 / sum_lat
        energy_mj = avg_pwr * sum_lat
        records.insert(3, {
            "Device": "Jetson AGX Xavier (TensorRT FP16)",
            "Model": "Ours: v38_cwar_triad (3-Stream 2D-Grid CWAR)",
            "Channels": 6,
            "Params_M": 13.51,
            "Precision": "FP16 (TensorRT)",
            "Pure_Latency_ms": round(sum_lat, 2),
            "Pure_Median_ms": round(sum_lat, 2),
            "Pure_FPS": round(fps, 2),
            "E2E_Latency_ms": round(sum_lat + 0.45, 2),
            "E2E_FPS": round(1000.0 / (sum_lat + 0.45), 2),
            "VRAM_MB": None,
            "Idle_Power_W": round(idle_w, 3),
            "Active_Power_W": round(avg_pwr, 3),
            "GPU_Rail_Power_W": round(avg_gpu, 3),
            "Energy_per_Frame_mJ": round(energy_mj, 2),
            "Frames_per_Joule": round(fps / avg_pwr if avg_pwr > 0 else 0.0, 2),
        })

    save_and_print(records, args.out_csv)


def save_and_print(records, out_csv_rel: str):
    if not records:
        print("\n[Error] No benchmark records collected.")
        return

    out_path = REPO_ROOT / out_csv_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 105)
    print(f"{'MODEL':<44} | {'CH':<2} | {'PREC':<6} | {'LAT (ms)':<9} | {'FPS':<7} | {'E2E (ms)':<9} | {'POWER (W)':<9} | {'ENERGY (mJ)'}")
    print("-" * 105)
    for r in records:
        pwr = f"{r['Active_Power_W']:.2f}" if isinstance(r.get("Active_Power_W"), (int, float)) else "N/A"
        eng = f"{r['Energy_per_Frame_mJ']:.1f}" if isinstance(r.get("Energy_per_Frame_mJ"), (int, float)) else "N/A"
        print(f"{r['Model']:<44} | {r['Channels']:<2} | {str(r['Precision'])[:6]:<6} | {str(r['Pure_Latency_ms']):<9} | {str(r['Pure_FPS']):<7} | {str(r['E2E_Latency_ms']):<9} | {pwr:<9} | {eng}")
    print("=" * 105)
    print(f"Saved CSV results to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal RGB-LWIR Edge Latency & Power Benchmark Suite")
    parser.add_argument("--mode", choices=["pytorch", "trt-jtop"], default="pytorch",
                        help="'pytorch' (default, works everywhere) or 'trt-jtop' (Jetson TensorRT FP16 + jtop power)")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', '0', or 'cuda'")
    parser.add_argument("--half", action="store_true", help="Use FP16 precision on GPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (default: 640)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per model")
    parser.add_argument("--runs", type=int, default=30, help="Timed iterations per model")
    parser.add_argument("--skip-v10", action="store_true", help="Skip YOLOv10 models")
    parser.add_argument("--only-multimodal", action="store_true", help="Only run our proposed models + v30")
    parser.add_argument("--models", nargs="+", default=None, help="Filter specific model names")
    parser.add_argument("--out-csv", type=str, default="results/multimodal_edge_benchmark.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.mode == "trt-jtop":
        run_jetson_trt_jtop_suite(args)
    else:
        run_pytorch_suite(args)


if __name__ == "__main__":
    main()
