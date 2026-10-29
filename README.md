# Asymmetric Dual-Stream RGB–LWIR Attention Fusion & 2D-Grid CWAR for Transit Security Surveillance

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blue.svg?style=flat)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official repository for:
1. **Multimodal RGB–LWIR Paper:** *"Asymmetric Dual-Stream RGB–LWIR Attention Fusion and 2D-Grid Anchor-Aligned Confidence-Weighted Regression for Transit Hub Security Surveillance"* (**92.33% Test mAP@50**, **69.02%–69.49% Test mAP@50–95**, **92.89% Val mAP@50**).
2. **Single-Modality Thermal Paper:** *"Overcoming Radiant Heat Diffusion: Stage-Selective Dilated Attention in Lightweight YOLO for Real-Time Edge Thermal Surveillance"* (**88.15% Test mAP@50**).

---

## ⚡ Quick Copy-Paste Commands for Edge Device Latency & Power Benchmarking

Clone this repository onto any edge device, workstation, or cloud server and run the exact command below for your target platform:

### 1. Clone & Setup (Run on Any Edge Device)
```bash
git clone https://github.com/Carbe-mo/rgb-thermal-fusion.git
cd rgb-thermal-fusion
pip install ultralytics torch torchvision numpy pandas
```

---

### 2. Run Inference Latency & Throughput Suite on Edge CPU (Raspberry Pi 5, Raspberry Pi 4B, Rock Pi 4C+, Laptop/Server CPU)
Benchmarks all multimodal dual-stream models (`v36_main`, `v36_eca_rgb`, `v38_ca_dilated`, `v38_cwar_triad`), our thermal champion (`v30`), and all YOLO baselines (`YOLOv8n/s`, `YOLOv10n/s`, `YOLO11n/s`) in isolated crash-safe subprocesses:

```bash
# Run full suite (all 11 models, 30 runs each) and save to results/multimodal_edge_benchmark.csv:
python3 scripts/benchmark_multimodal_edge.py --device cpu --runs 30

# Fast mode — run ONLY our proposed multimodal models + v30 thermal champion (10 runs each):
python3 scripts/benchmark_multimodal_edge.py --device cpu --runs 10 --only-multimodal
```
> **Note for ARMv8.0 boards (Raspberry Pi 4B / Rock Pi 4C+):** If `OpenBLAS` gives a coretype warning, prefix with `OPENBLAS_CORETYPE=ARMV8`:
> ```bash
> OPENBLAS_CORETYPE=ARMV8 python3 scripts/benchmark_multimodal_edge.py --device cpu --runs 20 --skip-v10
> ```

---

### 3. Run Inference Latency, Throughput & Onboard Power Suite on GPU (NVIDIA Jetson AGX Xavier / Orin / Nano, RTX GPU, A100)

#### Option A: PyTorch GPU FP16 Benchmark (+ Automatic `jtop` Power Rail Telemetry on Jetson)
```bash
python3 scripts/benchmark_multimodal_edge.py --device 0 --half --runs 100
```

#### Option B: Jetson TensorRT FP16 Engine Build (`trtexec`) + `jtop` Hardware Power Rails
Automatically compiles `.onnx` weights in `weights/` into TensorRT FP16 `.engine` files and measures GPU compute latency, FPS, Idle/Active Board Power (W), GPU Rail Power (W), Energy per Frame (mJ/frame), and Frames/Joule:
```bash
python3 scripts/benchmark_multimodal_edge.py --mode trt-jtop --out-csv results/jetson_multimodal_trt_power.csv
```

---

### 4. Synchronized In-Line Joulescope JS220 High-Speed DC Power & Energy Profiling
When the edge device (Raspberry Pi 5 / Rock Pi 4C+ / RPi 4B) is wired through a **Joulescope JS220** connected to your host PC over SSH:

```bash
# Run from the host PC (automatically SSHes into the edge board and captures 2 MS/s Joulescope telemetry):
python scripts/automated_power_suite.py <EDGE_DEVICE_IP>
```
Or to trigger a single model manually on the edge device during standalone power measurement:
```bash
# 6-channel Multimodal Dual-Stream Models:
python3 -u scripts/measure_edge_power.py --weights weights/v36_main_best.pt --channels 6 --runs 10
python3 -u scripts/measure_edge_power.py --weights weights/v36_eca_rgb_best.pt --channels 6 --runs 10
python3 -u scripts/measure_edge_power.py --weights weights/v38_ca_dilated_best.pt --channels 6 --runs 10
python3 -u scripts/measure_edge_power.py --weights CWAR_TRIAD --channels 6 --runs 10

# 3-channel Single-Modality Thermal Models:
python3 -u scripts/measure_edge_power.py --weights weights/v30_champion_best.pt --channels 3 --runs 10
```

---

## 📊 Multimodal RGB–LWIR Detection Benchmark (1,067 Paired Frames)

Evaluated on the official validation split (213 images, 776 instances) and held-out test split (108 images, 390 instances):

| Model / Fusion Architecture | Sensing Modality | Params (M) | GFLOPs | Val mAP@50 | Val mAP@50–95 | Test Precision | Test Recall | **Test mAP@50** | **Test mAP@50–95** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n (Stock Baseline)** | LWIR Only (3-ch) | 3.01 | 8.1 | 86.25% | 60.41% | 82.74% | 83.35% | 86.22% | 62.92% |
| **v30 (Hybrid Dilated ECA Champion)** | LWIR Only (3-ch) | 3.01 | 8.1 | 87.52% | 63.40% | 87.81% | 83.40% | 88.15% | 64.73% |
| **v34 (Early 4-ch Stem Concat)** | RGB + LWIR (4-ch) | 3.01 | 8.1 | 84.20% | 59.74% | 81.76% | 82.47% | 83.68% | 58.67% |
| **Ours: `v38_ca_dilated` (Sym-Dil + NeckECA)** | RGB + LWIR (6-ch) | 4.50 | 11.6 | 89.33% | 65.52% | 87.32% | **86.12%** | 88.74% | 66.52% |
| **Ours: `v36_main` (Asym CA+ECA Dual-Stream)** | RGB + LWIR (6-ch) | 4.51 | 11.6 | **90.53%** | **66.11%** | 91.50% | 85.05% | 89.75% | **66.99%** |
| **Ours: `v36_eca_rgb` (Asym ECA Dual-Stream)** | RGB + LWIR (6-ch) | 4.50 | 11.6 | 89.75% | 65.86% | **93.01%** | 82.74% | **89.86%** | 66.76% |
| **Ours: `3_Core4_MACAF` (Class-Aware Routing)** | RGB + LWIR (6-ch) | 16.52 | 42.9 | 92.68% | 68.76% | 92.13% | 86.47% | 92.12% | **69.47%** |
| 👑 **Ours: `v38_cwar_triad` (2D-Grid CWAR Triad)** | **RGB + LWIR (6-ch)** | **13.51** | **34.8** | **92.86%** | **68.80%** | **91.79%** | **87.44%** | **92.33%** | **69.02%** |

---

## 📁 Pretrained Weights Included in `weights/`

All PyTorch (`.pt`) and ONNX (`.onnx`) weights are pre-exported in `weights/` so no training or export is required on edge devices:
- `weights/v36_main_best.pt` & `weights/v36_main_best.onnx` — Asymmetric Dual-Stream (`CA+ECA` RGB $d=1$, Dilated LWIR $d=2$)
- `weights/v36_eca_rgb_best.pt` & `weights/v36_eca_rgb_best.onnx` — Asymmetric Dual-Stream (`ECA` RGB $d=1$, Dilated LWIR $d=2$)
- `weights/v38_ca_dilated_best.pt` & `weights/v38_ca_dilated_best.onnx` — Symmetric Dilated Dual-Stream ($d=2$) + `PANetNeckECA`
- `weights/v30_champion_best.pt` & `weights/v30_champion_best.onnx` — Single-Modality Thermal Hybrid Dilated ECA Champion
- `weights/yolov8n.pt`, `yolov8s.pt`, `yolov10n.pt`, `yolov10s.pt`, `yolo11n.pt`, `yolo11s.pt` (and `.onnx` / `.espdl` quantized models)
