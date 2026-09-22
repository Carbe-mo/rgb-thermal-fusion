# Multi-Modal RGB-Thermal Fusion Object Detection (DualX-YOLO)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%2B-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An advanced multi-modal object detection framework fusing **Visible Spectrum (RGB)** and **Thermal Infrared (TIR)** imagery. Features early 4-channel pixel fusion and an asymmetric dual-backbone architecture with multi-scale **dynamic spatial-channel gated attention** (`GatedFusion`).

---

## 🌟 Key Results (FLIR Benchmark)

| Model Architecture | Modality | Params (M) | GFLOPs | Test mAP@50 (%) | Test mAP@50-95 (%) | Test Precision (%) | Test Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-Stream YOLOv8n (Baseline) | Thermal | 3.01 | 8.8 | 86.93 | 61.92 | 85.08 | 78.64 |
| Single-Stream YOLO11n (Baseline) | Thermal | 2.59 | 6.6 | 87.37 | 62.55 | 83.24 | 82.76 |
| **Early Fusion (v34, 4-Channel)** | RGB + TIR | 3.01 | 8.1 | 83.68 | 58.67 | 81.76 | **82.47** |
| 🏆 **Dual-Backbone Gated Fusion (v35)** | **RGB + TIR** | **4.45** | **16.2** | **87.54** | **64.84** | **90.68** | 81.05 |

### Per-Class Performance Gains (v35 Dual-Backbone vs. Baseline)
- **Trolley**: **97.4%** (+2.1% improvement)
- **Baggage / Luggage**: **92.1%** (+17.6% improvement)
- **Unattended Baggage**: **84.3%** (+6.6% improvement)
- **Overall Detection Precision**: **90.68%** (+5.6% improvement)

---

## 📐 Architectures

### 1. Dual-Backbone Multi-Scale Gated Fusion (v35)
```
Input: 6-Channel Tensor [RGB (0:3) | Thermal (3:6)]
   │
   ├─── Stream A: Visible RGB Backbone (Extracts texture, color, and fine edges)
   │        ├── P3_rgb (64 ch)  ──┐
   │        ├── P4_rgb (128 ch) ──┼────────┐
   │        └── P5_rgb (256 ch) ──┼────────┼────────┐
   │                              ▼        ▼        ▼
   │                         [GatedFusion] [GatedFusion] [GatedFusion]
   │                              ▲        ▲        ▲
   └─── Stream B: Thermal Dilated ECA ──┘        │        │
            ├── P3_th (64 ch)   ─────────────────┘        │
            ├── P4_th (128 ch)  ──────────────────────────┘
            └── P5_th (256 ch)
                                  │        │        │
                                  ▼        ▼        ▼
                              Shared PANet Feature Neck
                                          │
                                          ▼
                                   Detection Head
```

### 2. Early Fusion (v34)
- Integrates visual texture and thermal heatmaps at the pixel level into a 4-channel tensor `[R, G, B, T_gray]`.
- First convolution initialized via **weight surgery**, preserving pretrained ImageNet weights with zero random disruption.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Carbe-mo/rgb-thermal-fusion.git
cd rgb-thermal-fusion
pip install -r requirements.txt
```

### Data Preparation
To generate the multi-channel training sets from paired RGB and thermal directories:

1. **Generate 4-channel RGBA images (for Early Fusion)**:
   ```bash
   python scripts/prepare_fusion_dataset.py --thermal_dir /path/to/thermal --rgb_dir /path/to/rgb --out_dir data/fusion
   ```
2. **Generate 6-channel native `.npy` arrays (for Dual Fusion)**:
   ```bash
   python scripts/prepare_dual_fusion_dataset.py --thermal_dir /path/to/thermal --rgb_dir /path/to/rgb --out_dir data/fusion_dual
   ```

### Training

- **Train Dual-Backbone Gated Fusion (v35)**:
  ```bash
  python train_dual_fusion.py --data data/fusion_dual_data.yaml --epochs 100 --batch 16 --device 0
  ```

- **Train Early Fusion (v34)**:
  ```bash
  python train_early_fusion.py --data data/fusion_data.yaml --epochs 100 --batch 16 --device 0
  ```

### Evaluation
Evaluate the trained models directly on test splits:
```bash
python scripts/evaluate_fusion.py --model weights/v35_dual_fusion_best.pt --data data/fusion_dual_data.yaml --split test
```

---

## 📦 Model Weights
Pretrained and fine-tuned checkpoints are provided directly in `weights/`:
- `v35_dual_fusion_best.pt`: Fine-tuned dual-backbone checkpoint (mAP50-95 = 64.84%)
- `v35_dual_fusion_best.onnx`: Exported ONNX graph for cross-platform inference
- `v35_dual_fusion_pretrained.pt`: Pretrained dual-stream weights transfer checkpoint
- `v34_early_fusion_best.pt`: Fine-tuned early fusion checkpoint
- `v34_early_fusion_pretrained.pt`: Surgical 4-channel transfer checkpoint

---

## 📄 Citation & Attribution
If you use this architecture or multi-modal fusion pipeline in your research, please cite:
```bibtex
@article{mohit2026dualx,
  title={DualX-YOLO: Multi-Modal Visible-Thermal Object Detection with Receptive Field Expansion and Dynamic Gated Attention},
  author={Mohit, Satya},
  year={2026}
}
```
