"""
scripts/prepare_dual_fusion_dataset.py
======================================
Prepares 6-channel dataset for Dual-Backbone RGB-Thermal Fusion (v35):
- Channels 0-2: Visible RGB image
- Channels 3-5: Thermal pseudo-color image
- Saves as native NumPy .npy arrays [480, 640, 6] uint8
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import argparse

def prepare_dual_dataset(thermal_dir: str, rgb_dir: str, out_dir: str, splits_dir: str = None):
    thermal_path = Path(thermal_dir)
    rgb_path = Path(rgb_dir)
    out_path = Path(out_dir)

    img_out = out_path / 'images'
    lbl_out = out_path / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    thermal_files = sorted(list((thermal_path / 'images').glob('*.jpg')))
    print(f"Preparing 6-channel arrays for {len(thermal_files)} image pairs...")

    for t_path in tqdm(thermal_files, desc="Writing 6-channel .npy arrays"):
        stem = t_path.stem
        match = re.search(r'\d+', stem)
        if not match:
            continue

        idx_str = match.group()
        rgb_idx = int(idx_str) + 1
        rgb_stem = stem[:match.start()] + str(rgb_idx).zfill(len(idx_str)) + stem[match.end():]
        r_file = rgb_path / f"{rgb_stem}.jpg"

        if not r_file.exists():
            continue

        t_img = cv2.imread(str(t_path))        # [480, 640, 3] BGR Thermal
        rgb_img = cv2.imread(str(r_file))      # [480, 640, 3] BGR Visual

        if t_img is None or rgb_img is None:
            continue

        if t_img.shape[:2] != rgb_img.shape[:2]:
            rgb_img = cv2.resize(rgb_img, (t_img.shape[1], t_img.shape[0]))

        # Stack into 6-channel array: [BGR_rgb, BGR_thermal]
        arr_6ch = np.concatenate([rgb_img, t_img], axis=2).astype(np.uint8)  # [480, 640, 6]

        npy_path = img_out / f"{stem}.npy"
        np.save(str(npy_path), arr_6ch)

        jpg_path = img_out / f"{stem}.jpg"
        if not jpg_path.exists():
            cv2.imwrite(str(jpg_path), t_img)

        # Copy label
        l_file = thermal_path / 'labels' / f"{stem}.txt"
        if l_file.exists():
            shutil.copy(l_file, lbl_out / f"{stem}.txt")

    if splits_dir and Path(splits_dir).exists():
        s_path = Path(splits_dir)
        for split_name in ['train.txt', 'val.txt', 'test.txt']:
            src = s_path / split_name
            if src.exists():
                with open(src, 'r') as f:
                    stems = [l.strip() for l in f if l.strip()]

                lines = [str(img_out / f"{s}.jpg") for s in stems]
                with open(out_path / split_name, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
        print(f"Splits written to {out_path}")

    print(f"Successfully prepared 6-channel dataset at {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_dir', type=str, default=r"D:\Yolo\Data\project_flir\thermal-yolo\data\thermal")
    parser.add_argument('--rgb_dir', type=str, default=r"D:\Yolo\Data\project_flir\100_FLIR")
    parser.add_argument('--out_dir', type=str, default=r"D:\Yolo\Data\project_flir\rgb-thermal-fusion\data\fusion_dual")
    parser.add_argument('--splits_dir', type=str, default=r"D:\Yolo\Data\project_flir\thermal-yolo\data\splits")
    args = parser.parse_args()
    prepare_dual_dataset(args.thermal_dir, args.rgb_dir, args.out_dir, args.splits_dir)
