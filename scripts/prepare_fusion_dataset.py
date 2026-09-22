"""
scripts/prepare_fusion_dataset.py
=================================
Prepares 4-channel fusion dataset:
- Combines Thermal FLIR{N}.jpg (grayscale) + Visual RGB FLIR{N+1}.jpg
- Produces 4-channel BGRA PNGs [R, G, B, T_gray]
- Copies FLIR annotation text files and splits
"""

import os
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
import re
import argparse

def prepare_dataset(thermal_dir: str, rgb_dir: str, out_dir: str, splits_dir: str = None):
    thermal_path = Path(thermal_dir)
    rgb_path = Path(rgb_dir)
    out_path = Path(out_dir)

    img_out = out_path / 'images'
    lbl_out = out_path / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    thermal_images = list((thermal_path / 'images').glob('*.jpg'))
    print(f"Found {len(thermal_images)} thermal images in {thermal_path / 'images'}")

    for t_path in tqdm(thermal_images, desc="Generating 4-channel PNGs"):
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

        t_img = cv2.imread(str(t_path))
        r_img = cv2.imread(str(r_file))

        if t_img is None or r_img is None:
            continue

        if t_img.shape[:2] != r_img.shape[:2]:
            r_img = cv2.resize(r_img, (t_img.shape[1], t_img.shape[0]))

        t_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(r_img)

        fusion_img = cv2.merge([b, g, r, t_gray])
        cv2.imwrite(str(img_out / f"{stem}.png"), fusion_img)

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
                lines = [str(img_out / f"{s}.png") for s in stems]
                with open(out_path / split_name, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
        print(f"Splits written to {out_path}")

    print(f"Prepared 4-channel fusion dataset at {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_dir', type=str, default=r"D:\Yolo\Data\project_flir\thermal-yolo\data\thermal")
    parser.add_argument('--rgb_dir', type=str, default=r"D:\Yolo\Data\project_flir\100_FLIR")
    parser.add_argument('--out_dir', type=str, default=r"D:\Yolo\Data\project_flir\rgb-thermal-fusion\data\fusion")
    parser.add_argument('--splits_dir', type=str, default=r"D:\Yolo\Data\project_flir\thermal-yolo\data\splits")
    args = parser.parse_args()
    prepare_dataset(args.thermal_dir, args.rgb_dir, args.out_dir, args.splits_dir)
