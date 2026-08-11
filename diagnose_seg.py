#!/usr/bin/env python3
"""
diagnose_seg.py
Verify ADE20K→Stanford mapping by comparing GT and Pred side-by-side.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent))
from models.segmentation import get_segmentation_model
from benchmark import load_stanford_sample, map_ade20k_to_stanford, STANFORD_CLASSES

# Stanford colors (RGB)
STANFORD_COLORS = np.array([
    [0, 0, 0],        # 0  <UNK>
    [128, 128, 128],  # 1  beam
    [139, 69, 19],    # 2  board
    [0, 128, 0],      # 3  bookcase
    [0, 255, 255],    # 4  ceiling
    [255, 0, 0],      # 5  chair
    [255, 165, 0],    # 6  clutter
    [255, 255, 0],    # 7  column
    [0, 0, 255],      # 8  door
    [128, 0, 128],    # 9  floor
    [255, 192, 203],  # 10 sofa
    [0, 128, 128],    # 11 table
    [128, 128, 0],    # 12 wall
    [0, 255, 0],      # 13 window
], dtype=np.uint8)

def colorize(seg):
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(STANFORD_COLORS):
        out[seg == i] = c
    return out

# Load one sample
data_dir = Path('/home/william/data/stanford/area_1')
rgb_files = sorted((data_dir / 'pano' / 'rgb').glob('*_domain_rgb.png'))
gt_files = sorted((data_dir / 'pano' / 'semantic').glob('*_domain_semantic.png'))

rgb_path = rgb_files[0]
gt_path = gt_files[0]

img_bgr, gt_seg = load_stanford_sample(rgb_path, gt_path, 'segmentation')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Predict
model = get_segmentation_model('segformer_b4')
pred_ade = model.predict(img_rgb)
pred_stanford = map_ade20k_to_stanford(pred_ade)

# Class distributions
print("\n=== GT Class Distribution ===")
for i, name in enumerate(STANFORD_CLASSES):
    pct = 100 * np.mean(gt_seg == i)
    if pct > 1:
        print(f"  {i:2d} {name:12s}: {pct:5.1f}%")

print("\n=== PRED Class Distribution ===")
for i, name in enumerate(STANFORD_CLASSES):
    pct = 100 * np.mean(pred_stanford == i)
    if pct > 1:
        print(f"  {i:2d} {name:12s}: {pct:5.1f}%")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img_rgb)
axes[0].set_title("RGB Input")
axes[0].axis('off')

axes[1].imshow(colorize(gt_seg))
axes[1].set_title("Ground Truth (Stanford)")
axes[1].axis('off')

axes[2].imshow(colorize(pred_stanford))
axes[2].set_title("Prediction (ADE20K→Stanford)")
axes[2].axis('off')

# Legend
patches = [mpatches.Patch(color=c/255.0, label=f"{i}: {n}")
           for i, (c, n) in enumerate(zip(STANFORD_COLORS, STANFORD_CLASSES))]
fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=8)

plt.tight_layout()
plt.savefig('/home/william/seg_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: /home/william/seg_diagnosis.png")