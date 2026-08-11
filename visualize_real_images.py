#!/usr/bin/env python3
"""
visualize_real_images.py
Generate qualitative visualizations for real-world images.
FIX: ADE20K predictions are mapped to Stanford labels before colorization.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent))
from models.depth import get_depth_model
from models.segmentation import get_segmentation_model

# ── Stanford colormap (RGB) ──
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

STANFORD_NAMES = [
    '<UNK>', 'beam', 'board', 'bookcase', 'ceiling',
    'chair', 'clutter', 'column', 'door', 'floor',
    'sofa', 'table', 'wall', 'window'
]

# ── ADE20K → Stanford mapping (same as benchmark.py) ──
ADE20K_TO_STANFORD = {
    0: 12,   # wall       → wall
    3: 9,    # floor      → floor
    5: 4,    # ceiling    → ceiling
    8: 13,   # windowpane → window
    14: 8,   # door       → door
    15: 11,  # table      → table
    19: 5,   # chair      → chair
    23: 10,  # sofa       → sofa
    24: 3,   # shelf      → bookcase
    42: 7,   # column     → column
    62: 3,   # bookcase   → bookcase
}


def map_ade20k_to_stanford(pred_ade: np.ndarray) -> np.ndarray:
    """Map ADE20K class indices to Stanford label space."""
    pred_stanford = np.zeros_like(pred_ade, dtype=np.uint8)
    for ade_id, stanford_id in ADE20K_TO_STANFORD.items():
        pred_stanford[pred_ade == ade_id] = stanford_id
    return pred_stanford


def colorize_depth(depth: np.ndarray, cmap=cv2.COLORMAP_PLASMA) -> np.ndarray:
    """Normalize depth to [0,255] and apply colormap."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        d_max = d_min + 1.0
    norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cmap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """Map Stanford class indices to RGB colors."""
    h, w = seg.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(STANFORD_COLORS)):
        color[seg == cls] = STANFORD_COLORS[cls]
    return color


def add_legend(ax, names=STANFORD_NAMES, colors=STANFORD_COLORS):
    """Add a compact legend to the right of the axis."""
    # Only show classes that actually appear in the image
    patches = []
    for i, (c, n) in enumerate(zip(colors, names)):
        patches.append(mpatches.Patch(color=c / 255.0, label=f"{i}: {n}"))
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=6, frameon=False)


def process_one_image(rgb_path: Path, out_dir: Path, device):
    """Run all models on one image and save a multi-panel figure."""
    img_bgr = cv2.imread(str(rgb_path))
    if img_bgr is None:
        print(f"  [SKIP] Cannot read {rgb_path}")
        return

    # Resize to standard panorama size
    img_bgr = cv2.resize(img_bgr, (2048, 1024))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Depth models ──
    depth_models = {
        'DA2-Small': get_depth_model('da2_small', checkpoint_dir='/home/william/Depth-Anything-V2/checkpoints'),
        'DA2-Base':  get_depth_model('da2_base',  checkpoint_dir='/home/william/Depth-Anything-V2/checkpoints'),
        'DA2-Large': get_depth_model('da2_large', checkpoint_dir='/home/william/Depth-Anything-V2/checkpoints'),
        'UniFuse':   get_depth_model('unifuse'),
    }

    depth_preds = {}
    for name, model in depth_models.items():
        print(f"    Depth: {name}...")
        pred = model.predict(img_bgr)
        depth_preds[name] = pred
        del model
        torch.cuda.empty_cache()

    # ── Segmentation models ──
    seg_models = {
        'SegFormer-B4': get_segmentation_model('segformer_b4', device),
        'Mask2Former':  get_segmentation_model('mask2former', device),
    }

    seg_preds = {}
    for name, model in seg_models.items():
        print(f"    Seg: {name}...")
        pred_ade = model.predict(img_rgb)           # ADE20K indices
        pred_stanford = map_ade20k_to_stanford(pred_ade)  # ← FIX: map to Stanford
        seg_preds[name] = pred_stanford
        del model
        torch.cuda.empty_cache()

    # ── Plot ──
    n_depth = len(depth_preds)
    n_seg = len(seg_preds)
    n_rows = 1 + n_depth + n_seg
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows))

    if n_rows == 1:
        axes = [axes]

    # Row 0: RGB
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Input: {rgb_path.name}", fontsize=10)
    axes[0].axis('off')

    # Rows 1..n_depth: Depth predictions
    for i, (name, pred) in enumerate(depth_preds.items(), start=1):
        axes[i].imshow(colorize_depth(pred))
        axes[i].set_title(
            f"Depth — {name}  (min={pred.min():.2f}, max={pred.max():.2f}, mean={pred.mean():.2f})",
            fontsize=9)
        axes[i].axis('off')

    # Rows after depth: Segmentation predictions (NOW CORRECTLY MAPPED)
    for i, (name, pred) in enumerate(seg_preds.items(), start=1 + n_depth):
        axes[i].imshow(colorize_segmentation(pred))
        axes[i].set_title(f"Segmentation — {name}", fontsize=9)
        axes[i].axis('off')
        add_legend(axes[i])

    plt.tight_layout()
    out_path = out_dir / f"{rgb_path.stem}_viz.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  type=str, default='/home/william/data/validation/')
    parser.add_argument('--output_dir', type=str, default='/home/william/outputs_real/')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob('*.png')) + sorted(input_dir.glob('*.jpg')) + sorted(input_dir.glob('*.jpeg'))
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Found {len(image_paths)} images. Device: {device}\n")

    for p in image_paths:
        print(f"Processing {p.name}...")
        process_one_image(p, output_dir, device)

    print(f"\n🎉 All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()