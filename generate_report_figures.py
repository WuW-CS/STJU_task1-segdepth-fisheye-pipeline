#!/usr/bin/env python3
"""
generate_report_figures.py
Create side-by-side figures for the LaTeX report.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent))
from models.depth import get_depth_model
from models.segmentation import get_segmentation_model
from benchmark import load_stanford_sample, map_ade20k_to_stanford

# Stanford colormap (RGB)
STANFORD_COLORS = np.array([
    [0, 0, 0], [128, 128, 128], [139, 69, 19], [0, 128, 0],
    [0, 255, 255], [255, 0, 0], [255, 165, 0], [255, 255, 0],
    [0, 0, 255], [128, 0, 128], [255, 192, 203], [0, 128, 128],
    [128, 128, 0], [0, 255, 0],
], dtype=np.uint8)

STANFORD_NAMES = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair',
                  'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']


def colorize_depth(depth):
    d_min, d_max = depth.min(), depth.max()
    norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def colorize_seg(seg):
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(STANFORD_COLORS):
        out[seg == i] = c
    return out


def add_legend(ax):
    patches = [mpatches.Patch(color=c / 255.0, label=f"{i}: {n}")
               for i, (c, n) in enumerate(zip(STANFORD_COLORS, STANFORD_NAMES))]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=5, frameon=False)


def main():
    data_dir = Path('/home/william/data/stanford/area_1')
    rgb_files = sorted((data_dir / 'pano' / 'rgb').glob('*_domain_rgb.png'))
    depth_files = sorted((data_dir / 'pano' / 'depth').glob('*_domain_depth.png'))
    seg_files = sorted((data_dir / 'pano' / 'semantic').glob('*_domain_semantic.png'))

    device = torch.device('cuda')

    # Load models once
    print("Loading models...")
    da2 = get_depth_model('da2_small', checkpoint_dir='/home/william/Depth-Anything-V2/checkpoints')
    unifuse = get_depth_model('unifuse')
    segformer = get_segmentation_model('segformer_b4', device)
    mask2former = get_segmentation_model('mask2former', device)

    out_dir = Path('/home/william/report_figures')
    out_dir.mkdir(exist_ok=True)

    # Process 3 representative samples
    for idx in [0, 5, 10]:
        rgb_path = rgb_files[idx]
        depth_path = depth_files[idx]
        seg_path = seg_files[idx]

        # Load GT
        img_bgr, gt_depth = load_stanford_sample(rgb_path, depth_path, 'depth')
        _, gt_seg = load_stanford_sample(rgb_path, seg_path, 'segmentation')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Predictions
        pred_da2 = da2.predict(img_bgr)
        pred_unifuse = unifuse.predict(img_bgr)
        pred_seg = map_ade20k_to_stanford(segformer.predict(img_rgb))
        pred_mask2 = map_ade20k_to_stanford(mask2former.predict(img_rgb))

        # ── Figure 1: Depth ──
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title("RGB Input")
        axes[0].axis('off')

        axes[1].imshow(colorize_depth(gt_depth))
        axes[1].set_title(f"GT Depth\nmean={gt_depth[gt_depth>0].mean():.2f}m")
        axes[1].axis('off')

        axes[2].imshow(colorize_depth(pred_da2))
        axes[2].set_title(f"DA2-Small\nmean={pred_da2.mean():.2f}m")
        axes[2].axis('off')

        axes[3].imshow(colorize_depth(pred_unifuse))
        axes[3].set_title(f"UniFuse\nmean={pred_unifuse.mean():.2f}m")
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(out_dir / f"depth_sample{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved depth_sample{idx}.png")

        # ── Figure 2: Segmentation ──
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title("RGB Input")
        axes[0].axis('off')

        axes[1].imshow(colorize_seg(gt_seg))
        axes[1].set_title("GT Segmentation")
        axes[1].axis('off')

        axes[2].imshow(colorize_seg(pred_seg))
        axes[2].set_title("SegFormer-B4")
        axes[2].axis('off')
        add_legend(axes[2])

        axes[3].imshow(colorize_seg(pred_mask2))
        axes[3].set_title("Mask2Former")
        axes[3].axis('off')
        add_legend(axes[3])

        plt.tight_layout()
        plt.savefig(out_dir / f"seg_sample{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved seg_sample{idx}.png")

    print(f"\n🎉 All report figures saved to {out_dir}")


if __name__ == '__main__':
    main()