#!/usr/bin/env python3
"""
visualize_stanford_depth.py
Verify Stanford depth GT format: load, inspect stats, and save a pretty figure.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def load_stanford_depth(path: Path):
    """Load Stanford2D3D depth PNG (16-bit) and convert to metres."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    # Mark invalid pixels
    invalid = arr >= 65535
    arr[invalid] = 0
    # Scale: values are in millimetres (confirmed by dataset docs)
    depth_m = arr / 1000.0
    return depth_m, invalid


def main():
    depth_dir = Path('/home/william/data/stanford/area_1/pano/depth')
    depth_files = sorted(depth_dir.glob('*.png'))
    if not depth_files:
        raise ValueError(f"No depth files in {depth_dir}")

    # Pick first 3 samples
    for i, dp in enumerate(depth_files[:3]):
        depth, invalid = load_stanford_depth(dp)
        rgb_path = dp.parent.parent / 'rgb' / dp.name.replace('domain_depth', 'domain_rgb')
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (2048, 1024))

        # Normalize depth for visualization (percentile to avoid outliers)
        d_valid = depth[depth > 0]
        d_min = np.percentile(d_valid, 1)
        d_max = np.percentile(d_valid, 99)
        
        vis = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
        vis = (vis * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
        vis_color = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)

        # Figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        if rgb is not None:
            axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis('off')

        axes[1].imshow(vis_color)
        axes[1].set_title(f"Depth GT (1-99%ile)\nmin={depth.min():.2f}, max={depth.max():.2f}, mean={d_valid.mean():.2f}")
        axes[1].axis('off')

        # Invalid mask
        axes[2].imshow(invalid, cmap='gray')
        axes[2].set_title(f"Invalid mask\n{invalid.mean()*100:.1f}% invalid")
        axes[2].axis('off')

        plt.tight_layout()
        out = f'/home/william/stanford_depth_check_{i}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"✅ Saved: {out}")

        # Print detailed stats
        print(f"\n  File: {dp.name}")
        print(f"  Shape: {depth.shape}")
        print(f"  Valid pixels: {np.sum(depth > 0)} / {depth.size}")
        print(f"  Depth range (valid): {d_valid.min():.3f}m – {d_valid.max():.3f}m")
        print(f"  Mean / Median: {d_valid.mean():.3f}m / {np.median(d_valid):.3f}m")
        print(f"  Histogram (metres):")
        hist, bins = np.histogram(d_valid, bins=[0, 1, 2, 3, 5, 10, 20, 50])
        for j in range(len(hist)):
            print(f"    {bins[j]:.1f}–{bins[j+1]:.1f}m: {hist[j]} px ({100*hist[j]/len(d_valid):.1f}%)")


if __name__ == '__main__':
    main()