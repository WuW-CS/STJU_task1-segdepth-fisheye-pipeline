"""
diagnose_depth.py
Debug script: compare GT depth vs prediction on a single Stanford sample.
Run this AFTER fixing INTER_NEAREST to verify the real problem.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from models.depth import get_depth_model
from benchmark import load_stanford_sample


def analyze_sample(rgb_path, depth_path, model_name='da2_small', checkpoint_dir=None):
    img_bgr, gt = load_stanford_sample(rgb_path, depth_path, 'depth')

    # Load model
    if model_name.startswith('da2'):
        model = get_depth_model(model_name, checkpoint_dir=checkpoint_dir)
    else:
        model = get_depth_model(model_name)

    pred = model.predict(img_bgr)

    # Valid mask
    valid = gt > 0
    gt_valid = gt[valid]
    pred_valid = pred[valid]

    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Image: {rgb_path.name}")
    print("=" * 60)

    print(f"\n  GT shape: {gt.shape}, PRED shape: {pred.shape}")
    print(f"  Valid pixels: {valid.sum()} / {gt.size} ({100*valid.mean():.1f}%)")

    print(f"\n  GT  (valid)  min={gt_valid.min():.3f} max={gt_valid.max():.3f} mean={gt_valid.mean():.3f} median={np.median(gt_valid):.3f}")
    print(f"  PRED (valid) min={pred_valid.min():.3f} max={pred_valid.max():.3f} mean={pred_valid.mean():.3f} median={np.median(pred_valid):.3f}")

    # Scale hypothesis: what if pred is off by a constant factor?
    ratios = gt_valid / (pred_valid + 1e-8)
    print(f"\n  GT/PRED ratio: min={ratios.min():.3f} max={ratios.max():.3f} mean={ratios.mean():.3f} median={np.median(ratios):.3f}")

    # Optimal scale factor (minimizes MSE)
    scale = np.sum(gt_valid * pred_valid) / np.sum(pred_valid ** 2)
    pred_scaled = pred_valid * scale
    abs_rel_scaled = np.mean(np.abs(gt_valid - pred_scaled) / gt_valid)
    print(f"  Optimal scale factor: {scale:.4f}")
    print(f"  AbsRel AFTER optimal scaling: {abs_rel_scaled:.4f}")

    # Raw AbsRel
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    print(f"  AbsRel BEFORE scaling: {abs_rel:.4f}")

    # Error distribution
    rel_err = np.abs(gt_valid - pred_valid) / gt_valid
    print(f"\n  Relative error distribution:")
    for p in [50, 75, 90, 95, 99]:
        print(f"    {p}th percentile: {np.percentile(rel_err, p):.3f}")

    # Correlation
    corr = np.corrcoef(gt_valid, pred_valid)[0, 1]
    print(f"\n  Pearson correlation (GT, PRED): {corr:.4f}")

    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis('off')

    # GT depth (percentile normalized)
    d_min, d_max = np.percentile(gt_valid, 1), np.percentile(gt_valid, 99)
    gt_vis = np.clip((gt - d_min) / (d_max - d_min + 1e-8), 0, 1)
    axes[0, 1].imshow(gt_vis, cmap='plasma')
    axes[0, 1].set_title(f"GT Depth\nmean={gt_valid.mean():.2f}m")
    axes[0, 1].axis('off')

    # Pred depth (same normalization)
    p_min, p_max = np.percentile(pred_valid, 1), np.percentile(pred_valid, 99)
    pred_vis = np.clip((pred - p_min) / (p_max - p_min + 1e-8), 0, 1)
    axes[0, 2].imshow(pred_vis, cmap='plasma')
    axes[0, 2].set_title(f"PRED Depth ({model_name})\nmean={pred_valid.mean():.2f}m")
    axes[0, 2].axis('off')

    # Scatter plot GT vs PRED
    sample_idx = np.random.choice(len(gt_valid), min(5000, len(gt_valid)), replace=False)
    axes[1, 0].scatter(gt_valid[sample_idx], pred_valid[sample_idx], alpha=0.1, s=1)
    axes[1, 0].plot([0, gt_valid.max()], [0, gt_valid.max()], 'r--', label='y=x')
    axes[1, 0].set_xlabel("GT Depth (m)")
    axes[1, 0].set_ylabel("PRED Depth (m)")
    axes[1, 0].set_title(f"GT vs PRED\ncorr={corr:.3f}")
    axes[1, 0].legend()

    # Relative error map
    err_map = np.zeros_like(gt)
    err_map[valid] = rel_err
    err_vis = np.clip(err_map / np.percentile(rel_err, 95), 0, 1)
    axes[1, 1].imshow(err_vis, cmap='hot')
    axes[1, 1].set_title(f"Relative Error Map\nmean={rel_err.mean():.3f}")
    axes[1, 1].axis('off')

    # Histogram of GT/PRED ratio
    axes[1, 2].hist(ratios, bins=50, range=(0, 5), alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(ratios.mean(), color='r', linestyle='--', label=f'mean={ratios.mean():.2f}')
    axes[1, 2].set_xlabel("GT / PRED Ratio")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Scale Factor Distribution")
    axes[1, 2].legend()

    plt.tight_layout()
    out = f'/home/william/diagnose_depth_{model_name}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {out}")


def main():
    data_dir = Path('/home/william/data/stanford/area_1')
    rgb_files = sorted((data_dir / 'pano' / 'rgb').glob('*_domain_rgb.png'))
    depth_files = sorted((data_dir / 'pano' / 'depth').glob('*_domain_depth.png'))

    if not rgb_files or not depth_files:
        raise ValueError("No files found")

    # Test on first sample
    #analyze_sample(rgb_files[0], depth_files[0], 'da2_small',
    #              checkpoint_dir='/home/william/Depth-Anything-V2/checkpoints')

    # Test UniFuse on same sample
    for i in range(1,4):
        analyze_sample(rgb_files[i*5], depth_files[i*5], 'unifuse')
    
    


if __name__ == '__main__':
    main()