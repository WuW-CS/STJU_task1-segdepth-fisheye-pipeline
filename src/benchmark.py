"""
benchmark.py
Runs segmentation and depth benchmarks separately on a set of images.
Produces a CSV results table + visualization figures.

Usage (once dataset is ready):
    python src/benchmark.py --task segmentation --data_dir /path/to/stanford/area_3
    python src/benchmark.py --task depth        --data_dir /path/to/stanford/area_3
"""

import argparse
import time
import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Local modules
import sys
sys.path.append(str(Path(__file__).parent))
from metrics import SegmentationMetrics, DepthMetrics
from models.segmentation import get_segmentation_model
from models.depth import get_depth_model
from metrics import SegmentationMetrics, DepthMetrics, InferenceTimer


# 
# CONSTANTS
# 

# Stanford 2D-3D-S has 13 semantic classes (+ 1 unknown = index 0)
STANFORD_NUM_CLASSES = 14
STANFORD_IGNORE_INDEX = 0   # <UNK> class

# Stanford class names (index 0 = <UNK>, ignored in metrics)
STANFORD_CLASSES = [
    '<UNK>', 'beam', 'board', 'bookcase', 'ceiling',
    'chair', 'clutter', 'column', 'door', 'floor',
    'sofa', 'table', 'wall', 'window'
]

# ADE20K (150 classes) → Stanford 13 classes
# -1 means "not supported" → will be mapped to IGNORE_INDEX
ADE20K_TO_STANFORD = {
    0  : 12,  # wall       → wall
    2  : 4,   # ceiling    → ceiling
    3  : 9,   # floor      → floor
    5  : 13,  # window     → window
    6  : 11,  # table      → table
    7  : 5,   # chair      → chair
    14 : 8,   # door       → door
    15 : 10,  # sofa       → sofa
    16 : 3,   # bookcase   → bookcase
    19 : 2,   # board      → board
    33 : 7,   # column     → column
    34 : 1,   # beam       → beam
    # All others → IGNORE_INDEX (unsupported)
}

# Segmentation models to benchmark
SEG_MODELS = ['segformer_b2', 'segformer_b4', 'mask2former']

# Depth models to benchmark
DEPTH_MODELS = ['da2_small', 'da2_base', 'da2_large']


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_stanford_sample(rgb_path: Path, gt_path: Path, task: str):
    """
    Load one Stanford 2D-3D-S sample.

    For segmentation: gt_path points to a semantic PNG
    For depth:        gt_path points to a depth EXR or PNG

    Returns:
        image_bgr : np.ndarray (H, W, 3)
        gt        : np.ndarray (H, W)
    """
    image_bgr = cv2.imread(str(rgb_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {rgb_path}")

    if task == 'segmentation':
        # Stanford semantic PNGs encode class index as RGB base-256 integer
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        # Decode: index = R + G*256 + B*256^2
        gt = (gt_img[:, :, 2].astype(np.int32)
              + gt_img[:, :, 1].astype(np.int32) * 256
              + gt_img[:, :, 0].astype(np.int32) * 256**2)
        gt = gt.astype(np.int32)

    elif task == 'depth':
        # Stanford depth stored as 16-bit PNG (millimetres) or EXR
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_ANYDEPTH)
        gt = gt_img.astype(np.float32) / 1000.0  # mm → metres

    return image_bgr, gt


def map_ade20k_to_stanford(pred: np.ndarray) -> np.ndarray:
    """
    Remap ADE20K predictions (0-149) to Stanford class indices (0-13).
    Unsupported classes → STANFORD_IGNORE_INDEX (0 = <UNK>).

    Args:
        pred : np.ndarray (H, W) — ADE20K predictions

    Returns:
        remapped : np.ndarray (H, W) — Stanford class indices
    """
    remapped = np.full_like(pred, fill_value=STANFORD_IGNORE_INDEX)
    for ade_idx, stanford_idx in ADE20K_TO_STANFORD.items():
        remapped[pred == ade_idx] = stanford_idx
    return remapped


def collect_samples(data_dir: Path, task: str, n_samples: int = 50):
    """
    Collect up to n_samples equirectangular image pairs from Stanford area.

    Expects this structure inside data_dir:
        data/
        ├── rgb/        ← *_domain_equirectangular.png
        ├── semantic/   ← *_domain_semantic.png
        └── depth/      ← *_domain_depth.png

    Returns:
        samples : list of (rgb_path, gt_path) tuples
    """
    rgb_dir = data_dir / 'data' / 'rgb'
    if task == 'segmentation':
        gt_dir  = data_dir / 'data' / 'semantic'
        gt_suffix = '_domain_semantic.png'
    else:
        gt_dir  = data_dir / 'data' / 'depth'
        gt_suffix = '_domain_depth.png'

    rgb_files = sorted(rgb_dir.glob('*_domain_equirectangular.png'))[:n_samples]

    samples = []
    for rgb_path in rgb_files:
        # Build corresponding GT path from RGB filename
        stem = rgb_path.stem.replace('_domain_equirectangular', '')
        gt_path = gt_dir / (stem + gt_suffix)
        if gt_path.exists():
            samples.append((rgb_path, gt_path))

    print(f"Found {len(samples)} valid samples for {task} benchmark.")
    return samples

# 
# BENCHMARK FUNCTIONS
# 

def run_segmentation_benchmark(data_dir: Path, output_dir: Path,
                                n_samples: int = 50):
    """
    Benchmark all segmentation models on n_samples from data_dir.
    Saves results to CSV and a comparison figure.
    """
    print("\n" + "="*60)
    print("  SEGMENTATION BENCHMARK")
    print("="*60)

    samples = collect_samples(data_dir, 'segmentation', n_samples)
    results = []

    for model_name in SEG_MODELS:
        print(f"\n--- {model_name} ---")
        model = get_segmentation_model(model_name)

        # Measure inference time on first image
        img_bgr, _ = load_stanford_sample(samples[0][0], samples[0][1],
                                           'segmentation')
        timing = InferenceTimer.measure(model.predict, img_bgr)
        mean_ms = timing['means_ms']

        # Accumulate metrics over all samples
        all_miou, all_acc, all_f1 = [], [], []

        for rgb_path, gt_path in samples:
            img_bgr, gt_seg = load_stanford_sample(rgb_path, gt_path,
                                                    'segmentation')
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Predict
            pred_ade = model.predict(img_rgb)

            # Remap ADE20K → Stanford
            pred_stanford = map_ade20k_to_stanford(pred_ade)

            # Resize pred to GT size if needed
            if pred_stanford.shape != gt_seg.shape:
                pred_stanford = cv2.resize(
                    pred_stanford.astype(np.float32),
                    (gt_seg.shape[1], gt_seg.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            # Compute metrics
            m = SegmentationMetrics.compute_all_metrics(
                gt_seg, pred_stanford,
                num_classes=STANFORD_NUM_CLASSES,
                ignore_index=STANFORD_IGNORE_INDEX
            )
            all_miou.append(m['mean_iou'])
            all_acc.append(m['pixel_accuracy'])

        row = {
            'model'         : model_name,
            'mIoU'          : round(float(np.mean(all_miou)), 4),
            'pixel_acc'     : round(float(np.mean(all_acc)),  4),
            'inference_ms'  : round(mean_ms, 1),
            'n_samples'     : len(samples),
            'label_mapping' : 'ADE20K→Stanford (unsupported→ignore)',
        }
        results.append(row)
        print(f"  mIoU={row['mIoU']:.4f} | "
              f"Acc={row['pixel_acc']:.4f} | "
              f"Time={row['inference_ms']}ms")

        # Free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV
    csv_path = output_dir / 'segmentation_benchmark.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Results saved to {csv_path}")

    # Plot
    _plot_segmentation_results(results, output_dir)
    return results


def run_depth_benchmark(data_dir: Path, output_dir: Path,
                        checkpoint_dir: Path, n_samples: int = 50):
    """
    Benchmark all depth models on n_samples from data_dir.
    Saves results to CSV and a comparison figure.
    """
    print("\n" + "="*60)
    print("  DEPTH BENCHMARK")
    print("="*60)

    samples = collect_samples(data_dir, 'depth', n_samples)
    results = []

    for model_name in DEPTH_MODELS:
        print(f"\n--- {model_name} ---")
        model = get_depth_model(model_name, checkpoint_dir=str(checkpoint_dir))

        # Inference time
        img_bgr, _ = load_stanford_sample(samples[0][0], samples[0][1], 'depth')
        timing = InferenceTimer.measure(model.predict,img_bgr)
        mean_ms = timing['mean_ms']

        # Accumulate metrics
        all_absrel, all_rmse, all_d1 = [], [], []

        for rgb_path, gt_path in samples:
            img_bgr, gt_depth = load_stanford_sample(rgb_path, gt_path, 'depth')

            pred_depth = model.predict(img_bgr)

            # Resize pred to GT size if needed
            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(
                    pred_depth,
                    (gt_depth.shape[1], gt_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            m = DepthMetrics.compute_all_metrics(gt_depth, pred_depth)
            all_absrel.append(m['abs_rel'])
            all_rmse.append(m['rmse'])
            all_d1.append(m['δ1'])

        row = {
            'model'        : model_name,
            'AbsRel'       : round(float(np.nanmean(all_absrel)), 4),
            'RMSE'         : round(float(np.nanmean(all_rmse)),   4),
            'delta_1'      : round(float(np.nanmean(all_d1)),     4),
            'inference_mean_ms'  : timing['mean_ms'],
            'inference_std_ms'   : timing['std_ms'],
            'inference_min_ms'   : timing['min_ms'],
            'n_samples'    : len(samples),
        }
        results.append(row)
        print(f"  AbsRel={row['AbsRel']:.4f} | "
              f"RMSE={row['RMSE']:.4f} | "
              f"δ1={row['delta_1']:.4f} | "
              f"Time={row['inference_ms']}ms")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV
    csv_path = output_dir / 'depth_benchmark.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Results saved to {csv_path}")

    _plot_depth_results(results, output_dir)
    return results


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def _plot_segmentation_results(results: list, output_dir: Path):
    """Bar chart comparing mIoU and pixel accuracy across models."""
    models   = [r['model'] for r in results]
    miou     = [r['mIoU'] for r in results]
    acc      = [r['pixel_acc'] for r in results]
    times    = [r['inference_ms'] for r in results]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(x, miou, color='steelblue')
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_title('mIoU (higher is better)')
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, acc, color='darkorange')
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_title('Pixel Accuracy (higher is better)')
    axes[1].set_ylim(0, 1)

    axes[2].bar(x, times, color='green')
    axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=15)
    axes[2].set_title('Inference time ms (lower is better)')

    plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_benchmark.png', dpi=150)
    plt.close()
    print(f"✅ Figure saved to {output_dir / 'segmentation_benchmark.png'}")


def _plot_depth_results(results: list, output_dir: Path):
    """Bar chart comparing depth metrics across models."""
    models  = [r['model'] for r in results]
    absrel  = [r['AbsRel'] for r in results]
    rmse    = [r['RMSE'] for r in results]
    d1      = [r['delta_1'] for r in results]
    times   = [r['inference_ms'] for r in results]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].bar(x, absrel, color='crimson')
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_title('AbsRel (lower is better)')

    axes[1].bar(x, rmse, color='purple')
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_title('RMSE in metres (lower is better)')

    axes[2].bar(x, d1, color='teal')
    axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=15)
    axes[2].set_title('δ<1.25 (higher is better)')
    axes[2].set_ylim(0, 1)

    axes[3].bar(x, times, color='green')
    axes[3].set_xticks(x); axes[3].set_xticklabels(models, rotation=15)
    axes[3].set_title('Inference time ms (lower is better)')

    plt.tight_layout()
    plt.savefig(output_dir / 'depth_benchmark.png', dpi=150)
    plt.close()
    print(f"✅ Figure saved to {output_dir / 'depth_benchmark.png'}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run segmentation or depth benchmark')
    parser.add_argument('--task',           type=str, required=True,
                        choices=['segmentation', 'depth'],
                        help='Which benchmark to run')
    parser.add_argument('--data_dir',       type=str, required=True,
                        help='Path to Stanford 2D-3D-S area folder')
    parser.add_argument('--output_dir',     type=str, default='./outputs',
                        help='Where to save results')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./Depth-Anything-V2/checkpoints',
                        help='Path to depth model checkpoints (depth task only)')
    parser.add_argument('--n_samples',      type=int, default=50,
                        help='Number of images to benchmark')
    args = parser.parse_args()

    data_dir       = Path(args.data_dir)
    output_dir     = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == 'segmentation':
        run_segmentation_benchmark(data_dir, output_dir, args.n_samples)
    else:
        run_depth_benchmark(data_dir, output_dir, checkpoint_dir, args.n_samples)