"""
Evaluation metrics for semantic segmentation and depth estimation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import cv2


class SegmentationMetrics:
    """
    Metrics for semantic segmentation evaluation.
    """
    
    @staticmethod
    def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Pixel accuracy: (correct pixels) / (total pixels).
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def mean_intersection_over_union(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        num_classes: int,
        ignore_index: int = 255
    ) -> Dict[str, float]:
        """
        Compute IoU per class and mIoU.
        
        Args:
            y_true: Ground truth (H, W) class indices.
            y_pred: Predicted (H, W) class indices.
            num_classes: Total number of classes.
            ignore_index: Class index to ignore (e.g., 255 for background).
            
        Returns:
            Dictionary with 'per_class_iou' and 'miou'.
        """
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove ignore_index pixels
        valid_mask = y_true_flat != ignore_index
        y_true_flat = y_true_flat[valid_mask]
        y_pred_flat = y_pred_flat[valid_mask]
        
        # Compute confusion matrix
        conf_mat = confusion_matrix(
            y_true_flat, 
            y_pred_flat,
            labels=list(range(num_classes))
        )
        
        # Compute IoU per class
        iou_per_class = {}
        for cls in range(num_classes):
            tp = conf_mat[cls, cls]
            fp = conf_mat[:, cls].sum() - tp
            fn = conf_mat[cls, :].sum() - tp
            
            if (tp + fp + fn) > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = float('nan')  # Class not present
            iou_per_class[cls] = iou
        
        # Mean IoU (ignoring NaN)
        valid_ious = [iou for iou in iou_per_class.values() if not np.isnan(iou)]
        miou = np.mean(valid_ious) if valid_ious else 0.0
        
        return {
            'per_class_iou': iou_per_class,
            'miou': miou
        }
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
        ignore_index: int = 255
    ) -> Dict[str, float]:
        """
        Compute all segmentation metrics.
        """
        acc = SegmentationMetrics.pixel_accuracy(y_true, y_pred)
        iou_results = SegmentationMetrics.mean_intersection_over_union(
            y_true, y_pred, num_classes, ignore_index
        )
        
        return {
            'pixel_accuracy': acc,
            'mean_iou': iou_results['miou'],
            'per_class_iou': iou_results['per_class_iou']
        }
    @staticmethod
    def mean_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_index: int = 255) -> float:
        """
        Mean F1-score across classes (macro average, ignoring absent classes).
        F1_c = 2*TP / (2*TP + FP + FN)
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        valid = y_true_flat != ignore_index
        y_true_flat = y_true_flat[valid]
        y_pred_flat = y_pred_flat[valid]
        
        conf_mat = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(num_classes)))
        f1_scores = []
        
        for c in range(num_classes):
            tp = conf_mat[c, c]
            fp = conf_mat[:, c].sum() - tp
            fn = conf_mat[c, :].sum() - tp
            denom = 2*tp + fp + fn
            if denom > 0:
                f1_scores.append(2*tp / denom)
        
        return float(np.mean(f1_scores)) if f1_scores else 0.0


class DepthMetrics:
    """
    Metrics for monocular depth estimation.
    """
    
    @staticmethod
    def abs_relative_error(depth_true: np.ndarray, depth_pred: np.ndarray) -> float:
        """Absolute Relative Error: |d_true - d_pred| / d_true."""
        mask = depth_true > 0
        if not np.any(mask):
            return float('nan')
        return np.mean(np.abs(depth_true[mask] - depth_pred[mask]) / depth_true[mask])
    
    @staticmethod
    def rmse(depth_true: np.ndarray, depth_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        mask = depth_true > 0
        if not np.any(mask):
            return float('nan')
        diff = depth_true[mask] - depth_pred[mask]
        return np.sqrt(np.mean(diff ** 2))
    
    @staticmethod
    def rmse_log(depth_true: np.ndarray, depth_pred: np.ndarray) -> float:
        """RMSE in log space."""
        mask = depth_true > 0
        if not np.any(mask):
            return float('nan')
        log_diff = np.log(depth_true[mask] + 1e-8) - np.log(depth_pred[mask] + 1e-8)
        return np.sqrt(np.mean(log_diff ** 2))
    
    @staticmethod
    def threshold_accuracy(
        depth_true: np.ndarray,
        depth_pred: np.ndarray,
        threshold: float = 1.25
    ) -> Dict[str, float]:
        """
        Threshold accuracy: % of pixels where max(d_true/d_pred, d_pred/d_true) < threshold.
        
        Returns:
            Dictionary with accuracies for thresholds: δ1, δ2, δ3.
        """
        mask = (depth_true > 0) & (depth_pred>0)
        if not np.any(mask):
            return {'δ1': float('nan'), 'δ2': float('nan'), 'δ3': float('nan')}
        
        depth_true_masked = depth_true[mask]
        depth_pred_masked = depth_pred[mask]
        
        ratio = np.maximum(
            depth_true_masked / (depth_pred_masked + 1e-8),
            depth_pred_masked / (depth_true_masked + 1e-8)
        )
        
        return {
            'δ1': np.mean(ratio < threshold).item(),
            'δ2': np.mean(ratio < threshold ** 2).item(),
            'δ3': np.mean(ratio < threshold ** 3).item()
        }
    
    @staticmethod
    def compute_all_metrics(
        depth_true: np.ndarray,
        depth_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all depth metrics.
        """
        abs_rel = DepthMetrics.abs_relative_error(depth_true, depth_pred)
        rmse_val = DepthMetrics.rmse(depth_true, depth_pred)
        rmse_log_val = DepthMetrics.rmse_log(depth_true, depth_pred)
        threshold_vals = DepthMetrics.threshold_accuracy(depth_true, depth_pred)
        
        return {
            'abs_rel': abs_rel,
            'rmse': rmse_val,
            'rmse_log': rmse_log_val,
            'δ1': threshold_vals['δ1'],
            'δ2': threshold_vals['δ2'],
            'δ3': threshold_vals['δ3']
        }


def evaluate_pipeline(
    segmentation_true: np.ndarray,
    segmentation_pred: np.ndarray,
    depth_true: np.ndarray,
    depth_pred: np.ndarray,
    num_classes: int = 150,
    ignore_index: int = 255,
    max_depth: float = 20.0
) -> Dict[str, Dict[str, float]]:
    """
    Full evaluation of both segmentation and depth.
    
    Args:
        segmentation_true: Ground truth segmentation (H, W).
        segmentation_pred: Predicted segmentation (H, W).
        depth_true: Ground truth depth (H, W) in meters.
        depth_pred: Predicted depth (H, W) in meters.
        num_classes: Number of segmentation classes.
        ignore_index: Class index to ignore.
        max_depth: Maximum depth for clipping.
    
    Returns:
        Dictionary with 'segmentation' and 'depth' metrics.
    """
    # Clip depth
    depth_true_clipped = np.clip(depth_true, 0, max_depth)
    depth_pred_clipped = np.clip(depth_pred, 0, max_depth)
    
    # Segmentation metrics
    seg_metrics = SegmentationMetrics.compute_all_metrics(
        segmentation_true,
        segmentation_pred,
        num_classes=num_classes,
        ignore_index=ignore_index
    )
    
    # Depth metrics
    depth_metrics = DepthMetrics.compute_all_metrics(
        depth_true_clipped,
        depth_pred_clipped
    )
    
    return {
        'segmentation': seg_metrics,
        'depth': depth_metrics
    }


def print_metrics(metrics: Dict[str, Dict[str, float]], title: str = "Evaluation Results"):
    """
    Pretty print metrics.
    """
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
    print("\n--- Segmentation ---")
    print(f"  Pixel Accuracy:  {metrics['segmentation']['pixel_accuracy']:.4f}")
    print(f"  mIoU:            {metrics['segmentation']['mean_iou']:.4f}")
    
    print("\n--- Depth ---")
    print(f"  Abs Relative Error:  {metrics['depth']['abs_rel']:.4f}")
    print(f"  RMSE:                {metrics['depth']['rmse']:.4f}")
    print(f"  RMSE (log):          {metrics['depth']['rmse_log']:.4f}")
    print(f"  δ1:                  {metrics['depth']['δ1']:.4f}")
    print(f"  δ2:                  {metrics['depth']['δ2']:.4f}")
    print(f"  δ3:                  {metrics['depth']['δ3']:.4f}")
    print("=" * 60)



import time

class InferenceTimer:
    """Measure model inference time."""
    
    @staticmethod
    def measure(model_fn, *args, n_runs=5, **kwargs) -> Dict[str, float]:
        """
        Run model_fn n_runs times and return timing stats.
        
        Args:
            model_fn : callable — the inference function
            n_runs   : int — number of runs for averaging
        
        Returns:
            dict with mean_ms, std_ms, min_ms
        """
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model_fn(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # convert to ms
        
        return {
            'mean_ms' : round(float(np.mean(times)), 2),
            'std_ms'  : round(float(np.std(times)),  2),
            'min_ms'  : round(float(np.min(times)),  2),
        }


# ADE20K (150 classes) → Stanford 2D-3D-S (13 classes)
# If a category is not supported, mark as -1 (unsupported)
ADE20K_TO_STANFORD = {
    # ADE20K_id : Stanford_id
    0  : 1,   # wall → wall
    3  : 2,   # floor → floor
    5  : 3,   # ceiling → ceiling
    6  : 4,   # bed → furniture
    7  : 5,   # windowpane → window
    14 : 6,   # door → door
    15 : 7,   # table → table
    18 : 8,   # curtain → furniture
    19 : 9,   # chair → chair
    23 : 10,  # sofa → sofa
    25 : 11,  # shelf → bookcase
    33 : 12,  # column → column
    67 : 13,  # beam → beam
    # All others → -1 (unsupported, will be ignored in metrics)
}

def map_labels(pred: np.ndarray, label_map: dict, ignore_index: int = 255) -> np.ndarray:
    """
    Remap predicted label indices using label_map.
    Unsupported classes → ignore_index.
    
    Args:
        pred      : np.ndarray (H, W) — predicted labels (ADE20K indices)
        label_map : dict {source_id: target_id}
        ignore_index : int — value assigned to unsupported classes
    
    Returns:
        remapped : np.ndarray (H, W)
    """
    remapped = np.full_like(pred, fill_value=ignore_index)
    for src, tgt in label_map.items():
        remapped[pred == src] = tgt
    return remapped