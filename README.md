# Task 1 — Panoramic Vision Pipeline: Semantic Segmentation & Depth Estimation

**Summer Research Internship 2026 — Project 178**
Shanghai Jiao Tong University × EPFL

Supervised by Prof. Chong Han and PhD Zitong Fang.

---

## Overview

This repository contains the benchmarking pipeline for Task 1 of Project 178 (THz ISAC). The goal is to evaluate open-source pretrained models for **semantic segmentation** and **metric depth estimation** on indoor 360° equirectangular panoramas, as a first step toward multi-modal environment reconstruction fusing panoramic vision with THz sensing data (Task 2).

All models are evaluated **zero-shot** (no fine-tuning) on the [Stanford 2D-3D-Semantics](http://3dsemantics.stanford.edu/) dataset, and qualitatively on 15 real-world panoramas captured at the SJTU lab with a QooCam 8K camera.

---

## Repository Structure
task1/
├── benchmark.py # Main benchmark orchestrator (seg + depth)
├── metrics.py # mIoU, AbsRel, RMSE, δ1, InferenceTimer
├── pipeline.py # Single-image inference + visualisation
├── panorama_utils.py # Equirectangular ↔ perspective conversions (numpy/cv2)
├── visualize_real_images.py # Qualitative results on 15 SJTU images
├── visualize_stanford_depth.py # GT depth format verification + colourmap export
├── generate_report_figures.py # Side-by-side GT vs Pred figures for report
├── diagnose_depth.py # Per-image depth diagnostic (correlation, scale bias)
├── diagnose_seg.py # Per-class segmentation diagnostic
├── download_depth_model.py # Download DA2 checkpoints via hf-mirror.com
├── download_stanford.py # Download Stanford 2D-3D-S via Redivis API
├── models/
│ ├── segmentation.py # SegFormer-B2/B4, Mask2Former wrappers
│ └── depth.py # DA2 Small/Base/Large, UniFuse, HoHoNet, EGFormer wrappers
└── report/
├── SJTU_midterm_report.pdf
└── SJTU_task1_report.pdf


---

## Models Benchmarked

### Segmentation (ADE20K pretrained, zero-shot transfer to Stanford)

| Model | Params | Source |
|---|---|---|
| SegFormer-B2 | ~25M | `nvidia/segformer-b2-finetuned-ade-512-512` |
| SegFormer-B4 | ~64M | `nvidia/segformer-b4-finetuned-ade-512-512` |
| Mask2Former | ~432M | `facebook/mask2former-swin-base-ade-semantic` |

### Depth (zero-shot or in-domain)

| Model | Training data | Eval type |
|---|---|---|
| DA2-Small/Base/Large | Hypersim (synthetic indoor) | Zero-shot |
| UniFuse | Stanford2D3D | In-domain |

**Models not evaluated** (see report Section 4 for details):
- HoHoNet — official weights unavailable (404)
- EGFormer — DDP checkpoint key mismatch
- Trans4PASS+ — requires legacy mmcv (CUDA 11.1 / PyTorch 1.8)
- ZoeDepth — `torch.hub` blocked by server firewall

---

## Key Results

### Depth (Stanford area\_1, 50 samples)

| Model | AbsRel ↓ | RMSE ↓ | δ₁ ↑ | Protocol |
|---|---|---|---|---|
| DA2-Small | 0.964 | 0.925 | 0.083 | Zero-shot, no alignment |
| DA2-Base | 0.969 | 0.938 | 0.064 | Zero-shot, no alignment |
| DA2-Large | 0.980 | 0.954 | 0.056 | Zero-shot, no alignment |
| UniFuse | 0.124 | 0.244 | — | In-domain, median alignment |

### Segmentation (Stanford area\_1, 50 samples)

| Model | mIoU ↑ | Pixel Acc. ↑ | Inference (ms) |
|---|---|---|---|
| SegFormer-B2 | 0.206 | 0.545 | 110 |
| SegFormer-B4 | 0.248 | 0.639 | 30 |
| Mask2Former | 0.225 | 0.541 | 103 |

---

## Setup

### Requirements

```bash
conda create -n sjtu_task1 python=3.10
conda activate sjtu_task1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
pip install opencv-python-headless numpy matplotlib pillow
pip install huggingface_hub scipy scikit-learn redivis
pip install "numpy<2"  # required for opencv-python-headless 4.8 compatibility
```

### Depth Anything V2

```bash
git clone https://gh-proxy.com/https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2 && pip install -r requirements.txt && cd ..
```

Download checkpoints (Hypersim metric variant, via hf-mirror):

```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_depth_model.py
```

### UniFuse

```bash
git clone https://gh-proxy.com/https://github.com/Jokay/UniFuse-Unidirectional-Fusion.git
```

Download the Stanford2D3D pretrained weights from the [UniFuse release page](https://github.com/Jokay/UniFuse-Unidirectional-Fusion) and place at:
UniFuse-Unidirectional-Fusion/PretrainedModels/model.pth


### Stanford 2D-3D-S Dataset

Register at [Stanford SDSS / Redivis](https://redivis.com/datasets/sgdg-stanford2d3d), then:

```bash
export REDIVIS_API_TOKEN="your_token"
python download_stanford.py
```

Expected structure after extraction:

data/stanford/area_1/pano/
├── rgb/ # *_domain_rgb.png
├── semantic/ # *_domain_semantic.png
└── depth/ # *_domain_depth.png
data/stanford/semantic_labels.json


---

## Running the Benchmark

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com

# Segmentation benchmark
python benchmark.py \
    --task segmentation \
    --data_dir /path/to/data/stanford/area_1 \
    --output_dir ./outputs \
    --n_samples 50

# Depth benchmark
python benchmark.py \
    --task depth \
    --data_dir /path/to/data/stanford/area_1 \
    --output_dir ./outputs \
    --checkpoint_dir /path/to/Depth-Anything-V2/checkpoints \
    --n_samples 50
```

Results are saved as CSV and PNG bar charts in `outputs/`.

## Qualitative Visualisation (real-world images)

```bash
python visualize_real_images.py \
    --input_dir /path/to/data/validation/ \
    --output_dir ./outputs_real/
```

---

## Implementation Notes

**Image preprocessing.** All images are resized to 2048×1024 before inference (factor-4 downscale from original 4096×2048). GT segmentation masks use `INTER_NEAREST`; GT depth maps use `INTER_LINEAR` with post-resize validity mask re-application.

**Depth GT format.** Stanford depth maps are 16-bit PNG (uint16, millimetres). Invalid pixels = 65535 → set to 0. Conversion: `depth_m = uint16 / 1000`. Valid range: 0 < depth ≤ 10 m.

**Label mapping.** ADE20K (150 classes) → Stanford (13 classes) via verified index correspondence. Unmapped classes → `<UNK>` (ignored in mIoU). See `benchmark.py` `ADE20K_TO_STANFORD` dict.

**UniFuse evaluation protocol.** Median alignment (scale prediction median to GT median per image) is applied before computing metrics, following the official `evaluate.py`. This is standard for models that may have a global scale bias. DA2 is reported without alignment.

**Server constraints.** GitHub, Google Drive, and HuggingFace are blocked on the lab server. Use `gh-proxy.com` for GitHub clones and `hf-mirror.com` for HuggingFace downloads. Git pushes are done via `git bundle` + SCP to a local machine.

---

## Known Issues

- `δ₁ = 0.00` for UniFuse despite AbsRel = 0.12 — suspected clipping artefact in metric implementation, under investigation.
- HoHoNet wrapper implemented in `models/depth.py` but cannot be instantiated (weights unavailable).
- EGFormer wrapper implemented but not functional (DDP key mismatch).

---




