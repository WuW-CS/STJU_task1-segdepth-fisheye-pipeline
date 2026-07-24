# STJU_task1-segdepth-fisheye-pipeline
Developping a Joint semantic segmentation and depth estimation on fisheye panoramas model


**Joint Semantic Segmentation and Monocular Depth Estimation on Fisheye Images**

---

## Project Objective

This project is **Task 1** of the summer research internship at SJTU (Shanghai Jiao Tong University), supervised by Professor Chong Han.

The goal is to develop a computer vision pipeline for indoor environment analysis from **fisheye images**. The system must:

- Perform **semantic segmentation** at pixel level (identify doors, floor, ceiling, walls, etc.)
- Perform **monocular depth estimation** to understand the 3D geometry of the scene

This pipeline serves as the **perception module** for a future multimodal AI agent that will combine these outputs with THz sensing data.

---

## Architecture

The pipeline uses two pre-trained models, fine-tuned on fisheye data:

| Task | Model | Pre-training Dataset |
|------|-------|---------------------|
| Semantic Segmentation | **SegFormer** (HuggingFace Transformers) | ADE20K (indoor scenes) |
| Depth Estimation | **Depth Anything V2 Metric Hypersim Small** | Hypersim (indoor, metric) |

### Pipeline Steps

1. **Preprocessing** : Fisheye undistortion using OpenCV (`cv2.undistort`)
2. **Segmentation** : Pixel-wise labeling with SegFormer
3. **Depth Estimation** : Metric depth map generation with Depth Anything V2
4. **Visualization** : Overlay results for qualitative analysis

---


---

## Important: Models and Data

The following files are **excluded from Git versioning** due to their size:

- `checkpoints/*.pth` — Model weights (several hundred MB)
- `data/*.jpg`, `data/*.png` — Fisheye images
- `venv/` — Python virtual environment

### How to Get the Weights

Models are downloaded automatically via scripts:

- **Depth Anything V2** :
  ```bash
  python download_depth_model.py

- **SegFormer** :
    loaded directly from HuggingFace Transformers

Note: The Depth-Anything-V2/ repository is cloned locally but not tracked in this repo. You must clone it separately (see Installation).


## References

    SegFormer : Xie et al., SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers, NeurIPS 2021

    Depth Anything V2 : Yang et al., Depth Anything V2, arXiv 2024

    MiDaS : Ranftl et al., Towards Robust Monocular Depth Estimation, TPAMI 2022

## Author

    WuW-CS: Wu William Yuhao — Summer Research Internship at SJTU, 2026
    Supervisor: Prof. Chong Han, Phd Zitong Fan