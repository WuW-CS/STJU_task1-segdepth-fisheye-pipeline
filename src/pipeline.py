# %% [markdown]
# # Fisheye Pipeline - Segmentation + Depth
# ## Execution on Colab with GPU

# %%
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# ============================================
# 1. PATHS TO DRIVE
# ============================================

# Main project directory in Drive
PROJECT_DIR = Path('/content/drive/MyDrive/SJTU/STJU_task1-segdepth-fisheye-pipeline')
DEPTH_DIR = Path('/content/drive/MyDrive/SJTU/Depth-Anything-V2')

# Test image (from Depth Anything assets)
img_path = DEPTH_DIR / 'assets/examples/demo01.jpg'

# Depth Anything checkpoint
checkpoint_path = DEPTH_DIR / 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'

# Output directory
output_dir = PROJECT_DIR / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Image: {img_path}")
print(f"📁 Checkpoint: {checkpoint_path}")
print(f"📁 Output: {output_dir}")

# ============================================
# 2. ADD Depth-Anything-V2 TO PYTHONPATH
# ============================================

sys.path.append(str(DEPTH_DIR))
from depth_anything_v2.dpt import DepthAnythingV2

# ============================================
# 3. GPU / CPU DETECTION
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# ============================================
# 4. LOAD IMAGE
# ============================================

image_bgr = cv2.imread(str(img_path))
if image_bgr is None:
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print(f"📷 Image loaded: {image_rgb.shape}")

# ============================================
# 5. DEPTH ESTIMATION
# ============================================

print("⏳ Loading Depth Anything V2...")

model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384]
)

# Load weights
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()
print("✅ Depth model loaded")

# Inference
depth_map = model.infer_image(image_bgr)
print(f"📊 Depth map shape: {depth_map.shape}")

# ============================================
# 6. SEGMENTATION
# ============================================

print("⏳ Loading SegFormer...")

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
seg_model = seg_model.to(device)
seg_model.eval()
print("✅ Segmentation model loaded")

# Prepare image
pil_image = Image.fromarray(image_rgb)
inputs = processor(images=pil_image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Inference
with torch.no_grad():
    outputs = seg_model(**inputs)

logits = outputs.logits
upsampled = torch.nn.functional.interpolate(
    logits,
    size=image_rgb.shape[:2],
    mode='bilinear',
    align_corners=False
)
seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
print(f"📊 Segmentation map shape: {seg_map.shape}")

# ============================================
# 7. VISUALIZATION
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#Original image
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

#Segmentation
seg_plot = axes[1].imshow(seg_map, cmap='tab20')
axes[1].set_title('Semantic Segmentation')
axes[1].axis('off')
plt.colorbar(seg_plot, ax=axes[1], fraction=0.046, pad=0.04)

#Depth
depth_plot = axes[2].imshow(depth_map, cmap='plasma')
axes[2].set_title('Depth Estimation (meters)')
axes[2].axis('off')
cbar = plt.colorbar(depth_plot, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label('Depth (meters)', rotation=270, labelpad=15)


plt.tight_layout()
plt.savefig(str(output_dir / 'result.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Result saved to {output_dir / 'result.png'}")