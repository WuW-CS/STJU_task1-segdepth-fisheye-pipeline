import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('./Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# 1. Download images
img_path = './Depth-Anything-V2/assets/examples/demo01.jpg'
image_bgr = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print(f"Image chargée : {image_rgb.shape}")

# 2. DEPTH ESTIMATION
model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384]
)
model.load_state_dict(torch.load(
    './Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
    map_location='cpu'
))
model.eval()
print("Modèle depth chargé ✓")

depth_map = model.infer_image(image_bgr)
print(f"Depth map shape : {depth_map.shape}")

# 3. SEGMENTATION
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
seg_model.eval()
print("Modèle segmentation chargé ✓")

pil_image = Image.fromarray(image_rgb)
inputs = processor(images=pil_image, return_tensors="pt")

with torch.no_grad():
    outputs = seg_model(**inputs)

logits = outputs.logits
upsampled = torch.nn.functional.interpolate(
    logits,
    size=image_rgb.shape[:2],
    mode='bilinear',
    align_corners=False
)
seg_map = upsampled.argmax(dim=1).squeeze().numpy()
print(f"Segmentation map shape : {seg_map.shape}")

# 4. VISUALISATION 
os.makedirs('./outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image_rgb)
axes[0].set_title('Image originale')
axes[0].axis('off')

axes[1].imshow(seg_map, cmap='tab20')
axes[1].set_title('Segmentation sémantique')
axes[1].axis('off')

axes[2].imshow(depth_map, cmap='plasma')
axes[2].set_title('Depth estimation (mètres)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('./outputs/result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Résultat sauvegardé dans outputs/result.png ✓")