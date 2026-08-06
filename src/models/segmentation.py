"""
models/segmentation.py
Unified interface for all segmentation models used in the benchmark.
Models: SegFormer-B2, SegFormer-B4, Mask2Former (ADE20K)
"""

import torch
import numpy as np
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


class BaseSegmentationModel:
    """
    Abstract base class: all segmentation models implement predict().
    """
    def __init__(self, device=None):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.model  = None
        self.processor = None
        self.name = "base"

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Run segmentation on a single RGB image.

        Args:
            image_rgb : np.ndarray (H, W, 3) uint8 — RGB image

        Returns:
            seg_map : np.ndarray (H, W) int — predicted class indices
        """
        raise NotImplementedError


class SegFormerModel(BaseSegmentationModel):
    """
    SegFormer fine-tuned on ADE20K (150 classes).
    Supports B2 and B4 variants.

    B2 : nvidia/segformer-b2-finetuned-ade-512-512  (~25M params, faster)
    B4 : nvidia/segformer-b4-finetuned-ade-512-512  (~64M params, more accurate)
    """

    VARIANTS = {
        'b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
    }

    def __init__(self, variant='b2', device=None):
        super().__init__(device)
        assert variant in self.VARIANTS, f"variant must be one of {list(self.VARIANTS)}"
        self.name = f'SegFormer-{variant.upper()}'
        model_id = self.VARIANTS[variant]

        print(f"Loading {self.name} from {model_id}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ {self.name} loaded on {self.device}")

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Upsample logits to original image size
        logits = outputs.logits  # (1, num_classes, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image_rgb.shape[:2],  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        return seg_map.astype(np.int32)


class Mask2FormerModel(BaseSegmentationModel):
    """
    Mask2Former fine-tuned on ADE20K (150 classes).
    Used in related work — required baseline per PhD instructions.

    Model: facebook/mask2former-swin-base-ade-semantic
    """

    MODEL_ID = 'facebook/mask2former-swin-base-ade-semantic'

    def __init__(self, device=None):
        super().__init__(device)
        self.name = 'Mask2Former'

        print(f"Loading {self.name} from {self.MODEL_ID}...")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ {self.name} loaded on {self.device}")

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mask2Former uses a different post-processing API
        seg_result = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image_rgb.shape[:2]]
        )
        seg_map = seg_result[0].cpu().numpy()
        return seg_map.astype(np.int32)


def get_segmentation_model(name: str, device=None) -> BaseSegmentationModel:
    """
    Factory function — instantiate a segmentation model by name.

    Args:
        name   : str — one of 'segformer_b2', 'segformer_b4', 'mask2former'
        device : torch.device or None

    Returns:
        model : BaseSegmentationModel

    Example:
        model = get_segmentation_model('mask2former')
        seg_map = model.predict(image_rgb)
    """
    registry = {
        'segformer_b2' : lambda: SegFormerModel('b2', device),
        'segformer_b4' : lambda: SegFormerModel('b4', device),
        'mask2former'  : lambda: Mask2FormerModel(device),
    }
    assert name in registry, f"Unknown model '{name}'. Choose from: {list(registry)}"
    return registry[name]()