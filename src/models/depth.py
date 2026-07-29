"""
models/depth.py
Unified interface for all depth estimation models used in the benchmark.
Models: Depth Anything V2 (Small, Base, Large), HoHoNet, UniFuse
"""

import torch
import numpy as np
import cv2
import sys
from pathlib import Path


class BaseDepthModel:
    """
    Abstract base class all depth models implement predict().
    """
    def __init__(self, device=None):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.model = None
        self.name  = "base"

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a single BGR image.

        Args:
            image_bgr : np.ndarray (H, W, 3) uint8 — BGR image (OpenCV format)

        Returns:
            depth_map : np.ndarray (H, W) float32 — depth in metres
        """
        raise NotImplementedError


class DepthAnythingV2Model(BaseDepthModel):
    """
    Depth Anything V2 — metric depth estimation (indoor, Hypersim fine-tuned).

    Variants:
        small : ~25M params  — fastest, least accurate
        base  : ~97M params  — good balance
        large : ~335M params — most accurate, needs strong GPU

    Encoder configs per variant (fixed by training):
        small : features=64,  out_channels=[48, 96, 192, 384]
        base  : features=128, out_channels=[96, 192, 384, 768]
        large : features=256, out_channels=[256, 512, 1024, 1024]
    """

    CONFIGS = {
        'small': {
            'encoder'      : 'vits',
            'features'     : 64,
            'out_channels' : [48, 96, 192, 384],
            'checkpoint'   : 'depth_anything_v2_metric_hypersim_vits.pth',
            'repo_id'      : 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Small',
        },
        'base': {
            'encoder'      : 'vitb',
            'features'     : 128,
            'out_channels' : [96, 192, 384, 768],
            'checkpoint'   : 'depth_anything_v2_metric_hypersim_vitb.pth',
            'repo_id'      : 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Base',
        },
        'large': {
            'encoder'      : 'vitl',
            'features'     : 256,
            'out_channels' : [256, 512, 1024, 1024],
            'checkpoint'   : 'depth_anything_v2_metric_hypersim_vitl.pth',
            'repo_id'      : 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Large',
        },
    }

    def __init__(self, variant='small', checkpoint_dir='./checkpoints', device=None):
        super().__init__(device)
        assert variant in self.CONFIGS, \
            f"variant must be one of {list(self.CONFIGS)}"

        self.name    = f'DepthAnythingV2-{variant.capitalize()}'
        cfg          = self.CONFIGS[variant]
        ckpt_path    = Path(checkpoint_dir) / cfg['checkpoint']

        # Auto-download if checkpoint not found
        if not ckpt_path.exists():
            print(f"Checkpoint not found locally, downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id    = cfg['repo_id'],
                filename   = cfg['checkpoint'],
                local_dir  = checkpoint_dir,
            )

        # Import DepthAnythingV2 from cloned repo
        da2_path = Path(checkpoint_dir).parent
        if str(da2_path) not in sys.path:
            sys.path.append(str(da2_path))
        from depth_anything_v2.dpt import DepthAnythingV2

        print(f"Loading {self.name}...")
        self.model = DepthAnythingV2(
            encoder      = cfg['encoder'],
            features     = cfg['features'],
            out_channels = cfg['out_channels'],
        )
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ {self.name} loaded on {self.device}")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        infer_image handles preprocessing internally (resize, normalize).
        Returns metric depth in metres.
        """
        depth = self.model.infer_image(image_bgr)
        return depth.astype(np.float32)


class HoHoNetModel(BaseDepthModel):
    """
    HoHoNet — joint depth estimation + semantic segmentation
    specifically designed for equirectangular (360°) indoor images.

    Paper: HoHoNet: 360 Indoor Holistic Understanding with Latent
           Horizontal Features (CVPR 2021)
    GitHub: https://github.com/sunset1995/HoHoNet

    Note: HoHoNet requires a separate clone and pretrained weights.
          This wrapper assumes the repo is cloned at hohonet_dir.
    """

    def __init__(self, hohonet_dir: str, checkpoint_path: str, device=None):
        """
        Args:
            hohonet_dir     : str — path to cloned HoHoNet repo
            checkpoint_path : str — path to pretrained .pth weights
            device          : torch.device or None
        """
        super().__init__(device)
        self.name = 'HoHoNet'

        if str(hohonet_dir) not in sys.path:
            sys.path.insert(0, str(hohonet_dir))

        print(f"Loading {self.name}...")
        try:
            from model import HoHoNet
            import yaml

            # Load default config from repo
            cfg_path = Path(hohonet_dir) / 'config' / 'mp3d_depth.yaml'
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            self.model = HoHoNet(**cfg['model'])
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ {self.name} loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Could not import HoHoNet. Make sure the repo is cloned at "
                f"{hohonet_dir}. Error: {e}"
            )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        # HoHoNet expects RGB, normalized, tensor (1, 3, H, W)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (1024, 512))  # HoHoNet input size

        tensor = torch.from_numpy(image_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)

        with torch.no_grad():
            depth = self.model(tensor)

        depth = depth.squeeze().cpu().numpy()
        # Resize back to original resolution
        depth = cv2.resize(
            depth,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        return depth.astype(np.float32)


def get_depth_model(name: str, **kwargs) -> BaseDepthModel:
    """
    Factory function — instantiate a depth model by name.

    Args:
        name   : str — one of:
                 'da2_small', 'da2_base', 'da2_large', 'hohonet'
        kwargs : passed to the model constructor

    Returns:
        model : BaseDepthModel

    Examples:
        model = get_depth_model('da2_small',
                                checkpoint_dir='./Depth-Anything-V2/checkpoints')
        model = get_depth_model('hohonet',
                                hohonet_dir='./HoHoNet',
                                checkpoint_path='./HoHoNet/ckpt/mp3d.pth')
    """
    registry = {
        'da2_small' : lambda: DepthAnythingV2Model('small', **kwargs),
        'da2_base'  : lambda: DepthAnythingV2Model('base',  **kwargs),
        'da2_large' : lambda: DepthAnythingV2Model('large', **kwargs),
        'hohonet'   : lambda: HoHoNetModel(**kwargs),
    }
    assert name in registry, \
        f"Unknown model '{name}'. Choose from: {list(registry)}"
    return registry[name]()