"""
models/depth.py
Unified interface for all depth estimation models used in the benchmark.
Models: Depth Anything V2 (Small, Base, Large), HoHoNet, UniFuse, EGFormer
"""

import torch
import numpy as np
import cv2
import sys
import skimage.transform
from pathlib import Path


class BaseDepthModel:
    """
    Abstract base class: all depth models implement predict().
    """
    def __init__(self, device=None, **kwargs):
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


class DepthAnythingV2PanoramicModel(BaseDepthModel):
    """
    Depth Anything V2 wrapper for equirectangular panoramas.
    Splits the panorama into perspective sub-views and stitches back.
    NOTE: This approach was abandoned because parallax and metric scale
    inconsistencies lead to worse results than the naive baseline.
    """

    def __init__(self, variant='small', checkpoint_dir='./checkpoints',
                 device=None, n_views=8, fov_deg=90.0,
                 persp_h=512, persp_w=512):
        super().__init__(device)
        self.name = f'DepthAnythingV2-{variant.capitalize()}-Panoramic'
        self.n_views = n_views
        self.fov_deg = fov_deg
        self.persp_h = persp_h
        self.persp_w = persp_w

        self.persp_model = DepthAnythingV2Model(
            variant=variant,
            checkpoint_dir=checkpoint_dir,
            device=device
        )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        from panorama_utils import infer_depth_panoramic
        return infer_depth_panoramic(
            self.persp_model, image_bgr,
            n_views=self.n_views,
            fov_deg=self.fov_deg,
            persp_h=self.persp_h,
            persp_w=self.persp_w
        )


class HoHoNetModel(BaseDepthModel):
    """
    HoHoNet — joint depth estimation + semantic segmentation
    specifically designed for equirectangular (360°) indoor images.

    Paper: HoHoNet: 360 Indoor Holistic Understanding with Latent
           Horizontal Features (CVPR 2021)
    GitHub: https://github.com/sunset1995/HoHoNet

    NOTE: Official weights are dead (404). This wrapper is kept for
    completeness but cannot be used until weights are recovered.
    """

    def __init__(self, hohonet_dir: str, checkpoint_path: str, device=None):
        super().__init__(device)
        self.name = 'HoHoNet'

        if str(hohonet_dir) not in sys.path:
            sys.path.insert(0, str(hohonet_dir))

        print(f"Loading {self.name}...")
        try:
            from model import HoHoNet
            import yaml

            # Load default config from repo
            cfg_path = Path(hohonet_dir) / 'config' / 'mp3d_depth' / 'HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml'
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


class UniFuseModel(BaseDepthModel):
    """
    UniFuse — 360° panorama depth estimation via equirectangular + cubemap fusion.
    Trained on Stanford2D3D. Expects ImageNet-normalized inputs.
    """

    def __init__(self, device=None, **kwargs):
        super().__init__(device)
        self.name = 'UniFuse'
        unifuse_root = Path('/home/william/UniFuse-Unidirectional-Fusion')
        unifuse_code_dir = unifuse_root / 'UniFuse'
        checkpoint_path = unifuse_root / 'PretrainedModels' / 'model.pth'

        if str(unifuse_code_dir) not in sys.path:
            sys.path.insert(0, str(unifuse_code_dir))

        print(f"Loading {self.name}...")
        from networks import UniFuse, Equi

        model_dict = torch.load(checkpoint_path, map_location=self.device)

        Net_dict = {"UniFuse": UniFuse, "Equi": Equi}
        Net = Net_dict[model_dict['net']]

        self.in_h = model_dict['height']
        self.in_w = model_dict['width']
        self.max_depth = float(model_dict.get('max_depth', 10.0))

        self.model = Net(
            model_dict['layers'],
            self.in_h,
            self.in_w,
            max_depth=self.max_depth,
            fusion_type=model_dict['fusion'],
            se_in_fusion=model_dict['se_in_fusion']
        )

        model_state_dict = self.model.state_dict()
        self.model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})

        self.model = self.model.to(self.device).eval()
        print(f"✅ {self.name} loaded on {self.device}")

    def _equirect_to_cubemap(self, image_rgb: np.ndarray, face_size: int = 256):
        """
        Convert an equirectangular panorama to a horizontally-concatenated cubemap.
        UniFuse expects shape (1, 3, face_size, face_size * 6).
        Face order: front, right, back, left, top, bottom.
        """
        from panorama_utils import equirect_to_perspective
        faces = []
        params = [
            (0, 0),           # front
            (-np.pi / 2, 0),  # right
            (np.pi, 0),       # back
            (np.pi / 2, 0),   # left
            (0, np.pi / 2),   # top
            (0, -np.pi / 2),  # bottom
        ]
        for yaw, pitch in params:
            face = equirect_to_perspective(
                image_rgb, fov_deg=90, out_h=face_size, out_w=face_size,
                yaw=yaw, pitch=pitch, roll=0
            )
            faces.append(face)

        # Concatenate horizontally: (H, W * 6, 3)
        cube_img = np.concatenate(faces, axis=1)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        cube_norm = cube_img.astype(np.float32) / 255.0
        cube_norm = (cube_norm - mean) / std

        # (1, 3, H, W * 6)
        cube_tensor = torch.from_numpy(cube_norm).permute(2, 0, 1).unsqueeze(0)
        return cube_tensor

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        image_rgb_resized = cv2.resize(image_rgb, (self.in_w, self.in_h))

        # ImageNet normalization constants
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Equirectangular branch: (1, 3, H, W)
        equi_arr = image_rgb_resized.astype(np.float32) / 255.0
        equi_arr = (equi_arr - mean) / std
        equi_tensor = torch.from_numpy(equi_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Cubemap branch: (1, 3, H, W * 6)
        cube_tensor = self._equirect_to_cubemap(image_rgb_resized, face_size=self.in_h // 2)
        cube_tensor = cube_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(equi_tensor, cube_tensor)
            depth = outputs["pred_depth"].squeeze().cpu().numpy()

        # Resize back to original resolution
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


class EGFormerModel(BaseDepthModel):
    """
    EGFormer — Equirectangular Geometry-biased Transformer for 360° depth.
    ICCV 2023.

    NOTE: Currently non-functional due to DDP state_dict mismatch and timm
    dependency issues. Kept in registry for future fix.
    """

    def __init__(self, device=None, **kwargs):
        super().__init__(device)
        self.name = 'EGFormer'
        self.egformer_dir = Path('/home/william/EGformer')
        self.checkpoint_path = self.egformer_dir / 'pretrained_models' / 'EGformer_pretrained.pkl'

        if str(self.egformer_dir) not in sys.path:
            sys.path.insert(0, str(self.egformer_dir))

        print(f"Loading {self.name}...")
        from models.egformer import EGDepthModel
        self.model = EGDepthModel(hybrid=False)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict):
            state = ckpt.get('state_dict', ckpt.get('model', ckpt))
            self.model.load_state_dict(state)
        else:
            self.model = ckpt

        self.model = self.model.to(self.device).eval()
        print(f"✅ {self.name} loaded on {self.device}")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        # Same preprocessing as in trainer.py sample()
        input_h, input_w = 512, 1024
        image_resized = skimage.transform.resize(image_rgb, [input_h, input_w])
        image_resized = image_resized.astype(np.float32)

        tensor = torch.from_numpy(image_resized).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            depth = self.model(tensor)

        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)

class ZoeDepthModel(BaseDepthModel):
    """
    ZoeDepth — SOTA metric depth estimation (indoor).
    Uses a DPT backbone with metric bins. Much better than DA2 on indoor scenes.
    """

    def __init__(self, device=None, **kwargs):
        super().__init__(device)
        self.name = 'ZoeDepth'
        print(f"Loading {self.name}...")

        # torch.hub downloads automatically (uses hf-mirror if configured)
        self.model = torch.hub.load(
            "isl-org/ZoeDepth",
            "ZoeD_N",
            pretrained=True,
            trust_repo=True
        )
        self.model = self.model.to(self.device).eval()
        print(f"✅ {self.name} loaded on {self.device}")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ZoeDepth expects 384x512 (H, W)
        img_resized = cv2.resize(img_rgb, (512, 384))
        tensor = torch.from_numpy(img_resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(tensor)

        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


def get_depth_model(name: str, **kwargs) -> BaseDepthModel:
    """
    Factory function — instantiate a depth model by name.

    Args:
        name   : str — one of:
                 'da2_small', 'da2_base', 'da2_large',
                 'unifuse', 'egformer', 'hohonet'
        kwargs : passed to the model constructor

    Returns:
        model : BaseDepthModel

    Examples:
        model = get_depth_model('da2_small',
                                checkpoint_dir='./Depth-Anything-V2/checkpoints')
        model = get_depth_model('unifuse')
    """
    registry = {
        'da2_small'      : lambda: DepthAnythingV2Model('small', **kwargs),
        'da2_base'       : lambda: DepthAnythingV2Model('base',  **kwargs),
        'da2_large'      : lambda: DepthAnythingV2Model('large', **kwargs),
        'da2_small_pano' : lambda: DepthAnythingV2PanoramicModel('small', **kwargs),
        'da2_base_pano'  : lambda: DepthAnythingV2PanoramicModel('base',  **kwargs),
        'da2_large_pano' : lambda: DepthAnythingV2PanoramicModel('large', **kwargs),
        'hohonet'        : lambda: HoHoNetModel(**kwargs),
        'unifuse'        : lambda: UniFuseModel(**kwargs),
        'egformer'       : lambda: EGFormerModel(**kwargs),
        'zoedepth': lambda: ZoeDepthModel(**kwargs),
    }
    assert name in registry, f"Unknown model '{name}'. Choose from: {list(registry)}"
    return registry[name]()