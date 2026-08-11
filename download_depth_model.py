"""
Download Depth Anything V2 Metric Hypersim Small weights.
Works both locally and on Colab (with Drive paths).
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# ============================================================
# 1. DÉTECTION DE L'ENVIRONNEMENT
# ============================================================

def get_checkpoint_dir():
    """
    Returns the correct checkpoint directory:
    - If running on Colab: uses Drive path
    - If running locally: uses local ./Depth-Anything-V2/checkpoints
    """
    # Check if we're on Colab (Drive is mounted)
    if os.path.exists('/content/drive/MyDrive'):
        # Colab path
        base_dir = Path('/content/drive/MyDrive/SJTU')
        checkpoint_dir = base_dir / 'Depth-Anything-V2' / 'checkpoints'
    else:
        # Local path
        checkpoint_dir = Path('./Depth-Anything-V2/checkpoints')
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

# ============================================================
# 2. TÉLÉCHARGEMENT
# ============================================================

def download_model():
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_path = checkpoint_dir / 'depth_anything_v2_metric_hypersim_vits.pth'
    
    # Check if already downloaded
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {checkpoint_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return
    
    print(f"📥 Downloading Depth Anything V2 Metric Hypersim Small...")
    print(f"   Target: {checkpoint_path}")
    
    # Download from HuggingFace
    hf_hub_download(
        repo_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        filename="depth_anything_v2_metric_hypersim_vits.pth",
        local_dir=str(checkpoint_dir),  # Saves in Depth-Anything-V2/checkpoints/
        local_dir_use_symlinks=False,
    )
    
    print(f"✅ Download complete: {checkpoint_path}")
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")

# ============================================================
# 3. EXÉCUTION
# ============================================================

if __name__ == "__main__":
    download_model()