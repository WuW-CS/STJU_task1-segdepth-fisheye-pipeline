from huggingface_hub import hf_hub_download
import os

os.makedirs("./Depth-Anything-V2/checkpoints", exist_ok=True)

print("Téléchargement du modèle Depth Anything V2 metric indoor...")
hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
    filename="depth_anything_v2_metric_hypersim_vits.pth",
    local_dir="./Depth-Anything-V2/checkpoints"
)
print("Téléchargement terminé ✓")