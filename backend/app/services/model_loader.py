"""
Model Loader — Downloads ML models from Hugging Face Hub if not present locally.
Set env variable HF_REPO_ID to your Hugging Face repo, e.g. "VatsalNegi/careersphere-models"
"""

import os
import joblib
from pathlib import Path

# Base directory for models
BASE_DIR = Path(__file__).resolve().parents[3] / "ml" / "models"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Model files to download
MODEL_FILES = [
    "career_model.pkl",
    "career_encoders.pkl",
    "mental_health_model.pkl",
    "mental_health_encoders.pkl",
    "burnout_model.pkl",
    "burnout_encoders.pkl",
]

HF_REPO_ID = os.getenv("HF_REPO_ID", "")


def download_models_if_needed():
    """Download all model files from Hugging Face Hub if not present locally."""
    missing = [f for f in MODEL_FILES if not (BASE_DIR / f).exists()]

    if not missing:
        print("✅ All ML models found locally.")
        return

    if not HF_REPO_ID:
        raise RuntimeError(
            f"❌ Missing ML model files: {missing}\n"
            "Set the HF_REPO_ID environment variable to your Hugging Face repo "
            "(e.g. 'YourUsername/careersphere-models') so models can be auto-downloaded."
        )

    print(f"📥 Downloading {len(missing)} model file(s) from Hugging Face Hub: {HF_REPO_ID}")
    try:
        from huggingface_hub import hf_hub_download
        for filename in missing:
            print(f"   Downloading {filename}...")
            local_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(BASE_DIR),
                repo_type="model",
            )
            print(f"   ✅ Saved to {local_path}")
        print("✅ All models downloaded successfully!")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download models from Hugging Face: {e}")


def get_model_path(filename: str) -> str:
    """Return the absolute path to a model file."""
    return str(BASE_DIR / filename)
