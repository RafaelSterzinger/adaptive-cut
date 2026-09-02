"""Where to Cut? Adaptive Thresholds for Density-Ratio Few-Shot Segmentation.

Builds on FROST (Park, 2026): https://github.com/jhpark-ai/FROST (Apache-2.0).
"""

from .encoder import DINOv3FeatureExtractor, build_encoder
from .model import FROST

__all__ = ["FROST", "DINOv3FeatureExtractor", "build_encoder", "build_frost"]


def build_frost(weights_path: str, image_size: int = 1024, device: str = "cuda") -> FROST:
    """Build the model from a DINOv3 ViT-L/16 checkpoint path. No other knobs:
    every hyper-parameter is a fixed constant in ``model/model.py`` / ``model/threshold.py``."""
    raw = build_encoder(weights_path, device=device)
    enc = DINOv3FeatureExtractor(raw, image_size=image_size, patch_size=16)
    return FROST(encoder=enc, raw_encoder=raw, image_size=image_size,
                 device=device).to(device).eval()
