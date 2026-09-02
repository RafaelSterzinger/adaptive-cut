"""Image / mask preprocessing (from FROST, Park 2026).

ImageNet normalization, resize to the model's input resolution, and helpers
to move binary masks between image resolution and the patch grid.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int) -> transforms.Compose:
    """Resize -> ToTensor -> ImageNet-normalize."""
    return transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def load_image(
    image: "str | Image.Image",
    transform: transforms.Compose,
    device: "torch.device | str",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load a path or PIL image as (1, C, H, W). Returns (tensor, orig (H, W))."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    orig_size = (image.height, image.width)
    return transform(image).unsqueeze(0).to(device), orig_size


def load_mask(
    mask: "str | Image.Image | torch.Tensor",
    image_size: int,
    device: "torch.device | str",
) -> torch.Tensor:
    """Load a binary mask, nearest-resized to (image_size, image_size). Returns (1, H, W) bool."""
    if isinstance(mask, torch.Tensor):
        mask_tensor = mask.unsqueeze(0) if mask.ndim == 2 else mask
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError("Tensor mask must have shape (H,W) or (1,H,W)")
        mask_tensor = mask_tensor.to(device)
        if mask_tensor.dtype != torch.bool:
            mask_tensor = mask_tensor > 0
    else:
        if isinstance(mask, str):
            mask = Image.open(mask)
        mask_tensor = torch.tensor(
            np.array(mask) > 0, dtype=torch.bool).unsqueeze(0).to(device)
    return F.interpolate(
        mask_tensor.unsqueeze(0).float(),
        size=(image_size, image_size), mode="nearest",
    ).squeeze(0) > 0.5


def downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Downsample a (1, 1, H, W) binary mask to the patch grid (h, w) bool.

    Bilinear>0.5 first; if that empties the mask, fall back to nearest, then to
    a single centre pixel, so a tiny FG object never vanishes on the grid.
    """
    down = F.interpolate(mask.float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
    if down.sum() == 0:
        down = F.interpolate(mask.float(), size=(h, w), mode="nearest")[0, 0] > 0.5
        if down.sum() == 0:
            if not (mask[0, 0] > 0).any():
                return down
            center = torch.argwhere(mask[0, 0] > 0).float().mean(dim=0)
            scale = mask.shape[-1] // w
            cy, cx = (center / scale).int()
            down[cy, cx] = True
    return down


def upsample_mask(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Upsample a patch-grid binary mask to (H, W) via bilinear>0.5."""
    return F.interpolate(
        mask[None, None].float(), size=(H, W), mode="bilinear", align_corners=False,
    )[0, 0] > 0.5
