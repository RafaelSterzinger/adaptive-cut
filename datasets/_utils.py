# Vendored from INSID3 utils/data.py (Cuttano et al., CVPR 2026), Apache-2.0.
from typing import List

import numpy as np
import pycocotools.mask as mask_util


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon annotations to a binary mask."""
    if len(polygons) == 0:
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)
