"""Episodic few-shot segmentation benchmarks, following the INSID3 protocol.

The loaders are vendored from INSID3 (Cuttano, Trivigno, Reich, Cremers,
Masone, Roth, "Training-Free In-Context Segmentation with DINOv3", CVPR 2026;
https://github.com/visinf/INSID3, Apache-2.0). Credit for the episode
sampling, the split files and the data layout goes to them.
"""
from types import SimpleNamespace

from .coco import build as build_coco
from .isaid import build as build_isaid
from .isic import build as build_isic
from .lung import build as build_lung
from .lvis import build as build_lvis
from .paco_part import build as build_paco_part
from .pascal_part import build as build_pascal_part
from .suim import build as build_suim

_BUILDERS = {
    "coco": build_coco,
    "lvis": build_lvis,
    "pascal_part": build_pascal_part,
    "paco_part": build_paco_part,
    "isaid": build_isaid,
    "isic": build_isic,
    "lung": build_lung,
    "suim": build_suim,
}

# number of folds per benchmark (the paper reports the mean over folds)
FOLDS = {"coco": 4, "lvis": 10, "pascal_part": 4, "paco_part": 4,
         "isaid": 3, "isic": 1, "lung": 1, "suim": 1}

# classes per benchmark, for the class-summed IoU meter
NCLASS = {"coco": 80, "lvis": 1203, "paco_part": 448, "lung": 2,
          "isaid": 15, "isic": 3, "suim": 7}


def build_dataset(name: str, data_root: str, fold: int, shots: int):
    if name not in _BUILDERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(_BUILDERS)}")
    return _BUILDERS[name](SimpleNamespace(data_root=data_root, fold=fold, shots=shots))
