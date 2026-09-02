#!/usr/bin/env python
"""1-shot demo: segment a query image from one annotated reference.

    python demo.py --weights dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --ref-image ref.jpg --ref-mask ref_mask.png --query-image query.jpg --out pred.png
"""
import argparse
import os

import numpy as np
from PIL import Image

from model import build_frost


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.environ.get("DINO_WEIGHTS", ""), help="DINOv3 ViT-L/16 checkpoint")
    p.add_argument("--ref-image", required=True)
    p.add_argument("--ref-mask", required=True, help="nonzero = foreground")
    p.add_argument("--query-image", required=True)
    p.add_argument("--out", default="pred.png")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not args.weights:
        p.error("pass --weights or set $DINO_WEIGHTS")

    model = build_frost(args.weights, device=args.device)
    mask = model.segment_one(
        Image.open(args.ref_image).convert("RGB"),
        Image.open(args.ref_mask),
        Image.open(args.query_image).convert("RGB"),
    )
    Image.fromarray((mask.cpu().numpy().astype(np.uint8) * 255), mode="L").save(args.out)
    print(f"wrote {args.out}: {int(mask.sum())} foreground pixels")


if __name__ == "__main__":
    main()
