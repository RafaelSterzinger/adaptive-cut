#!/usr/bin/env python
"""Episodic few-shot segmentation evaluation of one benchmark at one shot count.

Every fold of the benchmark is run in turn (per-fold re-seeding, so each fold
is reproducible on its own), and the mean over folds is what the paper's table
reports. Metric: foreground IoU accumulated over all episodes of a fold,
averaged over classes (mIoU), plus FB-IoU -- as in INSID3 and FROST.

    python eval.py --dataset isic --shots 1 --output results/k1/isic/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import FOLDS, NCLASS, build_dataset  # noqa: E402
from model import build_frost  # noqa: E402


def classify_prediction(pred_mask, gt_mask, tgt_ignore_idx=None):
    """Per-episode (2, 1) intersection and union for [BG, FG]."""
    pred = pred_mask.float().unsqueeze(0)
    gt = (gt_mask > 0).to(pred.device).float().unsqueeze(0)
    if tgt_ignore_idx is not None:
        tgt_ignore_idx = tgt_ignore_idx.to(gt.device)
        assert torch.logical_and(tgt_ignore_idx.bool(), gt.bool()).sum() == 0
        tgt_ignore_idx = tgt_ignore_idx * 255
        gt = gt + tgt_ignore_idx
        pred[gt == 255] = 255
    area_inter, area_pred, area_gt = [], [], []
    for _pred, _gt in zip(pred, gt):
        inter = _pred[_pred == _gt]
        if inter.numel() == 0:
            area_inter.append(torch.tensor([0.0, 0.0], device=_pred.device))
        else:
            area_inter.append(torch.histc(inter, bins=2, min=0, max=1))
        area_pred.append(torch.histc(_pred, bins=2, min=0, max=1))
        area_gt.append(torch.histc(_gt, bins=2, min=0, max=1))
    area_inter = torch.stack(area_inter).t()
    area_pred = torch.stack(area_pred).t()
    area_gt = torch.stack(area_gt).t()
    return area_inter, area_pred + area_gt - area_inter


class AverageMeter:
    """Class-summed IoU meter over every class of the benchmark."""

    def __init__(self, dataset_name, class_ids, device):
        self.class_ids_interest = torch.tensor(class_ids, device=device)
        self.nclass = NCLASS.get(dataset_name, max(class_ids) + 1)
        self.intersection_buf = torch.zeros(2, self.nclass, device=device)
        self.union_buf = torch.zeros(2, self.nclass, device=device)

    def update(self, inter_b, union_b, class_id):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())

    def compute_iou(self):
        ones = torch.ones_like(self.union_buf)
        iou = self.intersection_buf.float() / torch.max(
            torch.stack([self.union_buf, ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100
        fb_iou = (
            self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1)
            / self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)
        ).mean() * 100
        return float(miou.item()), float(fb_iou.item())


def seed_all(seed: int) -> None:
    # np.random drives the loaders' episode sampling: the seed fixes the episodes.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_fold(model, args, fold, device):
    seed_all(args.seed)
    dataset = build_dataset(args.dataset, args.data_root, fold, args.shots)
    # num_workers=0: sampling runs in this process, under the seed above.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda x: x[0])
    meter = AverageMeter(args.dataset, dataset.class_ids, device)
    n_done = 0
    for idx, batch in enumerate(tqdm(loader, ncols=80, desc=f"{args.dataset} k={args.shots} fold={fold}")):
        if args.max_episodes > 0 and idx >= args.max_episodes:
            break
        for ref_img, ref_mask in zip(batch["ref_imgs"], batch["ref_masks"]):
            model.set_reference(ref_img, ref_mask)
        model.set_target(batch["tgt_img"])
        pred_mask = model.segment()

        # Evaluate at the prediction's resolution: GT and ignore masks are nearest-resized onto it.
        tgt_mask = F.interpolate(batch["tgt_mask"].unsqueeze(0).unsqueeze(0).float(),
                                 size=pred_mask.shape, mode="nearest").squeeze(0).squeeze(0) > 0.5
        tgt_ignore_idx = batch.get("tgt_ignore_idx")
        if tgt_ignore_idx is not None:
            tgt_ignore_idx = F.interpolate(tgt_ignore_idx.unsqueeze(0).unsqueeze(0).float(),
                                           size=pred_mask.shape, mode="nearest").squeeze(0).squeeze(0).long()
        area_inter, area_union = classify_prediction(pred_mask, tgt_mask, tgt_ignore_idx)
        meter.update(area_inter.to(device), area_union.to(device), batch["class_id"].to(device))
        n_done += 1
    miou, fbiou = meter.compute_iou()
    print(f"{args.dataset} k={args.shots} fold={fold}: mIoU = {miou:.2f}, fbIoU = {fbiou:.2f}")
    return {"fold": fold, "episodes": n_done, "miou": miou, "fbiou": fbiou}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(FOLDS))
    p.add_argument("--shots", type=int, required=True)
    p.add_argument("--folds", default="", help="comma-separated folds; default: all folds of the benchmark")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", default=os.environ.get("DINO_WEIGHTS", ""),
                   help="DINOv3 ViT-L/16 checkpoint (default: $DINO_WEIGHTS)")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""),
                   help="benchmark root (default: $DATA_ROOT)")
    p.add_argument("--output", required=True, help="results.json to write")
    args = p.parse_args()
    if not args.weights:
        sys.exit("pass --weights or set $DINO_WEIGHTS")
    if not args.data_root:
        sys.exit("pass --data-root or set $DATA_ROOT")
    folds = ([int(f) for f in args.folds.split(",") if f.strip()] if args.folds
             else list(range(FOLDS[args.dataset])))

    device = torch.device(args.device)
    model = build_frost(args.weights, image_size=args.image_size, device=str(device))

    per_fold = [run_fold(model, args, fold, device) for fold in folds]
    result = {
        "dataset": args.dataset, "shots": args.shots, "seed": args.seed,
        "image_size": args.image_size, "max_episodes": args.max_episodes,
        "folds": folds, "per_fold": per_fold,
        "miou": sum(r["miou"] for r in per_fold) / len(per_fold),
        "fbiou": sum(r["fbiou"] for r in per_fold) / len(per_fold),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.dataset} k={args.shots}: mIoU = {result['miou']:.2f} "
          f"(mean over {len(folds)} fold(s)) -> {out}")


if __name__ == "__main__":
    main()
