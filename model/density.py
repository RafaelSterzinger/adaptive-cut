"""Kernel density ratio on the unit sphere (from FROST, Park 2026).

Each class is a von Mises-Fisher kernel density estimate over its support
anchors; a query patch is scored by  ell(t) = log p(t|FG) - log p(t|BG).
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def _chunked_lse_sim(A: torch.Tensor, B: torch.Tensor, sigma: float,
                     log_count: float) -> torch.Tensor:
    """Memory-safe ``logsumexp((A @ B.T - 1)/sigma, dim=1) - log_count``.

    Splits A's rows so the (rows x anchors) block stays within a fixed element
    budget; logsumexp reduces over the full anchor dim per chunk, so the result
    is bit-identical to the un-chunked form.
    """
    Nt = A.shape[0]
    Nc = max(B.shape[0], 1)
    s = max(sigma, 1e-6)
    budget = 150_000_000  # ~600 MB fp32 per block
    chunk = max(64, min(Nt, int(budget // Nc)))
    if chunk >= Nt:
        sim = (A @ B.T - 1.0) / s
        return torch.logsumexp(sim, dim=1) - log_count
    parts = []
    for i in range(0, Nt, chunk):
        blk = (A[i:i + chunk] @ B.T - 1.0) / s
        parts.append(torch.logsumexp(blk, dim=1))
    return torch.cat(parts, 0) - log_count


def kde_log_posterior(
    feat_tgt: torch.Tensor,   # (N_t, D), L2-normalized
    feat_fg: torch.Tensor,    # (N_fg, D), L2-normalized
    feat_bg: torch.Tensor,    # (N_bg, D), L2-normalized
    sigma: float,
) -> torch.Tensor:
    """Per-patch log density ratio  log p(t|FG) - log p(t|BG). Returns (N_t,)."""
    log_p_fg = _chunked_lse_sim(feat_tgt, feat_fg, sigma, math.log(max(feat_fg.shape[0], 1)))
    log_p_bg = _chunked_lse_sim(feat_tgt, feat_bg, sigma, math.log(max(feat_bg.shape[0], 1)))
    return log_p_fg - log_p_bg
