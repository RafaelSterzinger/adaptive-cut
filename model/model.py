"""Density-ratio few-shot segmentation with an adaptive decision threshold.

Pipeline (all closed-form, no training):

  A. frozen DINOv3 ViT-L/16 patch tokens -> L2-normalise -> SVD positional
     debias -> within-class shrinkage whitening (from the support set)
  B. score field  S = ell + lambda * fwd
       ell : vMF-kernel log density ratio  log p(t|FG) - log p(t|BG)
       fwd : support->query attention log-mass (centred)
  C. adaptive threshold: Glasbey valley of the score histogram inside the
     window {S > 0} (or the whole field if its valley is deeper), shrunk by
     1 / (1 + a * pi_Otsu)
  D. sub-patch upsample -> threshold -> AND with the reciprocal-kNN veto

The pipeline builds on FROST (Park, 2026; https://github.com/jhpark-ai/FROST,
Apache-2.0): stage A and the kernel density ratio are FROST's; the score
fusion, the adaptive threshold and the simplified veto are this work's.
"""
from __future__ import annotations

import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data import build_transform, downsample_mask, load_image, load_mask, upsample_mask
from .density import kde_log_posterior
from .threshold import PI_SHRINK_A, windowed_valley_tau

# --- hyper-parameters (fixed across all datasets, domains and shot counts) ---
SVD_COMPONENTS = 250    # rank of the positional subspace projected out
SHRINK_LAMBDA = 0.95    # whitening shrinkage toward (mean diagonal) * I
BACKWARD_TOP_K = 3      # reciprocal-kNN veto: top-k support patches per query patch
CAND_BWD_MIN_FG = 1     # ... of which at least this many must be foreground
ATT_T = 0.1             # attention temperature of the fwd term
SIGMA = 0.05            # vMF kernel bandwidth of the density ratio
SCORE_LAMBDA = 0.5      # weight of fwd in the score  S = ell + lambda * fwd


class FROST(nn.Module):
    """Training-free few-shot segmenter.

    Args:
        encoder: ``model.encoder.DINOv3FeatureExtractor``.
        raw_encoder: the underlying hub backbone (noise-image forward for the
            positional basis). Defaults to ``encoder.encoder``.
        image_size: input resolution (default 1024 -> 64x64 token grid).
        device: torch device string.
    """

    def __init__(self, encoder: nn.Module, raw_encoder: nn.Module | None = None,
                 image_size: int = 1024, device: str = "cuda"):
        super().__init__()
        self._extract_features = encoder
        self._raw_encoder = raw_encoder if raw_encoder is not None else encoder.encoder
        self.device = device
        self.image_size = image_size
        self._transform = build_transform(image_size)
        self.positional_basis = self._build_positional_basis(device)
        self.reset_state()

    # ──────── episode state ────────
    def reset_state(self) -> None:
        self._ref_raw: list = []      # (PIL image, (1, H, W) bool mask) per support
        self._tgt_image = None
        self._orig_tgt_size = None
        self._post_ell = None         # score field (h, w)
        self._post_cand = None        # statistics window (h, w)
        self._post_veto = None        # reciprocal-kNN veto (h, w)

    def set_reference(self, image, mask) -> None:
        """Add a support image + binary mask (nonzero = FG). Call k times for k-shot."""
        mask_tensor = load_mask(mask, self.image_size, self.device)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        self._ref_raw.append((image, mask_tensor))

    def set_target(self, image) -> None:
        """Set the query image."""
        img_tensor, self._orig_tgt_size = load_image(image, self._transform, self.device)
        self._tgt_image = img_tensor

    @torch.no_grad()
    def segment(self) -> torch.Tensor:
        """Segment the query from the set support(s). Returns (H, W) bool at the query's size."""
        if not self._ref_raw or self._tgt_image is None:
            raise RuntimeError("segment() requires reference image(s)+mask(s) and a target.")
        ref_images = torch.cat(
            [load_image(img, self._transform, self.device)[0] for img, _ in self._ref_raw], dim=0)
        ref_masks = torch.cat([m for _, m in self._ref_raw], dim=0)
        pred = self.predict_mask(ref_images, ref_masks, self._tgt_image)
        self.reset_state()
        return pred

    @torch.no_grad()
    def segment_one(self, ref_pil, ref_mask, query_pil) -> torch.Tensor:
        """Convenience 1-shot call."""
        self.reset_state()
        self.set_reference(ref_pil, ref_mask)
        self.set_target(query_pil)
        return self.segment()

    # ──────── A: positional debias ────────
    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Top-r SVD directions of a noise-image token map = positional subspace."""
        from torchvision.transforms.functional import normalize
        enc = self._raw_encoder
        _dt = next(enc.parameters()).dtype
        noise_img = normalize(
            torch.zeros(1, 3, self.image_size, self.image_size),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ).to(device, dtype=_dt)
        noise_fmaps = enc.to(device).get_intermediate_layers(
            noise_img, n=1, reshape=True)[0].float()
        noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)
        E = einops.rearrange(noise_fmaps, "b c h w -> c (b h w)")
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        return U[:, :SVD_COMPONENTS].contiguous()

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)
        basis = self.positional_basis.to(X.device).to(X.dtype)
        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
        return F.normalize(X_deb, p=2, dim=2)

    # ──────── A: within-class shrinkage whitening ────────
    @torch.no_grad()
    def _shrinkage_whiten(self, feat_refs_deb, feat_tgt_deb, ref_masks, S, hs, ws):
        """f' = Sigma_W^{-1/2} f, Sigma_W the pooled FG/BG support covariance
        shrunk toward (mean diagonal) * I. No post-whitening renorm."""
        fg_list, bg_list = [], []
        for s_ in range(S):
            mask_s = downsample_mask(ref_masks[s_:s_ + 1], hs, ws)
            fg_s = feat_refs_deb[0, s_, :, mask_s]
            if fg_s.shape[1] > 0:
                fg_list.append(fg_s)
            bg_s = feat_refs_deb[0, s_, :, ~mask_s]
            if bg_s.shape[1] > 0:
                bg_list.append(bg_s)
        if not (fg_list and bg_list):
            return feat_refs_deb, feat_tgt_deb
        fg_all = torch.cat(fg_list, dim=1).float()   # (C, N_fg)
        bg_all = torch.cat(bg_list, dim=1).float()   # (C, N_bg)
        C = fg_all.shape[0]
        Xf = fg_all - fg_all.mean(dim=1, keepdim=True)
        Xb = bg_all - bg_all.mean(dim=1, keepdim=True)
        nrm = max(fg_all.shape[1] + bg_all.shape[1] - 2, 1)
        Sig = (Xf @ Xf.T + Xb @ Xb.T) / nrm
        mu = Sig.diagonal().mean()
        lam = SHRINK_LAMBDA
        Sig = (1 - lam) * Sig + lam * mu * torch.eye(C, device=Sig.device, dtype=Sig.dtype)
        evals, evecs = torch.linalg.eigh(Sig)
        inv_sqrt = evals.clamp(min=1e-6).rsqrt()
        W = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T
        W = W.to(feat_refs_deb.dtype)
        fr = feat_refs_deb.permute(0, 1, 3, 4, 2) @ W  # (B,n,h,w,C)
        ft = feat_tgt_deb.permute(0, 2, 3, 1) @ W      # (B,h,w,C)
        return fr.permute(0, 1, 4, 2, 3), ft.permute(0, 3, 1, 2)

    # ──────── B: correspondence term fwd ────────
    def _correspondence_fields(self, feat_refs, feat_tgt, ref_masks, S, hs, ws, h, w):
        """fwd: support->query attention log-mass, centred so mean_q exp(fwd) = 1.
        Computed on the unwhitened debiased features. Returns (h, w)."""
        C = feat_tgt.shape[0]
        tq = F.normalize(feat_tgt.reshape(C, -1).T.float(), p=2, dim=1)
        fg_u = []
        for s in range(S):
            mask_s = downsample_mask(ref_masks[s:s + 1], hs, ws)
            fg_u.append(feat_refs[s][:, mask_s].T)
        fg_u = F.normalize(torch.cat(fg_u, 0).float(), p=2, dim=1)
        n_fg, n_q = fg_u.shape[0], tq.shape[0]
        sim_fg = tq @ fg_u.T / ATT_T
        att = torch.softmax(sim_fg, dim=0)
        fwd = torch.log(att.sum(dim=1) + 1e-12) - math.log(n_fg / n_q)
        return fwd.view(h, w)

    # ──────── D: reciprocal-kNN veto ────────
    def _locate_candidates(self, sim_maps, ref_masks, h, w) -> torch.Tensor:
        """Per support view, a query patch votes FG if at least CAND_BWD_MIN_FG
        of its top-k most similar support patches are FG; majority over views."""
        k = len(sim_maps)
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]                        # (Hs, Ws, h, w)
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)  # (h, w, Hs, Ws)
            ref_mask_m = downsample_mask(ref_masks[m:m + 1], Hs, Ws).squeeze(0)
            flat = sim_t_to_r.reshape(h, w, -1)
            K_eff = min(BACKWARD_TOP_K, flat.shape[-1])
            _, top_idx = flat.topk(K_eff, dim=-1)
            rows = top_idx // Ws
            cols = top_idx % Ws
            fg_votes = ref_mask_m[rows, cols].sum(dim=-1)
            votes += (fg_votes >= CAND_BWD_MIN_FG).to(torch.int32)
        return votes >= math.ceil(k / 2)

    # ──────── full forward ────────
    @torch.no_grad()
    def predict_mask(self, ref_images: torch.Tensor, ref_masks: torch.Tensor,
                     tgt_image: torch.Tensor) -> torch.Tensor:
        """ref_images (S, C, H, W), ref_masks (S, H, W) bool, tgt_image (1, C, H, W).
        Returns (H, W) bool at the query's original size."""
        self._post_ell = self._post_cand = self._post_veto = None
        S = ref_images.shape[0]

        # A: features -> L2 -> positional debias (support and query in one batch)
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)   # (1, S+1, C, H, W)
        fmaps_norm = F.normalize(self._extract_features(imgs), p=2, dim=2)
        fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]
        _, _, C, hs, ws = feat_refs_deb.shape
        h, w = feat_tgt_deb.shape[-2:]
        ref_masks = ref_masks.unsqueeze(1)                              # (S, 1, H, W)

        # No FG patch in any support view: all-background.
        if not any(downsample_mask(ref_masks[s:s + 1], hs, ws).any() for s in range(S)):
            return self._all_background(tgt_image)

        # B: fwd on the unwhitened features
        fwd = self._correspondence_fields(
            feat_refs_deb[0], feat_tgt_deb[0], ref_masks, S, hs, ws, h, w)

        # A: shrinkage whitening
        feat_refs_deb, feat_tgt_deb = self._shrinkage_whiten(
            feat_refs_deb, feat_tgt_deb, ref_masks, S, hs, ws)

        # D: veto from whitened features
        sim_maps = [torch.einsum("bchw,bcxy->bhwxy", feat_refs_deb[:, m], feat_tgt_deb)
                    for m in range(S)]
        candidate_mask = self._locate_candidates(sim_maps, ref_masks, h, w)

        # B: FG / BG anchors
        fg_collect, bg_collect = [], []
        for s_ in range(S):
            feat_s = feat_refs_deb[0, s_]
            mask_s = downsample_mask(ref_masks[s_:s_ + 1], hs, ws)
            fg_s = feat_s[:, mask_s]
            if fg_s.shape[1] > 0:
                fg_collect.append(fg_s.T)
            bg_s = feat_s[:, ~mask_s]
            if bg_s.shape[1] > 0:
                bg_collect.append(bg_s.T)
        if not bg_collect:
            # The support says the whole image is FG: score from fwd alone.
            self._post_ell = fwd
            self._post_cand = fwd > 0
            return self._finalize_mask(tgt_image)
        fg_anchors = F.normalize(torch.cat(fg_collect, dim=0).float(), p=2, dim=1)
        bg_anchors = F.normalize(torch.cat(bg_collect, dim=0).float(), p=2, dim=1)
        feat_t_white = F.normalize(
            feat_tgt_deb[0].reshape(C, -1).permute(1, 0).float(), p=2, dim=1)

        # B: density ratio, score fusion, statistics window
        ell = kde_log_posterior(feat_t_white, fg_anchors, bg_anchors, SIGMA).view(h, w)
        score = ell + SCORE_LAMBDA * fwd.to(ell.device)
        self._post_ell = score
        self._post_cand = score > 0
        if candidate_mask.sum() > 0:      # an empty veto vetoes nothing
            self._post_veto = candidate_mask
        return self._finalize_mask(tgt_image)

    # ──────── C: adaptive threshold ────────
    @torch.no_grad()
    def _decide_threshold(self, ell_up, cand) -> float:
        """Windowed-valley threshold on the upsampled score field (no ground
        truth enters). Falls back to tau = 0 on an empty window."""
        cand_in = upsample_mask(cand.to(ell_up.device), *ell_up.shape)
        vals = ell_up[cand_in & torch.isfinite(ell_up)]
        if vals.numel() == 0:
            return 0.0
        m_sorted = np.sort(vals.detach().cpu().numpy().astype(np.float64))
        m_full = None
        full = ell_up[torch.isfinite(ell_up)]
        if full.numel() > 0:
            m_full = np.sort(full.detach().cpu().numpy().astype(np.float64))
        return float(windowed_valley_tau(m_sorted, a=PI_SHRINK_A, m_full=m_full))

    # ──────── D: upsample -> threshold -> veto ────────
    def _all_background(self, tgt_image: torch.Tensor) -> torch.Tensor:
        self._post_ell = self._post_cand = self._post_veto = None
        return torch.zeros(tuple(self._orig_tgt_size), dtype=torch.bool,
                           device=tgt_image.device)

    @torch.no_grad()
    def _finalize_mask(self, tgt_image: torch.Tensor) -> torch.Tensor:
        H, W = tgt_image.shape[-2:]
        ell = self._post_ell.to(tgt_image.device).float()
        ell_up = F.interpolate(ell[None, None], size=(H, W),
                               mode="bilinear", align_corners=False)[0, 0]
        tau = self._decide_threshold(ell_up, self._post_cand)
        up = ell_up > tau
        if self._post_veto is not None:
            up = up & upsample_mask(self._post_veto, H, W)
        self._post_ell = self._post_cand = self._post_veto = None
        return upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
