"""Adaptive decision threshold for the score field (the paper's "where to cut").

The Bayes threshold tau = 0 of a density ratio is optimal only under equal
class priors. Instead, the cut is read from the score histogram of each
episode:

    tau = t / (1 + a * pi_Otsu)

where t is the Glasbey valley of the histogram (Otsu's threshold if the
smoothed histogram is not bimodal) and pi_Otsu is Otsu's induced foreground
fraction. The histogram is taken inside the statistics window {S > 0}, or over
the whole field when the whole field's valley is deeper. One constant, a.
"""
from __future__ import annotations

import math

import numpy as np

HIST_BINS = 256             # histogram resolution for every binned estimator
VALLEY_MAX_SMOOTH = 10000   # convergence backstop for the valley smoother
PI_SHRINK_A = 1.0           # the 'a' of tau = t / (1 + a*pi_Otsu)


def hist_of(m_sorted: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(m_sorted[0]), float(m_sorted[-1]), bins + 1)
    if edges[0] >= edges[-1]:
        edges = np.linspace(edges[0] - 1.0, edges[0] + 1.0, bins + 1)
    h = np.histogram(m_sorted, bins=edges)[0].astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h


def otsu_binned(m_sorted: np.ndarray) -> dict:
    """Otsu's between-class-variance threshold, plus the induced pi estimate."""
    c, h = hist_of(m_sorted, HIST_BINS)
    w = h / h.sum()
    cw = np.cumsum(w)
    cm = np.cumsum(w * c)
    mean_t = cm[-1]
    w0 = np.clip(cw, 1e-12, 1 - 1e-12)
    between = (mean_t * w0 - cm) ** 2 / (w0 * (1 - w0))
    k = int(np.argmax(between[:-1]))
    t = float(c[k])
    pi_hat = float((m_sorted > t).sum() / m_sorted.size)
    return {"pi": pi_hat, "t_native": t}


def _n_maxima(h: np.ndarray) -> int:
    keep = np.r_[True, np.diff(h) != 0]
    z = h[keep]
    if z.size == 1:
        return 1
    n = 0
    for i in range(z.size):
        left = z[i - 1] if i > 0 else -math.inf
        right = z[i + 1] if i < z.size - 1 else -math.inf
        if z[i] > left and z[i] > right:
            n += 1
    return n


def valley_glasbey(centers: np.ndarray, h: np.ndarray) -> dict:
    """Glasbey minimum method: smooth to two modes, cut at the valley."""
    hs = h.astype(np.float64).copy()
    k = np.array([1 / 3, 1 / 3, 1 / 3])
    it = 0
    while _n_maxima(hs) > 2 and it < VALLEY_MAX_SMOOTH:
        hs = np.convolve(hs, k, mode="same")
        it += 1
    if _n_maxima(hs) != 2:
        return {"bimodal": False, "t": math.nan, "iters": it, "depth": 0.0}
    peaks = []
    keep_idx = np.r_[True, np.diff(hs) != 0].nonzero()[0]
    z = hs[keep_idx]
    for i in range(z.size):
        left = z[i - 1] if i > 0 else -math.inf
        right = z[i + 1] if i < z.size - 1 else -math.inf
        if z[i] > left and z[i] > right:
            peaks.append(keep_idx[i])
    lo, hi = peaks[0], peaks[1]
    j = lo + int(np.argmin(hs[lo:hi + 1]))
    depth = 1.0 - hs[j] / max(min(hs[lo], hs[hi]), 1e-12)
    return {"bimodal": True, "t": float(centers[j]), "iters": it,
            "depth": float(depth)}


def _valley_of(m_sorted: np.ndarray) -> dict:
    centers, h = hist_of(m_sorted, HIST_BINS)
    return valley_glasbey(centers, h)


def windowed_valley_tau(m_sorted: np.ndarray, a: float = PI_SHRINK_A,
                        m_full: np.ndarray | None = None) -> float:
    """Glasbey valley of the (windowed) score histogram, shrunk by 1/(1+a*pi).

    ``m_sorted`` are the scores inside the statistics window, ``m_full`` the
    whole field's. The histogram is chosen per episode by Glasbey valley
    depth: the window is kept only when its valley is strictly deeper; the
    whole field wins ties. Falls back to Otsu when the smoothed histogram is
    not bimodal. pi is Otsu's induced foreground fraction on the chosen array.
    """
    m, vt = m_sorted, _valley_of(m_sorted)
    if m_full is not None:
        vt_full = _valley_of(m_full)
        if not float(vt["depth"]) > float(vt_full["depth"]):
            m, vt = m_full, vt_full
    ot = otsu_binned(m)
    bimodal = bool(vt["bimodal"] and np.isfinite(vt["t"]))
    t = float(vt["t"]) if bimodal else float(ot["t_native"])
    return t / (1.0 + a * float(ot["pi"]))
