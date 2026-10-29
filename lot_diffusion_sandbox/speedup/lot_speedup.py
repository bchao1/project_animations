"""Token count → LoT speedup lookup for every clip in lot_diffusion_sandbox (fit of the paper's measured timings).

    import sys; sys.path.insert(0, "<…>/lot_diffusion_sandbox/speedup")
    from lot_speedup import speedup
    speedup(1546)                      # image, of 4,096 tokens → ≈ 2.0
    speedup(27795, modality="video")   # of 75,600 tokens       → ≈ 3.2

Model: relative time per denoising step 1/s = a + b·f + c·f², f = tokens / full, constrained to s(1) = 1
(constant part = text tokens / VAE / overheads, linear = MLPs and projections, quadratic = attention), all three
≥ 0. Non-negative least squares in 1/s over the "fit": true points of speedup_points.json, per modality. It interpolates between and
extrapolates beyond the measured budgets; values far outside the measured range (images below ~750 tokens, video
below ~25k) are extrapolations — label them "≈". Video has only 4 points (25–66% tokens) and its constant term fits to 0, so
below ~25% the video curve grows like 1/f. Regenerate the table / plot with `python fit_speedup.py`.
"""
import json
import os

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
POINTS = json.load(open(os.path.join(HERE, "speedup_points.json")))
FULL = {m: POINTS[m]["full"] for m in ("image", "video")}


def _fit(modality):
    """(a, b, c) of 1/s = a + b f + c f², with a + b + c = 1."""
    pts = [p for p in POINTS[modality]["points"] if p["fit"]]
    f = np.array([p["tokens"] for p in pts]) / FULL[modality]
    y = 1.0 / np.array([p["speedup"] for p in pts])
    # substitute b = 1 − a − c:  y − f = a (1 − f) + c (f² − f); a, c ≥ 0 (and b ≥ 0, checked) so the curve stays
    # physical when extrapolated (unconstrained, the 4 video points give a < 0 and s blows up below ~20% tokens)
    A = np.c_[1 - f, f * f - f]
    (a, c), _ = nnls(A, y - f)
    assert 1 - a - c >= 0, (a, c)
    return float(a), float(1 - a - c), float(c)


COEF = {m: _fit(m) for m in ("image", "video")}


def speedup(tokens, full=None, modality="image"):
    """Estimated wall-clock speedup of LoT over full resolution at `tokens` (of `full`, default per modality)."""
    full = FULL[modality] if full is None else full
    a, b, c = COEF[modality]
    f = np.asarray(tokens, float) / full
    s = 1.0 / (a + b * f + c * f * f)
    return float(s) if np.ndim(s) == 0 else s
