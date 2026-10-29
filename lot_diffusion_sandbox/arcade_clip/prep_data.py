"""Derive per-frame token levels for the platformer video and stage counts for the giraffe depth example.

Platformer: generated-grid frames colour each token cell (green 1x1, yellow 2x2, red 4x4) on the 80 x 45 latent token grid.
Writes data/platformer_levels.npy (81, 45, 80) uint8 and prints token counts per stage.
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "data/clips/vd_platformer_grid")

levels = np.zeros((81, 45, 80), np.uint8)
for f in range(81):
    a = np.asarray(Image.open(os.path.join(GRID, f"{f + 1:04d}.jpg")).convert("RGB")).astype(int)
    c = a[8::16, 8::16][:45, :80]                       # 16-px token cells, sample centres
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    lv = np.full(c.shape[:2], 2, np.uint8)               # yellow
    lv[(g > r + 40)] = 1                                 # green
    lv[(r > g + 60)] = 4                                 # red
    levels[f] = lv
np.save(os.path.join(HERE, "data/platformer_levels.npy"), levels)


def count(lv, cap):
    """tokens per frame when merges are capped at `cap` (1, 2, 4)."""
    n = 0
    for s in (1, 2, 4):
        m = (np.minimum(lv, cap) == s).sum()
        n += m / (s * s)
    return n


for cap in (1, 2, 4):
    tot = sum(count(levels[f], cap) for f in [0] + list(range(1, 81, 4)))   # one frame per latent frame
    print("platformer cap", cap, "tokens/video", tot, "compression", 75600 / tot)
u, c = np.unique(levels, return_counts=True)
print("level hist", dict(zip(u.tolist(), c.tolist())))
# 4x4 alignment check: every red cell's aligned 4x4 block is all red
bad = 0
for f in range(81):
    lv = levels[f]
    for i in range(0, 44, 4):
        for j in range(0, 80, 4):
            blk = lv[i:i + 4, j:j + 4]
            if (blk == 4).any() and not (blk == 4).all():
                bad += 1
print("misaligned 4x4 blocks", bad)

d = json.load(open(os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")))
sizes = [r1 - r0 for r0, r1, c0, c1 in d["rects"]]
for cap in (1, 2, 4, 8):
    print("giraffe cap", cap, "tokens", sum((s // min(s, cap)) ** 2 for s in sizes))
