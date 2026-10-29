"""V2 v3 (post 2): image and video side by side — quadtree subdivision → noisy token layout → image / first frame →
mean-colour layout → (video: frames stack in time) → both settle. No captions.

v3 (user feedback on v2, 2026-10-01): no subtitles; the subdivision is drawn in ../../quadtree_clip's style (dim grey
detail silhouette, thin white cell outlines appearing level by level); then the same generation steps as V1 (noisy
token layout → 2-s denoise inside the token boxes → mean-colour layout); counts as in V1 ("4,096 tokens · 100% of
4,096" on the full grid, counting down when the tokens form). No thumbnails.
Left: corgi (painted detail map, 1,546 of 4,096 tokens, physics_left_v1.npz reused). Right: latent frame 1 of
spacetime item 31 (dof layout), then the v16 colour volume vol11c_31 from its t = VOL0 on (unchanged frames).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SB, "spacetime_clip"))
os.environ["LOT_COLOR"] = "image"
os.environ.setdefault("LOT_ITEM", "31")
import spacetime_common_v8 as ST      # noqa: E402

GEN = os.path.join(HERE, "data", "generated")
FRAMES = os.path.join(GEN, "frames_v3")
PHYS_L = os.path.join(GEN, "physics_left_v1.npz")
VOL = ST.VOL                                  # vol11c_31
FPS = 60
N = 64

# ------------------------------------------------------------------ timeline (global s)
SPLIT = {32: 0.4, 16: 0.8, 8: 1.2, 4: 1.6, 2: 2.0, 1: 2.4}     # children of nominal size s appear
SPLIT_LEN = 0.3
UNDER_OUT = (3.7, 4.2)            # silhouette + outlines fade as the noisy tokens appear
NOISY = (3.7, 4.2)                # quadtree cells → noise-filled token boxes
COUNT = (3.7, 4.7)                # counts run down from the full grid to the layout
DENOISE = (4.7, 6.7)              # 2 s: noise → image (left) / latent frame 1 (right) inside the boxes
OPEN = (6.2, 6.7)                 # box gaps close
TOKENIZE = (7.4, 8.2)             # diagonal sweep → mean-colour token layout
VOL0 = 1.0                        # spacetime v16 time at which the volume takes over (camera still face-on)
PRE = 8.6                         # global time of v16 t = VOL0
XFADE = (PRE - 0.1, PRE + 0.2)


def g(tv):
    """spacetime v16 time → global time."""
    return PRE + tv - VOL0


SHRINK = (g(ST.SHRINK[0]), g(ST.SHRINK[1]))
FALL0 = g(ST.FALL[0])
LOFF = FALL0 - 4.6                # settle2d v3 local time = t − LOFF (its fall starts at 4.6)
LIQ0 = g(ST.LIQUEFY[0])
GHOST1 = g(ST.GHOST[1])
COLLAPSE = (g(ST.COLLAPSE[0]), g(ST.COLLAPSE[1]))
DURATION = g(ST.DURATION)         # 21.4 s

ease, smooth = ST.ease, ST.smooth
LINE = (235, 235, 235)            # quadtree outlines (quadtree_clip: white)
SIL = 110                         # dim silhouette grey (quadtree_clip mask_dim)

# ------------------------------------------------------------------ geometry (manim units; 135 px / unit)
PX = 135.0
L_C, L_SIDE = (-3.35, 0.15), 4.8          # room on the left for the % label
V14_C, V14_W = (ST.REGION[0], ST.REGION[1]), ST.REGION[2]
R_C, R_W = (3.55, 0.15), 7.0
R_S = R_W / V14_W
R_H = ST.REGION[3] * R_S
HEAD_Y = 3.45
COUNT_Y = -2.78


def to_px(x, y):
    return (x + 64 / 9) * PX, (4.0 - y) * PX


def v2_from_v14(p):
    return np.array([R_C[0] + (p[0] - V14_C[0]) * R_S, R_C[1] + (p[1] - V14_C[1]) * R_S, 0.0])


def vol_px(tv, w):
    proj, view, _ = ST.matrices(tv)
    q = proj @ view @ np.array([*w, 1.0])
    x, y = q[0] / q[3], q[1] / q[3]
    return (x + 1) / 2 * ST.RES[0], (1 - y) / 2 * ST.RES[1]


def left_level_y(level):
    return L_C[1] - L_SIDE / 2 + level * L_SIDE / N


# ------------------------------------------------------------------ data
def corgi_rects():
    return np.array(json.load(open(os.path.join(SB, "data", "layouts", "paint0.json")))["rects"], int)


CORGI_IMG = os.path.join(SB, "data", "img", "paint0_result.png")
CORGI_MAP = os.path.join(SB, "data", "img", "paint0_canvas.png")


def quadtree(rects, H, W, root):
    """Nodes (y0, y1, x0, x1, size, is_leaf, born): cells split into four until they are a leaf of `rects`."""
    leaves = {tuple(r) for r in map(tuple, rects)}
    out = []

    def rec(y0, x0, s, born):
        if y0 >= H or x0 >= W:
            return
        y1, x1 = min(y0 + s, H), min(x0 + s, W)
        leaf = (y0, y1, x0, x1) in leaves
        out.append((y0, y1, x0, x1, s, leaf, born))
        if not leaf:
            h = s // 2
            for dy in (0, h):
                for dx in (0, h):
                    rec(y0 + dy, x0 + dx, h, SPLIT[h])
    for y in range(0, H, root):
        for x in range(0, W, root):
            rec(y, x, root, SPLIT.get(root, 0.0) if root < 64 else 0.0)
    assert sum(n[5] for n in out) == len(leaves)
    return out
