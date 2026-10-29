"""V2 (post 2): quadtree of an image and a video, side by side, then both settle (2D particles left, 3D right).

Left: the site's running example (girl riding a corgi, painted detail map, 1,546 of 4,096 tokens). Right:
../../spacetime_clip v16 colour variant (item 31, 21 latent frames × 45 × 80, 27,795 of 75,600 tokens), its GL volume
frames vol11c_31/ reused unchanged after a quadtree prefix of PRE seconds. Every token, in both panels, is filled with
its mean image / video colour (user, 2026-10-01: colourised tokens, not the blue / green extent colours).
Quadtrees are read off the exact LoT leaves (all leaves are aligned powers of two): a cell splits into four until it
is a leaf; a leaf takes its size colour the moment it is created. Video leaves are per latent frame (no token spans
time), so the right panel's quadtree is drawn on latent frame 1 only, then the frames step out in time.
Left physics: settle2d v3 recipe on the corgi (sim_left_v1.py), time-shifted so its fall starts with the 3D fall;
v16's 3D physics (sim_physics_v3, spacetime_common_v8) is the same recipe on the same schedule relative to the fall.
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
FRAMES = os.path.join(GEN, "frames_v1")
PHYS_L = os.path.join(GEN, "physics_left_v1.npz")
VOL = ST.VOL                                  # vol11c_31: v16, mean-colour tokens
FPS = 60
PRE = 3.0                                     # quadtree prefix before spacetime v14's t = 0
DURATION = PRE + ST.DURATION                  # 16.8 s
N = 64

# ------------------------------------------------------------------ timeline (global s)
SPLIT = {32: 0.30, 16: 0.65, 8: 1.00, 4: 1.35, 2: 1.70, 1: 2.05}    # children of nominal size s appear
SPLIT_LEN = 0.30
UNDER_OUT = (2.3, 2.9)                        # dim detail-map underlay fades out
XFADE = (PRE - 0.05, PRE + 0.3)               # right: drawn quadtree → GL volume frame 0
THUMBS = (PRE, PRE + 0.5)                     # generated image / video thumbnails + counts fade in
SHRINK = (PRE + ST.SHRINK[0], PRE + ST.SHRINK[1])
FALL0 = PRE + ST.FALL[0]
LOFF = FALL0 - 4.6                            # settle2d v3 local time = t − LOFF (its fall starts at 4.6)
LIQ0 = PRE + ST.LIQUEFY[0]
GHOST1 = PRE + ST.GHOST[1]
COLLAPSE = (PRE + ST.COLLAPSE[0], PRE + ST.COLLAPSE[1])
CAPS = [(0.0, "Like a quadtree in graphics, LoT splits only where the layout asks for detail."),
        (PRE, "An image gets one token grid; a video gets one per latent frame."),
        (PRE + 2.8, "Stacked in time, the video's fine tokens follow her."),
        (SHRINK[0], "Now shrink every token to a single cell…"),
        (PRE + 8.6, "…and let them settle: the height is the token fraction.")]

ease, smooth = ST.ease, ST.smooth

LINE = (95, 95, 95)                           # quadtree cell outlines

# ------------------------------------------------------------------ geometry (manim units; 1080p: 135 px / unit)
PX = 135.0
L_C, L_SIDE = (-3.6, -0.55), 3.6              # left tile centre, side
V14_C, V14_W = (ST.REGION[0], ST.REGION[1]), ST.REGION[2]
R_C, R_W = (3.6, -0.45), 6.7                  # right: volume region centre, width
R_S = R_W / V14_W
R_H = ST.REGION[3] * R_S
IMG_TH = ((-5.6, 2.75), 1.3)                  # thumbnail centre, height
VID_TH = ((1.55, 2.75), 1.3)


def to_px(x, y):
    return (x + 64 / 9) * PX, (4.0 - y) * PX


def v2_from_v14(p):
    """spacetime v14 manim point → V2 manim point (the volume region is scaled and moved)."""
    return np.array([R_C[0] + (p[0] - V14_C[0]) * R_S, R_C[1] + (p[1] - V14_C[1]) * R_S, 0.0])


def vol_px(t, w):
    """World point → pixel in a vol11c frame (ST.RES) at spacetime time t."""
    proj, view, _ = ST.matrices(t)
    q = proj @ view @ np.array([*w, 1.0])
    x, y = q[0] / q[3], q[1] / q[3]
    return (x + 1) / 2 * ST.RES[0], (1 - y) / 2 * ST.RES[1]


def left_level_y(level):
    """manim y of `level` rows above the left tile's floor."""
    return L_C[1] - L_SIDE / 2 + level * L_SIDE / N


# ------------------------------------------------------------------ data
def corgi_rects():
    return np.array(json.load(open(os.path.join(SB, "data", "layouts", "paint0.json")))["rects"], int)


CORGI_IMG = os.path.join(SB, "data", "img", "paint0_result.png")
CORGI_MAP = os.path.join(SB, "data", "img", "paint0_canvas.png")


def quadtree(rects, H, W, root):
    """Nodes (y0, y1, x0, x1, size, is_leaf, born) of the quadtree whose leaves are `rects`, roots = blocks of
    nominal size `root` tiling H × W (clipped). born = time the node's box appears (its parent's split)."""
    leaves = {tuple(r) for r in map(tuple, rects)}
    out = []

    def rec(y0, x0, s, born):
        y1, x1 = min(y0 + s, H), min(x0 + s, W)
        if y0 >= H or x0 >= W:
            return
        leaf = (y0, y1, x0, x1) in leaves
        out.append((y0, y1, x0, x1, s, leaf, born))
        if not leaf:
            assert s > 1, (y0, y1, x0, x1)
            h = s // 2
            for dy in (0, h):
                for dx in (0, h):
                    rec(y0 + dy, x0 + dx, h, SPLIT[h])
    for y in range(0, H, root):
        for x in range(0, W, root):
            rec(y, x, root, SPLIT.get(root, 0.0) if root < 64 else 0.0)
    got = sum(n[5] for n in out)
    assert got == len(leaves), (got, len(leaves))
    return out
