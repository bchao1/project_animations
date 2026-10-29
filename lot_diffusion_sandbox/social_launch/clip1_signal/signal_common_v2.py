"""V1 v2 (post 1): control signal → token layout (signal colours) → noisy token layout → image → mean-colour layout →
shrink and settle (settle2d v3 recipe, re-simulated for the new images by sim_v2.py).

v2 (user feedback on v1, 2026-10-01): 3 columns × 2 rows by signal (depth map | semantic mask | bounding box), better
image picks (fine tokens on the subject), the bounding box in orange so it differs from the blue mask, the signal is
on screen from frame 0 with its full 64 × 64 grid labelled "4,096 tokens", holds between the steps (2 s on the token
layout, 2-s denoise), no captions or footnotes (only column names and token counts), bigger tiles filling the frame,
no speedup labels.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "settle2d_clip"))
import settle2d_common_v3 as B          # noqa: E402  (data loader, physics schedule)

ROOT = B.ROOT
GEN = os.path.join(HERE, "data", "generated")
TILES = os.path.join(GEN, "tiles_v2")
PHYS = os.path.join(GEN, "physics_v2.npz")
N, FPS, FULL = B.N, B.FPS, B.FULL
# (id, source, subject); index = 2 · column + row
ITEMS = [("000000000283", "dof", "wine"), ("000000000294", "dof", "man"),
         ("000000000154", "sam3_mask", "zebra"), ("000000000201", "sam3_mask", "snowboard"),
         ("000000000025", "sam3_box", "giraffe"), ("000000000307", "sam3_box", "dog")]
NI = len(ITEMS)
COLUMNS = ["depth map", "semantic mask", "bounding box"]

# ------------------------------------------------------------------ geometry (1080p pixels; manim = px / 135)
TP = 384                                   # tile side (6 px per cell)
CELL = TP / N
COL_X = [960 - 580, 960, 960 + 580]        # tile centres (room for the % labels on the left)
ROW_Y0 = [83, 577]                         # tile tops: 26-px top margin + 43-px header + 14-px gap
HEAD_Y = 47.5                              # column names (centre)
LABEL_DY = 49                              # count centre below the tile: 10-px gap + half its 78-px height


def tile_origin_px(i):
    return COL_X[i // 2] - TP / 2, ROW_Y0[i % 2]


def px_to_m(x, y):
    return x / 135.0 - 64 / 9, 4.0 - y / 135.0


# ------------------------------------------------------------------ timeline (s)
STAGGER = 0.06
SWEEP = (1.0, 2.0)                 # signal (+ full 64 × 64 grid) → token layout in the signal's colours
TO_NOISE = (4.0, 4.5)              # after a 2-s hold: token fills → noise ("noisy token layout")
DENOISE = (4.5, 6.5)               # 2 s: noise → generated image inside the token boxes
OPEN = (6.0, 6.5)                  # box gaps close: the clean image
OFFSET = 5.6                       # settle2d v3 time = clip time − OFFSET from here on
TOKENIZE = (B.TOKENIZE[0] + OFFSET, B.TOKENIZE[1] + OFFSET)     # 7.6 – 8.6: image → mean-colour layout
SHRINK = (B.SHRINK[0] + OFFSET, B.SHRINK[1] + OFFSET)           # 9.3 – 10.1
FALL0 = B.FALL[0] + OFFSET                                      # 10.2
LIQUEFY0 = B.LIQUEFY[0] + OFFSET
COLLAPSE = (B.COLLAPSE[0] + OFFSET, B.COLLAPSE[1] + OFFSET)     # 15.4 – 16.2
DURATION = 17.5

ease, smooth = B.ease, B.smooth

BOX_FILL, BOX_EDGE = (245, 150, 70), (205, 105, 35)            # bounding box: orange (mask stays blue)
CUE_BLUE = np.array([93, 166, 214])


def load_item(i):
    iid, src, _ = ITEMS[i]
    img = Image.open(os.path.join(ROOT, "data", "img", f"img_{iid}_{src}_generated_4b.png")).convert("RGB")
    import json
    rects = np.array(json.load(open(os.path.join(ROOT, "data", "layouts", f"img_{iid}_{src}.json")))["rects"], int)
    return img, rects


def load_cue(i):
    iid, src, _ = ITEMS[i]
    a = np.asarray(Image.open(os.path.join(ROOT, "data", "img", f"img_{iid}_{src}_cue.png")).convert("RGB"), np.float32)
    if src == "sam3_box":                    # recolour the blue box orange, darker 6-px edge
        inside = np.abs(a - CUE_BLUE).sum(-1) < 40
        out = np.where(inside[..., None], np.array(BOX_FILL, np.float32), a)
        e = np.zeros_like(inside)
        for d in range(1, 7):
            e |= inside & ~(np.roll(inside, d, 0) & np.roll(inside, -d, 0) & np.roll(inside, d, 1) & np.roll(inside, -d, 1))
        out[e] = BOX_EDGE
        a = out
    return Image.fromarray(a.astype(np.uint8))


def patch_settle():
    """Point settle2d_common_v3 at these six items (for its Tokens class and sim2d_v3)."""
    B.ITEMS, B.NI, B.load_item = ITEMS, NI, load_item


class Tokens(B.Tokens):
    """settle2d tokens + the mean control-signal colour of every token."""

    def __init__(self, i):
        patch_settle()
        super().__init__(i)
        self.src = ITEMS[i][1]
        self.cue = load_cue(i)
        a = np.asarray(self.cue.resize((N * 16, N * 16), Image.BOX), np.float64)
        cell = a.reshape(N, 16, N, 16, 3).mean((1, 3))
        self.cue_color = np.array([cell[a0:a1, b0:b1].mean((0, 1)) for a0, a1, b0, b1 in self.rects])
