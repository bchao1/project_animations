"""V1 (post 1): control signal → token layout in the signal's colours → generated image → token layout in the
image's colours → shrink and settle. Six images of ../../settle2d_clip (v3 physics, reused unchanged).

New opening (0 – OFFSET + TOKENIZE[0]); from the image → layout sweep on, every tile frame is settle2d v3's
tile_frame(i, t − OFFSET), so the physics file ../../settle2d_clip/data/generated/physics2d_v3.npz is reused as is.
Cues: ../../data/img/img_<id>_<source>_cue.png (depth maps for dof, blue-on-white masks / box).
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SETTLE = os.path.join(HERE, "..", "..", "settle2d_clip")
sys.path.insert(0, SETTLE)
import settle2d_common_v3 as B          # noqa: E402  (settle2d v3: data, geometry, physics timeline)

ROOT = B.ROOT
GEN = os.path.join(HERE, "data", "generated")
TILES = os.path.join(GEN, "tiles_v1")
N, NI, FPS, STAGGER = B.N, B.NI, B.FPS, B.STAGGER
ITEMS, CENTERS, TILE, CELL = B.ITEMS, B.CENTERS, B.TILE, B.CELL
SIGNAL_NAME = {"dof": "depth map", "sam3_mask": "semantic mask", "sam3_box": "bounding box"}

OFFSET = 2.4                       # settle2d v3 time = clip time − OFFSET (from the image → layout sweep on)
DURATION = B.DURATION + OFFSET     # 16.9 s

# ------------------------------------------------------------------ new opening (s)
SIG_IN = (0.0, 0.6)                # control signal fades in
CUE_TOK = (1.0, 2.0)               # diagonal sweep: signal → token layout, fill = mean signal colour per token
TO_NOISE = (2.6, 3.0)              # token fills → noise (inside the token boxes)
DENOISE = (3.0, 4.0)               # noise → generated image, still inside the token boxes
OPEN = (3.6, 4.1)                  # box gaps close: the clean full-resolution image
TOKENIZE = (B.TOKENIZE[0] + OFFSET, B.TOKENIZE[1] + OFFSET)     # 4.4 – 5.4: image → mean-colour layout (settle2d)
SHRINK = (B.SHRINK[0] + OFFSET, B.SHRINK[1] + OFFSET)
LIQUEFY0 = B.LIQUEFY[0] + OFFSET
COLLAPSE = (B.COLLAPSE[0] + OFFSET, B.COLLAPSE[1] + OFFSET)
SPEED_T = B.SPEED_T + OFFSET
CAPS = [(0.0, "Six control signals: depth maps, semantic masks, a bounding box…"),
        (CUE_TOK[0], "…each turned into a multi-resolution token layout."),
        (TO_NOISE[0], "LoT Diffusion generates an image from each layout."),
        (TOKENIZE[0], "The same layouts, now in the images' colours."),
        (SHRINK[0], "Shrink every token to a single cell…"),
        (B.SETTLE[0] + OFFSET, "…and let them settle: the height is the token fraction."),
        (SPEED_T, "25–35% of the tokens, 2.1–2.4× faster.")]

ease, smooth, speedup, level_y = B.ease, B.smooth, B.speedup, B.level_y


def load_cue(i):
    iid, src, _ = ITEMS[i]
    return Image.open(os.path.join(ROOT, "data", "img", f"img_{iid}_{src}_cue.png")).convert("RGB")


class Tokens(B.Tokens):
    """settle2d tokens + the mean control-signal colour of every token."""

    def __init__(self, i):
        super().__init__(i)
        self.src = ITEMS[i][1]
        self.cue = load_cue(i)
        a = np.asarray(self.cue.resize((N * 16, N * 16), Image.BOX), np.float64)
        cell = a.reshape(N, 16, N, 16, 3).mean((1, 3))
        self.cue_color = np.array([cell[a0:a1, b0:b1].mean((0, 1)) for a0, a1, b0, b1 in self.rects])
