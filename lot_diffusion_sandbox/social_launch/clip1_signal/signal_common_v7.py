"""V1 v7 (post 1): control signal → token layout (signal colours) → noisy token layout → image → mean-colour layout →
shrink and settle (settle2d v3 recipe, re-simulated for the new images by sim_v2.py).

v2 (user feedback on v1, 2026-10-01): 3 columns × 2 rows by signal (depth map | semantic mask | bounding box), better
image picks (fine tokens on the subject), the bounding box in orange so it differs from the blue mask, the signal is
on screen from frame 0 with its full 64 × 64 grid labelled "4,096 tokens", holds between the steps (2 s on the token
layout, 2-s denoise), no captions or footnotes (only column names and token counts), bigger tiles filling the frame,
no speedup labels.
v3 (user, 2026-10-01): a clear speedup line under every count, in the second highlight colour (purple), from the
shared lookup ../../speedup/lot_speedup.py; tiles 384 → 352 px to fit the third line.
v4 (user, 2026-10-01): before the settle only the token count shows under a tile; the % appears only after the
settle (on the left tick) and the speedup only then (under the tile) — two label lines, so the tiles grow to 390 px.
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
TILES = os.path.join(GEN, "tiles_v7")
PHYS = os.path.join(GEN, "physics_v6.npz")
N, FPS, FULL = B.N, B.FPS, B.FULL
# (id, source, subject); index = 2 · column + row
ITEMS = [("000000000025", "dof", "giraffe"), ("000000000283", "dof", "wine"),
         ("000000000034", "sam3_mask", "zebra"), ("000000000092", "sam3_mask", "cake"),
         ("000000000042", "sam3_box", "dog"), ("000000000142", "sam3_box", "sandwich")]
NI = len(ITEMS)
COLUMNS = ["depth map", "semantic mask", "bounding box"]

# ------------------------------------------------------------------ geometry (1080p pixels; manim = px / 135)
TP = 390                                   # tile side
CELL = TP / N
COL_X = [960 - 580, 960, 960 + 580]        # tile centres (room for the % labels on the left)
ROW_Y0 = [64, 570]                         # tile tops: 20-px margin + 34-px header + 10-px gap; 14 px under row 1's label
HEAD_Y = 37                                # column names (centre)
LABEL_DY = 55                              # label centre below the tile: 8-px gap + half its 94-px height (count + speedup)


def tile_origin_px(i):
    return COL_X[i // 2] - TP / 2, ROW_Y0[i % 2]


def px_to_m(x, y):
    return x / 135.0 - 64 / 9, 4.0 - y / 135.0


# ------------------------------------------------------------------ timeline (s)
STAGGER = 0.06
SWEEP = (0.0, 0.0)                 # (v1–v6: full grid → layout sweep; v7 starts on the layout)
TO_NOISE = (2.5, 3.0)              # after a 2.5-s hold on the tokenized signal: token fills → noise
DENOISE = (3.0, 5.0)               # 2 s: noise → generated image inside the token boxes
OPEN = (4.5, 5.0)                  # box gaps close: the clean image
OFFSET = 4.1                       # settle2d v3 time = clip time − OFFSET from here on (v6: 5.6)
TOKENIZE = (B.TOKENIZE[0] + OFFSET, B.TOKENIZE[1] + OFFSET)     # 6.1 – 7.1: image → mean-colour layout
SHRINK = (B.SHRINK[0] + OFFSET, B.SHRINK[1] + OFFSET)
FALL0 = B.FALL[0] + OFFSET
LIQUEFY0 = B.LIQUEFY[0] + OFFSET
COLLAPSE = (B.COLLAPSE[0] + OFFSET, B.COLLAPSE[1] + OFFSET)
DURATION = 16.0

ease, smooth = B.ease, B.smooth
sys.path.insert(0, os.path.join(ROOT, "speedup"))
from lot_speedup import speedup        # noqa: E402  (shared token → speedup lookup)

CUE_BLUE = np.array([93, 166, 214])
GROUND = ((18, 22, 52), (40, 30, 78))                          # dark navy → indigo (turbo's low end), top → bottom
MASK_A, MASK_B, MASK_RIM = (24, 178, 196), (64, 226, 168), (190, 255, 236)   # teal → mint, light rim
BOX_A, BOX_B, BOX_EDGE, BOX_FILL_A = (251, 190, 60), (236, 96, 42), (255, 226, 150), 0.88   # warm turbo end, light edge


def load_item(i):
    iid, src, _ = ITEMS[i]
    img = Image.open(os.path.join(ROOT, "data", "img", f"img_{iid}_{src}_generated.png")).convert("RGB")
    import json
    rects = np.array(json.load(open(os.path.join(ROOT, "data", "layouts", f"img_{iid}_{src}.json")))["rects"], int)
    return img, rects


def _ground(n):
    t = np.linspace(0, 1, n)[:, None, None]
    g = (1 - t) * np.array(GROUND[0], np.float32) + t * np.array(GROUND[1], np.float32)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    vig = 1 - 0.35 * np.clip((xx ** 2 + yy ** 2) / 0.5, 0, 1)[..., None]
    return np.broadcast_to(g, (n, n, 3)) * vig


def load_cue(i):
    """v7 styling: depth map as is; mask / box on the dark ground (1024 × 1024)."""
    from PIL import ImageDraw, ImageFilter
    iid, src, _ = ITEMS[i]
    a = np.asarray(Image.open(os.path.join(ROOT, "data", "img", f"img_{iid}_{src}_cue.png")).convert("RGB"), np.float32)
    if src == "dof":
        return Image.fromarray(a.astype(np.uint8))
    n = a.shape[0]
    inside = np.abs(a - CUE_BLUE).sum(-1) < 40
    out = _ground(n).copy()
    if src == "sam3_mask":                   # smooth the latent-grid steps, teal → mint gradient, light rim, soft glow
        m8 = Image.fromarray((inside * 255).astype(np.uint8))
        m = np.asarray(m8.filter(ImageFilter.GaussianBlur(9)), np.float32) / 255
        m = np.clip((m - 0.5) * 8 + 0.5, 0, 1)
        mi = Image.fromarray((m * 255).astype(np.uint8))
        glow = np.asarray(mi.filter(ImageFilter.GaussianBlur(24)), np.float32) / 255
        ero = np.asarray(mi.filter(ImageFilter.MinFilter(13)), np.float32) / 255
        t = np.linspace(0, 1, n)[:, None, None]
        fill = (1 - t) * np.array(MASK_A, np.float32) + t * np.array(MASK_B, np.float32)
        fill = np.broadcast_to(fill, (n, n, 3))
        out = out + 0.35 * glow[..., None] * (np.array(MASK_A, np.float32) - out)
        fill = fill + 0.85 * (m - ero)[..., None] * (np.array(MASK_RIM, np.float32) - fill)
        out = out * (1 - m[..., None]) + fill * m[..., None]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    ys, xs = np.nonzero(inside)              # bounding box: warm turbo gradient (amber → red-orange), light edge
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    t = (((yy - y0) / max(1, y1 - y0 - 1) + (xx - x0) / max(1, x1 - x0 - 1)) / 2)[..., None]
    fill = (1 - t) * np.array(BOX_A, np.float32) + t * np.array(BOX_B, np.float32)
    out[y0:y1, x0:x1] += BOX_FILL_A * (fill - out[y0:y1, x0:x1])
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], outline=BOX_EDGE, width=12)
    return im


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
