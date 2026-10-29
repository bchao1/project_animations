"""Tile frames of the 2D settle clip, v2 (= v1 on the v2 physics) → data/generated/tiles_v2/NNNN.png (1920 × 1080 RGBA, 60 fps).

Per image (settle2d_common_v1.py timeline): noise → generated image (noise fades while the image sharpens) →
a diagonal sweep turns the image into its LoT layout (token fill = mean image colour over the token's patch) →
every token shrinks to one 1 × 1 token at a cell of its footprint → squares at the physics positions
(sim2d_v2.py). The dim tile frame (the full-resolution 64 × 64 = 100% height) appears with the tokenization.
Drawn at 2× and box-downsampled. Text, ticks and captions are added by animation_lot_settle2d_v2.py.
Usage: python render_tiles_v1.py [first last]      (all frames by default; existing frames are skipped)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import settle2d_common_v2 as C

SS = 2
S = C.CELL * SS                    # pixels per cell at 2×
TP = int(round(C.N * S))           # tile pixels at 2×
GHOST = (58, 58, 58)
NF = int(round(C.DURATION * C.FPS))

_st = {}


def state():
    if not _st:
        toks = [C.Tokens(i) for i in range(C.NI)]
        d = np.load(C.PHYS)
        pos, tile = d["pos"], d["tile"]
        rng = np.random.default_rng(3)
        _st["toks"] = toks
        _st["pos"] = [pos[:, tile == i] for i in range(C.NI)]
        _st["img"] = [np.asarray(tk.img.resize((TP, TP), Image.LANCZOS), np.float32) for tk in toks]
        _st["blur"] = [np.asarray(tk.img.resize((TP, TP), Image.LANCZOS).filter(ImageFilter.GaussianBlur(28)),
                                  np.float32) for tk in toks]
        _st["noise"] = [rng.normal(118, 62, (TP, TP, 3)).astype(np.float32) for _ in toks]
        yy, xx = np.mgrid[0:TP, 0:TP] / TP
        _st["diag"] = (xx + yy) / 2
        _st["mosaic"] = [np.asarray(draw_boxes(tk.lay_lo, tk.lay_hi, tk.color, tk.rects), np.float32) for tk in toks]
    return _st


def draw_boxes(lo, hi, color, rects=None, frame=0.0):
    """RGB tile (2×) with one rounded box per token; lo / hi in cell space, y down from the top."""
    im = Image.new("RGB", (TP, TP))
    dr = ImageDraw.Draw(im)
    for k in range(len(lo)):
        x0, y0 = lo[k] * S
        x1, y1 = hi[k] * S
        side = min(x1 - x0, y1 - y0)
        r = int(max(1, 0.12 * side))
        dr.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=r, fill=tuple(int(c) for c in color[k]))
    if frame > 0:
        g = tuple(int(c * frame) for c in GHOST)
        dr.rectangle([0, 0, TP - 1, TP - 1], outline=g, width=3)
    return im


def tile_frame(i, t):
    st = state()
    tk = st["toks"][i]
    d = C.STAGGER * i
    g0, g1 = C.GENERATE[0] + d, C.GENERATE[1] + d
    k0, k1 = C.TOKENIZE[0] + d, C.TOKENIZE[1] + d
    if t < k0:                                                   # generation: noise → image, blur → sharp
        a = C.ease(t, g0, g1)
        b = C.ease(t, g0 + 0.3 * (g1 - g0), g1)
        clean = (1 - b) * st["blur"][i] + b * st["img"][i]
        x = a * clean + (1 - a) * st["noise"][i]
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).convert("RGBA")
    if t < C.SHRINK[0]:                                          # tokenization sweep
        u = (t - k0) / (k1 - k0)
        band = 0.25
        f = u * (1 + band) - band
        w = C.smooth((f - st["diag"]) / band + 1)[..., None]
        x = (1 - w) * st["img"][i] + w * st["mosaic"][i]
        im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
        fr = C.ease(t, k0, k1)
        if fr > 0:
            ImageDraw.Draw(im).rectangle([0, 0, TP - 1, TP - 1], outline=tuple(int(c * fr) for c in GHOST), width=3)
        return im.convert("RGBA")
    if t < C.FALL[0]:                                            # shrink to unit tokens
        s = C.ease(t, *C.SHRINK)
        lo = (1 - s) * tk.lay_lo + s * tk.unit_lo
        hi = (1 - s) * tk.lay_hi + s * tk.unit_hi
        return draw_boxes(lo, hi, tk.color, frame=1.0).convert("RGBA")
    P = st["pos"][i]
    u = min((t - C.FALL[0]) * C.FPS, len(P) - 1.0)
    a = int(u)
    b = min(a + 1, len(P) - 1)
    c = P[a] + (u - a) * (P[b] - P[a])                          # (n, 2): x, y above the floor
    h = 0.5 - C.M
    ctr = np.c_[c[:, 0], C.N - c[:, 1]]
    return draw_boxes(ctr - h, ctr + h, tk.color, frame=1.0).convert("RGBA")


def render(f):
    out = os.path.join(C.TILES, f"{f:04d}.png")
    if os.path.exists(out):
        return
    t = f / C.FPS
    canvas = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    for i in range(C.NI):
        im = tile_frame(i, t).resize((int(round(C.N * C.CELL)),) * 2, Image.BOX)
        x0, y0 = C.tile_origin_px(i)
        canvas.paste(im, (int(round(x0)), int(round(y0))))
    canvas.save(out)


if __name__ == "__main__":
    os.makedirs(C.TILES, exist_ok=True)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, NF - 1)
    with Pool(int(os.environ.get("WORKERS", 48))) as pool:
        pool.map(render, range(a, b + 1), chunksize=4)
    print(f"wrote {C.TILES} frames {a}–{b}")
