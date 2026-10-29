"""Tile frames of V1 v3 (= v2 drawing, v3 geometry) → data/generated/tiles_v3/NNNN.png (1920 × 1080 RGB, 60 fps). Drawn at 2× and box-downsampled.

Per tile (signal_common_v2.py timeline): control signal with its full 64 × 64 grid (frame 0) → diagonal sweep into
the token layout, fill = mean signal colour → 2-s hold → fills turn to noise (noisy token layout) → 2-s denoise into
the generated image inside the token boxes, gaps close → hold → diagonal sweep into the mean-image-colour layout →
shrink to 1 × 1 tokens → physics (sim_v2.py). Text is added by animation_signal_settle_v2.py.
Usage: python render_tiles_v2.py [first last]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import signal_common_v3 as C

SS = 2
P = C.TP * SS                       # tile pixels at 2×
S = P / C.N                         # pixels per cell at 2×
GHOST = (58, 58, 58)
NF = int(round(C.DURATION * C.FPS))
_st = {}


def draw_boxes(lo, hi, color):
    im = Image.new("RGB", (P, P))
    dr = ImageDraw.Draw(im)
    for k in range(len(lo)):
        x0, y0 = lo[k] * S
        x1, y1 = hi[k] * S
        r = int(max(1, 0.12 * min(x1 - x0, y1 - y0)))
        dr.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=r, fill=tuple(int(c) for c in color[k]))
    return im


def frame_box(im, a):
    if a > 0:
        ImageDraw.Draw(im).rectangle([0, 0, P - 1, P - 1], outline=tuple(int(c * a) for c in GHOST), width=3)
    return im


def state():
    if not _st:
        toks = [C.Tokens(i) for i in range(C.NI)]
        rng = np.random.default_rng(11)
        d = np.load(C.PHYS)
        yy, xx = np.mgrid[0:P, 0:P] / P
        grid = Image.new("L", (P, P), 0)
        dg = ImageDraw.Draw(grid)
        for k in range(C.N + 1):
            dg.line([(k * S, 0), (k * S, P)], fill=255, width=2)
            dg.line([(0, k * S), (P, k * S)], fill=255, width=2)
        _st.update(
            toks=toks, pos=[d["pos"][:, d["tile"] == i] for i in range(C.NI)], diag=(xx + yy) / 2,
            grid=np.asarray(grid, np.float32)[..., None] / 255.0,
            cue=[np.asarray(tk.cue.resize((P, P), Image.LANCZOS), np.float32) for tk in toks],
            cue_mosaic=[np.asarray(draw_boxes(tk.lay_lo, tk.lay_hi, tk.cue_color), np.float32) for tk in toks],
            mosaic=[np.asarray(draw_boxes(tk.lay_lo, tk.lay_hi, tk.color), np.float32) for tk in toks],
            box_mask=[np.asarray(draw_boxes(tk.lay_lo, tk.lay_hi, np.full((tk.n, 3), 255.0)), np.float32)[..., :1]
                      / 255.0 for tk in toks],
            img=[np.asarray(tk.img.resize((P, P), Image.LANCZOS), np.float32) for tk in toks],
            blur=[np.asarray(tk.img.resize((P, P), Image.LANCZOS).filter(ImageFilter.GaussianBlur(28)), np.float32)
                  for tk in toks],
            noise=[rng.normal(118, 62, (P, P, 3)).astype(np.float32) for _ in toks])
    return _st


def sweep(a, b, u, diag):
    band = 0.25
    f = u * (1 + band) - band
    w = C.smooth((f - diag) / band + 1)[..., None]
    return (1 - w) * a + w * b


def arr(x):
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def tile_frame(i, t):
    st = state()
    tk = st["toks"][i]
    d = C.STAGGER * i
    if t < C.TO_NOISE[0] + d:                                    # signal + full grid → signal-colour layout
        u = (t - C.SWEEP[0] - d) / (C.SWEEP[1] - C.SWEEP[0])
        gridded = st["cue"][i] * (1 - 0.45 * st["grid"])
        return arr(sweep(gridded, st["cue_mosaic"][i], np.clip(u, 0, 1), st["diag"]))
    if t < C.TOKENIZE[0] + d:                                    # noisy token layout → image
        a = C.ease(t, C.TO_NOISE[0] + d, C.TO_NOISE[1] + d)
        b = C.ease(t, C.DENOISE[0] + d, C.DENOISE[1] + d)
        b2 = C.ease(t, C.DENOISE[0] + d + 0.3 * (C.DENOISE[1] - C.DENOISE[0]), C.DENOISE[1] + d)
        clean = (1 - b2) * st["blur"][i] + b2 * st["img"][i]
        fill = (1 - a) * st["cue_mosaic"][i] + a * ((1 - b) * st["noise"][i] + b * clean)
        o = C.ease(t, C.OPEN[0] + d, C.OPEN[1] + d)
        return arr(fill * ((1 - o) * st["box_mask"][i] + o))
    if t < C.SHRINK[0]:                                          # image → mean-colour layout
        u = (t - C.TOKENIZE[0] - d) / (C.TOKENIZE[1] - C.TOKENIZE[0])
        return frame_box(arr(sweep(st["img"][i], st["mosaic"][i], np.clip(u, 0, 1), st["diag"])),
                         C.ease(t, C.TOKENIZE[0] + d, C.TOKENIZE[1] + d))
    if t < C.FALL0:                                              # shrink to unit tokens
        s = C.ease(t, *C.SHRINK)
        lo = (1 - s) * tk.lay_lo + s * tk.unit_lo
        hi = (1 - s) * tk.lay_hi + s * tk.unit_hi
        return frame_box(draw_boxes(lo, hi, tk.color), 1.0)
    pos = st["pos"][i]
    u = min((t - C.FALL0) * C.FPS, len(pos) - 1.0)
    a = int(u)
    b = min(a + 1, len(pos) - 1)
    c = pos[a] + (u - a) * (pos[b] - pos[a])
    ctr = np.c_[c[:, 0], C.N - c[:, 1]]
    h = 0.5 - C.B.M
    return frame_box(draw_boxes(ctr - h, ctr + h, tk.color), 1.0)


def render(f):
    out = os.path.join(C.TILES, f"{f:04d}.png")
    if os.path.exists(out):
        return
    t = f / C.FPS
    canvas = Image.new("RGB", (1920, 1080))
    for i in range(C.NI):
        im = tile_frame(i, t).resize((C.TP, C.TP), Image.BOX)
        x0, y0 = C.tile_origin_px(i)
        canvas.paste(im, (int(round(x0)), int(round(y0))))
    canvas.save(out)


if __name__ == "__main__":
    os.makedirs(C.TILES, exist_ok=True)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, NF - 1)
    with Pool(int(os.environ.get("WORKERS", 32))) as pool:
        pool.map(render, range(a, b + 1), chunksize=4)
    print(f"wrote {C.TILES} frames {a}–{b}")
