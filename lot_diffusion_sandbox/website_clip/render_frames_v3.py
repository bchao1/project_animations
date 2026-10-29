"""Tile frames of the website clip v3 → data/generated/frames_v3/NNNN.png (1024 × 1024 RGB, 60 fps), drawn at 2× and
box-downsampled. Drawing as in ../social_launch/clip1_signal/render_tiles_v7.py, one tile, web_common_v3 timeline.
Usage: python render_frames_v3.py [first last]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import web_common_v3 as C

S = C.S
SS = 2
P = C.TP * SS
CS = P / C.N                        # pixels per cell at 2×
GHOST = (58, 58, 58)
NF = int(round(C.DURATION * C.FPS))
_st = {}


def draw_boxes(lo, hi, color):
    im = Image.new("RGB", (P, P))
    dr = ImageDraw.Draw(im)
    for k in range(len(lo)):
        x0, y0 = lo[k] * CS
        x1, y1 = hi[k] * CS
        r = int(max(1, 0.12 * min(x1 - x0, y1 - y0)))
        dr.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=r, fill=tuple(int(c) for c in color[k]))
    return im


def frame_box(im, a):
    if a > 0:
        ImageDraw.Draw(im).rectangle([0, 0, P - 1, P - 1], outline=tuple(int(c * a) for c in GHOST), width=4)
    return im


def state():
    if not _st:
        tk = S.Tokens(C.ITEM)
        d = np.load(C.PHYS)
        yy, xx = np.mgrid[0:P, 0:P] / P
        img = tk.img.resize((P, P), Image.LANCZOS)
        _st.update(
            tk=tk, pos=d["pos"][:, d["tile"] == C.ITEM], diag=(xx + yy) / 2,
            cue=np.asarray(tk.cue.resize((P, P), Image.LANCZOS), np.float32),
            mosaic=np.asarray(draw_boxes(tk.lay_lo, tk.lay_hi, tk.color), np.float32),
            box_mask=np.asarray(draw_boxes(tk.lay_lo, tk.lay_hi, np.full((tk.n, 3), 255.0)), np.float32)[..., :1] / 255,
            img=np.asarray(img, np.float32),
            blur=np.asarray(img.filter(ImageFilter.GaussianBlur(28 * SS * C.TP / (390 * 2))), np.float32),
            noise=np.random.default_rng(11).normal(118, 62, (P, P, 3)).astype(np.float32))
    return _st


def sweep(a, b, u, diag):
    band = 0.25
    f = u * (1 + band) - band
    w = C.smooth((f - diag) / band + 1)[..., None]
    return (1 - w) * a + w * b


def arr(x):
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def tile_frame(t):
    st = state()
    tk = st["tk"]
    if t < C.TO_NOISE[0]:                                        # depth map inside its token boxes
        return arr(st["cue"] * st["box_mask"])
    if t < C.TOKENIZE[0]:                                        # noisy token layout → image
        a = C.ease(t, *C.TO_NOISE)
        b = C.ease(t, *C.DENOISE)
        b2 = C.ease(t, C.DENOISE[0] + 0.3 * (C.DENOISE[1] - C.DENOISE[0]), C.DENOISE[1])
        clean = (1 - b2) * st["blur"] + b2 * st["img"]
        fill = (1 - a) * st["cue"] + a * ((1 - b) * st["noise"] + b * clean)
        o = C.ease(t, *C.OPEN)
        return arr(fill * ((1 - o) * st["box_mask"] + o))
    if t < C.SHRINK[0]:                                          # image → mean-colour layout
        u = (t - C.TOKENIZE[0]) / (C.TOKENIZE[1] - C.TOKENIZE[0])
        return frame_box(arr(sweep(st["img"], st["mosaic"], np.clip(u, 0, 1), st["diag"])), C.ease(t, *C.TOKENIZE))
    if t < C.FALL0:                                              # shrink to unit tokens
        s = C.ease(t, *C.SHRINK)
        return frame_box(draw_boxes((1 - s) * tk.lay_lo + s * tk.unit_lo, (1 - s) * tk.lay_hi + s * tk.unit_hi,
                                    tk.color), 1.0)
    pos = st["pos"]
    u = min(C.sim_time(t) * C.FPS, len(pos) - 1.0)
    a = int(u)
    b = min(a + 1, len(pos) - 1)
    c = pos[a] + (u - a) * (pos[b] - pos[a])
    ctr = np.c_[c[:, 0], C.N - c[:, 1]]
    h = 0.5 - C.B.M
    return frame_box(draw_boxes(ctr - h, ctr + h, tk.color), 1.0)


def render(f):
    out = os.path.join(C.FRAMES, f"{f:04d}.png")
    if os.path.exists(out):
        return
    canvas = Image.new("RGB", (C.W, C.W))
    canvas.paste(tile_frame(f / C.FPS).resize((C.TP, C.TP), Image.BOX), (C.X0, C.Y0))
    canvas.save(out)


if __name__ == "__main__":
    os.makedirs(C.FRAMES, exist_ok=True)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, NF - 1)
    with Pool(int(os.environ.get("WORKERS", 32))) as pool:
        pool.map(render, range(a, b + 1), chunksize=2)
    print(f"wrote {C.FRAMES} frames {a}–{b}")
