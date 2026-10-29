"""Tile frames of V1 → data/generated/tiles_v1/NNNN.png (1920 × 1080 RGBA, 60 fps).

Opening per image (signal_common_v1.py): control signal → diagonal sweep into its token layout (fill = mean signal
colour per token) → the fills turn to noise inside the token boxes → noise denoises to the generated image while the
box gaps close → from TOKENIZE on, settle2d v3's tile_frame(i, t − OFFSET) (image → mean-colour layout → shrink →
physics). Drawn at 2× and box-downsampled; text is added by animation_signal_settle_v1.py.
Usage: python render_tiles_v1.py [first last]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

import signal_common_v1 as C
import render_tiles_v3 as R                      # settle2d v3 tile renderer (on sys.path via signal_common_v1)

NF = int(round(C.DURATION * C.FPS))
TP, S = R.TP, R.S
_st = {}


def state():
    if not _st:
        toks = [C.Tokens(i) for i in range(C.NI)]
        rng = np.random.default_rng(11)
        _st["toks"] = toks
        _st["cue"] = [np.asarray(tk.cue.resize((TP, TP), Image.LANCZOS), np.float32) for tk in toks]
        _st["cue_mosaic"] = [np.asarray(R.draw_boxes(tk.lay_lo, tk.lay_hi, tk.cue_color), np.float32) for tk in toks]
        _st["box_mask"] = [np.asarray(R.draw_boxes(tk.lay_lo, tk.lay_hi, np.full((tk.n, 3), 255.0)), np.float32)[..., :1]
                           / 255.0 for tk in toks]
        _st["noise"] = [rng.normal(118, 62, (TP, TP, 3)).astype(np.float32) for _ in toks]
    return _st


def opening(i, t):
    st, rs = state(), R.state()                  # rs: settle2d images, blurs, diagonal ramp
    d = C.STAGGER * i
    if t < C.CUE_TOK[0] + d:                      # control signal fades in
        x = C.ease(t, C.SIG_IN[0] + d, C.SIG_IN[1] + d) * st["cue"][i]
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).convert("RGBA")
    if t < C.TO_NOISE[0] + d:                     # diagonal sweep → signal-colour token layout
        u = (t - C.CUE_TOK[0] - d) / (C.CUE_TOK[1] - C.CUE_TOK[0])
        band = 0.25
        f = u * (1 + band) - band
        w = C.smooth((f - rs["diag"]) / band + 1)[..., None]
        x = (1 - w) * st["cue"][i] + w * st["cue_mosaic"][i]
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).convert("RGBA")
    # generation inside the token boxes: fill → noise → image, then the gaps close
    a = C.ease(t, C.TO_NOISE[0] + d, C.TO_NOISE[1] + d)
    b = C.ease(t, C.DENOISE[0] + d, C.DENOISE[1] + d)
    b2 = C.ease(t, C.DENOISE[0] + d + 0.3 * (C.DENOISE[1] - C.DENOISE[0]), C.DENOISE[1] + d)
    clean = (1 - b2) * rs["blur"][i] + b2 * rs["img"][i]
    fill = (1 - a) * st["cue_mosaic"][i] + a * ((1 - b) * st["noise"][i] + b * clean)
    o = C.ease(t, C.OPEN[0] + d, C.OPEN[1] + d)
    m = (1 - o) * st["box_mask"][i] + o
    x = fill * m
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).convert("RGBA")


def tile_frame(i, t):
    if t < C.TOKENIZE[0] + C.STAGGER * i:
        return opening(i, t)
    return R.tile_frame(i, t - C.OFFSET)


def render(f):
    out = os.path.join(C.TILES, f"{f:04d}.png")
    if os.path.exists(out):
        return
    t = f / C.FPS
    canvas = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    for i in range(C.NI):
        im = tile_frame(i, t).resize((int(round(C.N * C.CELL)),) * 2, Image.BOX)
        x0, y0 = C.B.tile_origin_px(i)
        canvas.paste(im, (int(round(x0)), int(round(y0))))
    canvas.save(out)


if __name__ == "__main__":
    os.makedirs(C.TILES, exist_ok=True)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, NF - 1)
    with Pool(int(os.environ.get("WORKERS", 48))) as pool:
        pool.map(render, range(a, b + 1), chunksize=4)
    print(f"wrote {C.TILES} frames {a}–{b}")
