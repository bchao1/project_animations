"""V2 background frames → data/generated/frames_v1/NNNN.png (1920 × 1080 RGB, 60 fps): left tile, right volume (v16 colour), thumbnails.

Left (2× SSAA): dim painted detail map + quadtree (outlines of every cell born so far, leaves filled in size colours)
→ outlines and map fade → static layout → tokens shrink to 1 × 1 → corgi physics (sim_left_v1.py).
Right: the same quadtree on latent frame 1, drawn where spacetime v14's GL camera shows it at t = 0, crossfading into
vol11c_31 frame 0; afterwards vol11c frame (t − PRE) unchanged. Text is added by animation_quadtree_settle_v1.py.
Usage: python render_frames_v1.py [first last]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

import q_common_v1 as Q

SS = 2
NF = int(round(Q.DURATION * Q.FPS))
LP = int(round(Q.L_SIDE * Q.PX))             # left tile px at 1080p (486)
LS = LP * SS / Q.N                           # px per cell at 2×
RW, RH = int(round(Q.R_W * Q.PX)), int(round(Q.R_H * Q.PX))
M = 0.1
GHOST = (58, 58, 58)
_st = {}


def state():
    if not _st:
        R = Q.corgi_rects()
        y0, y1, x0, x1 = R.T
        rs, cs = y0 + (y1 - y0 - 1) // 2, x0 + (x1 - x0 - 1) // 2
        img = np.asarray(Image.open(Q.CORGI_IMG).convert("RGB").resize((Q.N * 16,) * 2, Image.BOX), np.float64)
        cell = img.reshape(Q.N, 16, Q.N, 16, 3).mean((1, 3))
        col = np.array([cell[a0:a1, b0:b1].mean((0, 1)) for a0, a1, b0, b1 in R])
        _st["L"] = dict(rects=R, col=col,
                        lay_lo=np.c_[x0 + M, y0 + M], lay_hi=np.c_[x1 - M, y1 - M],
                        unit_lo=np.c_[cs + M, rs + M], unit_hi=np.c_[cs + 1 - M, rs + 1 - M],
                        tree=Q.quadtree(R, 64, 64, 64), pos=np.load(Q.PHYS_L)["pos"])
        m = Image.open(Q.CORGI_MAP).convert("L").resize((LP * SS,) * 2, Image.LANCZOS)
        _st["L_col"] = {tuple(r): c for r, c in zip(R.tolist(), col)}
        _st["L_under"] = np.asarray(m, np.float32)[..., None] / 255.0 * np.array([150, 150, 150], np.float32)
        fr0 = Q.ST.decode()[0]
        _st["R_tree"] = Q.quadtree(fr0, Q.ST.H, Q.ST.W, 16)
        tk = Q.ST.Tokens(sim=False)
        _st["R_col"] = {tuple(r): c for r, c in zip(fr0.tolist(), tk.img_col[tk.k == 0])}   # leaf → mean colour
        _st["R_fine"] = fr0[(fr0[:, 1] - fr0[:, 0]) * (fr0[:, 3] - fr0[:, 2]) == 1]
        a = Q.vol_px(0.0, (-Q.ST.W / 2, Q.ST.H / 2, 0.0))
        b = Q.vol_px(0.0, (Q.ST.W / 2, -Q.ST.H / 2, 0.0))
        _st["R_box"] = (a[0], a[1], (b[0] - a[0]) / Q.ST.W)      # x0, y0, px per cell (vol frame pixels, 1×)
        fs = sorted(os.listdir(Q.ST.CLIP))
        _st["vid"] = [os.path.join(Q.ST.CLIP, f) for f in fs]
        th = int(round(Q.VID_TH[1] * Q.PX))
        _st["img_th"] = Image.open(Q.CORGI_IMG).convert("RGB").resize((th, th), Image.LANCZOS)
        _st["th"] = th
    return _st


def rbox(dr, x0, y0, x1, y1, fill, alpha=1.0):
    side = min(x1 - x0, y1 - y0)
    dr.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=int(max(1, 0.12 * side)),
                         fill=tuple(int(c * alpha) for c in fill))


def draw_tree(im, tree, colors, t, ox, oy, s, line_w, fade_lines):
    """Quadtree on `im` (cells of s px at offset ox, oy): outlines of born cells, leaves filled (mean colour) as born."""
    dr = ImageDraw.Draw(im)
    lw = 1 - fade_lines
    if lw > 0:
        for y0, y1, x0, x1, size, leaf, born in tree:
            a = Q.ease(t, born, born + Q.SPLIT_LEN) * lw
            if a > 0:
                dr.rectangle([ox + x0 * s, oy + y0 * s, ox + x1 * s, oy + y1 * s],
                             outline=tuple(int(c * a) for c in Q.LINE), width=line_w)
    for y0, y1, x0, x1, size, leaf, born in tree:
        if leaf:
            a = Q.ease(t, born, born + Q.SPLIT_LEN)
            if a > 0:
                rbox(dr, ox + (x0 + M) * s, oy + (y0 + M) * s, ox + (x1 - M) * s, oy + (y1 - M) * s,
                     colors[(y0, y1, x0, x1)], a)


def left(t):
    st = state()
    L = st["L"]
    P = LP * SS
    fl = Q.ease(t, *Q.UNDER_OUT)
    if t < Q.PRE:
        base = (st["L_under"] * (1 - fl)).astype(np.uint8)
        im = Image.fromarray(np.repeat(base, 3, 2) if base.shape[2] == 1 else base)
        draw_tree(im, L["tree"], st["L_col"], t, 0, 0, LS, 2, fl)
    else:
        im = Image.new("RGB", (P, P))
        dr = ImageDraw.Draw(im)
        if t < Q.FALL0:
            s = Q.ease(t, *Q.SHRINK)
            lo = (1 - s) * L["lay_lo"] + s * L["unit_lo"]
            hi = (1 - s) * L["lay_hi"] + s * L["unit_hi"]
        else:
            pos = L["pos"]
            u = min((t - Q.FALL0) * Q.FPS, len(pos) - 1.0)
            a = int(u)
            b = min(a + 1, len(pos) - 1)
            c = pos[a] + (u - a) * (pos[b] - pos[a])
            ctr = np.c_[c[:, 0], Q.N - c[:, 1]]
            lo, hi = ctr - (0.5 - M), ctr + (0.5 - M)
        for k in range(len(lo)):
            rbox(dr, lo[k, 0] * LS, lo[k, 1] * LS, hi[k, 0] * LS, hi[k, 1] * LS, L["col"][k])
    g = fl
    if g > 0:
        ImageDraw.Draw(im).rectangle([0, 0, P - 1, P - 1], outline=tuple(int(c * g) for c in GHOST), width=3)
    return im.resize((LP, LP), Image.BOX)


def right_prefix(t):
    st = state()
    W, H = Q.ST.RES
    im = Image.new("RGB", (W * SS, H * SS))
    x0, y0, s = st["R_box"]
    x0, y0, s = x0 * SS, y0 * SS, s * SS
    fl = Q.ease(t, *Q.UNDER_OUT)
    dr = ImageDraw.Draw(im)
    if fl < 1:
        g = tuple(int(c * (1 - fl)) for c in (150, 150, 150))
        for a0, a1, b0, b1 in st["R_fine"]:
            dr.rectangle([x0 + b0 * s, y0 + a0 * s, x0 + b1 * s, y0 + a1 * s], fill=g)
    dr.rectangle([x0, y0, x0 + Q.ST.W * s, y0 + Q.ST.H * s], outline=tuple(int(c * (1 - fl)) for c in Q.LINE),
                 width=2)
    draw_tree(im, st["R_tree"], st["R_col"], t, x0, y0, s, 2, fl)
    return im.resize((W, H), Image.BOX)


def vol(i):
    return Image.open(os.path.join(Q.VOL, f"{min(i, 827):04d}.png")).convert("RGB")


def right(t):
    if t < Q.XFADE[0]:
        im = right_prefix(t)
    elif t < Q.XFADE[1]:
        im = Image.blend(right_prefix(Q.PRE), vol(0), Q.ease(t, *Q.XFADE))
    else:
        im = vol(int(round((t - Q.PRE) * Q.FPS)))
    return im.resize((RW, RH), Image.LANCZOS)


def paste_c(canvas, im, cx, cy, alpha=1.0):
    x, y = Q.to_px(cx, cy)
    x, y = int(round(x - im.width / 2)), int(round(y - im.height / 2))
    if alpha < 1:
        im = Image.blend(Image.new("RGB", im.size), im, alpha)
    canvas.paste(im, (x, y))
    ImageDraw.Draw(canvas).rectangle([x - 1, y - 1, x + im.width, y + im.height],
                                     outline=tuple(int(c * alpha) for c in GHOST), width=1)


def render(f):
    out = os.path.join(Q.FRAMES, f"{f:04d}.png")
    if os.path.exists(out):
        return
    t = f / Q.FPS
    st = state()
    canvas = Image.new("RGB", (1920, 1080))
    r = right(t)
    x, y = Q.to_px(*Q.R_C)
    canvas.paste(r, (int(round(x - RW / 2)), int(round(y - RH / 2))))
    l_ = left(t)
    x, y = Q.to_px(*Q.L_C)
    canvas.paste(l_, (int(round(x - LP / 2)), int(round(y - LP / 2))))
    a = Q.ease(t, *Q.THUMBS)
    if a > 0:
        paste_c(canvas, st["img_th"], *Q.IMG_TH[0], alpha=a)
        k = Q.ST.video_frame(t - Q.PRE)
        v = Image.open(st["vid"][k]).convert("RGB").resize((int(round(st["th"] * 16 / 9)), st["th"]), Image.LANCZOS)
        paste_c(canvas, v, *Q.VID_TH[0], alpha=a)
    canvas.save(out)


if __name__ == "__main__":
    os.makedirs(Q.FRAMES, exist_ok=True)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, NF - 1)
    with Pool(int(os.environ.get("WORKERS", 48))) as pool:
        pool.map(render, range(a, b + 1), chunksize=4)
    print(f"wrote {Q.FRAMES} frames {a}–{b}")
