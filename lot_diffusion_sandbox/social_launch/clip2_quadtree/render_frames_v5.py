"""V2 v5 background frames (= v3 drawing, v5 geometry) → data/generated/frames_v5/NNNN.png (1920 × 1080 RGB, 60 fps). Panels drawn at 2×.

Both panels: dim silhouette + white quadtree outlines (born level by level) → noise-filled token boxes → 2-s
denoise into the image / latent frame 1 inside the boxes, gaps close → diagonal sweep into the mean-colour layout.
Left then waits, shrinks and settles (corgi physics); right crossfades into spacetime v16 colour volume frames from
v16 t = VOL0 on. Text is added by animation_quadtree_settle_v5.py.
Usage: python render_frames_v3.py [first last]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import q_common_v5 as Q

SS = 2
NF = int(round(Q.DURATION * Q.FPS))
LP = int(round(Q.L_SIDE * Q.PX))             # 648
RW, RH = int(round(Q.R_W * Q.PX)), int(round(Q.R_H * Q.PX))
M = 0.1
GHOST = (58, 58, 58)
_st = {}


def boxes(size, rects, col, s, ox=0.0, oy=0.0, im=None):
    im = im or Image.new("RGB", size)
    dr = ImageDraw.Draw(im)
    for (y0, y1, x0, x1), c in zip(rects, col):
        a0, b0, a1, b1 = ox + (x0 + M) * s, oy + (y0 + M) * s, ox + (x1 - M) * s, oy + (y1 - M) * s
        dr.rounded_rectangle([a0, b0, a1 - 1, b1 - 1], radius=int(max(1, 0.12 * min(a1 - a0, b1 - b0))),
                             fill=tuple(int(v) for v in c))
    return im


def panel_state(rects, img, under, H, W, s, size, ox, oy, root, rng):
    """Arrays for one panel's opening at 2× (cells of s px at offset ox, oy inside a canvas of `size`)."""
    cellm = np.asarray(img.resize((W * 16, H * 16), Image.BOX), np.float64).reshape(H, 16, W, 16, 3).mean((1, 3))
    col = np.array([cellm[a0:a1, b0:b1].mean((0, 1)) for a0, a1, b0, b1 in rects])
    full = Image.new("RGB", size)
    full.paste(img.resize((int(round(W * s)), int(round(H * s))), Image.LANCZOS), (int(round(ox)), int(round(oy))))
    blur = full.filter(ImageFilter.GaussianBlur(28))
    sil = Image.new("RGB", size)
    sil.paste(under.resize((int(round(W * s)), int(round(H * s))), Image.BOX), (int(round(ox)), int(round(oy))))
    mask = boxes(size, rects, np.full((len(rects), 3), 255.0), s, ox, oy)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    diag = ((xx - ox) / (W * s) + (yy - oy) / (H * s)) / 2
    return dict(rects=rects, col=col, tree=Q.quadtree(rects, H, W, root), s=s, ox=ox, oy=oy, H=H, W=W, size=size,
                img=np.asarray(full, np.float32), blur=np.asarray(blur, np.float32), sil=np.asarray(sil, np.float32),
                mask=np.asarray(mask, np.float32)[..., :1] / 255.0, diag=diag,
                mosaic=np.asarray(boxes(size, rects, col, s, ox, oy), np.float32),
                noise=rng.normal(118, 62, (size[1], size[0], 3)).astype(np.float32))


def state():
    if not _st:
        rng = np.random.default_rng(11)
        R = Q.corgi_rects()
        canvas = np.asarray(Image.open(Q.CORGI_MAP).convert("L"), np.float32) / 255.0 * Q.SIL
        under_l = Image.fromarray(np.repeat(canvas[..., None], 3, 2).astype(np.uint8))
        P = LP * SS
        _st["L"] = panel_state(R, Image.open(Q.CORGI_IMG).convert("RGB"), under_l, Q.N, Q.N, P / Q.N, (P, P), 0, 0,
                               64, rng)
        fr0 = Q.ST.decode()[0]
        sil = np.zeros((Q.ST.H, Q.ST.W), np.uint8)
        for a0, a1, b0, b1 in fr0[(fr0[:, 1] - fr0[:, 0]) * (fr0[:, 3] - fr0[:, 2]) == 1]:
            sil[a0:a1, b0:b1] = Q.SIL
        under_r = Image.fromarray(np.repeat(sil[..., None], 3, 2))
        a = Q.vol_px(Q.VOL0, (-Q.ST.W / 2, Q.ST.H / 2, 0.0))
        b = Q.vol_px(Q.VOL0, (Q.ST.W / 2, -Q.ST.H / 2, 0.0))
        s = (b[0] - a[0]) / Q.ST.W * SS
        frame1 = Image.open(os.path.join(Q.ST.CLIP, sorted(os.listdir(Q.ST.CLIP))[0])).convert("RGB")
        _st["R"] = panel_state(fr0, frame1, under_r, Q.ST.H, Q.ST.W, s, (Q.ST.RES[0] * SS, Q.ST.RES[1] * SS),
                               a[0] * SS, a[1] * SS, 16, rng)
        tk = Q.ST.Tokens(sim=False)
        _st["R"]["col"] = tk.img_col[tk.k == 0]               # exactly the volume's colours
        _st["R"]["mosaic"] = np.asarray(boxes(_st["R"]["size"], fr0, _st["R"]["col"], s, a[0] * SS, a[1] * SS),
                                        np.float32)
        y0, y1, x0, x1 = R.T
        rs, cs = y0 + (y1 - y0 - 1) // 2, x0 + (x1 - x0 - 1) // 2
        _st["L"].update(lay_lo=np.c_[x0 + M, y0 + M], lay_hi=np.c_[x1 - M, y1 - M],
                        unit_lo=np.c_[cs + M, rs + M], unit_hi=np.c_[cs + 1 - M, rs + 1 - M],
                        pos=np.load(Q.PHYS_L)["pos"])
    return _st


def arr(x):
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def opening(p, t):
    """One panel's opening at time t (2× canvas)."""
    if t < Q.TOKENIZE[0]:
        a = Q.ease(t, *Q.NOISY)
        b = Q.ease(t, *Q.DENOISE)
        b2 = Q.ease(t, Q.DENOISE[0] + 0.3 * (Q.DENOISE[1] - Q.DENOISE[0]), Q.DENOISE[1])
        clean = (1 - b2) * p["blur"] + b2 * p["img"]
        o = Q.ease(t, *Q.OPEN)
        m = (1 - o) * p["mask"] + o
        x = (1 - a) * p["sil"] + a * ((1 - b) * p["noise"] + b * clean) * m
        im = arr(x)
        lw = 1 - Q.ease(t, *Q.UNDER_OUT)
        if lw > 0:                                             # quadtree outlines, level by level
            dr = ImageDraw.Draw(im)
            s, ox, oy = p["s"], p["ox"], p["oy"]
            dr.rectangle([ox, oy, ox + p["W"] * s, oy + p["H"] * s], outline=tuple(int(c * lw) for c in Q.LINE),
                         width=3)                              # root cell (the video's 16 × 16 blocks come later)
            for y0, y1, x0, x1, size, leaf, born in p["tree"]:
                e = Q.ease(t, born, born + Q.SPLIT_LEN) * lw
                if e > 0:
                    dr.rectangle([ox + x0 * s, oy + y0 * s, ox + x1 * s, oy + y1 * s],
                                 outline=tuple(int(c * e) for c in Q.LINE), width=3 if size >= 4 else 2)
        return im
    u = np.clip((t - Q.TOKENIZE[0]) / (Q.TOKENIZE[1] - Q.TOKENIZE[0]), 0, 1)
    band = 0.25
    f = u * (1 + band) - band
    w = Q.smooth((f - p["diag"]) / band + 1)[..., None]
    return arr((1 - w) * p["img"] + w * p["mosaic"])


def left(t):
    st = state()
    L = st["L"]
    P = LP * SS
    if t < Q.TOKENIZE[1] + 0.05:
        im = opening(L, t)
    else:
        im = Image.new("RGB", (P, P))
        if t < Q.SHRINK[0]:
            lo, hi = L["lay_lo"], L["lay_hi"]
        elif t < Q.FALL0:
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
        dr = ImageDraw.Draw(im)
        S = L["s"]
        for k in range(len(lo)):
            x0, y0, x1, y1 = lo[k, 0] * S, lo[k, 1] * S, hi[k, 0] * S, hi[k, 1] * S
            dr.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=int(max(1, 0.12 * min(x1 - x0, y1 - y0))),
                                 fill=tuple(int(v) for v in L["col"][k]))
    fr = Q.ease(t, *Q.TOKENIZE)
    if fr > 0:
        ImageDraw.Draw(im).rectangle([0, 0, P - 1, P - 1], outline=tuple(int(c * fr) for c in GHOST), width=3)
    return im.resize((LP, LP), Image.BOX)


def vol(tv):
    return Image.open(os.path.join(Q.VOL, f"{min(int(round(tv * Q.FPS)), 827):04d}.png")).convert("RGB")


def right(t):
    st = state()
    if t < Q.XFADE[0]:
        im = opening(st["R"], t).resize(Q.ST.RES, Image.BOX)
    elif t < Q.XFADE[1]:
        im = Image.blend(opening(st["R"], Q.XFADE[0]).resize(Q.ST.RES, Image.BOX), vol(Q.VOL0),
                         Q.ease(t, *Q.XFADE))
    else:
        im = vol(t - Q.PRE + Q.VOL0)
    return im.resize((RW, RH), Image.LANCZOS)


def render(f):
    out = os.path.join(Q.FRAMES, f"{f:04d}.png")
    if os.path.exists(out):
        return
    t = f / Q.FPS
    canvas = Image.new("RGB", (1920, 1080))
    x, y = Q.to_px(*Q.R_C)
    canvas.paste(right(t), (int(round(x - RW / 2)), int(round(y - RH / 2))))
    x, y = Q.to_px(*Q.L_C)
    canvas.paste(left(t), (int(round(x - LP / 2)), int(round(y - LP / 2))))
    canvas.save(out)


if __name__ == "__main__":
    os.makedirs(Q.FRAMES, exist_ok=True)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, NF - 1)
    with Pool(int(os.environ.get("WORKERS", 32))) as pool:
        pool.map(render, range(a, b + 1), chunksize=4)
    print(f"wrote {Q.FRAMES} frames {a}–{b}")
