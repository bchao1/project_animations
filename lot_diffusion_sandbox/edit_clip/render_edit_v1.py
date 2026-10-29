"""LoT multi-turn in-context editing — 20-s clip, v1 (see storyboard.md).

No manim: PIL frames (2× SSAA in h) piped to ffmpeg, 48 workers. Data from prep_data.py (data/generated/).

    ~/miniconda3/envs/imp_tome/bin/python render_edit_v1.py h|l      → media/LoT_edit_v1_<1080p60|540p15>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_edit_v1.py still 1.0 4.5 …   → scratch/still_v1_<t>.png
"""
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "data", "generated")
VER = "v1"
DUR = 20.0
MODE = sys.argv[1] if len(sys.argv) > 1 else "l"
FPS, OUTS, SS = (60, 1.0, 2) if MODE in ("h", "still") else (15, 0.5, 1)
K = OUTS * SS                                  # 1080p px → canvas px
W, H = 1920, 1080

# ------------------------------------------------------------------ style (STYLE_NOTES.md)
WHITE = (255, 255, 255)
DIM = (138, 138, 138)
GHOST = (58, 58, 58)
AMBER = (255, 202, 40)
PURPLE = (224, 64, 251)
LORA = (255, 112, 67)
BOX = (30, 42, 51)
BOXS = (144, 164, 174)
FD = os.path.expanduser("~/.local/share/fonts")
FONTS = {"rm": f"{FD}/cmunrm.ttf", "bx": f"{FD}/cmunbx.ttf", "ti": f"{FD}/cmunti.ttf",
         "math": "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf"}
MATHCH = set("×⊙→≈∼−∈≪≤‖↓÷")


@lru_cache(None)
def font(style, px):
    return ImageFont.truetype(FONTS[style], max(1, int(round(px))))


# ------------------------------------------------------------------ data
TURNS = json.load(open(f"{GEN}/turns.json"))
TT = TURNS["turns"]                            # TT[t - 1] = turn t
TIMING = TURNS["timing"]
NPZ = dict(np.load(f"{GEN}/session.npz"))     # eager: a lazy NpzFile breaks under fork
OFF = NPZ["off"]
REGION = NPZ["region"]                         # (100, 64, 64) edit region of turn t at [t - 1]
N = 64
rng = np.random.default_rng(7)
NOISE = np.clip(128 + 55 * rng.standard_normal((4096, 1)) + 18 * rng.standard_normal((4096, 3)), 0, 255)  # grey noise


def leaves(t):
    return NPZ["leaves"][OFF[t - 1]:OFF[t]].astype(int)


def ctx_cols(t):
    return NPZ["ctx"][OFF[t - 1]:OFF[t]]


def gen_cols(t):
    return NPZ["gen"][OFF[t - 1]:OFF[t]]


def noise_cols(t):
    lv = leaves(t)
    return NOISE[lv[:, 0] * N + lv[:, 2]]


def inside(t):
    lv = leaves(t)
    return REGION[t - 1][lv[:, 0], lv[:, 2]]


@lru_cache(8)
def out_img(t):
    return Image.open(f"{GEN}/img/out_{t:03d}.png").convert("RGB")


@lru_cache(4)
def gen_img(t):
    return Image.open(f"{GEN}/img/gen_{t:03d}.png").convert("RGB")


@lru_cache(8)
def cell_cols(t):
    """64 × 64 mean colours of out_t (cells of 16 px at 1024²)."""
    return np.asarray(out_img(t).resize((N, N), Image.BOX), dtype=np.float32)


@lru_cache(4)
def region_mask(t, px):
    """Feathered edit-region alpha of turn t at px × px."""
    m = Image.fromarray(REGION[t - 1].astype(np.uint8) * 255).resize((px, px), Image.NEAREST)
    return m.filter(ImageFilter.GaussianBlur(px / 400))


# ------------------------------------------------------------------ geometry (1080p px)
IMG = (70, 215, 540)
CTX = (760, 160, 330)
TGT = (760, 590, 330)
DIT = (1250, 330, 240, 430)                    # x, y, w, h
RO = 1705                                      # readout centre x
CAP_Y = 1012
NOTE_LEAF = (40, 44, 24, 28)                   # a 4 × 4 token on the front bottle's label (turn 1)

# ------------------------------------------------------------------ timeline (s)
T_LIFT, T_SWAP, T_DROP = (1.1, 1.5), 1.7, (1.9, 2.3)
T_FLY1 = (3.0, 3.6)
T_SWEEP = (3.9, 4.9)
T_TGT = (5.0, 5.5)
T_ARROWS = 5.5
T_CALL = 5.8
T_CALL_OUT = (8.25, 8.6)
T_STEP0, T_STEPLEN = 8.7, 0.7
T_DECODE = (11.6, 12.0)
T_PASTE = (12.0, 12.7)
T_TGT_OUT = (12.7, 13.0)
T_TURN2 = 13.0
T_FLY2 = (13.1, 13.6)
T_CHAIN = (13.8, 16.6)
T_OUT = (16.8, 17.3)
T_ROWS = (17.3, 17.75, 18.2, 18.7)
CAPTIONS = [(0.0, "Same LoT LoRA: base → distilled model"),
            (3.0, "In-context editing, never trained"),
            (5.8, "Down-projection: 16 patches → 1 token"),
            (8.6, "4 steps; outside the edit, blend in the source"),
            (11.6, "The result is the next turn’s context"),
            (13.8, "100 chained turns, 1.8 s each"),
            (16.8, "Fewer steps × fewer tokens")]


def seg(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1))


def ease(x):
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


def chain_turn(t):
    u = seg(t, *T_CHAIN)
    return int(min(100, 2 + np.floor(99 * u ** 2)))


def fmt(n):
    return f"{n:,}"


# ------------------------------------------------------------------ canvas
class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (int(W * K), int(H * K)), (0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _reset(self, im):
        self.im = im
        self.d = ImageDraw.Draw(self.im)

    @contextmanager
    def fade(self, a):
        """Everything drawn inside the block is blended in with opacity a."""
        if a <= 0:
            snap = self.im.copy()
            yield
            self._reset(snap)
            return
        if a >= 1:
            yield
            return
        snap = self.im.copy()
        yield
        self._reset(Image.blend(snap, self.im, a))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, r=0.0):
        box = [x0 * K, y0 * K, x1 * K, y1 * K]
        w = max(1, int(round(width * K))) if outline else 0
        if r * K >= 1.5:
            self.d.rounded_rectangle(box, radius=r * K, fill=fill, outline=outline, width=w)
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=w)

    def line(self, pts, fill, width=2.0):
        self.d.line([(x * K, y * K) for x, y in pts], fill=fill, width=max(1, int(round(width * K))))

    def poly(self, pts, fill=None, outline=None, width=2.0):
        self.d.polygon([(x * K, y * K) for x, y in pts], fill=fill, outline=outline,
                       width=max(1, int(round(width * K))))

    def arrow(self, x0, y0, x1, y1, fill=WHITE, width=3.0, tip=16):
        v = np.array([x1 - x0, y1 - y0], float)
        L = np.linalg.norm(v)
        u = v / L
        n = np.array([-u[1], u[0]])
        b = np.array([x1, y1]) - u * tip
        self.line([(x0, y0), tuple(b)], fill, width)
        self.poly([(x1, y1), tuple(b + n * tip * 0.45), tuple(b - n * tip * 0.45)], fill=fill)

    def paste(self, im, x0, y0, size, mask=None):
        px = max(1, int(round(size * K)))
        im = im.resize((px, px), Image.BILINEAR if px < im.width else Image.LANCZOS)
        if mask is not None:
            mask = mask.resize((px, px), Image.BILINEAR)
        self.im.paste(im, (int(round(x0 * K)), int(round(y0 * K))), mask)

    def text(self, x, y, s, size, fill=WHITE, style="rm", anchor="m", max_w=None):
        """Mixed-font line: CMU Serif runs + Latin Modern Math for MATHCH. anchor l / m / r; y = visual middle."""
        def runs(px):
            out = []
            for ch in s:
                f = "math" if ch in MATHCH else style
                if out and out[-1][0] == f:
                    out[-1][1] += ch
                else:
                    out.append([f, ch])
            return [(font(f, px), r) for f, r in out]
        px = size * K
        rs = runs(px)
        width = sum(f.getlength(r) for f, r in rs)
        if max_w is not None and width > max_w * K:
            px *= max_w * K / width
            rs = runs(px)
            width = sum(f.getlength(r) for f, r in rs)
        cap = font(style, px).getbbox("H", anchor="ls")
        yb = y * K - (cap[1] + cap[3]) / 2
        xc = x * K - {"l": 0, "m": width / 2, "r": width}[anchor]
        for f, r in rs:
            self.d.text((xc, yb), r, font=f, fill=fill, anchor="ls")
            xc += f.getlength(r)
        return width / K


def draw_tokens(c, panel, lv, cols, outline_idx=None, outline_col=WHITE, dy=0.0):
    x0, y0, S = panel
    y0 = y0 + dy
    cell = S / N
    cols = np.clip(cols, 0, 255).astype(np.uint8)
    for k, (r0, r1, c0, c1) in enumerate(lv):
        e = r1 - r0
        side = e * cell - 0.2 * cell
        px, py = x0 + c0 * cell + 0.1 * cell, y0 + r0 * cell + 0.1 * cell
        c.rect(px, py, px + side, py + side, fill=tuple(cols[k]), r=0.1 * side if e > 1 else 0)
    if outline_idx is not None:
        for k in np.atleast_1d(outline_idx):
            r0, r1, c0, c1 = lv[k]
            e = r1 - r0
            px, py = x0 + c0 * cell, y0 + r0 * cell
            c.rect(px - 1, py - 1, px + e * cell + 1, py + e * cell + 1, outline=outline_col, width=2.5)


def draw_region_outline(c, panel, region, col=WHITE, width=3.0):
    x0, y0, S = panel
    cell = S / N
    r = np.pad(region, 1)
    segs = []
    for i, j in zip(*np.nonzero(r[1:-1, 1:-1])):
        if not r[i, j + 1]:
            segs.append(((j, i), (j + 1, i)))           # top edge
        if not r[i + 2, j + 1]:
            segs.append(((j, i + 1), (j + 1, i + 1)))   # bottom
        if not r[i + 1, j]:
            segs.append(((j, i), (j, i + 1)))           # left
        if not r[i + 1, j + 2]:
            segs.append(((j + 1, i), (j + 1, i + 1)))   # right
    for (a, b) in segs:
        c.line([(x0 + a[0] * cell, y0 + a[1] * cell), (x0 + b[0] * cell, y0 + b[1] * cell)], col, width)


def ghost(c, panel):
    x0, y0, S = panel
    c.rect(x0, y0, x0 + S, y0 + S, outline=GHOST, width=1.5)


def frame_box(c, panel, col=(70, 70, 70)):
    x0, y0, S = panel
    c.rect(x0 - 1, y0 - 1, x0 + S + 1, y0 + S + 1, outline=col, width=1.5)


# ------------------------------------------------------------------ pieces
def dit_box(c, t):
    x, y, w, h = DIT
    cx = x + w / 2
    glow = 0.0
    for k in range(4):
        s = T_STEP0 + k * T_STEPLEN
        if s <= t < s + 0.45:
            glow = 1 - (t - s) / 0.45
    sc = tuple(int(lerp(a, b, glow)) for a, b in zip(BOXS, WHITE))
    c.rect(x, y, x + w, y + h, fill=BOX, outline=sc, width=2 + 2 * glow, r=18)
    c.text(cx, y + 62, "FLUX.2 [klein]", 30)
    c.text(cx, y + 102, "9B", 30)
    distilled = t >= T_SWAP
    c.text(cx, y + 175, "distilled" if distilled else "base", 32, style="bx")
    c.text(cx, y + 215, "4 steps" if distilled else "50 steps + CFG", 26, DIM)
    if T_STEP0 <= t < T_STEP0 + 4 * T_STEPLEN + 0.5:
        k = min(4, int((t - T_STEP0) / T_STEPLEN) + 1)
        for i in range(4):
            col = WHITE if i < k else GHOST
            c.rect(cx - 66 + i * 36, y + 250, cx - 66 + i * 36 + 24, y + 274, fill=col, r=4)
    lift = 70 * (ease(seg(t, *T_LIFT)) - ease(seg(t, *T_DROP)))
    tw, th = w * 0.78, 48
    ty = y + h - 52 - lift
    c.rect(cx - tw / 2, ty - th / 2, cx + tw / 2, ty + th / 2, fill=LORA, r=8)
    c.text(cx, ty, "LoT LoRA", 26, (0, 0, 0), style="bx")
    c.text(cx, y + h + 34, "LoRA + projections trained on", 22, DIM)
    c.text(cx, y + h + 62, "the base model, text → image only", 22, DIM)


def instruction(c, t, k):
    s = f"turn {k}"
    w1 = font("rm", 34 * K).getlength(s) / K
    ins = f"“{TT[k - 1]['instruction']}”"
    w2 = font("ti", 34 * K).getlength(ins) / K
    gap = 26
    tot = w1 + gap + w2
    if tot > 1780:
        sc = 1780 / tot
    else:
        sc = 1.0
    x0 = W / 2 - tot * sc / 2
    c.text(x0, 80, s, 34 * sc, AMBER, anchor="l")
    c.text(x0 + (w1 + gap) * sc, 80, ins, 34 * sc, WHITE, style="ti", anchor="l")


def panel_rect(p):
    return np.array([p[0], p[1], p[2]], float)


def fly(c, im, a, b, u, mask=None):
    r = lerp(panel_rect(a), panel_rect(b), ease(u))
    c.paste(im, r[0], r[1], r[2], mask)


def count_line(c, panel, n, dy=0.0):
    x0, y0, S = panel
    c.text(x0 + S / 2, y0 + S + 22 + dy, f"{fmt(n)} of 4,096 tokens", 26, AMBER)


def label(c, panel, s):
    x0, y0, S = panel
    c.text(x0 + S / 2, y0 - 22, s, 26, DIM)


def sweep_state(t, k=1):
    """Context tokens during the diagonal sweep: list of (leaves, cols) and the token count."""
    lv = leaves(k)
    cols = ctx_cols(k)
    u = seg(t, *T_SWEEP)
    trig = (lv[:, 0] + lv[:, 2]) / 126.0 < u
    e = lv[:, 1] - lv[:, 0]
    cc = cell_cols(k - 1)
    out_l, out_c = [lv[trig | (e == 1)]], [cols[trig | (e == 1)]]
    for (r0, r1, c0, c1) in lv[~trig & (e > 1)]:
        ii, jj = np.mgrid[r0:r1, c0:c1]
        ii, jj = ii.ravel(), jj.ravel()
        out_l.append(np.stack([ii, ii + 1, jj, jj + 1], 1))
        out_c.append(cc[ii, jj])
    lv2 = np.concatenate(out_l)
    return lv2, np.concatenate(out_c), len(lv2)


def sigma_cols(t, k):
    """Target colours of turn k during the 4 steps (edit: smooth noise → gen; outside: snapped blend)."""
    nz, gc, sc = noise_cols(k), gen_cols(k), ctx_cols(k)
    ins = inside(k)[:, None]
    x = np.clip((t - T_STEP0) / (4 * T_STEPLEN), 0, 1)              # smooth progress
    done = np.clip(np.floor((t - T_STEP0 + 0.1) / T_STEPLEN), 0, 4) / 4   # snapped progress (blend)
    col_in = lerp(nz, gc, ease(x) if x < 1 else 1.0)
    col_out = lerp(nz, sc, done)
    flash = 0.0
    for j in range(1, 5):
        s = T_STEP0 + j * T_STEPLEN - 0.1
        if s <= t < s + 0.3:
            flash = 1 - (t - s) / 0.3
    return np.where(ins, col_in, col_out), flash


def readout(c, t):
    if t < T_ARROWS or t >= T_OUT[1]:
        return
    k = 1 if t < T_FLY2[1] else (2 if t < T_CHAIN[0] else chain_turn(t))
    n = TT[k - 1]["n_tokens"] + TT[k - 1]["n_ref"]
    c.text(RO, 395, fmt(n), 58, AMBER, style="bx")
    c.text(RO, 442, "image tokens per pass", 24, DIM)
    c.text(RO, 472, "full-res: 8,192", 24, DIM)
    if T_STEP0 <= t < T_CHAIN[0]:
        k1 = min(4, int((t - T_STEP0) / T_STEPLEN) + 1)
        dt = TT[0]["seconds"] * seg(t, T_STEP0, T_STEP0 + 4 * T_STEPLEN)
        c.text(RO, 580, f"step {k1} / 4", 32)
        c.text(RO, 632, f"{dt:.1f} s", 46, style="bx")
        c.text(RO, 675, "DiT time this turn", 22, DIM)
    elif t >= T_CHAIN[0]:
        cum = sum(x["seconds"] for x in TT[:k])
        c.text(RO, 580, f"turn {k} / 100", 32)
        c.text(RO, 632, f"{cum:.0f} s", 46, style="bx")
        c.text(RO, 675, "total DiT time", 22, DIM)


def callout(c, t):
    """Down-projection callout over the image panel: 16 patches → trapezoid → 1 token."""
    if not (T_CALL <= t < T_CALL_OUT[1]):
        return
    a = 1 - seg(t, *T_CALL_OUT)
    r0, r1, c0, c1 = NOTE_LEAF
    x0, y0, S = IMG
    cell = S / N
    bx, by, bs = x0 + c0 * cell, y0 + r0 * cell, 4 * cell
    lv = leaves(1)
    idx = int(np.nonzero((lv == np.array(NOTE_LEAF)).all(1))[0][0])
    tok_col = tuple(np.clip(ctx_cols(1)[idx], 0, 255).astype(int))
    with c.fade(a):
        dark = seg(t, T_CALL + 0.2, T_CALL + 0.6)
        if dark > 0:
            box = tuple(int(v * K) for v in (x0, y0, x0 + S, y0 + S))
            reg = c.im.crop(box)
            c.im.paste(Image.blend(reg, Image.new("RGB", reg.size), 0.78 * dark), box[:2])
        c.rect(bx - 1, by - 1, bx + bs + 1, by + bs + 1, outline=WHITE, width=2.5)
        # magnified patches
        g = ease(seg(t, T_CALL + 0.2, T_CALL + 0.8))
        GX, GY, GS, gap = 100, 365, 240, 8
        tgt = lerp(np.array([bx, by, bs]), np.array([GX, GY, GS]), g)
        im = out_img(0)
        sp = im.width / 1024
        if g > 0:
            ps = (tgt[2] - 3 * gap * g) / 4
            for i in range(4):
                for j in range(4):
                    crop = im.crop((int((c0 + j) * 16 * sp), int((r0 + i) * 16 * sp),
                                    int((c0 + j + 1) * 16 * sp), int((r0 + i + 1) * 16 * sp)))
                    c.paste(crop, tgt[0] + j * (ps + gap * g), tgt[1] + i * (ps + gap * g), ps)
        if t >= T_CALL + 0.8:
            c.text(GX + GS / 2, GY + GS + 30, "16 patches", 24, DIM)
        if t >= T_CALL + 1.0:
            c.poly([(372, GY), (470, GY + 85), (470, GY + GS - 85), (372, GY + GS)], fill=BOX, outline=BOXS,
                   width=2)
            c.text(421, GY - 28, "down-projection", 24, DIM)
        if t >= T_CALL + 1.3:
            c.rect(497, GY + GS / 2 - 36, 569, GY + GS / 2 + 36, fill=tok_col, r=7)
            c.rect(497, GY + GS / 2 - 36, 569, GY + GS / 2 + 36, outline=WHITE, width=2, r=7)
            c.text(533, GY + GS + 30, "1 token", 24, DIM)
        if t >= T_CALL + 1.5:
            cx0, cy0, cS = CTX
            cc = cS / N
            c.line([(569, GY + GS / 2), (cx0 + c0 * cc, cy0 + (r0 + 2) * cc)], (200, 200, 200), 1.5)


# ------------------------------------------------------------------ frame
def render(t):
    c = Canvas()
    # instruction / turn
    k_show = 1 if t < T_TURN2 else (2 if t < T_CHAIN[0] else chain_turn(t))
    with c.fade(1 - seg(t, *T_OUT)):
        if t < T_OUT[1]:
            instruction(c, t, k_show)

    # ---- image panel
    x0, y0, S = IMG
    if t < T_PASTE[1]:
        img_t = 0
    elif t < T_CHAIN[0]:
        img_t = 1
    else:
        img_t = chain_turn(t)
    c.paste(out_img(img_t), x0, y0, S)
    frame_box(c, IMG)
    c.text(x0 + S / 2, y0 - 22, "current image" if t < T_OUT[0] else f"after turn {img_t}", 26, DIM)
    if 0.5 <= t < T_OUT[0] and not (T_DECODE[0] <= t < T_TURN2):
        draw_region_outline(c, IMG, REGION[k_show - 1])

    # ---- context panel
    with c.fade(1 - seg(t, *T_OUT)):
        if t < T_OUT[1]:
            ghost(c, CTX)
            ghost(c, TGT)
            if t >= T_FLY1[0]:
                label(c, CTX, "context tokens · current image")
            if T_FLY1[1] <= t < T_FLY2[1]:
                if t < T_SWEEP[0]:
                    lv1 = np.array([[i, i + 1, j, j + 1] for i in range(N) for j in range(N)])
                    with c.fade(1.0):
                        draw_tokens(c, CTX, lv1, cell_cols(0).reshape(-1, 3))
                    a = 1 - seg(t, T_FLY1[1], T_FLY1[1] + 0.25)
                    if a > 0:
                        with c.fade(a):
                            c.paste(out_img(0), *CTX)
                    n = 4096
                    oi = None
                else:
                    lv2, cols2, n = sweep_state(t)
                    oi = None
                    if T_CALL <= t < T_CALL_OUT[1]:
                        oi = int(np.nonzero((lv2 == np.array(NOTE_LEAF)).all(1))[0][0])
                    draw_tokens(c, CTX, lv2, cols2, outline_idx=oi)
                count_line(c, CTX, n)
            elif t >= T_FLY2[1]:
                k = 2 if t < T_CHAIN[0] else chain_turn(t)
                draw_tokens(c, CTX, leaves(k), ctx_cols(k))
                count_line(c, CTX, TT[k - 1]["n_tokens"])
            if T_FLY1[0] <= t < T_FLY1[1]:
                fly(c, out_img(0), IMG, CTX, seg(t, *T_FLY1))
            if T_FLY2[0] <= t < T_FLY2[1]:
                draw_tokens(c, CTX, leaves(1), ctx_cols(1))
                fly(c, out_img(1), IMG, CTX, seg(t, *T_FLY2))

            # ---- target panel
            if t >= T_TGT[0]:
                label(c, TGT, "target tokens · noise")
            if T_TGT[0] <= t < T_TGT[1]:
                u = ease(seg(t, *T_TGT))
                draw_tokens(c, TGT, leaves(1), lerp(ctx_cols(1), noise_cols(1), u), dy=-(TGT[1] - CTX[1]) * (1 - u))
            elif T_TGT[1] <= t < T_DECODE[0]:
                cols, flash = sigma_cols(t, 1)
                draw_tokens(c, TGT, leaves(1), cols)
                if flash > 0:
                    out_idx = np.nonzero(~inside(1))[0]
                    with c.fade(0.4 * flash):
                        draw_tokens(c, TGT, leaves(1)[out_idx], np.full((len(out_idx), 3), 255.0))
                count_line(c, TGT, TT[0]["n_tokens"])
            elif T_DECODE[0] <= t < T_TGT_OUT[1]:
                a = 1 - seg(t, *T_TGT_OUT)
                with c.fade(a):
                    u = seg(t, *T_DECODE)
                    g = gen_img(1)
                    dimmed = Image.blend(Image.new("RGB", g.size), g, 0.3)
                    dimmed.paste(g, (0, 0), region_mask(1, g.width))
                    if u < 1:
                        cols, _ = sigma_cols(T_DECODE[0], 1)
                        draw_tokens(c, TGT, leaves(1), cols)
                    with c.fade(u):
                        c.paste(dimmed, *TGT)
                    count_line(c, TGT, TT[0]["n_tokens"])
            elif T_FLY2[1] <= t:
                k = 2 if t < T_CHAIN[0] else chain_turn(t)
                draw_tokens(c, TGT, leaves(k), noise_cols(k) if t < T_CHAIN[0] else gen_cols(k))
                count_line(c, TGT, TT[k - 1]["n_tokens"])

            # ---- arrows, DiT, readout
            if t >= T_ARROWS:
                c.arrow(CTX[0] + CTX[2] + 18, CTX[1] + CTX[2] / 2, DIT[0] - 12, DIT[1] + 150)
                c.arrow(TGT[0] + TGT[2] + 18, TGT[1] + TGT[2] / 2, DIT[0] - 12, DIT[1] + DIT[3] - 150)
            dit_box(c, t)
            readout(c, t)

    # paste flight (over everything)
    if T_PASTE[0] <= t < T_PASTE[1]:
        g = gen_img(1)
        fly(c, g, TGT, IMG, seg(t, *T_PASTE), mask=region_mask(1, g.width))

    callout(c, t)

    # ---- speed card
    if t >= T_ROWS[0]:
        speed_card(c, t)

    # caption
    cap = [s for s0, s in CAPTIONS if t >= s0][-1]
    c.text(W / 2, CAP_Y, cap, 42)
    im = c.im
    if SS > 1:
        im = im.resize((int(W * OUTS), int(H * OUTS)), Image.LANCZOS)
    return im


def speed_card(c, t):
    L, R = 760, 1840
    tm = TIMING
    base, full, lot = tm["base_full"][0], tm["full"][0], tm["k4"][0]
    navg = int(round(np.mean([x["n_tokens"] for x in TT])))
    c.text((L + R) / 2, 238, "seconds per edit turn · FLUX.2 [klein] 9B · 1024 × 1024", 28, DIM)
    rows = [(T_ROWS[0], 330, "full-res · base model · 50 steps + CFG", f"{base:.0f} s", WHITE),
            (T_ROWS[1], 470, "full-res · distilled · 4 steps", f"{full:.1f} s", WHITE),
            (T_ROWS[2], 610, "LoT · distilled · 4 steps", f"{lot:.1f} s", WHITE)]
    for t0, y, lab, val, col in rows:
        if t >= t0:
            c.text(L, y, lab, 34, anchor="l")
            c.text(R, y, val, 44, col, style="bx", anchor="r")
    if t >= T_ROWS[1]:
        c.text(R, 400, f"÷{base / full:.0f}  fewer steps", 28, PURPLE, anchor="r")
    if t >= T_ROWS[2]:
        c.text(R, 540, f"÷{full / lot:.1f}  fewer tokens", 28, PURPLE, anchor="r")
        w = c.text(L, 652, f"{navg} + {navg}", 26, AMBER, anchor="l")
        c.text(L + w + 8, 652, "of 4,096 + 4,096 tokens (target + context, avg.)", 26, DIM, anchor="l")
    if t >= T_ROWS[3]:
        c.text((L + R) / 2, 790, f"≈{base / lot:.0f}× faster per turn", 84, PURPLE, style="bx")
        c.text((L + R) / 2, 862, "1.8 s per edit vs. the full-res base model", 26, DIM)


# ------------------------------------------------------------------ main
def _frame(i):
    return np.asarray(render(i / FPS)).tobytes()


def main():
    if MODE == "still":
        os.makedirs(f"{HERE}/scratch", exist_ok=True)
        for s in sys.argv[2:]:
            render(float(s)).save(f"{HERE}/scratch/still_{VER}_{s}.png")
            print("wrote", s)
        return
    nf = int(round(DUR * FPS))
    res = "1080p60" if MODE == "h" else "540p15"
    os.makedirs(f"{HERE}/media", exist_ok=True)
    out = f"{HERE}/media/LoT_edit_{VER}_{res}.mp4"
    w, h = int(W * OUTS), int(H * OUTS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    with Pool(48) as p:
        for i, b in enumerate(p.imap(_frame, range(nf), chunksize=4)):
            ff.stdin.write(b)
            if i % 120 == 0:
                print(f"frame {i}/{nf}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
