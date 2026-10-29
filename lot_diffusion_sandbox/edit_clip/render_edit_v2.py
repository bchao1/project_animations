"""LoT multi-turn in-context editing — 20-s clip, v2 (see storyboard.md).

No manim: PIL frames (2× SSAA in h) piped to ffmpeg, 48 workers. Data from prep_data.py (data/generated/).

    ~/miniconda3/envs/imp_tome/bin/python render_edit_v2.py h|l      → media/LoT_edit_v2_<1080p60|540p15>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_edit_v2.py still 1.0 4.5 …   → scratch/still_v2_<t>.png
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
VER = "v2"
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
IMG = (30, 95, 780)
CTX = (850, 95, 390)
TGT = (850, 535, 390)
DIT = (1290, 250, 220, 520)                    # x, y, w, h (distilled model, final position)
BASEBOX = (920, 250, 220, 520)                 # base model, beat A only
RC, BAR = 1720, (1580, 1860)                   # readout centre x, % bar extent
INS_Y, CAP_Y, LEG_Y = 48, 1040, 935
PICKS = [((40, 44, 24, 28), "4 × 4 · background context", "16 patches → 1 token"),
         ((20, 22, 36, 38), "2 × 2 · ring around the edit", "4 patches → 1 token"),
         ((30, 31, 50, 51), "1 × 1 · the edited object", "full res: 1 patch = 1 token")]
CALL_Y = [250, 400, 720]

# ------------------------------------------------------------------ timeline (s)
T_TAGFLY, T_BASE_OUT = (1.0, 1.9), (2.8, 3.2)
T_FLY1 = (3.4, 3.9)
T_SWEEP = (4.2, 5.1)
T_TGT = (5.2, 5.7)
T_ARROWS = 5.7
T_CALL = 6.0
T_ROW = [6.2, 6.8, 7.4]
T_CALL_OUT = (8.7, 9.0)
T_STEP0, T_STEPLEN = 9.1, 0.65
T_DECODE = (11.8, 12.1)
T_PASTE = (12.1, 12.7)
T_TGT_OUT = (12.7, 12.95)
T_TURN2 = 13.0
T_FLY2 = (13.05, 13.5)
T_CHAIN = (13.6, 16.5)
T_OUT = (16.7, 17.1)
T_ROWS = (17.1, 17.6, 18.1, 18.6)
CAPTIONS = [(0.0, "Same LoT LoRA: base model → 4-step distilled model"),
            (3.4, "In-context editing, never trained"),
            (6.0, "Full res on the edit; context compressed 16 → 1"),
            (9.0, "4 steps; outside the edit, blend in the source"),
            (11.8, "The result is the next turn’s context"),
            (13.6, "100 chained turns, 1.8 s each"),
            (16.7, "Fewer steps × fewer tokens")]


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


def pct(n, full=4096):
    return int(round(100 * n / full))


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
def tag(c, x, y, s, size=24, fill=WHITE, bg=(12, 12, 12), style="rm"):
    """Text in a dark rounded box, top-left corner at (x, y) — overlay labels on panels."""
    w = font(style, size * K).getlength(s) / K
    c.rect(x, y, x + w + 20, y + size + 14, fill=bg, r=6)
    c.text(x + 10, y + (size + 14) / 2, s, size, fill, style=style, anchor="l")


def lora_tag(c, cx, cy, ghost_only=False):
    tw, th = 172, 46
    if ghost_only:
        c.rect(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2, outline=GHOST, width=2, r=8)
        return
    c.rect(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2, fill=LORA, r=8)
    c.text(cx, cy, "LoT LoRA", 26, (0, 0, 0), style="bx")


def model_box(c, box, distilled, t, glow=0.0):
    x, y, w, h = box
    cx = x + w / 2
    sc = tuple(int(lerp(a, b, glow)) for a, b in zip(BOXS, WHITE))
    c.rect(x, y, x + w, y + h, fill=BOX, outline=sc, width=2 + 2 * glow, r=18)
    c.text(cx, y + 50, "FLUX.2 [klein]", 28)
    c.text(cx, y + 85, "9B", 28)
    c.text(cx, y + 150, "distilled" if distilled else "base", 34, style="bx")
    c.text(cx, y + 192, "4 steps" if distilled else "50 steps + CFG", 26, DIM)


def tag_pos(box):
    x, y, w, h = box
    return x + w / 2, y + h - 52


def dit(c, t):
    """Beat A: LoRA tag flies base → distilled. Afterwards the distilled box runs the steps."""
    x, y, w, h = DIT
    cx = x + w / 2
    glow = 0.0
    for k in range(4):
        s = T_STEP0 + k * T_STEPLEN
        if s <= t < s + 0.4:
            glow = 1 - (t - s) / 0.4
    with c.fade(1 - seg(t, *T_BASE_OUT)):
        if t < T_BASE_OUT[1]:
            model_box(c, BASEBOX, False, t)
            bx, by = tag_pos(BASEBOX)
            lora_tag(c, bx, by, ghost_only=True)
            ax0, ax1, ay = BASEBOX[0] + BASEBOX[2] + 12, x - 12, y + h / 2
            c.arrow(ax0, ay, ax1, ay, fill=DIM, width=2.5)
            c.text((ax0 + ax1) / 2, ay - 34, "same LoRA", 22, DIM)
            c.text((ax0 + ax1) / 2, ay + 32, "no retraining", 22, DIM)
            c.text(BASEBOX[0] + BASEBOX[2] / 2, y + h + 34, "trained here:", 22, DIM)
            c.text(BASEBOX[0] + BASEBOX[2] / 2, y + h + 62, "text → image only", 22, DIM)
    model_box(c, DIT, True, t, glow)
    if T_STEP0 <= t < T_CHAIN[0]:
        k = min(4, int((t - T_STEP0) / T_STEPLEN) + 1)
        for i in range(4):
            c.rect(cx - 66 + i * 36, y + 228, cx - 42 + i * 36, y + 252, fill=WHITE if i < k else GHOST, r=4)
        dt = TT[0]["seconds"] * seg(t, T_STEP0, T_STEP0 + 4 * T_STEPLEN)
        c.text(cx, y + 315, f"{dt:.1f} s", 44, style="bx")
        c.text(cx, y + 357, "this turn", 22, DIM)
    elif t >= T_CHAIN[0]:
        k = chain_turn(t)
        cum = sum(v["seconds"] for v in TT[:k])
        c.text(cx, y + 315, f"{cum:.0f} s", 44, style="bx")
        c.text(cx, y + 357, f"total, {k} turns", 22, DIM)
    # LoRA tag: base → distilled along an arc
    u = ease(seg(t, *T_TAGFLY))
    (x0, y0), (x1, y1) = tag_pos(BASEBOX), tag_pos(DIT)
    lora_tag(c, lerp(x0, x1, u), lerp(y0, y1, u) - 150 * np.sin(np.pi * u))
    if t >= T_BASE_OUT[0]:
        c.text(cx, y + h + 34, "LoRA from the base model,", 22, DIM)
        c.text(cx, y + h + 62, "trained on text → image only", 22, DIM)


def instruction(c, k):
    s = f"turn {k}"
    ins = f"“{TT[k - 1]['instruction']}”"
    w1 = font("rm", 32 * K).getlength(s) / K
    w2 = font("ti", 32 * K).getlength(ins) / K
    gap = 24
    tot = w1 + gap + w2
    sc = min(1.0, 1860 / tot)
    x0 = W / 2 - tot * sc / 2
    c.text(x0, INS_Y, s, 32 * sc, AMBER, anchor="l")
    c.text(x0 + (w1 + gap) * sc, INS_Y, ins, 32 * sc, WHITE, style="ti", anchor="l")


def fly(c, im, a, b, u, mask=None):
    r = lerp(np.array(a, float), np.array(b, float), ease(u))
    c.paste(im, r[0], r[1], r[2], mask)


def readout(c, panel, title, n):
    """Count + % of full res beside a token panel."""
    y0 = panel[1]
    c.text(RC, y0 + 40, title, 26, DIM)
    c.text(RC, y0 + 112, fmt(n), 76, AMBER, style="bx")
    c.text(RC, y0 + 170, "of 4,096 tokens", 26, DIM)
    b0, b1 = BAR
    c.rect(b0, y0 + 200, b1, y0 + 220, fill=(40, 40, 40), r=4)
    c.rect(b0, y0 + 200, b0 + (b1 - b0) * n / 4096, y0 + 220, fill=AMBER, r=4)
    c.text(RC, y0 + 262, f"{pct(n)}% of full res", 34, AMBER, style="bx")


def sweep_state(t, k=1):
    """Context tokens during the diagonal sweep (4,096 1 × 1 → LoT layout) and the token count."""
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
    x = np.clip((t - T_STEP0) / (4 * T_STEPLEN), 0, 1)
    done = np.clip(np.floor((t - T_STEP0 + 0.1) / T_STEPLEN), 0, 4) / 4
    col_in = lerp(nz, gc, ease(x))
    col_out = lerp(nz, sc, done)
    flash = 0.0
    for j in range(1, 5):
        s = T_STEP0 + j * T_STEPLEN - 0.1
        if s <= t < s + 0.3:
            flash = 1 - (t - s) / 0.3
    return np.where(ins, col_in, col_out), flash


def pick_idx(lv):
    return [int(np.nonzero((lv == np.array(p)).all(1))[0][0]) for p, _, _ in PICKS]


def callout(c, t):
    """Over the dimmed image: one real token of each extent, its patches → down-projection → 1 token."""
    if not (T_CALL <= t < T_CALL_OUT[1]):
        return
    x0, y0, S = IMG
    cell = S / N
    im = out_img(0)
    sp = im.width / 1024
    cols = ctx_cols(1)
    idx = pick_idx(leaves(1))
    with c.fade(1 - seg(t, *T_CALL_OUT)):
        dark = seg(t, T_CALL, T_CALL + 0.3)
        box = tuple(int(v * K) for v in (x0, y0, x0 + S, y0 + S))
        reg = c.im.crop(box)
        c.im.paste(Image.blend(reg, Image.new("RGB", reg.size), 0.8 * dark), box[:2])
        for row, ((r0, r1, c0, c1), l1, l2) in enumerate(PICKS):
            t0 = T_ROW[row]
            if t < t0:
                continue
            e = r1 - r0
            bx, by, bs = x0 + c0 * cell, y0 + r0 * cell, e * cell
            # bright source block on the image
            crop = im.crop((int(c0 * 16 * sp), int(r0 * 16 * sp), int(c1 * 16 * sp), int(r1 * 16 * sp)))
            c.paste(crop, bx, by, bs)
            c.rect(bx - 2, by - 2, bx + bs + 2, by + bs + 2, outline=WHITE, width=2.5)
            P, gap = 36, 6
            bw = e * P + (e - 1) * gap
            cy = CALL_Y[row]
            g = ease(seg(t, t0, t0 + 0.45))
            gx, gy = lerp(bx, 240 - bw, g), lerp(by, cy - bw / 2, g)
            ps, gg = lerp(bs / e, P, g), gap * g
            for i in range(e):
                for j in range(e):
                    pc = im.crop((int((c0 + j) * 16 * sp), int((r0 + i) * 16 * sp),
                                  int((c0 + j + 1) * 16 * sp), int((r0 + i + 1) * 16 * sp)))
                    c.paste(pc, gx + j * (ps + gg), gy + i * (ps + gg), ps)
            if t >= t0 + 0.3:
                if e > 1:
                    c.poly([(262, cy - bw / 2), (346, cy - 28), (346, cy + 28), (262, cy + bw / 2)],
                           fill=BOX, outline=BOXS, width=2)
                    if row == 0:
                        c.text(304, cy - bw / 2 - 24, "down-projection", 22, DIM)
                else:
                    c.arrow(262, cy, 346, cy, fill=DIM, width=2.5, tip=14)
            if t >= t0 + 0.45:
                col = tuple(np.clip(cols[idx[row]], 0, 255).astype(int))
                c.rect(370, cy - 28, 426, cy + 28, fill=col, r=6)
                c.rect(370, cy - 28, 426, cy + 28, outline=WHITE, width=2, r=6)
                c.text(450, cy - 18, l1, 30, style="bx", anchor="l", max_w=350)
                c.text(450, cy + 22, l2, 26, DIM, anchor="l", max_w=350)


def legend(c):
    """Token sizes under the image: grey swatches with their patch sub-grid."""
    items = [(1, "1 × 1", "edited object, full res"), (2, "2 × 2", "ring, 4 → 1"), (4, "4 × 4", "background, 16 → 1")]
    xs = [40, 318, 548]
    for (e, a, b), x in zip(items, xs):
        s = 15.5 * e
        y = LEG_Y - s / 2
        c.rect(x, y, x + s, y + s, fill=(205, 205, 205), r=0.1 * s if e > 1 else 0)
        for i in range(1, e):
            c.line([(x + i * s / e, y + 2), (x + i * s / e, y + s - 2)], (120, 120, 120), 1.2)
            c.line([(x + 2, y + i * s / e), (x + s - 2, y + i * s / e)], (120, 120, 120), 1.2)
        c.text(x + s + 14, LEG_Y - 14, a, 26, style="bx", anchor="l")
        c.text(x + s + 14, LEG_Y + 16, b, 22, DIM, anchor="l")


def speed_card(c, t):
    L, R = 870, 1880
    base, full, lot = TIMING["base_full"][0], TIMING["full"][0], TIMING["k4"][0]
    navg = int(round(np.mean([v["n_tokens"] for v in TT])))
    c.text((L + R) / 2, 130, "seconds per edit turn · FLUX.2 [klein] 9B · 1024 × 1024", 30, DIM)
    rows = [(T_ROWS[0], 230, "full res · base model · 50 steps + CFG", f"{base:.0f} s",
             "100 passes (50 steps × 2 for CFG) · 8,192 tokens (100%)", DIM),
            (T_ROWS[1], 410, "full res · distilled · 4 steps", f"{full:.1f} s",
             "4 passes · 8,192 tokens (100%)", DIM),
            (T_ROWS[2], 590, "LoT · distilled · 4 steps", f"{lot:.1f} s",
             f"4 passes · {fmt(2 * navg)} tokens ({pct(2 * navg, 8192)}% of full res)", AMBER)]
    for t0, y, lab, val, sub, sc in rows:
        if t >= t0:
            c.text(L, y, lab, 38, anchor="l")
            c.text(R, y, val, 56, style="bx", anchor="r")
            c.text(L, y + 46, sub, 26, sc, anchor="l")
    if t >= T_ROWS[1]:
        c.text(R, 320, f"÷{base / full:.0f}  fewer passes", 32, PURPLE, anchor="r")
    if t >= T_ROWS[2]:
        c.text(R, 500, f"÷{full / lot:.1f}  fewer tokens", 32, PURPLE, anchor="r")
    if t >= T_ROWS[3]:
        c.text((L + R) / 2, 790, f"≈{base / lot:.0f}× faster per turn", 88, PURPLE, style="bx", max_w=1000)
        c.text((L + R) / 2, 870, f"{lot:.1f} s per edit, vs. {base:.0f} s for the full-res base model", 28, DIM)


# ------------------------------------------------------------------ frame
def render(t):
    c = Canvas()
    k_show = 1 if t < T_TURN2 else (2 if t < T_CHAIN[0] else chain_turn(t))
    out_a = 1 - seg(t, *T_OUT)
    with c.fade(out_a):
        if t < T_OUT[1]:
            instruction(c, k_show)

    # ---- image panel
    x0, y0, S = IMG
    img_t = 0 if t < T_PASTE[1] else (1 if t < T_CHAIN[0] else chain_turn(t))
    c.paste(out_img(img_t), x0, y0, S)
    frame_box(c, IMG)
    if 0.4 <= t < T_OUT[0] and not (T_DECODE[0] <= t < T_TURN2):
        draw_region_outline(c, IMG, REGION[k_show - 1])
    callout(c, t)
    tag(c, x0 + 12, y0 + 12, "current image" if t < T_OUT[0] else f"after {img_t} turns", 24)
    if t >= T_CALL_OUT[0]:
        with c.fade(seg(t, *T_CALL_OUT) * out_a):
            if t < T_OUT[1]:
                legend(c)

    with c.fade(out_a):
        if t < T_OUT[1]:
            # ---- context panel
            if t >= T_BASE_OUT[0]:
                with c.fade(seg(t, *T_BASE_OUT)):
                    ghost(c, CTX)
                    ghost(c, TGT)
            n_ctx = None
            if T_FLY1[1] <= t < T_FLY2[1]:
                if t < T_SWEEP[0]:
                    lv1 = np.array([[i, i + 1, j, j + 1] for i in range(N) for j in range(N)])
                    draw_tokens(c, CTX, lv1, cell_cols(0).reshape(-1, 3))
                    a = 1 - seg(t, T_FLY1[1], T_FLY1[1] + 0.25)
                    if a > 0:
                        with c.fade(a):
                            c.paste(out_img(0), *CTX)
                    n_ctx = 4096
                else:
                    lv2, cols2, n_ctx = sweep_state(t)
                    oi = None
                    if T_CALL <= t < T_CALL_OUT[1]:
                        oi = [int(np.nonzero((lv2 == np.array(p)).all(1))[0][0])
                              for (p, _, _), t0 in zip(PICKS, T_ROW) if t >= t0]
                        oi = oi or None
                    draw_tokens(c, CTX, lv2, cols2, outline_idx=oi)
            elif t >= T_FLY2[1]:
                k = 2 if t < T_CHAIN[0] else chain_turn(t)
                draw_tokens(c, CTX, leaves(k), ctx_cols(k))
                n_ctx = TT[k - 1]["n_tokens"]
            if T_FLY1[0] <= t < T_FLY1[1]:
                fly(c, out_img(0), IMG, CTX, seg(t, *T_FLY1))
            if T_FLY2[0] <= t < T_FLY2[1]:
                draw_tokens(c, CTX, leaves(1), ctx_cols(1))
                n_ctx = TT[0]["n_tokens"]
                fly(c, out_img(1), IMG, CTX, seg(t, *T_FLY2))
            if t >= T_FLY1[0]:
                tag(c, CTX[0] + 10, CTX[1] + 10, "context · current image", 22)
            if n_ctx is not None:
                readout(c, CTX, "context tokens", n_ctx)

            # ---- target panel
            n_tgt = None
            if T_TGT[0] <= t < T_TGT[1]:
                u = ease(seg(t, *T_TGT))
                draw_tokens(c, TGT, leaves(1), lerp(ctx_cols(1), noise_cols(1), u),
                            dy=-(TGT[1] - CTX[1]) * (1 - u))
            elif T_TGT[1] <= t < T_DECODE[0]:
                cols, flash = sigma_cols(t, 1)
                draw_tokens(c, TGT, leaves(1), cols)
                if flash > 0:
                    out_idx = np.nonzero(~inside(1))[0]
                    with c.fade(0.4 * flash):
                        draw_tokens(c, TGT, leaves(1)[out_idx], np.full((len(out_idx), 3), 255.0))
                n_tgt = TT[0]["n_tokens"]
            elif T_DECODE[0] <= t < T_TGT_OUT[1]:
                with c.fade(1 - seg(t, *T_TGT_OUT)):
                    u = seg(t, *T_DECODE)
                    g = gen_img(1)
                    dimmed = Image.blend(Image.new("RGB", g.size), g, 0.3)
                    dimmed.paste(g, (0, 0), region_mask(1, g.width))
                    if u < 1:
                        cols, _ = sigma_cols(T_DECODE[0], 1)
                        draw_tokens(c, TGT, leaves(1), cols)
                    with c.fade(u):
                        c.paste(dimmed, *TGT)
                n_tgt = TT[0]["n_tokens"]
            elif t >= T_FLY2[1]:
                k = 2 if t < T_CHAIN[0] else chain_turn(t)
                draw_tokens(c, TGT, leaves(k), noise_cols(k) if t < T_CHAIN[0] else gen_cols(k))
                n_tgt = TT[k - 1]["n_tokens"]
            if t >= T_TGT[0]:
                tag(c, TGT[0] + 10, TGT[1] + 10, "target · noise → edit", 22)
            if n_tgt is not None:
                readout(c, TGT, "target tokens", n_tgt)

            # ---- arrows, DiT
            if t >= T_ARROWS:
                c.arrow(CTX[0] + CTX[2] + 8, CTX[1] + CTX[2] / 2 + 40, DIT[0] - 8, DIT[1] + 140)
                c.arrow(TGT[0] + TGT[2] + 8, TGT[1] + TGT[2] / 2 - 40, DIT[0] - 8, DIT[1] + DIT[3] - 140)
            dit(c, t)

    if T_PASTE[0] <= t < T_PASTE[1]:
        g = gen_img(1)
        fly(c, g, TGT, IMG, seg(t, *T_PASTE), mask=region_mask(1, g.width))

    if t >= T_ROWS[0]:
        speed_card(c, t)

    cap = [s for s0, s in CAPTIONS if t >= s0][-1]
    c.text(W / 2, CAP_Y, cap, 40)
    im = c.im
    if SS > 1:
        im = im.resize((int(W * OUTS), int(H * OUTS)), Image.LANCZOS)
    return im


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
