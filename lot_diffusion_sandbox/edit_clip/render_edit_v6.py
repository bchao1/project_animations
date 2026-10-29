"""LoT multi-turn in-context editing — 20-s clip, v6 (see storyboard.md).

v6 (2026-10-14): the speed equation uses the matched timing run (one editor, idle GPU 8, 5-turn bottle session):
~/LoT-george-branch/results/2026-10-14_173643_timing-matched-base-vs-distilled-bottles. Base full 197.8 s,
distilled full 7.85 s, distilled LoT k4 1.86 s per turn (means of turns 1-5). Everything else as v5.

v5 (2026-10-14, slide deck): CAPTIONS = False — no bottom caption (one section; the slide freezes on the last frame).

No manim: PIL frames (2× SSAA in h) piped to ffmpeg, 48 workers. Data from prep_data.py (data/generated/).

    ~/miniconda3/envs/imp_tome/bin/python render_edit_v5.py h|l      → media/LoT_edit_v5_<1080p60|540p15>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_edit_v5.py still 1.0 4.5 …   → scratch/still_v5_<t>.png
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
VER = "v6"
CAPTIONS = False                               # v5: False = no bottom caption
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
# v6: matched run (see docstring); [seconds, mean tokens, mean ref tokens]
TIMING = {"base_full": [197.782, 4096, 4096], "full": [7.846, 4096, 4096], "k4": [1.858, 835.6, 835.6]}
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
P = 560                                         # three equal panels, centred row
PC, PT, PR = (60, 260, P), (680, 260, P), (1300, 260, P)
LAB_Y, CNT_Y, INS_Y, CAP_Y = 230, 860, 152, 1012
BOXL, BOXR = (460, 330, 340, 420), (1120, 330, 340, 420)    # beat A: base | distilled (x, y, w, h)
ZOOM_LEAF = (40, 44, 24, 28)                    # 4 × 4 token on the front bottle's label (turn 1)

# ------------------------------------------------------------------ timeline (s)
T_TAG, T_A_OUT = (1.0, 2.0), (3.1, 3.6)
T_B = 3.6
T_SWEEP = (4.0, 5.0)
T_TGT = (5.1, 5.6)
T_ZOOM, T_ZOOM_OUT = 6.2, (8.3, 8.6)
T_STEP0, T_STEPLEN = 8.8, 0.5
T_RES = (10.8, 11.2)
T_SLIDE = (11.6, 12.3)
T_CHAIN = (12.4, 16.1)
T_E_OUT = (16.3, 16.8)
T_EQ = (16.9, 17.4, 17.9)
CAPTION_LINES = [(0.0, "The LoT LoRA from the base model transfers to the distilled model"),
            (3.6, "Trained only on text → image, in-context editing still works"),
            (6.2, "4 × 4 background → 1 token; the edit stays at full res"),
            (8.7, "4 steps; outside the edit, the source is kept"),
            (11.4, "Each result becomes the next turn’s context"),
            (14.0, "100 chained edits, 1.8 s each"),
            (16.3, "Distillation × LoT: 1.8 s per edit")]


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
def tagbox(c, x, y, s, size=28, fill=WHITE, anchor="l"):
    """Dark rounded text box (as the website clip); (x, y) = left / centre of the box's left or middle edge."""
    w = font("rm", size * K).getlength(s) / K + 24
    x0 = x if anchor == "l" else x - w / 2
    c.rect(x0, y - size * 0.75, x0 + w, y + size * 0.75, fill=(10, 10, 10), r=8)
    c.text(x0 + 12, y, s, size, fill, anchor="l")


def model_box(c, box, distilled):
    x, y, w, h = box
    cx = x + w / 2
    c.rect(x, y, x + w, y + h, fill=BOX, outline=BOXS, width=2, r=20)
    c.text(cx, y + 62, "FLUX.2 [klein] 9B", 32)
    c.text(cx, y + 170, "distilled" if distilled else "base", 48, style="bx")
    c.text(cx, y + 228, "4 steps" if distilled else "50 steps + CFG", 32, DIM)


def lora_tag(c, cx, cy, ghost_only=False):
    tw, th = 230, 60
    if ghost_only:
        c.rect(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2, outline=GHOST, width=2, r=10)
        return
    c.rect(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2, fill=LORA, r=10)
    c.text(cx, cy, "LoT LoRA", 34, (0, 0, 0), style="bx")


def beat_a(c, t):
    with c.fade(1 - seg(t, *T_A_OUT)):
        if t >= T_A_OUT[1]:
            return
        for box, lab, d in ((BOXL, "trained on text → image", False), (BOXR, "runs here, unchanged", True)):
            model_box(c, box, d)
            c.text(box[0] + box[2] / 2, box[1] - 40, lab, 30)
        tx = lambda b: (b[0] + b[2] / 2, b[1] + b[3] - 70)
        (x0, y0), (x1, y1) = tx(BOXL), tx(BOXR)
        lora_tag(c, x0, y0, ghost_only=True)
        c.arrow(BOXL[0] + BOXL[2] + 24, BOXL[1] + BOXL[3] / 2, BOXR[0] - 24, BOXR[1] + BOXR[3] / 2, fill=DIM)
        u = ease(seg(t, *T_TAG))
        lora_tag(c, lerp(x0, x1, u), lerp(y0, y1, u) - 170 * np.sin(np.pi * u))


CHANGE = {int(k): v for k, v in json.load(open(f"{GEN}/change_text.json")).items()}


def instruction(c, k):
    """v4: the edit as a top text box ("Change the … to …")."""
    s = f"“{CHANGE[k]}”"
    size = 34
    w = font("ti", size * K).getlength(s) / K
    cx = W / 2
    c.rect(cx - w / 2 - 22, INS_Y - size * 0.85, cx + w / 2 + 22, INS_Y + size * 0.85, fill=(26, 26, 26),
           outline=(85, 85, 85), width=1.5, r=10)
    c.text(cx, INS_Y, s, size, style="ti")


def turn_box(c, panel, k, size=44):
    """v4: large "turn N" box in each panel's bottom-right corner."""
    x0, y0, S = panel
    s = f"turn {k}"
    w = font("bx", size * K).getlength(s) / K
    bx1, by1 = x0 + S - 14, y0 + S - 14
    c.rect(bx1 - w - 28, by1 - size * 1.35, bx1, by1, fill=(12, 12, 12), r=10)
    c.text(bx1 - 14, by1 - size * 0.675, s, size, style="bx", anchor="r")


def count(c, panel, n):
    """Centred "N of 4,096 tokens (X%)" under a panel (amber number, white text)."""
    a, b = f"{fmt(n)}", f" of 4,096 tokens  ({pct(n)}%)"
    wa = font("bx", 32 * K).getlength(a) / K
    wb = font("rm", 32 * K).getlength(b) / K
    x0 = panel[0] + panel[2] / 2 - (wa + wb) / 2
    c.text(x0, CNT_Y, a, 32, AMBER, style="bx", anchor="l")
    c.text(x0 + wa, CNT_Y, b, 32, anchor="l")


def label(c, panel, s):
    c.text(panel[0] + panel[2] / 2, LAB_Y, s, 30)


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


def target_cols(t, k=1):
    """Target colours during the 4 steps: edit tokens go noise → result, the rest step to the source."""
    nz, gc, sc = noise_cols(k), gen_cols(k), ctx_cols(k)
    ins = inside(k)[:, None]
    x = ease(np.clip((t - T_STEP0) / (4 * T_STEPLEN), 0, 1))
    done = np.clip(np.floor((t - T_STEP0 + 0.05) / T_STEPLEN), 0, 4) / 4
    return np.where(ins, lerp(nz, gc, x), lerp(nz, sc, done))


def zoom(c, t):
    """Down-projection on one 4 × 4 context token: its 16 patches spread out, then merge into one token."""
    if not (T_ZOOM <= t < T_ZOOM_OUT[1]):
        return
    x0, y0, S = PC
    cell = S / N
    r0, r1, c0, c1 = ZOOM_LEAF
    tx, ty, ts = x0 + c0 * cell, y0 + r0 * cell, 4 * cell
    im = out_img(0)
    sp = im.width / 1024
    lv = leaves(1)
    idx = int(np.nonzero((lv == np.array(ZOOM_LEAF)).all(1))[0][0])
    col = tuple(np.clip(ctx_cols(1)[idx], 0, 255).astype(int))
    with c.fade(1 - seg(t, *T_ZOOM_OUT)):
        c.rect(tx - 2, ty - 2, tx + ts + 2, ty + ts + 2, outline=WHITE, width=3)
        g = ease(seg(t, T_ZOOM + 0.1, T_ZOOM + 0.5))          # spread out
        m = ease(seg(t, T_ZOOM + 0.9, T_ZOOM + 1.3))          # merge
        BS = 200                                              # magnified block side
        bx, by = x0 + 70, y0 + 40                             # magnified block (over the pool, top-left)
        cx, cy = lerp(tx, bx, g), lerp(ty, by, g)
        side = lerp(ts, BS, g)
        if g > 0:
            c.line([(tx + ts / 2, ty), (cx + side / 2, cy + side)], (200, 200, 200), 2)
            gap = 8 * g * (1 - m)
            ps = (side - 3 * gap) / 4
            if m < 1:
                for i in range(4):
                    for j in range(4):
                        pc = im.crop((int((c0 + j) * 16 * sp), int((r0 + i) * 16 * sp),
                                      int((c0 + j + 1) * 16 * sp), int((r0 + i + 1) * 16 * sp)))
                        c.paste(pc, cx + j * (ps + gap), cy + i * (ps + gap), ps)
            if m > 0:
                with c.fade(m):
                    c.rect(cx, cy, cx + side, cy + side, fill=col, r=0.1 * side)
            c.rect(cx - 2, cy - 2, cx + side + 2, cy + side + 2, outline=WHITE, width=2.5, r=0.1 * side * m)
            if t >= T_ZOOM + 0.5:
                tagbox(c, bx + BS / 2, by + BS + 40, "16 patches → 1 token" if m > 0.5 else "16 patches", 28, anchor="m")
        if t >= T_ZOOM + 1.5:
            draw_region_outline(c, PC, REGION[0], WHITE, 3)
            tagbox(c, x0 + 50 * cell, y0 + 50 * cell, "1 × 1 · full res", 28, anchor="m")


def speed_eq(c, t):
    base, full, lot = TIMING["base_full"][0], TIMING["full"][0], TIMING["k4"][0]
    navg = int(round(TIMING["k4"][1]))
    blocks = [(T_EQ[0], 430, f"{base / full:.0f}×", "fewer passes", "50 steps + CFG → 4 steps"),
              (T_EQ[1], 960, f"{full / lot:.1f}×", "fewer tokens", f"{pct(navg)}% of full-res tokens"),
              (T_EQ[2], 1490, f"≈{base / lot:.0f}×", "faster per edit", f"{base:.0f} s → {lot:.1f} s")]
    for i, (t0, x, big, l1, l2) in enumerate(blocks):
        if t < t0:
            continue
        c.text(x, 520, big, 110, PURPLE, style="bx")
        c.text(x, 625, l1, 40)
        c.text(x, 680, l2, 30, DIM)
        if i:
            c.text((x + blocks[i - 1][1]) / 2, 520, "×" if i == 1 else "=", 90)
    if t >= T_EQ[0]:
        c.text(W / 2, 360, "per edit turn · FLUX.2 [klein] 9B · 1024 × 1024", 32, DIM)


# ------------------------------------------------------------------ frame
def render(t):
    c = Canvas()
    if t < T_A_OUT[1]:
        beat_a(c, t)
    if T_B <= t < T_E_OUT[1]:
        with c.fade(1 - seg(t, *T_E_OUT)):
            main_beats(c, t)
    if t >= T_EQ[0]:
        speed_eq(c, t)
    if CAPTIONS:
        cap = [s for s0, s in CAPTION_LINES if t >= s0][-1]
        c.text(W / 2, CAP_Y, cap, 40)
    im = c.im
    if SS > 1:
        im = im.resize((int(W * OUTS), int(H * OUTS)), Image.LANCZOS)
    return im


def main_beats(c, t):
    chain = t >= T_SLIDE[1]
    k = 1 if not chain else (2 if t < T_CHAIN[0] else chain_turn(t))
    instruction(c, k)
    for p in (PC, PT, PR):
        ghost(c, p)
    label(c, PC, "context · current image")
    if t < T_STEP0 or t >= T_RES[0]:
        label(c, PT, "target")
    else:
        label(c, PT, f"target · step {min(4, int((t - T_STEP0) / T_STEPLEN) + 1)} / 4")
    label(c, PR, "result")

    # ---- context
    if not chain:
        with c.fade(1 - seg(t, T_SLIDE[0], T_SLIDE[0] + 0.3)):
            if t < T_SWEEP[0]:
                c.paste(out_img(0), *PC)
                count(c, PC, 4096)
            else:
                lv2, cols2, n = sweep_state(t)
                draw_tokens(c, PC, lv2, cols2)
                count(c, PC, n)
            zoom(c, t)
    else:
        draw_tokens(c, PC, leaves(k), ctx_cols(k))
        count(c, PC, TT[k - 1]["n_tokens"])

    # ---- target
    if not chain:
        with c.fade(1 - seg(t, T_SLIDE[0], T_SLIDE[0] + 0.3)):
            if T_TGT[0] <= t < T_TGT[1]:
                u = ease(seg(t, *T_TGT))
                x = lerp(PC[0], PT[0], u)
                draw_tokens(c, (x, PT[1], P), leaves(1), lerp(ctx_cols(1), noise_cols(1), u))
            elif t >= T_TGT[1]:
                draw_tokens(c, PT, leaves(1), target_cols(t))
                count(c, PT, TT[0]["n_tokens"])
    else:
        draw_tokens(c, PT, leaves(k), gen_cols(k))
        count(c, PT, TT[k - 1]["n_tokens"])

    for p in (PC, PT, PR):
        frame_box(c, p)
    # "+" and "→" in the gaps
    gy = PC[1] + P / 2
    c.text((PC[0] + P + PT[0]) / 2, gy, "+", 48)
    c.arrow(PT[0] + P + 12, gy, PR[0] - 12, gy, width=3, tip=16)

    # ---- result (drawn last: it slides over the other panels)
    if not chain:
        if t >= T_RES[0]:
            u = ease(seg(t, T_SLIDE[0] + 0.2, T_SLIDE[1]))
            with c.fade(seg(t, *T_RES)):
                c.paste(out_img(1), lerp(PR[0], PC[0], u), PR[1], P)
    else:
        c.paste(out_img(k), *PR)
    shown = [PC] if chain or t < T_SLIDE[0] else []
    if chain or T_TGT[1] <= t < T_SLIDE[0]:
        shown.append(PT)
    if chain or T_RES[0] <= t < T_SLIDE[0]:
        shown.append(PR)
    for p in shown:
        turn_box(c, p, k)


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
