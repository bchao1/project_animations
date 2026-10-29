"""Level-of-Token Diffusion — ~1-minute teaser, v1.

Source: "Level-of-Token Diffusion", anonymous ICLR 2027 submission 7014
(../7014_Level_of_Token_Diffusion.pdf, supplementary site mirrored in ../web/).
Running example: the site's painted demo "A girl riding a corgi" (LoT-FLUX.2); its 64 × 64 layout is rebuilt
from the site's token grid (quadtree of A.3, Eq. 23, reproduces the site's 1,312 / 140 / 79 / 15 leaves).

Render (see render.sh):
    manim -ql --media_dir media animation_lot_1min_v1.py <Scene>
Scenes, in order: S1_Insight, S2_Method, S3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # lot_diffusion_sandbox (read-only sources)
IMG = os.path.join(ROOT, "data", "img")
CLIPS = os.path.join(ROOT, "data", "clips")
WEB = os.path.join(ROOT, "web", "assets")
PDF = os.path.join(ROOT, "7014_Level_of_Token_Diffusion.pdf")
GEN = os.path.join(HERE, "data", "generated")                  # assets built on import

config.background_color = BLACK

# ------------------------------------------------------------------ style (as foveated_diffusion/teaser_1min v4)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × ⊙ − → ≈ …
HR = "#448AFF"                           # 1 × 1 (full resolution)
LR = "#00C853"                           # 2 × 2
LR3 = "#1B7F3B"                          # 4 × 4
LR4 = "#0E5A2A"                          # 8 × 8
EXT_COL = {1: HR, 2: LR, 4: LR3, 8: LR4}
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
RED = "#FF5252"
AMBER = "#FFCA28"
PURPLE = "#E040FB"
LORA = "#FF7043"
BOX = "#1E2A33"
PAINT = {"0": "#000000", "0.3": "#FF8A00", "0.6": "#FFD400", "1": "#FFFFFF"}   # brush colours of the site's canvas

CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤‖·↓")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def sub(x):
    return f'<span rise="-3000"><small><small>{x}</small></small></span>'


def sup(x):
    return f'<span rise="4500"><small><small>{x}</small></small></span>'


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def fit_width(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def img(path, h=None, w=None):
    m = ImageMobject(path)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def dit_box(center, w=1.5, h=2.0, lora=False, label="DiT"):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.15, fill_color=BOX,
                         fill_opacity=1, stroke_color="#90A4AE", stroke_width=2).move_to(center)
    g = VGroup(r, tx(label, 34).move_to(r.get_center() + (UP * 0.2 if lora else 0)))
    if lora:
        tag = RoundedRectangle(width=w * 0.62, height=0.38, corner_radius=0.08,
                               fill_color=LORA, fill_opacity=1, stroke_width=0)
        tag.move_to(r.get_bottom() + UP * 0.4)
        g.add(tag, tx("LoRA", 20, BLACK, bold=True).move_to(tag))
    return g


def fmt(n):
    return f"{n:,}"


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


N = 64                                   # FLUX.2 at 1024²: 64 × 64 token grid
QT = {8: 0.02, 4: 0.44, 2: 0.74}         # fitted split thresholds (A.3, Eq. 23); reproduce the site's leaf counts
SITE_LEAVES = {1: 1312, 2: 140, 4: 79, 8: 15}
TOKGRID = os.path.join(WEB, "b72bc22236f8abc9ffd53f8b980b6c031bad9faae21a35438deb4389cb7474fb.webp")
FIG7A_X = [109.1, 188.1, 267.1, 346.1, 425.1]     # panel left edges (pt) on PDF page 9, width 77.6
XT_T = [1.0, 0.75, 0.5, 0.25, 0.0]


def _layout_from_site():
    """Detail field D from the site's token-grid image (14 px per cell), then the quadtree of Eq. 23."""
    a = np.asarray(Image.open(TOKGRID).convert("RGB")).astype(float) / 255
    D = np.array([[a[i * 14 + 3:i * 14 + 11, j * 14 + 3:j * 14 + 11].mean() for j in range(N)] for i in range(N)])
    leaves = []

    def rec(i, j, b):
        if b > 1 and D[i:i + b, j:j + b].max() >= QT[b]:
            h = b // 2
            for di in (0, h):
                for dj in (0, h):
                    rec(i + di, j + dj, h)
        else:
            leaves.append((i, j, b))

    for i in range(0, N, 8):
        for j in range(0, N, 8):
            rec(i, j, 8)
    cnt = {b: sum(1 for *_, e in leaves if e == b) for b in (1, 2, 4, 8)}
    assert cnt == SITE_LEAVES, cnt
    return leaves


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    lp = gen("corgi_layout.json")
    if not os.path.exists(lp):
        tmp = lp + f".{os.getpid()}"
        json.dump(_layout_from_site(), open(tmp, "w"))
        os.replace(tmp, lp)
    if not os.path.exists(gen("fig7a_4.png")):
        import fitz
        page = fitz.open(PDF)[8]
        for k, x0 in enumerate(FIG7A_X):
            pix = page.get_pixmap(clip=fitz.Rect(x0 + 0.6, 97.4, x0 + 77.0, 174.5), matrix=fitz.Matrix(8, 8))
            tmp = gen(f"fig7a_{k}.{os.getpid()}.png")
            pix.save(tmp)
            os.replace(tmp, gen(f"fig7a_{k}.png"))
    if not os.path.exists(gen("xt_4.png")):
        x0 = np.asarray(Image.open(os.path.join(IMG, "paint0_result.png")).convert("RGB").resize((512, 512))) / 255.0
        eps = np.clip(0.5 + 0.28 * np.random.default_rng(7).standard_normal(x0.shape), 0, 1)
        for k, t in enumerate(XT_T):
            xt = (1 - t) * x0 + t * eps
            tmp = gen(f"xt_{k}.{os.getpid()}.png")
            Image.fromarray((np.clip(xt, 0, 1) * 255).astype(np.uint8)).save(tmp)
            os.replace(tmp, gen(f"xt_{k}.png"))


build_assets()
LEAVES = [tuple(l) for l in json.load(open(gen("corgi_layout.json")))]


# ------------------------------------------------------------------ Level-of-Token field
def tok(side, color):
    """Token of the given side. 1 × 1 tokens are plain squares (their 10% corner radius is sub-pixel)."""
    if side < 0.1:
        return Square(side_length=side, fill_color=color, fill_opacity=1, stroke_width=0)
    return RoundedRectangle(width=side, height=side, corner_radius=0.1 * side,
                            fill_color=color, fill_opacity=1, stroke_width=0)


def state_max(maxb):
    """Leaves with extent > maxb split into aligned maxb-blocks (maxb = 1: uniform grid)."""
    out = []
    for i, j, b in LEAVES:
        if b <= maxb:
            out.append((i, j, b))
        else:
            out += [(i + di, j + dj, maxb) for di in range(0, b, maxb) for dj in range(0, b, maxb)]
    return out


def state_min(minb):
    """Leaves with extent < minb are shown at their containing minb-block (minb = 8: all coarse)."""
    s = set()
    for i, j, b in LEAVES:
        if b >= minb:
            s.add((i, j, b))
        else:
            s.add((i - i % minb, j - j % minb, minb))
    return sorted(s)


class LoTField:
    """Token layout drawn on an n × n grid of size `size`; `morph` merges / splits blocks in place."""

    def __init__(self, blocks, size, center, n=N):
        self.n, self.size, self.c = n, size, size / n
        self.center = np.array([center[0], center[1], 0.0])
        self.toks = {b: self._make(b) for b in blocks}

    def pos(self, i, j, b):
        return self.center + np.array([(j + b / 2) * self.c - self.size / 2, self.size / 2 - (i + b / 2) * self.c, 0])

    def cell_rect(self, i, j, b, color, sw=3):
        return Square(side_length=b * self.c, color=color, stroke_width=sw).move_to(self.pos(i, j, b))

    def _make(self, blk):
        i, j, b = blk
        return tok(b * self.c - 0.2 * self.c, EXT_COL[b]).move_to(self.pos(i, j, b))

    def group(self):
        return VGroup(*self.toks.values())

    def count(self):
        return len(self.toks)

    def morph(self, blocks, focus=(0.5, 0.5), outward=True):
        """Animations from the current blocks to `blocks` (each new block either merges old ones or splits one)."""
        old = dict(self.toks)
        own_new = {}
        for nb in blocks:
            for di in range(nb[2]):
                for dj in range(nb[2]):
                    own_new[(nb[0] + di, nb[1] + dj)] = nb
        anims, made = [], {}
        merge_kids, split_kids = {}, {}
        for ob in old:
            nb = own_new[(ob[0], ob[1])]
            if nb == ob:
                made[nb] = old[ob]
            elif nb[2] > ob[2]:
                merge_kids.setdefault(nb, []).append(ob)
            else:
                split_kids[ob] = [own_new[(ob[0] + di, ob[1] + dj)]
                                  for di in range(0, ob[2], nb[2]) for dj in range(0, ob[2], nb[2])]
        for nb, kids in merge_kids.items():          # several old tokens → one new token
            m = self._make(nb)
            anims.append((nb, ReplacementTransform(VGroup(*[old[k] for k in kids]), m)))
            made[nb] = m
        for ob, kids in split_kids.items():          # one old token → several new tokens
            ms = [self._make(k) for k in kids]
            made.update(zip(kids, ms))
            anims.append((ob, ReplacementTransform(old[ob], VGroup(*ms))))
        self.toks = made
        fi, fj = focus[1] * self.n, focus[0] * self.n
        anims.sort(key=lambda a: ((a[0][0] + a[0][2] / 2 - fi) ** 2 + (a[0][1] + a[0][2] / 2 - fj) ** 2)
                   * (1 if outward else -1))
        return [a for _, a in anims]


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None

    # Captions stay up long enough to read: ~16 characters per second + 0.8 s (teaser pacing).
    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    @staticmethod
    def _split(s):
        """Split at the space nearest the middle that lies outside every markup element."""
        opened, spaces, n, k = 0, [], 0, 0
        while k < len(s):
            if s[k] == "<":
                end = s.index(">", k)
                opened += -1 if s[k + 1] == "/" else 1
                k = end + 1
                continue
            if opened == 0 and s[k] == " ":
                spaces.append((k, n))
            n += 1
            k += 1
        if not spaces:
            return [s]
        k = min(spaces, key=lambda p: abs(p[1] - n / 2))[0]
        return [s[:k], s[k + 1:]]

    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)   # never wrap inside “4 × 4”
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            parts = self._split(s)
            c = VGroup(*[tx(p, size - 2) for p in parts]).arrange(DOWN, buff=0.12)
            if c.width > 13.2:
                c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        if c.get_bottom()[1] < -3.85:
            c.shift(UP * (-3.85 - c.get_bottom()[1]))
        self.cap = c
        plain = re.sub(r"<[^>]+>", "", s)
        self.cap_t0 = self.renderer.time
        self.cap_need = len(plain) / 16.0 + 0.8
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def clear_all(self, keep=(), run_time=0.8):
        self.uncaption()
        mobs = [m for m in self.mobjects if m not in set(keep)]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)


# the girl's face and a patch of empty sky, in token coordinates (row, col) of the 64 × 64 grid
FACE = (13, 29)
SKY = (5, 5)
FOCUS = (0.47, 0.5)          # (u, v) of the subject, for LaggedStart order


def leaf_of(i, j):
    for (a, b, e) in LEAVES:
        if a <= i < a + e and b <= j < b + e:
            return (a, b, e)


def count_line(n, extra=""):
    return tx(f"<b>{fmt(n)}</b> tokens{extra}", 28)


# ================================================================== S1 Hook + insight
class S1_Insight(Base):
    def construct(self):
        # --- title with a preview of the layout: 8 × 8 tokens split down to the corgi layout
        title = tx("Level-of-Token Diffusion", 72, bold=True).move_to([0, 2.75, 0])
        subt = tx("adaptive, efficient generation from multiresolution token layouts", 32).move_to([0, 1.95, 0])
        fit_width(subt, 12.8)
        f0 = LoTField(state_min(8), 3.8, (0, -0.75))
        self.add(title, subt)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in f0.toks.values()], lag_ratio=0.02), run_time=1.0)
        for mb in (4, 2, 1):
            self.play(LaggedStart(*f0.morph(state_min(mb), FOCUS), lag_ratio=0.004), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(f0.group()), run_time=0.5)
        self.remove(title, subt)

        # --- hook: the image and the uniform grid
        S = 4.6
        L, R = np.array([-3.4, 0.1, 0]), np.array([3.4, 0.1, 0])
        pic = img(os.path.join(IMG, "paint0_result.png"), S).move_to(L)
        lab_l = tx("the image", 26, DIM).next_to(pic, UP, buff=0.25)
        field = LoTField(state_max(1), S, R)
        lab_r = tx("what the model computes", 26, DIM).move_to([R[0], lab_l.get_y(), 0])
        self.play(FadeIn(pic), run_time=0.5)
        self.add(lab_l, lab_r)
        self.caption("A diffusion transformer gives every patch the same token.")
        self.play(LaggedStart(*[FadeIn(m) for m in field.toks.values()], lag_ratio=0.0003), run_time=1.0)
        cnt_c = [R[0], -2.55, 0]
        cnt = count_line(4096, ", every denoising step").move_to(cnt_c)
        self.add(cnt)
        self.wait(0.6)

        def img_rect(i, j, color):
            c = S / N
            return Square(side_length=c, color=color, stroke_width=3).move_to(
                L + np.array([(j + 0.5) * c - S / 2, S / 2 - (i + 0.5) * c, 0]))

        bf_l, bs_l = img_rect(*FACE, AMBER), img_rect(*SKY, PURPLE)
        bf_r, bs_r = field.cell_rect(*FACE, 1, AMBER), field.cell_rect(*SKY, 1, PURPLE)
        for m in (bf_l, bs_l, bf_r, bs_r):
            m.set_z_index(10)
        tf = tx("face", 22, AMBER).next_to(bf_l, RIGHT, buff=0.12).set_z_index(10)
        ts = tx("sky", 22, PURPLE).next_to(bs_l, RIGHT, buff=0.12).set_z_index(10)
        self.caption("The girl's face and the empty sky cost exactly the same.")
        self.play(Create(bf_l), Create(bs_l), Create(bf_r), Create(bs_r), run_time=0.4)
        self.add(tf, ts)
        self._hold_caption()
        self.wait(0.4)

        # --- insight: painted detail map → merge passes
        cmap = img(os.path.join(IMG, "paint0_canvas.png"), S).move_to(L)
        edge = framed(cmap, GHOST)
        self.remove(tf, ts)
        self.play(FadeIn(cmap), FadeIn(edge), FadeOut(bf_l), FadeOut(bs_l), FadeOut(bf_r), run_time=0.8)
        self.remove(pic, lab_l)
        lab_l = tx("detail map, painted before generation", 26, DIM).move_to(lab_l)
        self.add(lab_l)
        leg = VGroup()
        for k, (v, c) in enumerate(PAINT.items()):
            sq = Square(side_length=0.26, fill_color=c, fill_opacity=1, stroke_color=DIM, stroke_width=1)
            leg.add(VGroup(sq, tx(v, 22).next_to(sq, RIGHT, buff=0.1)))
        leg.arrange(RIGHT, buff=0.35)
        leg = VGroup(tx("detail", 22, DIM), leg).arrange(RIGHT, buff=0.3).move_to([L[0], -2.55, 0])
        self.add(leg)
        self.caption("But where detail matters is often known in advance.")

        # the sky box follows its token as it merges
        self.caption("Level-of-Token layout: fine tokens where detail matters, coarse elsewhere.")
        for mb in (2, 4, 8):
            blk = [b for b in state_max(mb) if b[0] <= SKY[0] < b[0] + b[2] and b[1] <= SKY[1] < b[1] + b[2]][0]
            box = field.cell_rect(*blk, PURPLE).set_z_index(10)
            anims = field.morph(state_max(mb), FOCUS, outward=True)
            self.play(LaggedStart(*anims, lag_ratio=0.6 / max(len(anims), 1)),
                      Transform(bs_r, box), run_time=1.1)
            self.remove(cnt)
            cnt = count_line(field.count()).move_to(cnt_c)
            self.add(cnt)
            self.wait(0.35)
        self.play(FadeOut(bs_r), run_time=0.3)
        self.remove(cnt)
        cnt = tx(f"<b>{fmt(1546)}</b> tokens instead of {fmt(4096)}  ·  2.65× fewer", 28).move_to(cnt_c)
        fit_width(cnt, 6.4)
        self.add(cnt)
        el = VGroup()
        for b, n in SITE_LEAVES.items():
            sw = tok(0.26, EXT_COL[b])
            el.add(VGroup(sw, tx(f"{b} × {b}: {fmt(n)}", 20).next_to(sw, RIGHT, buff=0.08)))
        el.arrange(RIGHT, buff=0.28).move_to([R[0], -3.0, 0])
        fit_width(el, 6.2)
        self.add(el)
        self.caption("1,546 tokens instead of 4,096.")
        self.wait(1.5)
        self.uncaption()
        # keep the layout for the next scene: fade the rest
        self.play(*[FadeOut(m) for m in (cmap, edge, lab_l, leg, lab_r, cnt, el)], run_time=0.6)
        self.wait(0.1)


# ================================================================== S2 Method
class S2_Method(Base):
    def construct(self):
        S0 = 4.6
        field = LoTField(LEAVES, S0, (3.4, 0.1))
        self.add(field.group())
        S = 3.2
        FC = np.array([-4.65, 0.2, 0])
        self.play(field.group().animate.scale(S / S0).move_to(FC), run_time=0.9)
        field.size, field.c, field.center = S, S / N, FC
        lab_f = tx("LoT layout", 24, DIM).next_to(field.group(), UP, buff=0.2)
        self.add(lab_f)

        dit = dit_box([-1.35, 0.2, 0], w=1.6, h=2.2, lora=True)
        a1 = arrow([FC[0] + S / 2 + 0.05, 0.2, 0], dit.get_left(), sw=2)
        n_in = tx(fmt(1546), 22).next_to(a1, UP, buff=0.08)
        n_in2 = tx("tokens", 20, DIM).next_to(a1, DOWN, buff=0.08)
        self.caption("A pretrained DiT, fine-tuned once, reads only these tokens.")
        self.play(GrowArrow(a1), FadeIn(dit), run_time=0.5)
        self.add(n_in, n_in2)
        base = tx("pretrained FLUX.2 / Wan2.1 + LoRA", 20, DIM).next_to(dit, DOWN, buff=0.2)
        self.add(base)

        # one 4 × 4 token of grass
        cand = [l for l in LEAVES if l[2] == 4 and l[0] >= 52]
        g = min(cand, key=lambda l: abs(l[1] - 10))
        hl = field.cell_rect(*g, AMBER, sw=3).set_z_index(10)
        self.caption("A size-specific head returns full-resolution flow for each patch.")
        self.play(Create(hl), run_time=0.4)
        fly = field.toks[g].copy().set_z_index(11)
        self.play(fly.animate.move_to(dit.get_center()).scale(1.6), run_time=0.7)
        self.remove(fly)
        head = RoundedRectangle(width=0.95, height=0.62, corner_radius=0.08, fill_color=BOX, fill_opacity=1,
                                stroke_color=AMBER, stroke_width=2).move_to([0.3, 0.2, 0])
        head_l = VGroup(tx("head", 18), tx("4 × 4", 18)).arrange(DOWN, buff=0.04).move_to(head)
        a2 = arrow(dit.get_right(), head.get_left(), sw=2)
        side = 0.26
        fan = VGroup(*[tok(0.8 * side, HR).move_to([1.3 + (jj + 0.5) * side, 0.2 + (1.5 - ii) * side, 0])
                       for ii in range(4) for jj in range(4)])
        fan.move_to([1.72, 0.2, 0])
        a3 = arrow(head.get_right(), fan.get_left(), sw=2, buff=0.08)
        self.play(GrowArrow(a2), FadeIn(head), run_time=0.4)
        self.add(head_l)
        self.play(GrowArrow(a3), LaggedStart(*[FadeIn(t, scale=0.5) for t in fan], lag_ratio=0.05), run_time=0.8)
        fan_l = VGroup(tx("flow for all", 20, DIM), tx("16 positions", 20, DIM)).arrange(DOWN, buff=0.05)
        fan_l.next_to(fan, DOWN, buff=0.18)
        self.add(fan_l)

        # the full-resolution output, denoised (illustration)
        OC = np.array([4.65, 0.2, 0])
        frames = [img(gen(f"xt_{k}.png"), S).move_to(OC) for k in range(5)]
        lab_o = tx("full-resolution output", 24, DIM).next_to(frames[0], UP, buff=0.2)
        note = tx("illustration of denoising", 18, DIM).next_to(frames[0], DOWN, buff=0.15)
        self.caption("Every step runs on 38% of the tokens; the image stays full resolution.")
        self.play(FadeIn(frames[0]), run_time=0.4)
        self.add(lab_o, note)
        # the patch lands where its token sits
        tgt = OC + np.array([(g[1] + 2) * S / N - S / 2, S / 2 - (g[0] + 2) * S / N, 0])
        land = Square(side_length=4 * S / N, color=AMBER, stroke_width=3).move_to(tgt).set_z_index(10)
        self.play(ReplacementTransform(fan.copy(), land), run_time=0.6)
        for k in range(1, 5):
            self.play(FadeIn(frames[k]), run_time=0.45)
        self._hold_caption()
        self.wait(1.5)
        self.clear_all(run_time=0.6)
        self.wait(0.1)


# ================================================================== S3 Payoff + end card
def load_clip(name, h):
    fs = sorted(os.listdir(os.path.join(CLIPS, name)))
    arr = [np.asarray(Image.open(os.path.join(CLIPS, name, f)).convert("RGBA").resize((640, 360))) for f in fs]
    m = ImageMobject(arr[0].copy())
    m.set(height=h)
    return m, arr


class S3_Payoff(Base):
    def construct(self):
        # --- speed: one website video + the headline numbers
        h = 1.95
        lay, lay_f = load_clip("show03_lay", h)
        genv, gen_f = load_clip("show03_gen", h)
        lay.move_to([-3.5, 1.15, 0])
        genv.move_to([-3.5, -1.0, 0])
        def play_clip(m, frames):
            clock = {"t": 0.0}

            def upd(mob, dt):
                clock["t"] += dt
                mob.pixel_array = frames[min(int(clock["t"] * 15), len(frames) - 1)]   # 15 fps source
            return upd

        vl = tx("LoT-Wan2.1 14B  ·  semantic-mask layout", 22).next_to(lay, UP, buff=0.18)
        vn = tx("1.66× fewer tokens  ·  supplementary website", 18, DIM).next_to(genv, DOWN, buff=0.15)
        self.play(FadeIn(lay), FadeIn(genv), run_time=0.4)
        self.add(vl, vn)
        lay.add_updater(play_clip(lay, lay_f))
        genv.add_updater(play_clip(genv, gen_f))
        r1 = VGroup(tx("2.04×", 84, AMBER, bold=True), tx("faster images", 32))
        r2 = VGroup(tx("3.53×", 84, PURPLE, bold=True), tx("faster video", 32))
        for r in (r1, r2):
            r.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
        r1.move_to([3.2, 1.2, 0])
        r2.move_to([3.2, -0.25, 0]).align_to(r1, LEFT)
        r3 = VGroup(tx("FLUX.2 klein 4B · Wan2.1 14B", 22, DIM),
                    tx("at ≈ 3× fewer tokens (paper Tab. 1, 2)", 22, DIM)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        r3.next_to(r2, DOWN, buff=0.4).align_to(r1, LEFT)
        self.caption("Fewer tokens per step, faster generation.")
        self.add(r1, r2, r3)
        self.wait(5.0)
        lay.clear_updaters()
        genv.clear_updaters()
        self.clear_all(run_time=0.5)

        # --- quality at the same budget (paper Fig. 7a)
        s, pitch, y = 2.3, 2.52, 0.55
        names = [("Full res. FLUX.2", WHITE, "1.00×", ""), ("ToMe-SD", WHITE, "2.10×", "noisy low-res regions"),
                 ("DDiT", WHITE, "1.90×", "noisy low-res regions"),
                 ("Foveated Diffusion", WHITE, "1.63×", "distorted at token borders"),
                 ("Ours (LoT)", HR, "2.29×", "")]
        head = tx("image generation, same budget: 1,269 tokens  (paper Fig. 7a)", 22, DIM).move_to([0, 2.55, 0])
        self.add(head)
        ims = []
        for k, (n, c, sp, bad) in enumerate(names):
            m = img(gen(f"fig7a_{k}.png"), s).move_to([(k - 2) * pitch, y, 0])
            ims.append(m)
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.12), run_time=0.8)
        for k, (n, c, sp, bad) in enumerate(names):
            m = ims[k]
            self.add(fit_width(tx(n, 22, c, bold=(k == 4)), s + 0.1).next_to(m, UP, buff=0.14))
            self.add(tx(f"{sp} speedup", 22, c if k == 4 else DIM).next_to(m, DOWN, buff=0.14))
            if bad:
                self.add(fit_width(tx(bad, 18, RED), s).next_to(m, DOWN, buff=0.5))
        self.add(framed(ims[4], HR, 3))
        q = tx("HPSv3 at 2.94× fewer tokens:  full 10.43  ·  " + col("ours 9.55", HR)
               + "  ·  best baseline 7.84  (paper Tab. 1)", 22, DIM).move_to([0, -2.35, 0])
        self.add(fit_width(q, 12.8))
        self.caption("Same token budget: the baselines break, LoT keeps the detail.")
        self._hold_caption()
        self.wait(2.0)
        self.clear_all(run_time=0.5)

        # --- end card
        emb = LoTField(LEAVES, 1.9, (0, 2.3))
        t = tx("Level-of-Token Diffusion", 64, bold=True).move_to([0, 0.75, 0])
        tag = fit_width(tx("fine tokens where detail matters  ·  up to 2.04× faster images  ·  3.53× faster video",
                           30, HR), 12.8).move_to([0, -0.15, 0])
        a = tx("Anonymous ICLR 2027 submission 7014", 24).move_to([0, -1.2, 0])
        u = tx("iclr-submission7014.github.io/supplementary", 26, DIM).move_to([0, -1.85, 0])
        self.add(emb.group(), t, tag, a, u)
        self.wait(5.0)
