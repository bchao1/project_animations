"""Foveated Diffusion — ~5-minute teaser, v1 (condensed from the ≈9 min 20 s explainer v4).

Source: Chao, Yariv, Xiao, Wetzstein, "Foveated Diffusion: Efficient Spatially
Adaptive Image and Video Generation", arXiv 2603.23491 (2026).
All images in data/ are cropped from the paper PDF (foveated-diffusion/pdfs/).

Render (see render.sh):
    manim -ql --media_dir media animation_foveated_diffusion_5min_v1.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image, ImageFilter
from scipy.interpolate import CubicSpline
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × ⊙ − → ≈ …
HR = "#448AFF"
LR = "#00C853"
LR3 = "#1B7F3B"                          # third (coarsest) level, outlook only
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
RED = "#FF5252"
AMBER = "#FFCA28"
PURPLE = "#E040FB"
LORA = "#FF7043"
BOX = "#1E2A33"

HEADER_Y = 3.62
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


# common symbols
def Z(s=None, p=None, c=None):
    t = "<i>z</i>" + (sub(s) if s is not None else "") + (sup(p) if p else "")
    return col(t, c) if c else t


def X(p=None, c=None):
    t = "<i>x</i>" + (sup(p) if p else "")
    return col(t, c) if c else t


ZH0 = Z("0", "high", HR)
ZL0 = Z("0", "low", LR)
ZF0 = Z("0", "fov")
ZF1 = Z("1", "fov")
ZFT = Z("<i>t</i>", "fov")
XH = X("high", HR)
XL = X("low", LR)
XF = X("fov")
M_ = "<i>M</i>"


# ------------------------------------------------------------------ tokens
def token(side, fill=HR, stroke=None, sw=0.0, op=1.0):
    return RoundedRectangle(
        width=side, height=side, corner_radius=0.1 * side,
        fill_color=fill, fill_opacity=op,
        stroke_color=stroke if stroke else fill, stroke_width=sw)


def ghost(side, dashed=False):
    r = RoundedRectangle(width=side, height=side, corner_radius=0.1 * side,
                         stroke_color=GHOST, stroke_width=1.5, fill_opacity=0)
    return DashedVMobject(r, num_dashes=16) if dashed else r


class Grid:
    """h×w token grid. Cell margin 10% → HR token side 0.8c, LR (2×2) 1.6c."""

    def __init__(self, n, size, center):
        self.n, self.size, self.c = n, size, size / n
        self.center = np.array([center[0], center[1], 0.0])

    def cell(self, i, j):
        return self.center + np.array(
            [(j + 0.5) * self.c - self.size / 2, self.size / 2 - (i + 0.5) * self.c, 0])

    def block(self, bi, bj):
        return self.center + np.array(
            [(2 * bj + 1) * self.c - self.size / 2, self.size / 2 - (2 * bi + 1) * self.c, 0])

    @property
    def hs(self):
        return 0.8 * self.c

    @property
    def ls(self):
        return 1.6 * self.c


def fovea_blocks(nb, cy, cx, r):
    return {(bi, bj) for bi in range(nb) for bj in range(nb)
            if (bi + 0.5 - cy) ** 2 + (bj + 0.5 - cx) ** 2 <= r * r + 1e-9}


def seq_order(nb, fb):
    """Raster order over 2×2 blocks; a fovea block contributes 4 HR tokens."""
    out = []
    for bi in range(nb):
        for bj in range(nb):
            if (bi, bj) in fb:
                for di in (0, 1):
                    for dj in (0, 1):
                        out.append(("H", (2 * bi + di, 2 * bj + dj)))
            else:
                out.append(("L", (bi, bj)))
    return out


def fovea_outline(g, fb, color=WHITE, sw=3):
    sq = [Square(side_length=2 * g.c).move_to(g.block(*b)) for b in fb]
    u = sq[0]
    for s in sq[1:]:
        u = Union(u, s)
    u.set_stroke(color, sw).set_fill(opacity=0)
    return u


def patch_colors(path, n):
    a = np.asarray(Image.open(path).convert("RGB").resize((n, n), Image.BOX)) / 255.0
    return [[rgb_to_color(a[i, j]) for j in range(n)] for i in range(n)]


def noise_colors(n, seed):
    rng = np.random.default_rng(seed)
    v = np.clip(0.5 + 0.2 * rng.standard_normal((n, n, 3)) * np.array([1, 1, 1]), 0.08, 0.92)
    g = np.clip(0.5 + 0.2 * rng.standard_normal((n, n, 1)), 0.1, 0.9)
    v = 0.75 * g + 0.25 * v
    return [[rgb_to_color(v[i, j]) for j in range(n)] for i in range(n)]


class FoveaField:
    """Foveated token field whose fovea can move: blocks split 1→4 / merge 4→1."""

    def __init__(self, g, fov, r, lr_color=LR):
        self.g, self.nb, self.r, self.lr_color = g, g.n // 2, r, lr_color
        fb = fovea_blocks(self.nb, fov[0], fov[1], r)
        self.state = {(bi, bj): self._make((bi, bj), (bi, bj) in fb)
                      for bi in range(self.nb) for bj in range(self.nb)}

    def _make(self, b, is_hr):
        bi, bj = b
        if is_hr:
            return ("H", VGroup(*[token(self.g.hs, HR).move_to(self.g.cell(2 * bi + di, 2 * bj + dj))
                                  for di in (0, 1) for dj in (0, 1)]))
        return ("L", token(self.g.ls, self.lr_color).move_to(self.g.block(bi, bj)))

    def mobs(self):
        return [m for (_, m) in self.state.values()]

    def group(self):
        return VGroup(*self.mobs())

    def count(self):
        return sum(4 if s == "H" else 1 for (s, _) in self.state.values())

    def retarget(self, fov, r=None):
        r = self.r if r is None else r
        fb = fovea_blocks(self.nb, fov[0], fov[1], r)
        anims = []
        for b, (s, m) in list(self.state.items()):
            want = "H" if b in fb else "L"
            if want != s:
                new = self._make(b, want == "H")
                anims.append(ReplacementTransform(m, new[1]))
                self.state[b] = new
        return anims


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


def _gaze_blur(img, gx, gy, r0, k, smax):
    a = np.asarray(img).astype(np.float32)
    H, W = a.shape[:2]
    levels = [0, 1, 2, 3.5, 5, 7, 9]
    stack = [a] + [np.asarray(img.filter(ImageFilter.GaussianBlur(s))).astype(np.float32)
                   for s in levels[1:]]
    yy, xx = np.mgrid[0:H, 0:W]
    e = np.sqrt((xx - gx * W) ** 2 + (yy - gy * H) ** 2)
    sig = np.clip(k * np.maximum(0, e - r0), 0, smax)
    out = np.zeros_like(a)
    for li in range(len(levels) - 1):
        lo, hi = levels[li], levels[li + 1]
        m = (sig >= lo) & (sig <= hi)
        w = ((sig - lo) / (hi - lo))[..., None]
        out[m] = ((1 - w) * stack[li] + w * stack[li + 1])[m]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen("done_v1")):
        return
    # --- gaze-contingent blur demo (Scene 2)
    far = Image.open(os.path.join(DATA, "fig5_farmer_full.png")).convert("RGB")
    _gaze_blur(far, 0.50, 0.55, 40, 0.045, 9).save(gen("gaze_A.png"))
    _gaze_blur(far, 0.80, 0.22, 40, 0.045, 9).save(gen("gaze_B.png"))
    # --- foveated rendering equation (Scene 2)
    W = 512
    yy, xx = np.mgrid[0:W, 0:W]
    m = (((xx - 0.5 * W) ** 2 + (yy - 0.55 * W) ** 2) <= (0.24 * W) ** 2).astype(np.float32)
    Image.fromarray((m * 255).astype(np.uint8)).convert("RGB").save(gen("fr_M.png"))
    Image.fromarray(((1 - m) * 255).astype(np.uint8)).convert("RGB").save(gen("fr_1mM.png"))
    up = far.resize((W // 8, W // 8), Image.BICUBIC).resize((W, W), Image.NEAREST)
    up.save(gen("fr_up.png"))
    comp = m[..., None] * np.asarray(far) + (1 - m[..., None]) * np.asarray(up)
    Image.fromarray(comp.astype(np.uint8)).save(gen("fr_comp.png"))
    # --- attention matrix of the dog tokens (Scene 1)
    dog = np.asarray(Image.open(os.path.join(DATA, "fig5_dog_full.png")).convert("RGB")
                     .resize((16, 16), Image.BOX)) / 255.0
    f = dog.reshape(-1, 3)
    d2 = ((f[:, None, :] - f[None, :, :]) ** 2).sum(-1)
    A = np.exp(-d2 / 0.02)
    A = A / A.sum(1, keepdims=True)
    A = A / A.max(1, keepdims=True)
    base = np.array([0x44, 0x8A, 0xFF]) / 255.0
    rgb = (0.06 + 0.94 * A[..., None]) * base
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST).save(gen("attn.png"))
    # --- generation (Scene 7): lion output, fovea blocks centered at (3, 3)
    lion = Image.open(os.path.join(DATA, "lion_out.png")).convert("RGB")
    L = lion.size[0]
    fb = fovea_blocks(8, 3, 3, 2.0)
    mask = np.zeros((16, 16), np.float32)
    for (bi, bj) in fb:
        mask[2 * bi:2 * bi + 2, 2 * bj:2 * bj + 2] = 1
    Mp = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((L, L), Image.NEAREST)) / 255.0
    Image.fromarray((Mp * 255).astype(np.uint8)).convert("RGB").save(gen("lion_Mp.png"))
    Image.fromarray(((1 - Mp) * 255).astype(np.uint8)).convert("RGB").save(gen("lion_1mMp.png"))
    la = np.asarray(lion)
    rgba = np.dstack([la, (Mp * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(gen("lion_xhigh.png"))
    low = lion.resize((L // 2, L // 2), Image.BICUBIC)
    low_a = np.asarray(low)
    mlow = np.asarray(Image.fromarray(((1 - mask) * 255).astype(np.uint8)).resize((L // 2, L // 2), Image.NEAREST))
    Image.fromarray(np.dstack([low_a, mlow]), "RGBA").save(gen("lion_xlow_periph.png"))
    low.save(gen("lion_xlow.png"))
    upl = low.resize((L, L), Image.BICUBIC)
    upl.save(gen("lion_xlow_up.png"))
    comp = Mp[..., None] * la + (1 - Mp[..., None]) * np.asarray(upl)
    Image.fromarray(comp.astype(np.uint8)).save(gen("lion_comp.png"))
    # --- training (Scene 6)
    tr = Image.open(os.path.join(DATA, "lion_train.png")).convert("RGB")
    tr.resize((tr.size[0] // 2, tr.size[1] // 2), Image.BICUBIC).save(gen("lion_train_down.png"))
    open(gen("done_v1"), "w").write("ok")


build_assets()


def img(path, h):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    m.set(height=h)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


# ------------------------------------------------------------------ scene base
class Base(Scene):
    header_text = ""

    def setup(self):
        self.cap = None
        self.hdr = None

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

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
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)   # never wrap inside “16 × 16”
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            parts = self._split(s)
            c = VGroup(*[tx(p, size - 2) for p in parts]).arrange(DOWN, buff=0.12)
            if c.width > 13.2:
                c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        if c.get_bottom()[1] < -3.85:              # tall two-line captions: keep on screen
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

    def title_card(self, num, title, hold=2.4):
        a = tx(num, 30, DIM)
        b = tx(title, 50, bold=True)
        if b.width > 12.5:
            b.scale_to_fit_width(12.5)
        g = VGroup(a, b).arrange(DOWN, buff=0.3)
        self.add(g)
        self.wait(hold)
        self.remove(g)

    def clear_all(self, keep=(), run_time=0.8):
        self.uncaption()
        keep = set(keep) | {self.hdr}
        mobs = [m for m in self.mobjects if m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def coder(label, center, w=0.62, h=0.95, widen_right=True):
    """VAE encoder/decoder trapezoid."""
    s = h / 2
    t = h / 4
    if widen_right:   # decoder: narrow → wide
        pts = [[-w / 2, t, 0], [w / 2, s, 0], [w / 2, -s, 0], [-w / 2, -t, 0]]
    else:             # encoder: wide → narrow
        pts = [[-w / 2, s, 0], [w / 2, t, 0], [w / 2, -t, 0], [-w / 2, -s, 0]]
    p = Polygon(*pts, color="#B0BEC5", stroke_width=2, fill_color="#37474F", fill_opacity=1)
    p.move_to(center)
    return VGroup(p, tx(label, 28).move_to(center))


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



# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("Foveated Diffusion", 72, bold=True).move_to([0, 2.75, 0])
        subt = tx("Efficient Spatially Adaptive Image and Video Generation", 32).move_to([0, 1.95, 0])
        g = Grid(16, 3.8, (0, -0.75))
        field = FoveaField(g, (4, 4), 2.0)
        auth = tx("Brian Chao*, Lior Yariv*, Howard Xiao, Gordon Wetzstein  ·  Stanford University",
                  22, DIM).move_to([0, -3.3, 0])
        self.add(title, subt, auth)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in field.mobs()], lag_ratio=0.02),
                  run_time=1.5)
        path = [(4, 4), (3.5, 3), (3, 5), (4.5, 5.5), (5, 3.5), (4, 4)]
        for p in path[1:]:
            anims = field.retarget(p)
            if anims:
                self.play(*anims, run_time=0.5)
            else:
                self.wait(0.5)
        self.wait(0.6)
        self.play(FadeOut(field.group()), run_time=0.6)
        self.remove(title, subt, auth)
        self.wait(0.2)


# ================================================================== S01 Tokens
class S01_TokenCost(Base):
    def construct(self):
        self.title_card("Part 1 · The problem", "Every token costs the same", hold=1.6)
        self.header("1 · Every token costs the same")

        g = Grid(16, 4.6, (-3.5, 0.1))
        pic = img("fig5_dog_full.png", 4.6).move_to(g.center)
        self.play(FadeIn(pic), run_time=0.5)
        self.caption("A diffusion transformer (DiT) generates an image as a grid of latent tokens.")
        lines = VGroup(*[Line(g.center + [-2.3 + k * g.c, -2.3, 0], g.center + [-2.3 + k * g.c, 2.3, 0])
                         for k in range(17)],
                       *[Line(g.center + [-2.3, -2.3 + k * g.c, 0], g.center + [2.3, -2.3 + k * g.c, 0])
                         for k in range(17)]).set_stroke(WHITE, 1, 0.55)
        self.play(Create(lines, lag_ratio=0.02), run_time=1.0)
        cols = patch_colors(os.path.join(DATA, "fig5_dog_full.png"), 16)
        toks = {(i, j): token(g.hs, cols[i][j]).move_to(g.cell(i, j)) for i in range(16) for j in range(16)}
        tg = VGroup(*toks.values())
        self.play(LaggedStart(*[FadeIn(t) for t in tg], lag_ratio=0.004), run_time=1.2)
        self.play(FadeOut(pic), FadeOut(lines), run_time=0.5)

        # face vs background
        face, bg = (4, 8), (2, 1)
        hf = SurroundingRectangle(toks[face], buff=0.035, color=AMBER, stroke_width=4)
        hb = SurroundingRectangle(toks[bg], buff=0.035, color=PURPLE, stroke_width=4)
        kf = Square(0.32).set_stroke(AMBER, 4).move_to([1.0, 1.1, 0])
        kb = Square(0.32).set_stroke(PURPLE, 4).move_to([1.0, 0.3, 0])
        lf = tx("a token on the dog’s face", 26).next_to(kf, RIGHT, buff=0.25)
        lb = tx("a token on the plain door", 26).next_to(kb, RIGHT, buff=0.25)
        self.play(Create(hf), Create(kf), Create(hb), Create(kb), run_time=0.5)
        self.add(lf, lb)
        self.caption("Both tokens cost exactly the same compute, though only one carries the subject.", 1.0)
        self.play(FadeOut(VGroup(hf, hb, kf, kb)), run_time=0.3)
        self.remove(lf, lb)

        # recolor to HR, attention
        self.play(LaggedStart(*[t.animate.set_fill(HR) for t in tg], lag_ratio=0.003), run_time=1.0)
        cnt = tx(f"<i>h</i> × <i>w</i> = 16 × 16 = {col('256', HR)} tokens", 32).move_to([3.4, 2.55, 0])
        self.add(cnt)
        att = img(gen("attn.png"), 4.0).move_to([3.4, -0.1, 0])
        attf = framed(att, DIM)
        self.play(FadeIn(att), Create(attf), run_time=0.7)
        lab = tx("256 × 256 = 65,536 query–key pairs", 26).next_to(att, DOWN, buff=0.22)
        self.add(lab)
        self.caption("Every denoising step, self-attention compares every token with every other token.")
        q = (5, 8)
        k = q[0] * 16 + q[1]
        rowy = att.get_top()[1] - (k + 0.5) / 256 * 4.0
        hq = SurroundingRectangle(toks[q], buff=0.04, color=WHITE, stroke_width=3)
        hr_ = Rectangle(width=4.1, height=0.07, color=WHITE, stroke_width=2).move_to([3.4, rowy, 0])
        self.play(Create(hq), Create(hr_), run_time=0.5)
        for qn in [(2, 3), (11, 12)]:
            kn = qn[0] * 16 + qn[1]
            yn = att.get_top()[1] - (kn + 0.5) / 256 * 4.0
            self.play(hq.animate.move_to(toks[qn]), hr_.animate.set_y(yn), run_time=0.6)
            self.wait(0.2)
        self.uncaption()
        self.play(FadeOut(Group(att, attf, hq, hr_)), run_time=0.5)
        self.remove(lab, cnt)

        # scaling squares
        s1 = Square(0.9).set_fill(HR, 0.6).set_stroke(HR, 2).move_to([1.2, 0.9, 0])
        s2 = Square(3.6).set_fill(HR, 0.6).set_stroke(HR, 2).move_to([4.6, 0.0, 0])
        l1a = tx("<i>N</i> tokens", 24).next_to(s1, UP, buff=0.2)
        l1b = tx("<i>N</i>" + sup("2") + " pairs", 24).next_to(s1, DOWN, buff=0.2)
        l2a = tx("2× resolution: 4<i>N</i> tokens", 24).next_to(s2, UP, buff=0.2)
        l2b = tx("16 <i>N</i>" + sup("2") + " pairs", 24).next_to(s2, DOWN, buff=0.2)
        self.play(FadeIn(s1), run_time=0.4)
        self.add(l1a, l1b)
        self.play(GrowFromCenter(s2), run_time=1.0)
        self.add(l2a, l2b)
        self.caption("Double the resolution: 4× the tokens, 16× the attention. Video multiplies it again by the frames.")
        self.caption("But does the viewer actually <i>see</i> every one of these tokens equally well?")
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S02 Vision
class S02_Vision(Base):
    def construct(self):
        self.header("2 · Where you look matters")

        ax = Axes(x_range=[-40, 40, 10], y_range=[0, 1.05, 0.25], x_length=5.6, y_length=3.4,
                  tips=False, axis_config={"include_numbers": False, "color": DIM, "stroke_width": 2}
                  ).move_to([-3.6, 0.35, 0])
        xt = VGroup(*[tx(str(v), 20, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in (-40, -20, 0, 20, 40)])
        xl = tx("eccentricity (degrees from gaze)", 22, DIM).next_to(ax, DOWN, buff=0.55)
        yl = tx("visual acuity", 22, DIM).next_to(ax.c2p(-40, 1.05), UP, buff=0.15).shift(RIGHT * 0.6)
        band = Rectangle(width=ax.c2p(5, 0)[0] - ax.c2p(-5, 0)[0], height=3.4,
                         stroke_width=0, fill_color=HR, fill_opacity=0.18).move_to(ax.c2p(0, 0.525))
        curve = ax.plot(lambda e: 1 / (1 + abs(e) / 2.5), x_range=[-40, 40, 0.1], color=WHITE,
                        stroke_width=4, use_smoothing=False)
        self.play(Create(ax), run_time=0.6)
        self.add(xt, xl, yl)
        self.caption("Our eyes see sharply only at the center of gaze, the fovea; acuity falls off fast in the periphery.")
        self.play(Create(curve), run_time=1.4)
        self.play(FadeIn(band), run_time=0.4)
        fl = tx("fovea", 22, HR).move_to(ax.c2p(12, 0.95))
        pl = tx("periphery", 22, DIM).move_to(ax.c2p(-26, 0.32))
        pr = tx("periphery", 22, DIM).move_to(ax.c2p(26, 0.32))
        note = tx("schematic", 18, DIM, italic=True).next_to(ax.c2p(40, 1.05), UP, buff=0.1).shift(LEFT * 0.4)
        self.add(fl, pl, pr, note)

        pic = img("fig5_farmer_full.png", 4.6).move_to([3.4, 0.25, 0])
        self.play(FadeIn(pic), run_time=0.6)
        side = 4.6

        def pt(u, v):
            return pic.get_center() + np.array([(u - 0.5) * side, (0.5 - v) * side, 0])

        dot = Dot(pt(0.50, 0.55), radius=0.1, color=RED).set_z_index(10)
        self.add(dot)
        self.caption("Keep your eyes on the red dot.")
        blurA = img(gen("gaze_A.png"), 4.6).move_to(pic)
        self.play(FadeIn(blurA), run_time=1.8)
        self.bring_to_front(dot)
        circ = DashedVMobject(Circle(radius=(40 / 512) * side * 1.9, color=WHITE, stroke_width=2),
                              num_dashes=30).move_to(dot).set_z_index(10)
        self.play(Create(circ), run_time=0.4)
        self.caption("The blurred periphery is hard to notice: fine detail is only resolved near the gaze.")
        blurB = img(gen("gaze_B.png"), 4.6).move_to(pic)
        self.play(dot.animate.move_to(pt(0.80, 0.22)), circ.animate.move_to(pt(0.80, 0.22)),
                  FadeIn(blurB), run_time=1.3)
        self.remove(blurA)
        self.clear_all(run_time=0.6)

        # foveated rendering: x_fov = M ⊙ x_high + (1 − M) ⊙ Up(x_low)
        eq = tx(f"{XF} = {M_} ⊙ {XH} + (1 − {M_}) ⊙ Up({XL})", 42).move_to([0, 2.35, 0])
        self.add(eq)
        self.caption("Graphics exploits this with <b>foveated rendering</b>: sharp fovea, cheap periphery, blended by a mask <i>M</i>.")
        s = 1.9
        items = [img(gen("fr_M.png"), s), tx("⊙", 40), img("fig5_farmer_full.png", s), tx("+", 40),
                 img(gen("fr_1mM.png"), s), tx("⊙", 40), img(gen("fr_up.png"), s), tx("=", 40),
                 img(gen("fr_comp.png"), s)]
        Group(*items).arrange(RIGHT, buff=0.22).move_to([0, 0.0, 0])
        labs = [tx(M_, 26), tx(XH, 26), tx(f"1 − {M_}", 26), tx(f"Up({XL})", 26), tx(XF, 26)]
        for lb, it in zip(labs, items[0::2]):
            lb.next_to(it, DOWN, buff=0.22)
        frames = [framed(items[k], DIM, 1) for k in (0, 4)]
        for k in range(0, 9, 2):
            anims = [FadeIn(items[k])]
            if k in (0, 4):
                anims.append(Create(frames[k // 4]))
            self.play(*anims, run_time=0.4)
            self.add(labs[k // 2])
            if k + 1 < 9:
                self.add(items[k + 1])
        self.wait(1.0)
        self.clear_all(run_time=0.6)
        q = tx("Can a diffusion model spend its tokens the same way?", 40)
        if q.width > 12.5:
            q.scale_to_fit_width(12.5)
        self.add(q)
        self.wait(2.2)
        self.remove(q)
        self.wait(0.2)


# ================================================================== S03 Tokenization
class S03_Tokenization(Base):
    def construct(self):
        self.title_card("Part 2 · The method", "Foveated tokenization", hold=1.6)
        self.header("3 · Foveated tokenization")

        g = Grid(16, 5.0, (-3.4, 0.1))
        fb = fovea_blocks(8, 3, 4, 2.0)
        toks = {(i, j): token(g.hs, HR).move_to(g.cell(i, j)) for i in range(16) for j in range(16)}
        self.play(LaggedStart(*[FadeIn(t) for t in toks.values()], lag_ratio=0.003), run_time=1.0)

        # mask panel
        mg = Grid(16, 2.4, (3.6, 1.2))
        mcells = VGroup()
        for i in range(16):
            for j in range(16):
                on = (i // 2, j // 2) in fb
                mcells.add(Square(mg.c * 0.92).set_fill(WHITE if on else "#2B2B2B", 1).set_stroke(width=0)
                           .move_to(mg.cell(i, j)))
        mlab = tx(f"{M_} ∈ {{0, 1}}" + sup("<i>h</i>×<i>w</i>"), 30).next_to(mcells, LEFT, buff=0.35)
        m1 = tx(f"{M_}(<i>i</i>, <i>j</i>) = 1: fovea, keep high-res", 24).move_to([3.6, -0.55, 0])
        m0 = tx(f"{M_}(<i>i</i>, <i>j</i>) = 0: periphery", 24).move_to([3.6, -1.05, 0])
        outline = fovea_outline(g, fb)
        periph = [t for (i, j), t in toks.items() if (i // 2, j // 2) not in fb]
        self.play(FadeIn(mcells), run_time=0.5)
        self.add(mlab, m1, m0)
        self.caption("A binary foveation mask <i>M</i> marks where high-resolution tokens are kept.")
        self.play(*[t.animate.set_fill(opacity=0.3) for t in periph], Create(outline), run_time=1.0)
        self.uncaption()
        self.play(*[t.animate.set_fill(opacity=1.0) for t in periph], FadeOut(mcells), run_time=0.5)
        self.remove(mlab, m1, m0)

        # merge periphery 2×2 → 1
        self.caption("In the periphery, every 2 × 2 block of tokens becomes one low-resolution token.")
        lr = {}
        blocks = sorted([(bi, bj) for bi in range(8) for bj in range(8) if (bi, bj) not in fb],
                        key=lambda b: (b[0] + 0.5 - 3) ** 2 + (b[1] + 0.5 - 4) ** 2)
        anims = []
        for (bi, bj) in blocks:
            four = VGroup(*[toks[(2 * bi + di, 2 * bj + dj)] for di in (0, 1) for dj in (0, 1)])
            lr[(bi, bj)] = token(g.ls, LR).move_to(g.block(bi, bj))
            anims.append(ReplacementTransform(four, lr[(bi, bj)]))
        self.play(LaggedStart(*anims, lag_ratio=0.04), run_time=2.8)
        hr = {k: t for k, t in toks.items() if (k[0] // 2, k[1] // 2) in fb}

        # counting
        L1 = tx(f"<i>m</i> = {col('48', HR)} high-res tokens", 30).move_to([3.3, 2.2, 0])
        L2 = tx(f"(<i>h</i>·<i>w</i> − <i>m</i>) / 4 = 208 / 4 = {col('52', LR)} low-res", 28).move_to([3.3, 1.3, 0])
        L3 = tx("<i>L</i> = <i>m</i> + (<i>h</i>·<i>w</i> − <i>m</i>) / 4 = <b>100</b>", 34).move_to([3.3, 0.2, 0])
        L4 = tx("100 / 256 ≈ 39% of the full sequence", 30).move_to([3.3, -0.9, 0])
        for Lk in (L1, L2, L3, L4):
            if Lk.get_right()[0] > 6.8:
                Lk.shift(LEFT * (Lk.get_right()[0] - 6.8))
        L3b = SurroundingRectangle(L3, buff=0.18, color=WHITE, stroke_width=2)
        self.add(L1)
        self.play(*[Indicate(t, color=WHITE, scale_factor=1.25) for t in hr.values()], run_time=0.7)
        self.add(L2)
        self.play(*[Indicate(t, color=WHITE, scale_factor=1.15) for t in lr.values()], run_time=0.7)
        self.add(L3)
        self.play(Create(L3b), run_time=0.4)
        self.add(L4)
        self.caption("Sequence length <i>L</i> = <i>m</i> + (<i>h</i>·<i>w</i> − <i>m</i>)/4: here 100 tokens instead of 256.")
        self.uncaption()
        self.remove(L1, L2, L3, L4)
        self.play(FadeOut(L3b), FadeOut(outline), run_time=0.3)

        # flatten into a sequence (+ one-line RoPE beat)
        order = seq_order(8, fb)
        pitch, s = 0.125, 0.1
        x0 = -pitch * (len(order) - 1) / 2
        moves = []
        for k, (kind, key) in enumerate(order):
            t = hr[key] if kind == "H" else lr[key]
            target = token(s, HR if kind == "H" else LR).move_to([x0 + k * pitch, 2.75, 0])
            moves.append(Transform(t, target))
        self.caption("The DiT sees one sequence of 100 tokens. Phase-aligned RoPE (Wu et al., 2025) keeps positions consistent across the two scales.")
        self.play(LaggedStart(*moves, lag_ratio=0.01), run_time=2.2)

        # attention area comparison
        tl = np.array([-4.2, 2.2, 0])
        big = Square(4.4).set_fill(HR, 0.25).set_stroke(DIM, 2)
        big.move_to(tl + np.array([2.2, -2.2, 0]))
        sm_side = 4.4 * 100 / 256
        small = Square(sm_side).set_fill(HR, 0.85).set_stroke(WHITE, 2)
        small.move_to(tl + np.array([sm_side / 2, -sm_side / 2, 0]))
        self.play(FadeIn(big), run_time=0.5)
        la = tx("full: 256 × 256 = 65,536 pairs", 28, DIM).move_to([3.4, 0.9, 0])
        self.add(la)
        self.play(TransformFromCopy(big, small), run_time=1.0)
        lb = tx("foveated: 100 × 100 = 10,000 pairs", 28).move_to([3.4, 0.2, 0])
        lc = tx("≈ 15% of the attention work", 28, HR).move_to([3.4, -0.5, 0])
        self.add(lb, lc)
        self.caption("Attention shrinks quadratically in <i>L</i>, MLPs linearly, and the savings apply at every step.")
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S04 Naive
class S04_Naive(Base):
    def construct(self):
        self.header("4 · Just plug it in?  Naïve mixed-resolution denoising")

        s = 2.35
        xs = [-1.7, 0.95, 3.6]
        yt, yb = 1.35, -1.2
        names = ["horse", "guitar", "farmer"]
        notes = ["duplicated horse", "two guitars", "two hat brims"]
        rt = tx("Full high-res", 26).move_to([-4.35, yt, 0])
        rb = tx("Naïve\nmixed-res", 26).move_to([-4.35, yb, 0])
        tops = [img(f"fig5_{n}_full.png", s).move_to([x, yt, 0]) for n, x in zip(names, xs)]
        bots = [img(f"fig5_{n}_naive.png", s).move_to([x, yb, 0]) for n, x in zip(names, xs)]
        self.add(rt, rb)
        self.play(LaggedStart(*[FadeIn(t) for t in tops + bots], lag_ratio=0.12), run_time=1.4)
        self.caption("Feed foveated tokens to a pretrained DiT without training, and it breaks at the fovea boundary (white circles).")
        for b, n in zip(bots, notes):
            self.add(tx(n, 22, RED).next_to(b, DOWN, buff=0.12))
            self.wait(0.4)
        self.caption("It never saw a token covering a 2 × 2 area, so it draws the periphery at the wrong scale.")
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S05 Training
class S05_Training(Base):
    def construct(self):
        self.header("5 · Foveated training: mixed-resolution-consistent targets")

        x = img("lion_train.png", 1.9).move_to([-5.7, 1.3, 0])
        xd = img(gen("lion_train_down.png"), 0.95).move_to([-5.7, -1.6, 0])
        xl = tx("<i>x</i>", 28).next_to(x, UP, buff=0.15)
        xdl = tx("Down(<i>x</i>)", 24).next_to(xd, DOWN, buff=0.15)
        self.play(FadeIn(x), run_time=0.5)
        self.add(xl)
        self.caption("Take a training image <i>x</i> and a bicubically downsampled copy Down(<i>x</i>).")
        adown = arrow(x.get_bottom(), xd.get_top(), buff=0.1, sw=2)
        dl = tx("bicubic", 20, DIM).next_to(adown, RIGHT, buff=0.1)
        self.play(GrowArrow(adown), run_time=0.3)
        self.add(dl)
        self.play(FadeIn(xd), run_time=0.4)
        self.add(xdl)

        e1 = coder("E", [-3.95, 1.3, 0], widen_right=False)
        e2 = coder("E", [-3.95, -1.6, 0], widen_right=False)
        gh = Grid(16, 1.9, (-1.9, 1.3))
        gl = Grid(8, 1.9, (-1.9, -1.6))
        ch = patch_colors(os.path.join(DATA, "lion_train.png"), 16)
        cl = patch_colors(os.path.join(DATA, "lion_train.png"), 8)
        zh = {(i, j): token(gh.hs, ch[i][j], HR, 1.2).move_to(gh.cell(i, j)) for i in range(16) for j in range(16)}
        zl = {(i, j): token(gl.hs, cl[i][j], LR, 1.6).move_to(gl.cell(i, j)) for i in range(8) for j in range(8)}
        a1 = arrow(x.get_right(), e1.get_left(), sw=2)
        a2 = arrow(xd.get_right(), e2.get_left(), sw=2)
        a3 = arrow(e1.get_right(), [-2.9, 1.3, 0], sw=2)
        a4 = arrow(e2.get_right(), [-2.9, -1.6, 0], sw=2)
        self.caption("The VAE encoder <i>E</i> turns both into tokens: 16 × 16 high-res and 8 × 8 low-res.")
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(e1), FadeIn(e2), run_time=0.5)
        self.play(GrowArrow(a3), GrowArrow(a4),
                  LaggedStart(*[FadeIn(t) for t in zh.values()], lag_ratio=0.002),
                  LaggedStart(*[FadeIn(t) for t in zl.values()], lag_ratio=0.01), run_time=1.2)
        lzh = tx(f"{ZH0} = <i>E</i>(<i>x</i>)", 26).next_to(VGroup(*zh.values()), UP, buff=0.15)
        lzl = tx(f"{ZL0} = <i>E</i>(Down(<i>x</i>))", 26).next_to(VGroup(*zl.values()), UP, buff=0.15)
        self.add(lzh, lzl)

        # mask + merge
        fb = fovea_blocks(8, 4, 4, 2.0)
        mg = Grid(16, 1.1, (0.55, -0.15))
        mcells = VGroup(*[Square(mg.c * 0.92).set_fill(WHITE if (i // 2, j // 2) in fb else "#2B2B2B", 1)
                          .set_stroke(width=0).move_to(mg.cell(i, j)) for i in range(16) for j in range(16)])
        mlab = tx(M_, 28).next_to(mcells, UP, buff=0.15)
        gf = Grid(16, 2.6, (3.5, -0.15))
        ghosts = VGroup(*[ghost(gf.hs).move_to(gf.cell(i, j)) for i in range(16) for j in range(16)])
        lzf = tx(f"{ZF0} = Merge({ZH0}, {ZL0}, {M_})", 28).move_to([3.5, 1.65, 0])
        self.caption("Merge with the mask: high-res tokens inside the fovea, low-res tokens outside.")
        self.play(FadeIn(mcells), FadeIn(ghosts), run_time=0.5)
        self.add(mlab, lzf)
        unused_h = [t for (i, j), t in zh.items() if (i // 2, j // 2) not in fb]
        unused_l = [t for k, t in zl.items() if k in fb]
        self.play(*[t.animate.set_opacity(0.2) for t in unused_h + unused_l], run_time=0.5)
        fh, fl = {}, {}
        anims = []
        for (i, j), t in zh.items():
            if (i // 2, j // 2) in fb:
                fh[(i, j)] = token(gf.hs, ch[i][j], HR, 1.2).move_to(gf.cell(i, j))
                anims.append(TransformFromCopy(t, fh[(i, j)]))
        for (bi, bj), t in zl.items():
            if (bi, bj) not in fb:
                fl[(bi, bj)] = token(gf.ls, cl[bi][bj], LR, 1.6).move_to(gf.block(bi, bj))
                anims.append(TransformFromCopy(t, fl[(bi, bj)]))
        self.play(LaggedStart(*anims, lag_ratio=0.01), run_time=2.2)
        self.remove(ghosts)
        self.caption("Both parts encode the <i>same</i> image, so the target is consistent across resolutions by construction.")

        # flow matching
        keep = list(fh.values()) + list(fl.values())
        self.clear_all(keep=keep, run_time=0.6)
        gf2 = Grid(16, 2.8, (-4.3, 0.35))
        clean, noise = {}, {}
        nz16 = noise_colors(16, 3)
        nz8 = noise_colors(8, 4)
        tgt = []
        for (i, j), t in fh.items():
            clean[t] = ManimColor(ch[i][j])
            noise[t] = ManimColor(nz16[i][j])
            tgt.append(t.animate.move_to(gf2.cell(i, j)).scale(gf2.c / gf.c))
        for (bi, bj), t in fl.items():
            clean[t] = ManimColor(cl[bi][bj])
            noise[t] = ManimColor(nz8[bi][bj])
            tgt.append(t.animate.move_to(gf2.block(bi, bj)).scale(gf2.c / gf.c))
        self.play(*tgt, run_time=0.9)
        eq = tx(f"{ZFT} = (1 − <i>t</i>) {ZF0} + <i>t</i> {ZF1},    {ZF1} ∼ 𝒩(0, <i>I</i>)", 34).move_to([0, 2.55, 0])
        self.add(eq)
        tt = ValueTracker(0.0)
        for t in keep:
            t.add_updater(lambda m, t=t: m.set_fill(interpolate_color(clean[t], noise[t], tt.get_value()), 1))
        nl = NumberLine(x_range=[0, 1, 0.25], length=2.8, color=DIM, include_numbers=False,
                        stroke_width=2).move_to([-4.3, -1.65, 0])
        n0 = tx("0", 20, DIM).next_to(nl.n2p(0), DOWN, buff=0.12)
        n1 = tx("1", 20, DIM).next_to(nl.n2p(1), DOWN, buff=0.12)
        knob = always_redraw(lambda: Dot(nl.n2p(tt.get_value()), radius=0.08, color=WHITE))
        tlab = always_redraw(lambda: tx(f"<i>t</i> = {tt.get_value():.2f}", 24).next_to(nl, UP, buff=0.15))
        self.add(nl, n0, n1, knob, tlab)
        self.caption("Then train with standard flow matching, entirely on the short foveated token sequence.")
        self.play(tt.animate.set_value(1.0), run_time=1.5)
        self.play(tt.animate.set_value(0.55), run_time=0.9)

        dit = dit_box([0.2, 0.35, 0], w=1.7, h=2.2, lora=True)
        ain = arrow([-2.8, 0.35, 0], dit.get_left(), sw=2)
        ainl = tx(f"{ZFT}, <i>t</i>", 24).next_to(ain, UP, buff=0.1)
        self.play(FadeIn(dit), GrowArrow(ain), run_time=0.5)
        self.add(ainl)
        gv = Grid(16, 2.8, (4.4, 0.35))
        vt = []
        for (i, j), t in fh.items():
            d = np.array(noise[t].to_rgb()) - np.array(clean[t].to_rgb())
            vt.append(token(gv.hs, rgb_to_color(np.clip(0.5 + 0.9 * d, 0, 1)), HR, 1.2).move_to(gv.cell(i, j)))
        for (bi, bj), t in fl.items():
            d = np.array(noise[t].to_rgb()) - np.array(clean[t].to_rgb())
            vt.append(token(gv.ls, rgb_to_color(np.clip(0.5 + 0.9 * d, 0, 1)), LR, 1.6).move_to(gv.block(bi, bj)))
        vg = VGroup(*vt)
        aout = arrow(dit.get_right(), [2.9, 0.35, 0], sw=2)
        voutl = tx("<i>v</i>" + sub("θ") + f"({ZFT}, <i>t</i>)", 26).next_to(vg, DOWN, buff=0.2)
        self.play(GrowArrow(aout), LaggedStart(*[FadeIn(v) for v in vt], lag_ratio=0.005), run_time=1.0)
        self.add(voutl)
        loss = tx(f"ℒ = 𝔼 ‖ <i>v</i>{sub('θ')}({ZFT}, <i>t</i>) − ({ZF1} − {ZF0}) ‖{sup('2')}", 34).move_to([0, -2.45, 0])
        self.add(loss)
        self.caption("Only LoRA adapters (rank 32) are trained on a frozen DiT: FLUX.2 Klein 4B for images, Wan2.1 1.3B for video.")
        self.play(Indicate(dit[2], color=WHITE, scale_factor=1.15), run_time=0.7)
        self.uncaption()
        for t in keep:
            t.clear_updaters()
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S06 Generation
class S06_Generation(Base):
    def construct(self):
        self.header("6 · Foveated generation")

        fb = fovea_blocks(8, 3, 3, 2.0)
        g = Grid(16, 2.6, (-5.0, 0.6))
        ch = patch_colors(os.path.join(DATA, "lion_out.png"), 16)
        cl = patch_colors(os.path.join(DATA, "lion_out.png"), 8)
        nz16, nz8 = noise_colors(16, 11), noise_colors(8, 12)
        hr, lr, clean, noise = {}, {}, {}, {}
        for bi in range(8):
            for bj in range(8):
                if (bi, bj) in fb:
                    for di in (0, 1):
                        for dj in (0, 1):
                            i, j = 2 * bi + di, 2 * bj + dj
                            t = token(g.hs, nz16[i][j], HR, 1.2).move_to(g.cell(i, j))
                            hr[(i, j)] = t
                            clean[t], noise[t] = ManimColor(ch[i][j]), ManimColor(nz16[i][j])
                else:
                    t = token(g.ls, nz8[bi][bj], LR, 1.6).move_to(g.block(bi, bj))
                    lr[(bi, bj)] = t
                    clean[t], noise[t] = ManimColor(cl[bi][bj]), ManimColor(nz8[bi][bj])
        allt = list(hr.values()) + list(lr.values())
        self.play(LaggedStart(*[FadeIn(t) for t in allt], lag_ratio=0.01), run_time=0.9)
        lab = tx(f"{ZF1} ∼ 𝒩(0, <i>I</i>)", 28).next_to(VGroup(*allt), UP, buff=0.2)
        self.add(lab)
        self.caption("At inference, sample noise directly in foveated token space and denoise from <i>t</i> = 1 to 0: only 100 tokens per step.")

        dit = dit_box([-2.2, 0.6, 0], w=1.4, h=1.8, lora=True)
        a_in = arrow([-3.6, 0.9, 0], [dit.get_left()[0], 0.9, 0], sw=2)
        a_back = arrow([dit.get_left()[0], 0.3, 0], [-3.6, 0.3, 0], sw=2)
        self.play(FadeIn(dit), GrowArrow(a_in), GrowArrow(a_back), run_time=0.5)
        tt = ValueTracker(1.0)
        for t in allt:
            t.add_updater(lambda m, t=t: m.set_fill(interpolate_color(clean[t], noise[t], tt.get_value()), 1))
        tl = always_redraw(lambda: tx(f"<i>t</i> = {tt.get_value():.2f}", 28).move_to([-5.0, -1.2, 0]))
        self.add(tl)
        for k in range(8):
            v = 1 - (k + 1) / 8
            self.play(tt.animate(rate_func=smooth).set_value(v),
                      Indicate(dit[0], color="#90A4AE", scale_factor=1.05), run_time=0.4)
        for t in allt:
            t.clear_updaters()
        self.remove(tl)
        lab2 = tx(f"{ZF0}", 28).move_to(lab)
        self.remove(lab)
        self.add(lab2)

        # split
        gh = Grid(16, 1.6, (0.9, 1.55))
        gl = Grid(8, 1.6, (0.9, -1.25))
        gh_ghost = VGroup(*[ghost(gh.hs).move_to(gh.cell(i, j)) for i in range(16) for j in range(16)
                            if (i // 2, j // 2) not in fb])
        gl_ghost = VGroup(*[ghost(gl.hs).move_to(gl.cell(i, j)) for i in range(8) for j in range(8)
                            if (i, j) in fb])
        self.caption("Split the clean tokens by the mask and decode each part with the VAE decoder <i>D</i>.")
        self.play(FadeIn(gh_ghost), FadeIn(gl_ghost), run_time=0.3)
        mv = []
        for (i, j), t in hr.items():
            mv.append(TransformFromCopy(t, token(gh.hs, ch[i][j], HR, 1.2).move_to(gh.cell(i, j))))
        for (bi, bj), t in lr.items():
            mv.append(TransformFromCopy(t, token(gl.hs, cl[bi][bj], LR, 1.6).move_to(gl.cell(bi, bj))))
        self.play(LaggedStart(*mv, lag_ratio=0.01), run_time=1.4)
        lh = tx(ZH0, 26).move_to([0.9, 2.6, 0])
        ll = tx(ZL0, 26).move_to([0.9, -0.2, 0])
        self.add(lh, ll)

        # decode
        d1 = coder("D", [2.5, 1.55, 0])
        d2 = coder("D", [2.5, -1.25, 0])
        b1 = arrow([1.75, 1.55, 0], d1.get_left(), sw=2)
        b2 = arrow([1.75, -1.25, 0], d2.get_left(), sw=2)
        xh = img(gen("lion_xhigh.png"), 1.6).move_to([4.35, 1.55, 0])
        xhf = framed(xh, GHOST, 1)
        xlo = img(gen("lion_xlow.png"), 0.8).move_to([4.35, -1.25, 0])
        c1 = arrow(d1.get_right(), xhf.get_left(), sw=2)
        c2 = arrow(d2.get_right(), xlo.get_left(), sw=2)
        self.play(GrowArrow(b1), GrowArrow(b2), FadeIn(d1), FadeIn(d2), run_time=0.5)
        self.play(GrowArrow(c1), GrowArrow(c2), FadeIn(xh), Create(xhf), FadeIn(xlo), run_time=0.7)
        lxh = tx(f"{XH} = <i>D</i>({ZH0})", 24).move_to([4.35, 2.65, 0])
        lxl = tx(f"{XL} = <i>D</i>({ZL0})", 24).move_to([4.35, -0.4, 0])
        self.add(lxh, lxl)

        # blend
        self.clear_all(run_time=0.6)
        eq = tx(f"{XF} = {M_}′ ⊙ {XH} + (1 − {M_}′) ⊙ Up({XL}),    {M_}′ = Up({M_})", 38).move_to([0, 2.35, 0])
        self.add(eq)
        self.caption("Upsample the low-res image and blend it with the sharp fovea using the upsampled mask <i>M</i>′.")
        s = 1.9
        items = [img(gen("lion_Mp.png"), s), tx("⊙", 40), img(gen("lion_xhigh.png"), s), tx("+", 40),
                 img(gen("lion_1mMp.png"), s), tx("⊙", 40), img(gen("lion_xlow_up.png"), s), tx("=", 40),
                 img(gen("lion_comp.png"), s)]
        Group(*items).arrange(RIGHT, buff=0.22).move_to([0, 0.0, 0])
        labs = [tx(f"{M_}′", 26), tx(XH, 26), tx(f"1 − {M_}′", 26), tx(f"Up({XL})", 26), tx(XF, 26)]
        for lb, it in zip(labs, items[0::2]):
            lb.next_to(it, DOWN, buff=0.22)
        fr = [framed(items[k], DIM, 1) for k in (0, 2, 4)]
        for k in range(0, 9, 2):
            anims = [FadeIn(items[k])]
            if k in (0, 2, 4):
                anims.append(Create(fr[k // 2]))
            self.play(*anims, run_time=0.4)
            self.add(labs[k // 2])
            if k + 1 < 9:
                self.add(items[k + 1])

        # hero output
        out = items[8]
        self.uncaption()
        rest = [m for m in self.mobjects if m not in (out, self.hdr)]
        self.play(*[FadeOut(m) for m in rest], run_time=0.5)
        self.play(out.animate.set(height=4.9).move_to([-2.0, 0.1, 0]), run_time=1.0)
        gout = Grid(16, 4.9, (-2.0, 0.1))
        ol = fovea_outline(gout, fb, WHITE, 2.5)
        self.play(Create(ol), run_time=0.6)
        r1 = tx("fovea: 48 high-res tokens", 28, HR).move_to([3.6, 1.0, 0])
        r2 = tx("periphery: 52 low-res tokens", 28, LR).move_to([3.6, 0.3, 0])
        r3 = tx("100 tokens instead of 256", 30, bold=True).move_to([3.6, -0.6, 0])
        self.add(r1, r2, r3)
        self.caption("Full detail where the viewer looks, and every denoising step ran on 39% of the tokens.")
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S07 Masks
class S07_Masks(Base):
    def construct(self):
        self.header("7 · Foveation masks are just data")

        s = 1.62
        groups = ["Position", "Size", "Disjoint regions"]
        keys = ["position", "size", "disjoint"]
        gx = [-4.45, 0.0, 4.45]
        ys = [1.55, -0.13, -1.81]
        for gi in range(3):
            self.add(tx(groups[gi], 26, bold=True).move_to([gx[gi], 2.95, 0]))
            self.add(tx("Naïve", 22, DIM).move_to([gx[gi] - 0.85, 2.58, 0]))
            self.add(tx("Ours", 22).move_to([gx[gi] + 0.85, 2.58, 0]))
        ims = []
        for r in range(3):
            for gi in range(3):
                ims.append(img(f"fig9_{keys[gi]}{r}_naive.png", s).move_to([gx[gi] - 0.85, ys[r], 0]))
                ims.append(img(f"fig9_{keys[gi]}{r}_ours.png", s).move_to([gx[gi] + 0.85, ys[r], 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.04), run_time=1.8)
        self.caption("Trained on random masks, one model handles any fovea: same prompt and seed, different masks.")
        self.caption("Ours stays coherent as the fovea moves, grows, or splits in two; the naïve baseline breaks.")
        self.clear_all(run_time=0.6)

        # saliency-guided + applications
        s2 = 1.55
        xr, xs_ = -5.55, -3.7
        yy = [1.45, -0.2, -1.85]
        self.add(tx("Random\nmasks", 22).move_to([xr, 2.65, 0]))
        self.add(tx("Saliency-\nguided", 22).move_to([xs_, 2.65, 0]))
        col_a = [img(f"fig10_random{k}.png", s2).move_to([xr, yy[k], 0]) for k in range(3)]
        col_b = [img(f"fig10_saliency{k}.png", s2).move_to([xs_, yy[k], 0]) for k in range(3)]
        self.play(LaggedStart(*[FadeIn(m) for m in col_a + col_b], lag_ratio=0.08), run_time=1.0)
        self.caption("Masks from a saliency predictor (DeepGaze) put the salient object in the fovea.")
        apps = [("gaming", "VR gaming"), ("robotics", "robotics simulation"),
                ("dashcam", "driving simulation"), ("aerial", "aerial footage")]
        pos = [(0.4, 1.3), (3.9, 1.3), (0.4, -1.35), (3.9, -1.35)]
        self.add(tx("Applications", 24, bold=True).move_to([2.15, 2.75, 0]))
        appm = [img(f"fig10_{k}1.png", 2.1).move_to([px, py, 0]) for (k, _), (px, py) in zip(apps, pos)]
        self.play(LaggedStart(*[FadeIn(m) for m in appm], lag_ratio=0.2), run_time=1.0)
        for m, (_, name) in zip(appm, apps):
            self.add(tx(name, 22).next_to(m, DOWN, buff=0.1))
        self.caption("Only what matters (a weapon, a gripper, a van) is generated at full resolution.")
        self.clear_all(run_time=0.6)

        # video spline path
        rect = Rectangle(width=5.0, height=2.8, color=DIM, stroke_width=2).move_to([-2.3, 1.2, 0])
        self.play(Create(rect), run_time=0.4)
        cp = np.array([[0.2, 0.35, 0.16], [0.55, 0.7, 0.22], [0.82, 0.4, 0.14]])
        tks = np.array([0.0, 0.5, 1.0])
        cs = CubicSpline(tks, cp, bc_type="natural")

        def to_pt(u, v):
            return rect.get_corner(UL) + np.array([u * 5.0, -v * 2.8, 0])

        dots = VGroup(*[Dot(to_pt(p[0], p[1]), radius=0.07, color=AMBER) for p in cp])
        path = ParametricFunction(lambda t: to_pt(*cs(t)[:2]), t_range=[0, 1], color=AMBER, stroke_width=3)
        self.play(FadeIn(dots), Create(path), run_time=0.8)
        lab = tx("3 random control points (<i>x</i>, <i>y</i>, <i>r</i>)\n→ cubic spline gaze path", 24).move_to([3.6, 1.2, 0])
        self.add(lab)
        pr = ValueTracker(0.0)
        fovc = always_redraw(lambda: Circle(radius=float(cs(pr.get_value())[2]) * 2.8, color=WHITE, stroke_width=3)
                             .move_to(to_pt(*cs(pr.get_value())[:2])))
        self.add(fovc)
        self.caption("For video, the fovea travels along a smooth random gaze path, and its radius changes too.")
        self.play(pr.animate.set_value(1.0), run_time=2.2, rate_func=smooth)
        frames = [img(f"vid_fox_ours_{k}.png", 1.44) for k in range(5)]
        Group(*frames).arrange(RIGHT, buff=0.12).move_to([0, -1.95, 0])
        self.play(LaggedStart(*[FadeIn(f) for f in frames], lag_ratio=0.15), pr.animate.set_value(0.0),
                  run_time=1.2)
        self.caption("The generated fox stays one fox as the fovea (white circle) moves across the frame.")
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 3", "Results", hold=1.4)
        self.header("8 · Results")

        # speedup plot (curve vertices read from the paper’s Fig. 6)
        img_pts = [(100.0, 1.0), (90.9, 1.02), (82.4, 1.09), (78.4, 1.16), (74.5, 1.2), (70.8, 1.24),
                   (60.5, 1.41), (57.3, 1.45), (46.2, 1.66), (43.8, 1.69), (37.4, 1.88), (35.6, 1.9),
                   (30.9, 2.03), (29.7, 2.05), (27.6, 2.11), (26.9, 2.16), (25.7, 2.19), (25.0, 2.2)]
        vid_pts = [(100.0, 1.0), (92.7, 1.1), (85.8, 1.23), (79.2, 1.39), (73.0, 1.55), (61.8, 1.93),
                   (56.7, 2.13), (52.0, 2.36), (47.7, 2.55), (40.2, 3.02), (37.0, 3.1), (34.2, 3.35),
                   (31.8, 3.56), (29.7, 3.66), (28.0, 3.74), (26.7, 3.86), (25.2, 4.04), (25.0, 4.17)]
        ax = Axes(x_range=[0, 75, 25], y_range=[1, 4.5, 0.5], x_length=5.4, y_length=4.2, tips=False,
                  axis_config={"include_numbers": False, "color": DIM, "stroke_width": 2}).move_to([-3.4, 0.2, 0])
        xt = VGroup(*[tx(f"{100 - v}%", 20, DIM).next_to(ax.c2p(v, 1), DOWN, buff=0.15) for v in (0, 25, 50, 75)])
        yt = VGroup(*[tx(f"{v:g}×", 20, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (1, 2, 3, 4)])
        xl = tx("token ratio (fraction of full sequence)", 22, DIM).next_to(ax, DOWN, buff=0.55)
        yl = tx("speedup", 22, DIM).next_to(ax.c2p(0, 4.5), UP, buff=0.2)
        self.play(Create(ax), run_time=0.5)
        self.add(xt, yt, xl, yl)
        ci = VMobject(color=AMBER, stroke_width=4).set_points_as_corners([ax.c2p(100 - r, s) for r, s in img_pts])
        cv = VMobject(color=PURPLE, stroke_width=4).set_points_as_corners([ax.c2p(100 - r, s) for r, s in vid_pts])
        self.caption("Fewer tokens, faster generation: at 25% of the tokens, 2.2× for images and 4.2× for video.")
        self.play(Create(ci), Create(cv), run_time=1.8)
        e1 = tx("2.2×", 24, AMBER).next_to(ax.c2p(75, 2.2), RIGHT, buff=0.12)
        e2 = tx("4.2×", 24, PURPLE).next_to(ax.c2p(75, 4.17), RIGHT, buff=0.12)
        lg1 = tx("image", 22, AMBER).move_to(ax.c2p(12, 4.25))
        lg2 = tx("video", 22, PURPLE).move_to(ax.c2p(12, 3.85))
        self.add(e1, e2, lg1, lg2)

        # image quality (from Table 1)
        qx = 3.5
        q0 = tx("Image quality: HPSv2.1 ↑ (human preference)", 22, DIM).move_to([qx, 2.2, 0])
        q1 = tx("full high-res:  0.280", 28).move_to([qx, 1.35, 0])
        q2 = tx("<b>ours:  0.279 – 0.280</b>", 28, HR).move_to([qx, 0.65, 0])
        q3 = tx("naïve:  0.268 – 0.275", 28, DIM).move_to([qx, -0.05, 0])
        q4 = tx("at 26–43% of the tokens (1.6–2.1× faster)", 22, DIM).move_to([qx, -0.8, 0])
        self.add(q0, q1, q2, q3, q4)
        self.caption("Meanwhile, image quality stays at the full-resolution level; the naïve baseline falls behind.")
        self.clear_all(run_time=0.6)

        # user study
        steps = [("fixate", None), ("image A · 1 s", "fig5_farmer_full.png"), ("fixate", None),
                 ("image B · 1 s", "fig5_farmer_ours.png"), ("which looks better?", "Q")]
        sx = [-5.2, -2.6, 0.0, 2.6, 5.2]
        scr = []
        for (lbl, content), x in zip(steps, sx):
            box = Rectangle(width=1.7, height=1.0, color=DIM, stroke_width=2).set_fill(BLACK, 1).move_to([x, 2.0, 0])
            grp = Group(box)
            if content is None:
                grp.add(Dot(box.get_center(), radius=0.06, color=RED))
            elif content == "Q":
                grp.add(tx("A  or  B ?", 22).move_to(box))
            else:
                grp.add(img(content, 0.96).move_to(box))
            scr.append(grp)
        self.caption("A gaze-contingent user study: fixate the red dot, see two images for 1 s each, pick the better one.")
        self.play(LaggedStart(*[FadeIn(g_) for g_ in scr], lag_ratio=0.25), run_time=1.2)
        for (lbl, _), grp in zip(steps, scr):
            self.add(tx(lbl, 20, DIM).next_to(grp[0], DOWN, buff=0.12))
        for a, b in zip(sx[:-1], sx[1:]):
            self.add(arrow([a + 0.9, 2.0, 0], [b - 0.9, 2.0, 0], buff=0.05, sw=2))
        bars = [("Ours vs. Full high-res", 47.3, "p = 0.48: indistinguishable"),
                ("Ours vs. Naïve", 87.4, "p &lt; 0.0001"),
                ("Full high-res vs. Naïve", 90.8, "p &lt; 0.0001")]
        x0, W = -2.2, 6.2
        by = [0.25, -0.8, -1.85]
        for (name, p, note), y in zip(bars, by):
            self.add(tx(name, 24).move_to([-4.6, y, 0]))
            bg = Rectangle(width=W, height=0.5, stroke_width=0).set_fill(GHOST, 1)
            bg.move_to([x0 + W / 2, y, 0])
            fg = Rectangle(width=W * p / 100, height=0.5, stroke_width=0).set_fill(HR, 1)
            fg.move_to([x0 + W * p / 200, y, 0])
            self.add(bg)
            self.play(GrowFromEdge(fg, LEFT), run_time=0.5)
            self.add(tx(f"{p:.1f}%", 22).move_to(fg.get_right() + LEFT * 0.5))
            self.add(tx(note, 18, DIM).next_to(bg, DOWN, buff=0.08).align_to(bg, RIGHT))
        half = DashedLine([x0 + W / 2, 0.75, 0], [x0 + W / 2, -2.3, 0], color=WHITE, stroke_width=2)
        self.play(Create(half), run_time=0.4)
        self.add(tx("50%", 18).next_to(half, UP, buff=0.05))
        self.caption("Ours vs. full resolution: 47.3%, at chance. Viewers cannot tell them apart.")
        self.clear_all(run_time=0.6)

        # video
        rows = [("Full high-res", "fox_full"), ("Naïve", "fox_naive"), ("Ours", "fox_ours")]
        yv = [1.75, 0.4, -0.95]
        allf = []
        for (name, key), y in zip(rows, yv):
            self.add(tx(name, 24, WHITE if name == "Ours" else DIM, bold=(name == "Ours")).move_to([-5.7, y, 0]))
            fr = [img(f"vid_{key}_{k}.png", 1.22) for k in range(5)]
            Group(*fr).arrange(RIGHT, buff=0.1).move_to([1.0, y, 0])
            allf += fr
        self.play(LaggedStart(*[FadeIn(f) for f in allf], lag_ratio=0.05), run_time=1.4)
        vb1 = tx("VBench subject consistency:  full 0.941 · naïve 0.907 · <b>ours 0.945</b>", 24).move_to([0.3, -1.85, 0])
        vb2 = tx("3.5× faster at 38% of the tokens (Wan2.1 1.3B, 480p)", 24, HR).move_to([0.3, -2.4, 0])
        self.add(vb1, vb2)
        self.caption("Video: the naïve baseline spawns a second fox; ours stays consistent and runs 3.5× faster.")
        self.clear_all(run_time=0.6)
        self.wait(0.2)


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.header("9 · Limitations and outlook")

        a = img("artifact_image_annot.png", 4.0).move_to([-4.6, 0.3, 0])
        b = img("artifact_video_annot.png", 4.0).move_to([-1.5, 0.3, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        t1 = tx("Occasional color seams\nat the fovea boundary", 30, bold=True).move_to([3.4, 1.4, 0])
        t2 = tx("cause: two separate VAE decodes,\nthen a pixel-space blend", 24, DIM).move_to([3.4, 0.2, 0])
        t3 = tx("fix: a VAE that decodes\nmixed-resolution tokens directly", 24, DIM).move_to([3.4, -0.9, 0])
        self.add(t1, t2, t3)
        self.caption("Red circles: seams from the pixel-space blend, not from the diffusion model itself.")
        self.clear_all(run_time=0.6)

        # gaze following
        g2 = Grid(16, 4.4, (0, 0.2))
        field = FoveaField(g2, (4, 4), 2.0)
        self.play(LaggedStart(*[FadeIn(m, scale=0.7) for m in field.mobs()], lag_ratio=0.02), run_time=1.0)
        gaze = Dot(g2.center, radius=0.09, color=RED).set_z_index(10)
        self.add(gaze)
        self.add(tx("tokens follow an eye tracker", 24).move_to([0, -2.4, 0]))
        self.caption("Next: streaming video world models with an eye tracker, where the fovea follows your gaze.")
        path = [(3.2, 2.6), (2.6, 4.6), (4.2, 5.6), (5.4, 4.2), (4.6, 2.8), (4, 4)]
        for p in path:
            target = g2.center + np.array([p[1] * 2 * g2.c - g2.size / 2, g2.size / 2 - p[0] * 2 * g2.c, 0])
            anims = field.retarget(p)
            self.play(gaze.animate.move_to(target), *anims, run_time=0.55)
        self.clear_all(run_time=0.6)

        # summary
        self.add(tx("Foveated Diffusion in three ideas", 40, bold=True).move_to([0, 2.5, 0]))
        pts = [
            f"1.  Spend tokens by eccentricity:  <i>L</i> = <i>m</i> + (<i>h</i>·<i>w</i> − <i>m</i>) / 4",
            "2.  Post-train (LoRA) on targets built from one image at two resolutions",
            "3.  Keep each attention on one scale with phase-aligned RoPE",
        ]
        for k, p in enumerate(pts):
            self.add(tx(p, 27).move_to([0, 1.3 - 0.85 * k, 0]).align_to([-6.0, 0, 0], LEFT))
            self.wait(1.6)
        self.add(tx("≈ 2× faster images  ·  ≈ 4× faster video  ·  perceptually indistinguishable",
                    28, HR).move_to([0, -1.55, 0]))
        self.wait(3.0)
        self.clear_all(run_time=0.5)
        self.remove(self.hdr)
        t = tx("Foveated Diffusion", 60, bold=True).move_to([0, 0.9, 0])
        u = tx("arXiv 2603.23491  ·  bchao1.github.io/foveated-diffusion", 28, DIM).move_to([0, -0.1, 0])
        v = tx("Chao*, Yariv*, Xiao, Wetzstein · Stanford University", 24, DIM).move_to([0, -0.8, 0])
        self.add(t, u, v)
        self.wait(3.5)
