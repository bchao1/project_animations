"""Foveated Diffusion — explainer animation, v4.

Source: Chao, Yariv, Xiao, Wetzstein, "Foveated Diffusion: Efficient Spatially
Adaptive Image and Video Generation", arXiv 2603.23491 (2026).
All images in data/ are cropped from the paper PDF (foveated-diffusion/pdfs/).

Render (see render.sh):
    manim -ql --media_dir media animation_foveated_diffusion_v4.py <Scene>
Scenes, in order: S00_Title ... S10_Outlook.
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

    # Captions stay up long enough to read: ~15 characters per second + 1 s.
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
        self.cap_need = len(plain) / 15.0 + 1.0
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
                  run_time=2.0)
        path = [(4, 4), (3.5, 3), (3, 2.5), (3.5, 5), (4.5, 5.5), (5, 4), (4, 3.5), (4, 4)]
        for p in path[1:]:
            anims = field.retarget(p)
            if anims:
                self.play(*anims, run_time=0.55)
            else:
                self.wait(0.55)
        self.wait(1.0)
        self.play(FadeOut(field.group()), run_time=0.8)
        self.remove(title, subt, auth)
        self.wait(0.3)


# ================================================================== S01 Tokens
class S01_TokenCost(Base):
    def construct(self):
        self.title_card("Part 1", "Every token costs the same")
        self.header("1 · Every token costs the same")

        g = Grid(16, 4.6, (-3.5, 0.1))
        pic = img("fig5_dog_full.png", 4.6).move_to(g.center)
        self.play(FadeIn(pic), run_time=0.8)
        self.caption("A diffusion transformer (DiT) generates an image as a grid of latent tokens.")
        lines = VGroup(*[Line(g.center + [-2.3 + k * g.c, -2.3, 0], g.center + [-2.3 + k * g.c, 2.3, 0])
                         for k in range(17)],
                       *[Line(g.center + [-2.3, -2.3 + k * g.c, 0], g.center + [2.3, -2.3 + k * g.c, 0])
                         for k in range(17)]).set_stroke(WHITE, 1, 0.55)
        self.play(Create(lines, lag_ratio=0.02), run_time=1.6)
        cols = patch_colors(os.path.join(DATA, "fig5_dog_full.png"), 16)
        toks = {(i, j): token(g.hs, cols[i][j]).move_to(g.cell(i, j)) for i in range(16) for j in range(16)}
        tg = VGroup(*toks.values())
        self.play(LaggedStart(*[FadeIn(t) for t in tg], lag_ratio=0.004), run_time=2.0)
        self.play(FadeOut(pic), FadeOut(lines), run_time=0.8)
        self.caption("Each token encodes one small patch of the image as a latent feature vector.", 1.2)

        # face vs background
        face, bg = (4, 8), (2, 1)
        hf = SurroundingRectangle(toks[face], buff=0.035, color=AMBER, stroke_width=4)
        hb = SurroundingRectangle(toks[bg], buff=0.035, color=PURPLE, stroke_width=4)
        kf = Square(0.32).set_stroke(AMBER, 4).move_to([1.0, 1.1, 0])
        kb = Square(0.32).set_stroke(PURPLE, 4).move_to([1.0, 0.3, 0])
        lf = tx("a token on the dog’s face", 26).next_to(kf, RIGHT, buff=0.25)
        lb = tx("a token on the plain door", 26).next_to(kb, RIGHT, buff=0.25)
        self.play(Create(hf), Create(kf), run_time=0.5)
        self.add(lf)
        self.play(Create(hb), Create(kb), run_time=0.5)
        self.add(lb)
        self.caption("Both tokens cost exactly the same compute, although only one of them carries the subject.", 3.0)
        self.play(FadeOut(VGroup(hf, hb, kf, kb)), run_time=0.4)
        self.remove(lf, lb)

        # recolor to HR
        self.play(LaggedStart(*[t.animate.set_fill(HR) for t in tg], lag_ratio=0.003), run_time=1.8)
        cnt = tx(f"<i>h</i> × <i>w</i> = 16 × 16 = {col('256', HR)} tokens", 32).move_to([3.4, 2.55, 0])
        self.add(cnt)
        self.caption("A standard DiT processes all <i>h</i> × <i>w</i> tokens at full resolution, at every denoising step.", 3.0)

        att = img(gen("attn.png"), 4.0).move_to([3.4, -0.1, 0])
        attf = framed(att, DIM)
        self.play(FadeIn(att), Create(attf), run_time=1.0)
        lab = tx("256 × 256 = 65,536 query–key pairs", 26).next_to(att, DOWN, buff=0.22)
        self.add(lab)
        self.caption("Self-attention compares every token with every other token.", 1.0)
        q = (5, 8)
        k = q[0] * 16 + q[1]
        rowy = att.get_top()[1] - (k + 0.5) / 256 * 4.0
        hq = SurroundingRectangle(toks[q], buff=0.04, color=WHITE, stroke_width=3)
        hr_ = Rectangle(width=4.1, height=0.07, color=WHITE, stroke_width=2).move_to([3.4, rowy, 0])
        self.play(Create(hq), run_time=0.4)
        self.play(Create(hr_), run_time=0.6)
        self.caption("The highlighted row: one query token compared with all 256 keys.")
        for qn in [(2, 3), (9, 12), (13, 5), (7, 8)]:        # every token gets its own row
            kn = qn[0] * 16 + qn[1]
            yn = att.get_top()[1] - (kn + 0.5) / 256 * 4.0
            self.play(hq.animate.move_to(toks[qn]), hr_.animate.set_y(yn), run_time=0.7)
            self.wait(0.3)
        self.caption("With <i>N</i> tokens, attention costs <i>N</i> × <i>N</i>: it grows quadratically.", 1.5)
        self.play(FadeOut(Group(att, attf, hq, hr_)), run_time=0.6)
        self.remove(lab, cnt)

        # scaling squares
        s1 = Square(0.9).set_fill(HR, 0.6).set_stroke(HR, 2).move_to([1.2, 0.9, 0])
        s2 = Square(3.6).set_fill(HR, 0.6).set_stroke(HR, 2).move_to([4.6, 0.0, 0])
        l1a = tx("<i>N</i> tokens", 24).next_to(s1, UP, buff=0.2)
        l1b = tx("<i>N</i>" + sup("2") + " pairs", 24).next_to(s1, DOWN, buff=0.2)
        l2a = tx("2× resolution: 4<i>N</i> tokens", 24).next_to(s2, UP, buff=0.2)
        l2b = tx("16 <i>N</i>" + sup("2") + " pairs", 24).next_to(s2, DOWN, buff=0.2)
        self.play(FadeIn(s1), run_time=0.5)
        self.add(l1a, l1b)
        self.wait(0.6)
        self.play(GrowFromCenter(s2), run_time=1.4)
        self.add(l2a, l2b)
        self.caption("Double the resolution: 4× the tokens and 16× the attention cost.", 2.6)
        self.caption("Video multiplies the token count again by the number of frames: <i>f</i> × <i>h</i> × <i>w</i>.", 3.0)
        self.caption("But does the viewer actually <i>see</i> every one of these tokens equally well?", 3.0)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S02 Vision
class S02_Vision(Base):
    def construct(self):
        self.title_card("Part 2", "Where you look matters")
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
        self.play(Create(ax), run_time=0.8)
        self.add(xt, xl, yl)
        self.caption("Visual acuity peaks at the center of gaze (the fovea) and falls off rapidly with eccentricity.")
        self.play(Create(curve), run_time=2.2)
        self.play(FadeIn(band), run_time=0.5)
        fl = tx("fovea", 22, HR).move_to(ax.c2p(12, 0.95))
        pl = tx("periphery", 22, DIM).move_to(ax.c2p(-26, 0.32))
        pr = tx("periphery", 22, DIM).move_to(ax.c2p(26, 0.32))
        self.add(fl, pl, pr)
        note = tx("schematic", 18, DIM, italic=True).next_to(ax.c2p(40, 1.05), UP, buff=0.1).shift(LEFT * 0.4)
        self.add(note)
        self.wait(2.5)

        pic = img("fig5_farmer_full.png", 4.6).move_to([3.4, 0.25, 0])
        self.play(FadeIn(pic), run_time=0.8)
        side = 4.6

        def pt(u, v):
            return pic.get_center() + np.array([(u - 0.5) * side, (0.5 - v) * side, 0])

        dot = Dot(pt(0.50, 0.55), radius=0.1, color=RED).set_z_index(10)
        self.add(dot)
        self.caption("Keep your eyes on the red dot.", 1.5)
        blurA = img(gen("gaze_A.png"), 4.6).move_to(pic)
        self.play(FadeIn(blurA), run_time=2.5)
        self.bring_to_front(dot)
        circ = DashedVMobject(Circle(radius=(40 / 512) * side * 1.9, color=WHITE, stroke_width=2),
                              num_dashes=30).move_to(dot).set_z_index(10)
        self.play(Create(circ), run_time=0.6)
        self.caption("With the periphery blurred, the change is hard to notice: fine detail is only resolved near the gaze.", 3.8)
        blurB = img(gen("gaze_B.png"), 4.6).move_to(pic)
        self.play(dot.animate.move_to(pt(0.80, 0.22)), circ.animate.move_to(pt(0.80, 0.22)),
                  FadeIn(blurB), run_time=1.6)
        self.remove(blurA)
        self.caption("Move the gaze and the sharp region moves with it.", 2.5)
        self.clear_all()

        # foveated rendering: x_fov = M ⊙ x_high + (1 − M) ⊙ Up(x_low)
        self.caption("Computer graphics has exploited this for years: <b>foveated rendering</b>.", 1.0)
        eq = tx(f"{XF} = {M_} ⊙ {XH} + (1 − {M_}) ⊙ Up({XL})", 42).move_to([0, 2.35, 0])
        self.add(eq)
        s = 1.9
        items = [img(gen("fr_M.png"), s), tx("⊙", 40), img("fig5_farmer_full.png", s), tx("+", 40),
                 img(gen("fr_1mM.png"), s), tx("⊙", 40), img(gen("fr_up.png"), s), tx("=", 40),
                 img(gen("fr_comp.png"), s)]
        row = Group(*items).arrange(RIGHT, buff=0.22).move_to([0, 0.0, 0])
        labs = [tx(M_, 26), tx(XH, 26), tx(f"1 − {M_}", 26), tx(f"Up({XL})", 26), tx(XF, 26)]
        for lb, it in zip(labs, items[0::2]):
            lb.next_to(it, DOWN, buff=0.22)
        frames = [framed(items[k], DIM, 1) for k in (0, 4)]
        for k in range(0, 9, 2):
            anims = [FadeIn(items[k])]
            if k in (0, 4):
                anims.append(Create(frames[k // 4]))
            self.play(*anims, run_time=0.6)
            self.add(labs[k // 2])
            if k + 1 < 9:
                self.add(items[k + 1])
        self.caption("Render the fovea at full resolution, the periphery at low resolution, and blend with the mask <i>M</i>.", 3.5)
        self.caption("The periphery is cheap, and the viewer, looking at the fovea, barely notices.", 3.0)
        self.clear_all()
        q = tx("Can a diffusion model spend its tokens the same way?", 40)
        if q.width > 12.5:
            q.scale_to_fit_width(12.5)
        self.add(q)
        self.wait(2.8)
        self.remove(q)
        self.wait(0.3)


# ================================================================== S03 Tokenization
class S03_Tokenization(Base):
    def construct(self):
        self.title_card("Part 3", "Foveated tokenization")
        self.header("3 · Foveated tokenization")

        g = Grid(16, 5.0, (-3.4, 0.1))
        fb = fovea_blocks(8, 3, 4, 2.0)
        toks = {(i, j): token(g.hs, HR).move_to(g.cell(i, j)) for i in range(16) for j in range(16)}
        self.play(LaggedStart(*[FadeIn(t) for t in toks.values()], lag_ratio=0.003), run_time=1.5)
        self.caption("Start from the full grid of <i>h</i> × <i>w</i> = 256 high-resolution tokens.", 1.8)

        # mask panel
        mg = Grid(16, 2.4, (3.6, 1.2))
        mcells = VGroup()
        for i in range(16):
            for j in range(16):
                on = (i // 2, j // 2) in fb
                mcells.add(Square(mg.c * 0.92).set_fill(WHITE if on else "#2B2B2B", 1).set_stroke(width=0)
                           .move_to(mg.cell(i, j)))
        mlab = tx(f"{M_} ∈ {{0, 1}}" + sup("<i>h</i>×<i>w</i>"), 30).next_to(mcells, LEFT, buff=0.35)
        self.play(FadeIn(mcells), run_time=0.8)
        self.add(mlab)
        m1 = tx(f"{M_}(<i>i</i>, <i>j</i>) = 1: fovea, keep high-res", 24).move_to([3.6, -0.55, 0])
        m0 = tx(f"{M_}(<i>i</i>, <i>j</i>) = 0: periphery", 24).move_to([3.6, -1.05, 0])
        self.add(m1, m0)
        outline = fovea_outline(g, fb)
        periph = [t for (i, j), t in toks.items() if (i // 2, j // 2) not in fb]
        self.play(*[t.animate.set_fill(opacity=0.3) for t in periph], Create(outline), run_time=1.2)
        self.caption("A binary foveation mask <i>M</i> marks where high-resolution tokens are kept.", 3.2)
        self.play(*[t.animate.set_fill(opacity=1.0) for t in periph], run_time=0.5)
        self.play(FadeOut(mcells), run_time=0.5)
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
        self.play(LaggedStart(*anims, lag_ratio=0.04), run_time=4.0)
        self.wait(1.0)
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
        self.play(*[Indicate(t, color=WHITE, scale_factor=1.25) for t in hr.values()], run_time=1.0)
        self.add(L2)
        self.play(*[Indicate(t, color=WHITE, scale_factor=1.15) for t in lr.values()], run_time=1.0)
        self.add(L3)
        self.play(Create(L3b), run_time=0.6)
        self.caption("Total sequence length: <i>L</i> = <i>m</i> + (<i>h</i>·<i>w</i> − <i>m</i>)/4, with <i>L</i> ≪ <i>h</i>·<i>w</i> when the fovea is small.", 2.5)
        self.add(L4)
        self.wait(2.5)
        self.remove(L1, L2, L3, L4)
        self.play(FadeOut(L3b), FadeOut(outline), run_time=0.4)

        # flatten into a sequence
        self.caption("The transformer sees this as one sequence of 100 tokens, not 256.")
        order = seq_order(8, fb)
        pitch, s = 0.125, 0.1
        x0 = -pitch * (len(order) - 1) / 2
        moves = []
        seq = VGroup()
        for k, (kind, key) in enumerate(order):
            t = hr[key] if kind == "H" else lr[key]
            target = token(s, HR if kind == "H" else LR).move_to([x0 + k * pitch, 2.75, 0])
            moves.append(Transform(t, target))
            seq.add(t)
        self.play(LaggedStart(*moves, lag_ratio=0.01), run_time=3.0)
        self.wait(0.5)

        # attention area comparison
        tl = np.array([-4.2, 2.2, 0])
        big = Square(4.4).set_fill(HR, 0.25).set_stroke(DIM, 2)
        big.move_to(tl + np.array([2.2, -2.2, 0]))
        sm_side = 4.4 * 100 / 256
        small = Square(sm_side).set_fill(HR, 0.85).set_stroke(WHITE, 2)
        small.move_to(tl + np.array([sm_side / 2, -sm_side / 2, 0]))
        self.play(FadeIn(big), run_time=0.6)
        la = tx("full: 256 × 256 = 65,536 pairs", 28, DIM).move_to([3.4, 0.9, 0])
        self.add(la)
        self.wait(0.8)
        self.play(TransformFromCopy(big, small), run_time=1.2)
        lb = tx("foveated: 100 × 100 = 10,000 pairs", 28).move_to([3.4, 0.2, 0])
        lc = tx("≈ 15% of the attention work", 28, HR).move_to([3.4, -0.5, 0])
        self.add(lb, lc)
        self.caption("Attention shrinks quadratically in <i>L</i>; the MLP layers shrink linearly. Savings apply at every step.", 4.0)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S04 RoPE
class S04_RoPE(Base):
    def construct(self):
        self.title_card("Part 4", "Where is each token?  Mixed-resolution RoPE")
        self.header("4 · Mixed-resolution positional encoding")

        side, pitch = 0.8, 0.96
        xs = [(-3.5 + k) * pitch for k in range(8)]
        ya, yb, yc = 1.1, -0.3, -1.6
        theta = 0.62

        def tok_at(k, y, color, label):
            t = token(side, color).move_to([xs[k], y, 0])
            return VGroup(t, tx(str(label), 26).move_to(t))

        row_a = [tok_at(k, ya, HR, k) for k in range(8)]
        dials, hands = [], []
        for k in range(8):
            c = Circle(radius=0.3, color=DIM, stroke_width=2).move_to([xs[k], 2.4, 0])
            h = Line(c.get_center(), c.get_center() + RIGHT * 0.3, color=WHITE, stroke_width=3)
            dials.append(c)
            hands.append(h)
        la = tx("(a) uniform grid", 24, DIM).move_to([-5.6, ya, 0])
        self.play(LaggedStart(*[FadeIn(r[0]) for r in row_a], lag_ratio=0.08), run_time=1.0)
        self.add(*[r[1] for r in row_a], la)
        self.play(*[Create(c) for c in dials], *[Create(h) for h in hands], run_time=0.8)
        self.play(*[Rotate(hands[k], angle=theta * k, about_point=dials[k].get_center()) for k in range(8)],
                  run_time=1.8)
        self.caption("RoPE rotates each token’s features by an angle proportional to its grid index,", 2.2)
        self.caption("so attention between two tokens depends on their relative distance on the grid.", 2.5)

        # merge 4-7
        g45 = token(side, LR).move_to([xs[4], ya, 0])
        g67 = token(side, LR).move_to([xs[6], ya, 0])
        gh5 = ghost(side, dashed=True).move_to([xs[5], ya, 0])
        gh7 = ghost(side, dashed=True).move_to([xs[7], ya, 0])
        for k in (4, 5, 6, 7):
            self.remove(row_a[k][1])
        self.play(ReplacementTransform(VGroup(row_a[4][0], row_a[5][0]), g45),
                  ReplacementTransform(VGroup(row_a[6][0], row_a[7][0]), g67),
                  FadeIn(gh5), FadeIn(gh7),
                  *[FadeOut(hands[k]) for k in (4, 5, 6, 7)], run_time=1.2)
        qs = [tx("?", 30, LR).move_to(dials[k]) for k in (4, 5, 6, 7)]
        self.add(*qs)
        self.caption("With mixed resolution the grid is no longer uniform. Which index should a low-res token get?", 3.2)
        self.remove(*qs)
        self.play(*[FadeOut(dials[k]) for k in range(8)], *[FadeOut(hands[k]) for k in range(4)], run_time=0.5)
        la2 = tx("(a) mixed grid", 24, DIM).move_to(la)
        self.remove(la)
        self.add(la2)

        # (b) high-res query row
        lb = tx("high-res\nquery", 24).move_to([-5.6, yb, 0])
        rb = [tok_at(k, yb, HR, k) for k in range(4)] + [tok_at(4, yb, LR, 4), tok_at(6, yb, LR, 6)]
        gb = [ghost(side, True).move_to([xs[k], yb, 0]) for k in (5, 7)]
        self.add(lb)
        self.play(*[TransformFromCopy(row_a[k][0], rb[k][0]) for k in range(4)],
                  TransformFromCopy(g45, rb[4][0]), TransformFromCopy(g67, rb[5][0]),
                  FadeIn(VGroup(*gb)), run_time=1.2)
        self.add(*[r[1] for r in rb])
        self.caption("When a <b>high-res</b> token queries, low-res keys are placed on the fine grid: indices 4 and 6.", 3.8)

        # (c) low-res query row
        lc = tx("low-res\nquery", 24).move_to([-5.6, yc, 0])
        rc = [tok_at(0, yc, HR, 0), tok_at(2, yc, HR, 1), tok_at(4, yc, LR, 2), tok_at(6, yc, LR, 3)]
        gc = [ghost(side, True).move_to([xs[k], yc, 0]) for k in (1, 3, 5, 7)]
        self.add(lc)
        self.play(TransformFromCopy(row_a[0][0], rc[0][0]), TransformFromCopy(row_a[2][0], rc[1][0]),
                  TransformFromCopy(g45, rc[2][0]), TransformFromCopy(g67, rc[3][0]),
                  FadeIn(VGroup(*gc)), run_time=1.2)
        self.add(*[r[1] for r in rc])
        self.caption("When a <b>low-res</b> token queries, high-res keys are subsampled and re-indexed on the coarse grid: 0, 1, 2, 3.", 4.2)
        br1 = Line([xs[0] - side / 2, -2.2, 0], [xs[3] + side / 2, -2.2, 0], color=HR, stroke_width=3)
        br2 = Line([xs[4] - side / 2, -2.2, 0], [xs[7] + side / 2, -2.2, 0], color=LR, stroke_width=3)
        t1 = tx("high-res region", 22, HR).next_to(br1, DOWN, buff=0.12)
        t2 = tx("low-res region", 22, LR).next_to(br2, DOWN, buff=0.12)
        self.play(Create(br1), Create(br2), run_time=0.5)
        self.add(t1, t2)
        self.caption("Each attention computation then lives on a single, uniform scale that the pretrained DiT understands.", 3.6)
        self.caption("This is phase-aligned RoPE (Wu et al., 2025), used unchanged by Foveated Diffusion.", 3.0)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S05 Naive
class S05_Naive(Base):
    def construct(self):
        self.title_card("Part 5", "Just plug it in?")
        self.header("5 · Naïve mixed-resolution denoising")

        s = 2.35
        xs = [-1.7, 0.95, 3.6]
        yt, yb = 1.35, -1.2
        names = ["horse", "guitar", "farmer"]
        notes = ["duplicated horse", "two guitars", "two hat brims"]
        rt = tx("Full high-res", 26).move_to([-4.35, yt, 0])
        rb = tx("Naïve\nmixed-res", 26).move_to([-4.35, yb, 0])
        tops = [img(f"fig5_{n}_full.png", s).move_to([x, yt, 0]) for n, x in zip(names, xs)]
        bots = [img(f"fig5_{n}_naive.png", s).move_to([x, yb, 0]) for n, x in zip(names, xs)]
        self.add(rt)
        self.play(LaggedStart(*[FadeIn(t) for t in tops], lag_ratio=0.2), run_time=1.2)
        self.caption("With foveated tokens and phase-aligned RoPE, we could run a pretrained DiT as-is, with no training.", 2.0)
        self.add(rb)
        self.play(LaggedStart(*[FadeIn(b) for b in bots], lag_ratio=0.2), run_time=1.2)
        self.caption("It fails. White circles mark the fovea: content breaks exactly at the resolution boundary.", 2.0)
        for b, n in zip(bots, notes):
            self.add(tx(n, 22, RED).next_to(b, DOWN, buff=0.12))
            self.wait(0.6)
        self.wait(1.8)
        self.caption("The pretrained model never saw a token that covers a 2 × 2 area, so it draws the periphery at the wrong scale.", 4.0)
        self.caption("Objects get duplicated or fused across the boundary. The fix needs training, not just re-indexing.", 3.5)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S06 Training
class S06_Training(Base):
    def construct(self):
        self.title_card("Part 6", "Foveated training")
        self.header("6 · Foveated training: mixed-resolution-consistent targets")

        x = img("lion_train.png", 1.9).move_to([-5.7, 1.3, 0])
        xd = img(gen("lion_train_down.png"), 0.95).move_to([-5.7, -1.6, 0])
        xl = tx("<i>x</i>", 28).next_to(x, UP, buff=0.15)
        xdl = tx("Down(<i>x</i>)", 24).next_to(xd, DOWN, buff=0.15)
        self.play(FadeIn(x), run_time=0.6)
        self.add(xl)
        self.caption("Take a high-resolution training image <i>x</i>…", 1.5)
        adown = arrow(x.get_bottom(), xd.get_top(), buff=0.1, sw=2)
        dl = tx("bicubic", 20, DIM).next_to(adown, RIGHT, buff=0.1)
        self.play(GrowArrow(adown), run_time=0.4)
        self.add(dl)
        self.play(FadeIn(xd), run_time=0.5)
        self.add(xdl)
        self.caption("…and a bicubically downsampled copy Down(<i>x</i>) at half the resolution.", 2.2)

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
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(e1), FadeIn(e2), run_time=0.7)
        self.play(GrowArrow(a3), GrowArrow(a4),
                  LaggedStart(*[FadeIn(t) for t in zh.values()], lag_ratio=0.002),
                  LaggedStart(*[FadeIn(t) for t in zl.values()], lag_ratio=0.01), run_time=1.5)
        lzh = tx(f"{ZH0} = <i>E</i>(<i>x</i>)", 26).next_to(VGroup(*zh.values()), UP, buff=0.15)
        lzl = tx(f"{ZL0} = <i>E</i>(Down(<i>x</i>))", 26).next_to(VGroup(*zl.values()), UP, buff=0.15)
        self.add(lzh, lzl)
        self.caption("The VAE encoder <i>E</i> turns each into tokens: 16 × 16 high-res tokens and 8 × 8 low-res tokens.", 3.4)

        # mask + merge
        fb = fovea_blocks(8, 4, 4, 2.0)
        mg = Grid(16, 1.1, (0.55, -0.15))
        mcells = VGroup(*[Square(mg.c * 0.92).set_fill(WHITE if (i // 2, j // 2) in fb else "#2B2B2B", 1)
                          .set_stroke(width=0).move_to(mg.cell(i, j)) for i in range(16) for j in range(16)])
        mlab = tx(M_, 28).next_to(mcells, UP, buff=0.15)
        self.play(FadeIn(mcells), run_time=0.5)
        self.add(mlab)
        gf = Grid(16, 2.6, (3.5, -0.15))
        ghosts = VGroup(*[ghost(gf.hs).move_to(gf.cell(i, j)) for i in range(16) for j in range(16)])
        self.play(FadeIn(ghosts), run_time=0.4)
        lzf = tx(f"{ZF0} = Merge({ZH0}, {ZL0}, {M_})", 28).move_to([3.5, 1.65, 0])
        self.add(lzf)
        unused_h = [t for (i, j), t in zh.items() if (i // 2, j // 2) not in fb]
        unused_l = [t for k, t in zl.items() if k in fb]
        self.play(*[t.animate.set_opacity(0.2) for t in unused_h + unused_l], run_time=0.6)
        self.caption("Merge with the mask: keep high-res tokens inside the fovea and low-res tokens outside.")
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
        self.play(LaggedStart(*anims, lag_ratio=0.01), run_time=3.0)
        self.remove(ghosts)
        self.wait(0.6)
        self.caption("Both parts encode the <i>same</i> image, so the target is consistent across resolutions by construction.", 3.8)
        self.caption(f"Decoding and blending {ZF0} gives exactly the foveated image {XF}: nothing is left for the model to guess.", 3.5)

        # flow matching
        keep = list(fh.values()) + list(fl.values())
        self.clear_all(keep=keep)
        fov = VGroup(*keep)
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
        self.play(*tgt, run_time=1.2)
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
        self.caption("Training is standard flow matching: interpolate the clean foveated tokens toward Gaussian noise.")
        self.play(tt.animate.set_value(1.0), run_time=2.5)
        self.play(tt.animate.set_value(0.55), run_time=1.5)
        self.wait(0.5)

        dit = dit_box([0.2, 0.35, 0], w=1.7, h=2.2, lora=True)
        ain = arrow([-2.8, 0.35, 0], dit.get_left(), sw=2)
        ainl = tx(f"{ZFT}, <i>t</i>", 24).next_to(ain, UP, buff=0.1)
        self.play(FadeIn(dit), GrowArrow(ain), run_time=0.7)
        self.add(ainl)
        gv = Grid(16, 2.8, (4.4, 0.35))
        vt = []
        rng = np.random.default_rng(7)
        for (i, j), t in fh.items():
            d = np.array(noise[t].to_rgb()) - np.array(clean[t].to_rgb())
            vt.append(token(gv.hs, rgb_to_color(np.clip(0.5 + 0.9 * d, 0, 1)), HR, 1.2).move_to(gv.cell(i, j)))
        for (bi, bj), t in fl.items():
            d = np.array(noise[t].to_rgb()) - np.array(clean[t].to_rgb())
            vt.append(token(gv.ls, rgb_to_color(np.clip(0.5 + 0.9 * d, 0, 1)), LR, 1.6).move_to(gv.block(bi, bj)))
        vg = VGroup(*vt)
        aout = arrow(dit.get_right(), [2.9, 0.35, 0], sw=2)
        voutl = tx("<i>v</i>" + sub("θ") + f"({ZFT}, <i>t</i>)", 26).next_to(vg, DOWN, buff=0.2)
        self.play(GrowArrow(aout), LaggedStart(*[FadeIn(v) for v in vt], lag_ratio=0.005), run_time=1.2)
        self.add(voutl)
        self.caption("The DiT predicts the velocity field that carries noise back to the clean foveated tokens.", 3.0)
        loss = tx(f"ℒ = 𝔼 ‖ <i>v</i>{sub('θ')}({ZFT}, <i>t</i>) − ({ZF1} − {ZF0}) ‖{sup('2')}", 34).move_to([0, -2.45, 0])
        self.add(loss)
        self.caption("Same objective as the base model, computed entirely on the <i>L</i>-token foveated sequence.", 3.2)
        self.play(Indicate(dit[2], color=WHITE, scale_factor=1.15), run_time=0.8)
        self.caption("Only low-rank adapters (LoRA, rank 32) are trained on top of the frozen pretrained DiT.", 3.2)
        for t in keep:
            t.clear_updaters()
        self.clear_all()

        rows = [
            ("Images", "FLUX.2 Klein 4B · 90k images · 1024 × 1024"),
            ("Video", "Wan2.1 1.3B · Vchitect-T2V · 480p"),
            ("Both", "LoRA rank 32 · 10k steps · 8 × H100"),
            ("Masks", "random circles (images), random gaze paths (video)"),
        ]
        grp = VGroup()
        for i, (a, b) in enumerate(rows):
            la = tx(a, 30, WHITE, bold=True).move_to([0, 1.6 - 1.0 * i, 0]).align_to([-5.4, 0, 0], LEFT)
            lb = tx(b, 28).move_to([0, 1.6 - 1.0 * i, 0]).align_to([-3.2, 0, 0], LEFT)
            grp.add(la, lb)
        self.add(grp)
        self.caption("Post-training is cheap: a pretrained generator becomes a foveated one with a small LoRA.", 4.5)
        self.remove(grp)
        self.uncaption()
        self.wait(0.3)


# ================================================================== S07 Generation
class S07_Generation(Base):
    def construct(self):
        self.title_card("Part 7", "Foveated generation")
        self.header("7 · Foveated generation")

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
        self.play(LaggedStart(*[FadeIn(t) for t in allt], lag_ratio=0.01), run_time=1.2)
        lab = tx(f"{ZF1} ∼ 𝒩(0, <i>I</i>)", 28).next_to(VGroup(*allt), UP, buff=0.2)
        self.add(lab)
        self.caption("Sample Gaussian noise directly in the foveated token space: only <i>L</i> = 100 tokens.", 3.0)

        dit = dit_box([-2.2, 0.6, 0], w=1.4, h=1.8, lora=True)
        a_in = arrow([-3.6, 0.9, 0], [dit.get_left()[0], 0.9, 0], sw=2)
        a_back = arrow([dit.get_left()[0], 0.3, 0], [-3.6, 0.3, 0], sw=2)
        self.play(FadeIn(dit), GrowArrow(a_in), GrowArrow(a_back), run_time=0.7)
        tt = ValueTracker(1.0)
        for t in allt:
            t.add_updater(lambda m, t=t: m.set_fill(interpolate_color(clean[t], noise[t], tt.get_value()), 1))
        tl = always_redraw(lambda: tx(f"<i>t</i> = {tt.get_value():.2f}", 28).move_to([-5.0, -1.2, 0]))
        self.add(tl)
        self.caption("Denoise from <i>t</i> = 1 to <i>t</i> = 0, every step on the short foveated sequence.")
        for k in range(8):
            v = 1 - (k + 1) / 8
            self.play(tt.animate(rate_func=smooth).set_value(v),
                      Indicate(dit[0], color="#90A4AE", scale_factor=1.05), run_time=0.6)
        for t in allt:
            t.clear_updaters()
        self.remove(tl)
        lab2 = tx(f"{ZF0}", 28).move_to(lab)
        self.remove(lab)
        self.add(lab2)
        self.caption("The result is a clean foveated token sequence: fine detail in the fovea, coarse tokens outside.", 3.0)

        # split
        gh = Grid(16, 1.6, (0.9, 1.55))
        gl = Grid(8, 1.6, (0.9, -1.25))
        gh_ghost = VGroup(*[ghost(gh.hs).move_to(gh.cell(i, j)) for i in range(16) for j in range(16)
                            if (i // 2, j // 2) not in fb])
        gl_ghost = VGroup(*[ghost(gl.hs).move_to(gl.cell(i, j)) for i in range(8) for j in range(8)
                            if (i, j) in fb])
        split_eq = tx(f"({ZH0}, {ZL0}) = Split({ZF0}, {M_})", 26)
        self.caption(f"Split the sequence by the mask: ({ZH0}, {ZL0}) = Split({ZF0}, {M_}).")
        self.play(FadeIn(gh_ghost), FadeIn(gl_ghost), run_time=0.4)
        mv = []
        for (i, j), t in hr.items():
            mv.append(TransformFromCopy(t, token(gh.hs, ch[i][j], HR, 1.2).move_to(gh.cell(i, j))))
        for (bi, bj), t in lr.items():
            mv.append(TransformFromCopy(t, token(gl.hs, cl[bi][bj], LR, 1.6).move_to(gl.cell(bi, bj))))
        self.play(LaggedStart(*mv, lag_ratio=0.01), run_time=2.0)
        lh = tx(ZH0, 26).move_to([0.9, 2.6, 0])
        ll = tx(ZL0, 26).move_to([0.9, -0.2, 0])
        self.add(lh, ll)
        self.wait(1.0)

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
        self.play(GrowArrow(b1), GrowArrow(b2), FadeIn(d1), FadeIn(d2), run_time=0.6)
        self.play(GrowArrow(c1), GrowArrow(c2), FadeIn(xh), Create(xhf), FadeIn(xlo), run_time=0.9)
        lxh = tx(f"{XH} = <i>D</i>({ZH0})", 24).move_to([4.35, 2.65, 0])
        lxl = tx(f"{XL} = <i>D</i>({ZL0})", 24).move_to([4.35, -0.4, 0])
        self.add(lxh, lxl)
        self.caption("Decode each part separately with the VAE decoder <i>D</i>: a sharp fovea and a half-resolution image.", 3.6)

        # blend
        self.clear_all()
        eq = tx(f"{XF} = {M_}′ ⊙ {XH} + (1 − {M_}′) ⊙ Up({XL}),    {M_}′ = Up({M_})", 38).move_to([0, 2.35, 0])
        self.add(eq)
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
            self.play(*anims, run_time=0.55)
            self.add(labs[k // 2])
            if k + 1 < 9:
                self.add(items[k + 1])
        self.caption("Upsample the low-res image and blend it with the fovea using the upsampled mask <i>M</i>′.", 3.5)

        # hero output
        out = items[8]
        rest = [m for m in self.mobjects if m not in (out, self.hdr, self.cap)]
        self.uncaption()
        self.play(*[FadeOut(m) for m in rest], run_time=0.6)
        self.play(out.animate.set(height=4.9).move_to([-2.0, 0.1, 0]), run_time=1.2)
        gout = Grid(16, 4.9, (-2.0, 0.1))
        ol = fovea_outline(gout, fb, WHITE, 2.5)
        self.play(Create(ol), run_time=0.8)
        r1 = tx("fovea: 48 high-res tokens", 28, HR).move_to([3.6, 1.0, 0])
        r2 = tx("periphery: 52 low-res tokens", 28, LR).move_to([3.6, 0.3, 0])
        r3 = tx("100 tokens instead of 256", 30, bold=True).move_to([3.6, -0.6, 0])
        self.add(r1, r2)
        self.wait(0.8)
        self.add(r3)
        self.caption("Full detail where the viewer looks; every denoising step ran on 39% of the tokens.", 4.0)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S08 Masks
class S08_Masks(Base):
    def construct(self):
        self.title_card("Part 8", "Masks are just data")
        self.header("8 · Foveation masks are just data")

        s = 1.62
        groups = ["Position", "Size", "Disjoint regions"]
        keys = ["position", "size", "disjoint"]
        gx = [-4.45, 0.0, 4.45]
        ys = [1.55, -0.13, -1.81]
        self.caption("Trained with random masks, one model handles any fovea: same prompt and seed, different masks.")
        for gi in range(3):
            self.add(tx(groups[gi], 26, bold=True).move_to([gx[gi], 2.95, 0]))
            self.add(tx("Naïve", 22, DIM).move_to([gx[gi] - 0.85, 2.58, 0]))
            self.add(tx("Ours", 22).move_to([gx[gi] + 0.85, 2.58, 0]))
        for r in range(3):
            ims = []
            for gi in range(3):
                ims.append(img(f"fig9_{keys[gi]}{r}_naive.png", s).move_to([gx[gi] - 0.85, ys[r], 0]))
                ims.append(img(f"fig9_{keys[gi]}{r}_ours.png", s).move_to([gx[gi] + 0.85, ys[r], 0]))
            self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.1), run_time=1.2)
        self.wait(1.0)
        self.caption("Ours stays coherent as the fovea moves, grows, or splits in two; the naïve baseline breaks each time.", 4.5)
        self.clear_all()

        # saliency-guided
        s2 = 1.55
        xr, xs_ = -5.55, -3.7
        yy = [1.45, -0.2, -1.85]
        self.add(tx("Random\nmasks", 22).move_to([xr, 2.65, 0]))
        self.add(tx("Saliency-\nguided", 22).move_to([xs_, 2.65, 0]))
        col_a = [img(f"fig10_random{k}.png", s2).move_to([xr, yy[k], 0]) for k in range(3)]
        col_b = [img(f"fig10_saliency{k}.png", s2).move_to([xs_, yy[k], 0]) for k in range(3)]
        self.play(LaggedStart(*[FadeIn(m) for m in col_a + col_b], lag_ratio=0.08), run_time=1.4)
        self.caption("Train with masks from a saliency predictor (DeepGaze) and the salient object lands in the fovea.", 3.8)
        apps = [("gaming", "VR gaming"), ("robotics", "robotics simulation"),
                ("dashcam", "driving simulation"), ("aerial", "aerial footage")]
        pos = [(0.4, 1.3), (3.9, 1.3), (0.4, -1.35), (3.9, -1.35)]
        self.add(tx("Applications", 24, bold=True).move_to([2.15, 2.75, 0]))
        for (k, name), (px, py) in zip(apps, pos):
            m = img(f"fig10_{k}1.png", 2.1).move_to([px, py, 0])
            self.play(FadeIn(m), run_time=0.4)
            self.add(tx(name, 22).next_to(m, DOWN, buff=0.1))
        self.caption("Only what matters (a weapon, a gripper, a van) is generated at full resolution.", 3.5)
        self.caption("Bounding-box masks from an object detector (YOLO) work the same way, with no change to the model.", 3.5)
        self.clear_all()

        # video spline path
        rect = Rectangle(width=5.0, height=2.8, color=DIM, stroke_width=2).move_to([-2.3, 1.2, 0])
        self.play(Create(rect), run_time=0.5)
        rng = np.random.default_rng(5)
        cp = np.array([[0.2, 0.35, 0.16], [0.55, 0.7, 0.22], [0.82, 0.4, 0.14]])
        tks = np.array([0.0, 0.5, 1.0])
        cs = CubicSpline(tks, cp, bc_type="natural")

        def to_pt(u, v):
            return rect.get_corner(UL) + np.array([u * 5.0, -v * 2.8, 0])

        dots = VGroup(*[Dot(to_pt(p[0], p[1]), radius=0.07, color=AMBER) for p in cp])
        rings = VGroup(*[DashedVMobject(Circle(radius=p[2] * 2.8, color=AMBER, stroke_width=1.5), num_dashes=24)
                         .move_to(to_pt(p[0], p[1])) for p in cp])
        self.play(FadeIn(dots), Create(rings), run_time=0.8)
        path = ParametricFunction(lambda t: to_pt(*cs(t)[:2]), t_range=[0, 1], color=AMBER, stroke_width=3)
        self.play(Create(path), run_time=1.2)
        lab = tx("3 random control points (<i>x</i>, <i>y</i>, <i>r</i>)\n→ cubic spline gaze path", 24).move_to([3.6, 1.2, 0])
        self.add(lab)
        pr = ValueTracker(0.0)
        fovc = always_redraw(lambda: Circle(radius=float(cs(pr.get_value())[2]) * 2.8, color=WHITE, stroke_width=3)
                             .move_to(to_pt(*cs(pr.get_value())[:2])))
        self.add(fovc)
        self.caption("For video, the fovea moves smoothly: position and radius follow a cubic spline across frames.")
        self.play(pr.animate.set_value(1.0), run_time=3.0, rate_func=linear)
        self.play(pr.animate.set_value(0.0), run_time=2.0, rate_func=smooth)
        frames = [img(f"vid_fox_ours_{k}.png", 1.44) for k in range(5)]
        Group(*frames).arrange(RIGHT, buff=0.12).move_to([0, -1.95, 0])
        self.play(LaggedStart(*[FadeIn(f) for f in frames], lag_ratio=0.15), run_time=1.2)
        self.caption("Generated video: the fox stays one fox as the fovea (white circle) travels across the frame.", 4.0)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S09 Results
class S09_Results(Base):
    def construct(self):
        self.title_card("Part 9", "Results")
        self.header("9 · Results")

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
        self.play(Create(ax), run_time=0.7)
        self.add(xt, yt, xl, yl)
        ci = VMobject(color=AMBER, stroke_width=4).set_points_as_corners([ax.c2p(100 - r, s) for r, s in img_pts])
        cv = VMobject(color=PURPLE, stroke_width=4).set_points_as_corners([ax.c2p(100 - r, s) for r, s in vid_pts])
        self.caption("Speedup over full high-res generation as the periphery grows (fewer tokens, to the right).")
        self.play(Create(ci), Create(cv), run_time=2.5)
        e1 = tx("2.2×", 24, AMBER).next_to(ax.c2p(75, 2.2), RIGHT, buff=0.12)
        e2 = tx("4.2×", 24, PURPLE).next_to(ax.c2p(75, 4.17), RIGHT, buff=0.12)
        lg1 = tx("image", 22, AMBER).move_to(ax.c2p(12, 4.25))
        lg2 = tx("video", 22, PURPLE).move_to(ax.c2p(12, 3.85))
        self.add(e1, e2, lg1, lg2)
        self.caption("At 25% of the tokens: 2.2× faster for images and 4.2× for video, where 3D attention dominates the cost.", 4.0)

        # table 1
        hdr = [("tokens", 1.2), ("speedup", 2.65), ("naïve", 4.2), ("ours", 5.75)]
        yh = 1.9
        self.add(tx("Image quality: HPSv2.1 ↑ (human preference)", 22, DIM).move_to([3.5, 2.6, 0]))
        for h, x in hdr:
            self.add(tx(h, 24, DIM if h != "ours" else WHITE, bold=(h == "ours")).move_to([x, yh, 0]))
        self.add(Line([0.5, yh - 0.3, 0], [6.5, yh - 0.3, 0], color=GHOST, stroke_width=2))
        data = [("100%", "1.00×", "full high-res:", "0.280"), ("43%", "1.61×", "0.268", "0.279"),
                ("30%", "1.98×", "0.270", "0.279"), ("26%", "2.08×", "0.275", "0.280")]
        for r, row in enumerate(data):
            y = yh - 0.75 - 0.62 * r
            for (h, x), v in zip(hdr, row):
                c = WHITE if h in ("tokens", "speedup") else (DIM if h == "naïve" else HR)
                if r == 0 and h in ("naïve", "ours"):
                    c = DIM if h == "naïve" else WHITE
                self.add(tx(v, 22 if v == "full high-res:" else 26, c, bold=(h == "ours" and r > 0))
                         .move_to([x, y, 0]))
            self.wait(0.5)
        self.add(tx("FID weighs all pixels equally, so it is less\ninformative for foveated images (see paper).", 18, DIM)
                 .move_to([3.5, -1.55, 0]))
        self.caption("Quality tracks full-resolution generation at every ratio; the naïve baseline falls behind.", 4.0)
        self.clear_all()

        # qualitative
        s = 1.95
        cx = [-5.3, -3.25, -1.2, 1.2, 3.25, 5.3]
        yr = [1.2, -0.85]
        heads = ["Full high-res", "Naïve", "Ours"] * 2
        for x, h in zip(cx, heads):
            self.add(tx(h, 22, WHITE if h == "Ours" else DIM, bold=(h == "Ours")).move_to([x, 2.5, 0]))
        subj = [("horse", "dog"), ("guitar", "farmer")]
        for r in range(2):
            ims = []
            for gi in range(2):
                for mi, m in enumerate(["full", "naive", "ours"]):
                    ims.append(img(f"fig5_{subj[r][gi]}_{m}.png", s).move_to([cx[3 * gi + mi], yr[r], 0]))
            self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.1), run_time=1.2)
        self.caption("White circles mark the fovea. Ours matches the full-resolution look; naïve duplicates and misscales.", 5.0)
        self.clear_all()

        # user study
        self.caption("A gaze-contingent user study: two-alternative forced choice with a fixation dot.")
        steps = [("fixate · 5 s", None), ("image A · 1 s", "fig5_farmer_full.png"), ("fixate · 1 s", None),
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
            self.play(FadeIn(grp), run_time=0.35)
            self.add(tx(lbl, 20, DIM).next_to(box, DOWN, buff=0.12))
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
            self.play(GrowFromEdge(fg, LEFT), run_time=0.8)
            self.add(tx(f"{p:.1f}%", 22).move_to(fg.get_right() + LEFT * 0.5))
            self.add(tx(note, 18, DIM).next_to(bg, DOWN, buff=0.08).align_to(bg, RIGHT))
        half = DashedLine([x0 + W / 2, 0.75, 0], [x0 + W / 2, -2.3, 0], color=WHITE, stroke_width=2)
        self.play(Create(half), run_time=0.5)
        self.add(tx("50%", 18).next_to(half, UP, buff=0.05))
        self.caption("11 participants, 220 votes per pair: ours vs. full resolution is at chance, i.e. perceptually indistinguishable.", 5.0)
        self.clear_all()

        # video
        rows = [("Full high-res", "fox_full"), ("Naïve", "fox_naive"), ("Ours", "fox_ours")]
        yv = [1.75, 0.4, -0.95]
        for (name, key), y in zip(rows, yv):
            self.add(tx(name, 24, WHITE if name == "Ours" else DIM, bold=(name == "Ours")).move_to([-5.7, y, 0]))
            fr = [img(f"vid_{key}_{k}.png", 1.22) for k in range(5)]
            Group(*fr).arrange(RIGHT, buff=0.1).move_to([1.0, y, 0])
            self.play(LaggedStart(*[FadeIn(f) for f in fr], lag_ratio=0.12), run_time=1.0)
        vb1 = tx("VBench subject consistency:  full 0.941 · naïve 0.907 · <b>ours 0.945</b>", 24).move_to([0.3, -1.85, 0])
        vb2 = tx("3.5× faster at 38% of the tokens (Wan2.1 1.3B, 480p)", 24, HR).move_to([0.3, -2.4, 0])
        self.add(vb1, vb2)
        self.caption("Video: the naïve baseline spawns a second fox at the boundary; ours stays consistent and runs 3.5× faster.", 5.0)
        self.clear_all()
        self.wait(0.3)


# ================================================================== S10 Outlook
class S10_Outlook(Base):
    def construct(self):
        self.title_card("Part 10", "Limitations and outlook")
        self.header("10 · Limitations and outlook")

        a = img("artifact_image_annot.png", 4.0).move_to([-4.6, 0.3, 0])
        b = img("artifact_video_annot.png", 4.0).move_to([-1.5, 0.3, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.8)
        t1 = tx("Occasional color seams\nat the fovea boundary", 30, bold=True).move_to([3.4, 1.4, 0])
        t2 = tx("cause: two separate VAE decodes,\nthen a pixel-space blend", 24, DIM).move_to([3.4, 0.2, 0])
        t3 = tx("fix: a VAE that decodes\nmixed-resolution tokens directly", 24, DIM).move_to([3.4, -0.9, 0])
        self.add(t1)
        self.wait(0.8)
        self.add(t2)
        self.wait(0.8)
        self.add(t3)
        self.caption("Red circles: blending artifacts. They come from the pixel-space blend, not from the diffusion model.", 4.5)
        self.clear_all()

        # multi-level + gaze following
        g = Grid(16, 4.2, (-3.5, 0.25))
        mobs = []
        for qi in range(4):
            for qj in range(4):
                dq = max(abs(qi - 1.5), abs(qj - 1.5))
                if dq < 1:           # central 2×2 quads → split to 2×2 blocks
                    for bi in (2 * qi, 2 * qi + 1):
                        for bj in (2 * qj, 2 * qj + 1):
                            if (bi - 3.5) ** 2 + (bj - 3.5) ** 2 <= 2.0:
                                for di in (0, 1):
                                    for dj in (0, 1):
                                        mobs.append(token(g.hs, HR).move_to(g.cell(2 * bi + di, 2 * bj + dj)))
                            else:
                                mobs.append(token(g.ls, LR).move_to(g.block(bi, bj)))
                else:
                    c = g.center + np.array([(4 * qj + 2) * g.c - g.size / 2, g.size / 2 - (4 * qi + 2) * g.c, 0])
                    mobs.append(token(3.2 * g.c, LR3).move_to(c))
        self.play(LaggedStart(*[FadeIn(m, scale=0.7) for m in mobs], lag_ratio=0.02), run_time=1.6)
        self.add(tx("three levels: 1×, ½×, ¼×", 24).move_to([-3.5, -2.35, 0]))
        self.caption("The framework extends to more foveation levels; this needs multi-level phase-aligned RoPE.", 3.2)

        g2 = Grid(16, 4.2, (3.5, 0.25))
        field = FoveaField(g2, (4, 4), 2.0)
        self.play(LaggedStart(*[FadeIn(m, scale=0.7) for m in field.mobs()], lag_ratio=0.02), run_time=1.2)
        gaze = Dot(g2.center, radius=0.09, color=RED)
        self.add(gaze)
        self.add(tx("tokens follow an eye tracker", 24).move_to([3.5, -2.35, 0]))
        self.caption("Most impactful in a streaming video world model with an eye tracker: the fovea follows the gaze.")
        path = [(3.2, 2.6), (2.6, 4.6), (4.2, 5.6), (5.4, 4.2), (4.6, 2.8), (4, 4)]
        for p in path:
            target = g2.center + np.array([p[1] * 2 * g2.c - g2.size / 2, g2.size / 2 - p[0] * 2 * g2.c, 0])
            anims = field.retarget(p)
            self.play(gaze.animate.move_to(target), *anims, run_time=0.7)
        self.wait(0.8)
        self.clear_all()

        # summary
        self.add(tx("Foveated Diffusion in three ideas", 40, bold=True).move_to([0, 2.5, 0]))
        pts = [
            f"1.  Spend tokens by eccentricity:  <i>L</i> = <i>m</i> + (<i>h</i>·<i>w</i> − <i>m</i>) / 4",
            "2.  Post-train (LoRA) on targets built from one image at two resolutions",
            "3.  Keep each attention on one scale with phase-aligned RoPE",
        ]
        for k, p in enumerate(pts):
            self.add(tx(p, 27).move_to([0, 1.3 - 0.85 * k, 0]).align_to([-6.0, 0, 0], LEFT))
            self.wait(2.5)
        self.add(tx("≈ 2× faster images  ·  ≈ 4× faster video  ·  perceptually indistinguishable",
                    28, HR).move_to([0, -1.55, 0]))
        self.wait(4.0)
        self.clear_all(run_time=0.6)
        self.remove(self.hdr)
        t = tx("Foveated Diffusion", 60, bold=True).move_to([0, 0.9, 0])
        u = tx("arXiv 2603.23491  ·  bchao1.github.io/foveated-diffusion", 28, DIM).move_to([0, -0.1, 0])
        v = tx("Chao*, Yariv*, Xiao, Wetzstein · Stanford University", 24, DIM).move_to([0, -0.8, 0])
        self.add(t, u, v)
        self.wait(4.0)
