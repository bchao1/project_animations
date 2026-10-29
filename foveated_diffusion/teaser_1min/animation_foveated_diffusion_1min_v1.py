"""Foveated Diffusion — ~1-minute teaser, v1 (compact cut of teaser_5min v2).

Source: Chao, Yariv, Xiao, Wetzstein, "Foveated Diffusion: Efficient Spatially
Adaptive Image and Video Generation", arXiv 2603.23491 (2026).
Images are read from ../teaser_5min/data (paper figure crops, by absolute path).

Render (see render.sh):
    manim -ql --media_dir media animation_foveated_diffusion_1min_v1.py <Scene>
Scenes, in order: S1_Insight, S2_Method, S3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image, ImageFilter
from scipy.interpolate import CubicSpline
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/brianchc/project_animations/foveated_diffusion/teaser_5min/data"   # read-only source assets
GEN5 = os.path.join(DATA, "generated")                                            # cached by the 5-min script
GEN = os.path.join(HERE, "data", "generated")                                     # new assets of this teaser

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
def gen5(name):
    return os.path.join(GEN5, name)


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


# gaze points (u, v) in image coordinates; A and B match the cached 5-min blurs
GAZE = [(0.50, 0.55), (0.80, 0.22), (0.25, 0.70)]
GAZE_IMG = [gen5("gaze_A.png"), gen5("gaze_B.png"), gen("gaze_C.png")]


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen("gaze_C.png")):
        return
    far = Image.open(os.path.join(DATA, "fig5_farmer_full.png")).convert("RGB")
    tmp = gen(f"gaze_C.{os.getpid()}.png")
    _gaze_blur(far, GAZE[2][0], GAZE[2][1], 40, 0.045, 9).save(tmp)
    os.replace(tmp, gen("gaze_C.png"))


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



def fit_width(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def to_fov(uv):
    """Gaze (u, v) in image coords → fovea centre in 2×2-block units of a 16×16 grid."""
    return (uv[1] * 8, uv[0] * 8)


def count_text(field, center):
    nh = 4 * sum(1 for (s, _) in field.state.values() if s == "H")
    nl = sum(1 for (s, _) in field.state.values() if s == "L")
    t = tx(f"{col(str(nh), HR)} high-res + {col(str(nl), LR)} low-res = <b>{nh + nl}</b> tokens", 28)
    return fit_width(t, 6.2).move_to(center)


# ================================================================== S1 Hook + insight
class S1_Insight(Base):
    def construct(self):
        # --- title
        title = tx("Foveated Diffusion", 72, bold=True).move_to([0, 2.75, 0])
        subt = tx("Efficient Spatially Adaptive Image and Video Generation", 32).move_to([0, 1.95, 0])
        g0 = Grid(16, 3.8, (0, -0.75))
        f0 = FoveaField(g0, (4, 4), 2.0)
        self.add(title, subt)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in f0.mobs()], lag_ratio=0.02), run_time=1.2)
        for p in [(3.5, 3), (3, 5), (4.5, 4.5)]:
            anims = f0.retarget(p)
            if anims:
                self.play(*anims, run_time=0.5)
            else:
                self.wait(0.5)
        self.wait(0.4)
        self.play(FadeOut(f0.group()), run_time=0.5)
        self.remove(title, subt)

        # --- hook: gaze-contingent blur
        S = 4.6
        pic = img("fig5_farmer_full.png", S).move_to([-3.4, 0.0, 0])

        def pt(uv):
            return pic.get_center() + np.array([(uv[0] - 0.5) * S, (0.5 - uv[1]) * S, 0])

        lab_see = tx("what you see", 26, DIM).next_to(pic, UP, buff=0.25)
        self.play(FadeIn(pic), run_time=0.5)
        dot = Dot(pt(GAZE[0]), radius=0.1, color=RED).set_z_index(10)
        self.add(dot, lab_see)
        self.caption("Keep your eyes on the red dot.")
        blur = img(GAZE_IMG[0], S).move_to(pic)
        self.play(FadeIn(blur), run_time=1.5)
        circ = DashedVMobject(Circle(radius=(40 / 512) * S * 1.9, color=WHITE, stroke_width=2),
                              num_dashes=30).move_to(dot).set_z_index(10)
        self.play(Create(circ), run_time=0.3)
        self.caption("You only see full detail where you look.")

        # --- problem: every token at full resolution
        g = Grid(16, S, (3.4, 0.0))
        toks = {(i, j): token(g.hs, HR).move_to(g.cell(i, j)) for i in range(16) for j in range(16)}
        lab_comp = tx("what the model computes", 26, DIM).move_to([3.4, lab_see.get_y(), 0])
        self.caption("Yet a diffusion transformer pays for every token equally.")
        self.add(lab_comp)
        self.play(LaggedStart(*[FadeIn(t) for t in toks.values()], lag_ratio=0.004), run_time=1.0)
        cnt_c = [3.4, -2.72, 0]
        cnt = tx(f"{col('256', HR)} high-res tokens", 28).move_to(cnt_c)
        self.add(cnt)

        # --- insight: foveated tokens under the gaze
        field = FoveaField(g, to_fov(GAZE[0]), 2.0)
        c0 = to_fov(GAZE[0])
        order = sorted(field.state.items(),
                       key=lambda kv: (kv[0][0] + 0.5 - c0[0]) ** 2 + (kv[0][1] + 0.5 - c0[1]) ** 2)
        anims = []
        for (bi, bj), (s, m) in order:
            four = VGroup(*[toks[(2 * bi + di, 2 * bj + dj)] for di in (0, 1) for dj in (0, 1)])
            anims.append(ReplacementTransform(four, m))
        self.caption("Foveated Diffusion: high-res tokens only where you look.")
        self.play(LaggedStart(*anims, lag_ratio=0.02), run_time=1.5)
        self.remove(cnt)
        cnt = count_text(field, cnt_c)
        self.add(cnt)
        self.caption("Elsewhere, every 2 × 2 block becomes one token.")
        for k in (1, 2):
            nb = img(GAZE_IMG[k], S).move_to(pic)
            tgt = pt(GAZE[k])
            self.play(dot.animate.move_to(tgt), circ.animate.move_to(tgt), FadeIn(nb),
                      *field.retarget(to_fov(GAZE[k])), run_time=1.2)
            self.remove(blur)
            blur = nb
            self.remove(cnt)
            cnt = count_text(field, cnt_c)
            self.add(cnt)
            self.wait(0.7)
        self.clear_all(run_time=0.6)
        self.wait(0.1)


# ================================================================== S2 Method
class S2_Method(Base):
    def construct(self):
        x = img("lion_train.png", 1.9).move_to([-5.7, 1.3, 0])
        xd = img(gen5("lion_train_down.png"), 0.95).move_to([-5.7, -1.6, 0])
        xl = tx("image <i>x</i>", 26).next_to(x, UP, buff=0.15)
        xdl = tx("Down(<i>x</i>)", 24).next_to(xd, DOWN, buff=0.15)
        adown = arrow(x.get_bottom(), xd.get_top(), buff=0.1, sw=2)
        self.play(FadeIn(x), run_time=0.4)
        self.add(xl)
        self.caption("Training targets: encode each image at two resolutions.")
        self.play(GrowArrow(adown), FadeIn(xd), run_time=0.5)
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
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(e1), FadeIn(e2), run_time=0.4)
        self.play(GrowArrow(a3), GrowArrow(a4),
                  LaggedStart(*[FadeIn(t) for t in zh.values()], lag_ratio=0.002),
                  LaggedStart(*[FadeIn(t) for t in zl.values()], lag_ratio=0.01), run_time=1.0)
        lzh = tx("16 × 16 high-res", 24, HR).next_to(VGroup(*zh.values()), UP, buff=0.15)
        lzl = tx("8 × 8 low-res", 24, LR).next_to(VGroup(*zl.values()), UP, buff=0.15)
        self.add(lzh, lzl)

        # mask + merge
        fb = fovea_blocks(8, 4, 4, 2.0)
        mg = Grid(16, 1.1, (0.55, -0.15))
        mcells = VGroup(*[Square(mg.c * 0.92).set_fill(WHITE if (i // 2, j // 2) in fb else "#2B2B2B", 1)
                          .set_stroke(width=0).move_to(mg.cell(i, j)) for i in range(16) for j in range(16)])
        mlab = tx(f"mask {M_}", 24).next_to(mcells, UP, buff=0.15)
        gf = Grid(16, 2.6, (3.5, -0.15))
        ghosts = VGroup(*[ghost(gf.hs).move_to(gf.cell(i, j)) for i in range(16) for j in range(16)])
        lzf = tx("foveated target", 28, bold=True).next_to(ghosts, UP, buff=0.2)
        self.caption("Merge by the mask: a mixed-resolution-consistent target.")
        self.play(FadeIn(mcells), FadeIn(ghosts), run_time=0.4)
        self.add(mlab, lzf)
        unused_h = [t for (i, j), t in zh.items() if (i // 2, j // 2) not in fb]
        unused_l = [t for k, t in zl.items() if k in fb]
        self.play(*[t.animate.set_opacity(0.2) for t in unused_h + unused_l], run_time=0.4)
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
        self.play(LaggedStart(*anims, lag_ratio=0.01), run_time=1.8)
        self.remove(ghosts)
        same = tx("both from the <i>same</i> image", 22, DIM).next_to(VGroup(*fh.values(), *fl.values()), DOWN, buff=0.2)
        self.add(same)

        # post-train the DiT
        tgt = VGroup(*fh.values(), *fl.values())
        self.caption("Post-train a pretrained DiT with LoRA on these targets.")
        rest = [m for m in self.mobjects if m not in (tgt, lzf, *tgt)]
        self.play(*[FadeOut(m) for m in rest], run_time=0.5)
        self.play(VGroup(tgt, lzf).animate.shift(LEFT * 5.6), run_time=0.8)
        dit = dit_box([1.6, -0.15, 0], w=1.9, h=2.4, lora=True)
        ain = arrow([tgt.get_right()[0] + 0.1, -0.15, 0], dit.get_left(), sw=2)
        self.play(GrowArrow(ain), FadeIn(dit), run_time=0.5)
        base = VGroup(tx("pretrained base models:", 22, DIM),
                      tx("FLUX.2 Klein 4B (images)", 24), tx("Wan2.1 1.3B (video)", 24)
                      ).arrange(DOWN, buff=0.15).move_to([4.9, -0.15, 0])
        self.add(base)
        self.play(Indicate(dit[2], color=WHITE, scale_factor=1.15), run_time=0.7)
        self.clear_all(run_time=0.6)
        self.wait(0.1)


# ================================================================== S3 Payoff + end card
class S3_Payoff(Base):
    def construct(self):
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
        xl = tx("fraction of tokens kept", 22, DIM).next_to(ax, DOWN, buff=0.55)
        yl = tx("speedup", 22, DIM).next_to(ax.c2p(0, 4.5), UP, buff=0.2)
        self.play(Create(ax), run_time=0.4)
        self.add(xt, yt, xl, yl)
        ci = VMobject(color=AMBER, stroke_width=4).set_points_as_corners([ax.c2p(100 - r, s) for r, s in img_pts])
        cv = VMobject(color=PURPLE, stroke_width=4).set_points_as_corners([ax.c2p(100 - r, s) for r, s in vid_pts])
        self.caption("Fewer tokens per step, much faster generation.")
        self.play(Create(ci), Create(cv), run_time=1.4)
        e1 = tx("2.2×", 24, AMBER).next_to(ax.c2p(75, 2.2), RIGHT, buff=0.12)
        e2 = tx("4.2×", 24, PURPLE).next_to(ax.c2p(75, 4.17), RIGHT, buff=0.12)
        lg1 = tx("image", 22, AMBER).move_to(ax.c2p(12, 4.25))
        lg2 = tx("video", 22, PURPLE).move_to(ax.c2p(12, 3.85))
        self.add(e1, e2, lg1, lg2)
        r1 = VGroup(tx("2.2×", 84, AMBER, bold=True), tx("faster images", 32))
        r2 = VGroup(tx("4.2×", 84, PURPLE, bold=True), tx("faster video", 32))
        for r in (r1, r2):
            r.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
        r1.move_to([3.6, 1.35, 0])
        r2.move_to([3.6, -0.1, 0]).align_to(r1, LEFT)
        r3 = tx("with 25% of the tokens", 26, DIM).next_to(r2, DOWN, buff=0.45).align_to(r1, LEFT)
        self.add(r1, r2, r3)
        self.wait(1.2)
        self.clear_all(run_time=0.5)

        # quality: full / naive / ours + user study
        s = 2.9
        xs = [-3.3, 0.0, 3.3]
        yi = 1.0
        names = [("Full high-res", "full", WHITE), ("Naïve (no training)", "naive", DIM), ("Ours", "ours", HR)]
        ims = [img(f"fig5_horse_{k}.png", s).move_to([x, yi, 0]) for (_, k, _), x in zip(names, xs)]
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.15), run_time=0.8)
        for (n, _, c), m in zip(names, ims):
            self.add(tx(n, 26, c, bold=(n == "Ours")).next_to(m, UP, buff=0.18))
        self.add(tx("duplicated horse", 22, RED).next_to(ims[1], DOWN, buff=0.12))
        self.caption("Without training it breaks; ours matches full resolution.")

        yb, x0, W = -1.75, -2.2, 4.4
        head = tx("gaze-contingent user study: which image looks better?", 20, DIM).move_to([0, -1.15, 0])
        lbl = tx("ours vs. full high-res", 24).move_to([0, yb, 0])
        lbl.shift(RIGHT * (x0 - 0.25 - lbl.get_right()[0]))
        bg = Rectangle(width=W, height=0.45, stroke_width=0).set_fill(GHOST, 1).move_to([x0 + W / 2, yb, 0])
        fg = Rectangle(width=W * 0.473, height=0.45, stroke_width=0).set_fill(HR, 1).move_to([x0 + W * 0.473 / 2, yb, 0])
        self.caption("Viewers could not tell ours from full resolution.")
        self.add(head, lbl, bg)
        self.play(GrowFromEdge(fg, LEFT), run_time=0.6)
        self.add(tx("47.3%", 22).move_to(fg.get_right() + LEFT * 0.5))
        half = DashedLine([x0 + W / 2, yb + 0.35, 0], [x0 + W / 2, yb - 0.35, 0], color=WHITE, stroke_width=2)
        self.play(Create(half), run_time=0.3)
        self.add(tx("50% = chance", 18).next_to(half, DOWN, buff=0.06))
        self.add(tx("p = 0.48: indistinguishable", 20, DIM).next_to(bg, RIGHT, buff=0.2))
        self.clear_all(run_time=0.5)

        # end card
        t = tx("Foveated Diffusion", 64, bold=True).move_to([0, 1.5, 0])
        tag = fit_width(tx("≈ 2× faster images  ·  ≈ 4× faster video  ·  perceptually indistinguishable", 30, HR),
                        12.8).move_to([0, 0.55, 0])
        a = fit_width(tx("Brian Chao*, Lior Yariv*, Howard Xiao, Gordon Wetzstein  ·  Stanford University", 24),
                      12.8).move_to([0, -0.6, 0])
        u = tx("arXiv 2603.23491  ·  bchao1.github.io/foveated-diffusion", 26, DIM).move_to([0, -1.3, 0])
        self.add(t, tag, a, u)
        self.wait(5.0)
