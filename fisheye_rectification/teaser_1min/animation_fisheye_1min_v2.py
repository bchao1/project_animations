"""Self-Supervised Deep Learning for Fisheye Image Rectification (FE-GAN) — 1-minute teaser, v2.

Condensed from the 5-min teaser (../animation_fisheye_v3.py): helpers are copied, not imported.
Source: Chao*, Hsu*, Lee, Wang, "Self-Supervised Deep Learning for Fisheye Image Rectification", IEEE ICASSP 2020.
Assets are read (never written) from ../data (paper figure crops, website GIF frames) and ../data/generated (toy caches
built by the 5-min script). Render with ./render.sh; scenes in order: T1_Hook, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
import matplotlib
import fcntl
import os
import re

DATA = "/home/brianchc/project_animations/fisheye_rectification/data"
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as random_phase_wave_splatting)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ 𝔼 ‖ ∇ ∫ …
SYMFONT = "DejaVu Sans"                 # ✗ is in neither
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
ORANGE = "#FF9E40"
FISH = ORANGE        # fisheye / distorted
PIN = CYAN           # pinhole / distortion-free
BOX = "#1E2A33"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↑↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓‖∇↦ℒπ")
SYMCH = set("✗")


def mathify(s):
    out = []
    for c in s:
        if c in MATHCH:
            out.append(f'<span font_family="{MATHFONT}">{c}</span>')
        elif c in SYMCH:
            out.append(f'<span font_family="{SYMFONT}">{c}</span>')
        else:
            out.append(c)
    return "".join(out)


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


def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


TH = I("θ")
XT = I("x̃")
X_, Y_, G_, U_, W_, D_, P_, F_, R_ = (I(c) for c in "xyGUWDpfR")
FP = F_ + sub(I("p"))
RT = R_ + sub(TH)
RMT = R_ + sub("−" + TH)
E_X = "𝔼" + sub(X_) + "\u2009"
E_Y = "𝔼" + sub(Y_) + "\u2009"
DADV = D_ + sub("adv")
DCLS = D_ + sub("cls")
L1 = sub("1")
EQ_WARP = f"{XT}({P_}) = {X_}({P_} + {FP})"
EQ_ADV = f"ℒ{sub('adv')} = {E_Y}[log {DADV}({Y_})] + {E_X}[log(1 − {DADV}({G_}({X_})))]"
EQ_CLS = (f"ℒ{sub('cls')} = {E_Y}[log {DCLS}({Y_})] + {E_X}[log(1 − {DCLS}({X_}))] + "
          f"{E_X}[log(1 − {DCLS}({G_}({X_})))]")
EQ_GEO1 = f"ℒ{sub('geo')} = {E_X}‖{RT}({G_}({X_})) − {G_}({RT}({X_}))‖{L1}"
EQ_GEO2 = f"+ {E_X}‖{G_}({X_}) − {RMT}({G_}({RT}({X_})))‖{L1}"
EQ_FLOWC = f"ℒ{sub('flow')} = {E_X}‖{U_}({X_}) − {U_}({RT}({X_}))‖{L1}"
EQ_SMOOTH = f"ℒ{sub('smooth')} = {E_X} ∫ ‖∇{sub(I('x'))}{FP}‖{L1} + ‖∇{sub(I('y'))}{FP}‖{L1}"
EQ_DIR = f"|{FP} · {P_}| / (‖{FP}‖ ‖{P_}‖)"
EQ_VAR = f"Var({{ ‖{FP}‖ : ‖{P_}‖ = {I('r')} }})"
EQ_FULL = f"{G_} = arg min{sub(G_)} max{sub(D_)} ℒ({G_}, {D_}),   ℒ = sum of the losses (1)–(6)"
MATHCH.add("ℒ")

# ------------------------------------------------------------------ toy geometry (numpy)
# Normalised image coordinates: the image frame is [-1, 1]^2, focal length f = 1 (half the image width).
# World = the pinhole image plane (canvas); a ray at angle θ lands at radius tan θ there.
# Projection family r = tan(κθ)/κ: κ = 1 pinhole (r = f tan θ), κ → 0 equidistant fisheye (r = f θ).
L_CAN = 6.4          # canvas half-extent (tan of the fisheye corner angle √2 rad = 81°, is 6.3)
PPU = 256            # canvas pixels per unit
NS = 512             # displayed image resolution
NK = 48              # pinhole → fisheye morph steps
NSTR = 41            # flow-strength family steps (S07)
S_MAX = 1.6          # largest flow strength in the family


def draw_scene(Lc=L_CAN, ppu=PPU, ss=2, sc=0.5, voff=0.7, wall=(178, 92, 70), trim=(228, 214, 190), floors=2,
               sky=((205, 225, 242), (80, 140, 205))):
    """Procedural street facade full of straight lines, drawn on the pinhole image plane. Scene units × sc = normalised."""
    n = int(round(2 * Lc * ppu)) * ss
    s = ppu * ss
    X = lambda u: (u * sc + Lc) * s
    Y = lambda v: (Lc - (v - voff) * sc) * s
    Wd = lambda k: max(1, int(round(k * ss * sc / 0.6)))
    im = Image.new("RGB", (n, n))
    d = ImageDraw.Draw(im)
    for j in range(n):                                   # sky gradient
        v = (Lc - j / s) / sc + voff
        t = np.clip((v + 0.2) / 3.5, 0, 1)
        c = (1 - t) * np.array(sky[0]) + t * np.array(sky[1])
        d.line([(0, j), (n, j)], fill=tuple(int(x) for x in c))
    ground = -0.95
    top = ground + 0.95 + 0.74 * floors + 0.13
    for (a, b, tp, colr) in ((-5.0, -2.35, 2.2, (120, 132, 150)), (2.35, 5.0, 2.6, (132, 124, 118)),
                             (-9.5, -5.0, 1.4, (150, 150, 140)), (5.0, 9.5, 1.8, (110, 120, 135)),
                             (-16, -9.5, 3.2, (128, 118, 128)), (9.5, 16, 2.4, (140, 136, 120))):
        d.rectangle([X(a), Y(tp), X(b), Y(ground)], fill=colr)
        for u0 in np.arange(a + 0.2, b - 0.3, 0.5):
            for v0 in np.arange(ground + 0.35, tp - 0.35, 0.55):
                d.rectangle([X(u0), Y(v0 + 0.32), X(u0 + 0.28), Y(v0)], fill=(55, 65, 80),
                            outline=(215, 215, 210), width=Wd(4))
    d.rectangle([0, Y(ground), n, n], fill=(150, 146, 140))            # pavement
    for v in np.arange(ground - 0.25, -40, -0.25):
        d.line([(0, Y(v)), (n, Y(v))], fill=(108, 104, 98), width=Wd(3))
    for u in np.arange(-40, 40.01, 0.25):
        d.line([(X(u), Y(ground)), (X(u), n)], fill=(108, 104, 98), width=Wd(3))
    d.rectangle([X(-2.2), Y(top), X(2.2), Y(ground)], fill=wall)         # main facade
    tw = tuple(int(c * 0.9) for c in wall)
    d.rectangle([X(-0.42), Y(top + 0.8), X(0.42), Y(top)], fill=tw)      # clock tower
    d.polygon([(X(-0.5), Y(top + 0.8)), (X(0.5), Y(top + 0.8)), (X(0), Y(top + 1.15))], fill=(70, 80, 95))
    cy = top + 0.4
    d.ellipse([X(-0.2), Y(cy + 0.2), X(0.2), Y(cy - 0.2)], fill=(235, 230, 215), outline=(60, 50, 45), width=Wd(4))
    d.line([(X(0), Y(cy)), (X(0), Y(cy + 0.15))], fill=(40, 40, 40), width=Wd(4))
    d.line([(X(0), Y(cy)), (X(0.1), Y(cy))], fill=(40, 40, 40), width=Wd(4))
    bands = [top] + [ground + 0.83 + 0.74 * k for k in range(floors)]
    for v in bands:
        d.rectangle([X(-2.25), Y(v + 0.04), X(2.25), Y(v - 0.04)], fill=trim)
    for u in np.arange(-2.2, 2.21, 0.55):
        d.rectangle([X(u - 0.03), Y(top), X(u + 0.03), Y(ground)], fill=tuple(int(0.5 * a + 0.5 * b) for a, b in zip(wall, trim)))
    for u0 in np.arange(-2.2, 2.2 - 0.01, 0.55):
        for k in range(floors):
            v0 = ground + 0.93 + 0.74 * k
            v1 = v0 + 0.5
            a, b = u0 + 0.13, u0 + 0.42
            d.rectangle([X(a), Y(v1), X(b), Y(v0)], fill=(52, 70, 92), outline=(240, 232, 215), width=Wd(5))
            d.line([(X((a + b) / 2), Y(v1)), (X((a + b) / 2), Y(v0))], fill=(240, 232, 215), width=Wd(4))
            d.line([(X(a), Y((v0 + v1) / 2)), (X(b), Y((v0 + v1) / 2))], fill=(240, 232, 215), width=Wd(4))
    d.rectangle([X(-0.22), Y(-0.28), X(0.22), Y(ground)], fill=(90, 55, 40), outline=(240, 232, 215), width=Wd(5))
    d.line([(X(0), Y(-0.28)), (X(0), Y(ground))], fill=(240, 232, 215), width=Wd(4))
    for u0 in (-2.0, -1.2, 0.6, 1.4):
        d.rectangle([X(u0), Y(-0.3), X(u0 + 0.6), Y(-0.85)], fill=(70, 96, 110), outline=(240, 232, 215), width=Wd(5))
    for u in (-1.35, 1.35):
        d.rectangle([X(u - 0.025), Y(0.1), X(u + 0.025), Y(ground - 0.3)], fill=(35, 38, 42))
        d.ellipse([X(u - 0.08), Y(0.22), X(u + 0.08), Y(0.06)], fill=(255, 230, 150))
    im = im.resize((n // ss, n // ss), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def sample(img, u, v, extent):
    """Bilinear sample an image covering [-extent, extent]^2 at normalised coords (u right, v up); edges clamp."""
    n = img.shape[0]
    cc = (u + extent) / (2 * extent) * n - 0.5
    rr = (extent - v) / (2 * extent) * n - 0.5
    return np.clip(np.stack([map_coordinates(img[..., c], [rr, cc], order=1, mode="nearest")
                             for c in range(img.shape[2])], -1), 0, 1)


def grid(n):
    t = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(t, -t)
    return u, v


def down2(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, -1).mean((1, 3))


def proj_r(theta, kappa):
    """Image radius (f = 1) of a ray at angle theta for the family r = tan(κθ)/κ."""
    return theta if kappa < 1e-4 else np.tan(kappa * theta) / kappa


def inv_theta(r, kappa):
    return r if kappa < 1e-4 else np.arctan(kappa * r) / kappa


def render_view(canvas, kappa, n=NS):
    """Image of the scene through a lens of the κ family (2× supersampled)."""
    u, v = grid(2 * n)
    r = np.hypot(u, v) + 1e-12
    th = np.clip(inv_theta(r, kappa), 0, np.pi / 2 - 1e-3)
    s = np.tan(th) / r
    return down2(sample(canvas, u * s, v * s, L_CAN))


def world_to_image(wx, wy, kappa):
    rw = np.hypot(wx, wy) + 1e-12
    r = proj_r(np.arctan(rw), kappa)
    return wx * r / rw, wy * r / rw


def flow_true_uv(u, v):
    """Exact rectifying flow of the toy lens: x~(p) = x(p + f_p) with p + f_p = p · atan|p| / |p| (inward)."""
    r = np.hypot(u, v) + 1e-12
    s = np.arctan(r) / r
    return u * (s - 1), v * (s - 1)


def flow_xonly_uv(u, v):
    """A hand-made rectifier that only fixes horizontal bending (not rotation-equivariant)."""
    return np.arctan(u) - u, 0 * v


def warp(img, fu, fv, n):
    u, v = grid(n)
    return sample(img, np.clip(u + fu, -1, 1), np.clip(v + fv, -1, 1), 1.0)


def warp_fn(img, fn, n=NS):
    u, v = grid(2 * n)
    fu, fv = fn(u, v)
    return down2(warp(img, fu, fv, 2 * n))


def flow_rgb(fu, fv, fmax):
    h = (np.arctan2(fv, fu) + np.pi) / (2 * np.pi)
    m = np.clip(np.hypot(fu, fv) / fmax, 0, 1)
    hsv = np.stack([h, m ** 0.8, 0.35 + 0.65 * m ** 0.5], -1)
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def rgba(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


def flow_losses(fu, fv):
    """Toy versions of the paper's flow terms, flow in pixels of a 512² image:
    smooth = mean(|∂x f| + |∂y f|) (Eq. 5), align = mean |cos(f_p, p)| (Eq. 6, direction), var = mean over 64 radius
    bins of Var(|f_p|) (Eq. 6, circles)."""
    n = fu.shape[0]
    a, b = fu * n / 2, fv * n / 2
    sm = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(b, axis=1)).mean()
          + np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(b, axis=0)).mean())
    u, v = grid(n)
    r = np.hypot(u, v)
    m = np.hypot(a, b)
    ok = (r > 0.05) & (m > 1e-6)
    cos = np.abs(a * u + b * v)[ok] / (m * r)[ok]
    bins = np.minimum((r / 1.0 * 64).astype(int), 999)
    var = np.mean([m[(bins == k)].var() for k in range(64) if (bins == k).sum() > 4])
    return float(sm), float(cos.mean()), float(var)


# ------------------------------------------------------------------ cached assets (read only)
def gen(name):
    return os.path.join(GEN, name)


def read_stats():
    out = {}
    for line in open(gen("stats.txt")):
        k, *v = line.split(" ", 1)
        out[k] = eval(v[0])
    return out


assert os.path.exists(gen("done_v1")), "run the 5-min script once to build ../data/generated"


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def arr_img(a, h=None, w=None):
    m = ImageMobject(rgba(a))
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def live(m, fn):
    """Attach an updater that rewrites the image pixels every frame."""
    m.add_updater(lambda mob: setattr(mob, "pixel_array", rgba(fn())))
    return m


def interp(frames, v):
    v = float(np.clip(v, 0, len(frames) - 1))
    i0 = int(np.floor(v))
    i1 = min(i0 + 1, len(frames) - 1)
    w = v - i0
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def pt(x, y):
    return np.array([x, y, 0.0])


# ------------------------------------------------------------------ scene base (from random_phase_wave_splatting)
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None
        self.clock = ValueTracker(0)
        self.clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(self.clock)

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

    # Captions stay up long enough to read: ~17 characters per second + 0.8 s.
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
        s = s.replace(" × ", "\u00a0×\u00a0")
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
        self.cap_need = len(plain) / 17.0 + 0.8
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def hold(self, extra=0.0):
        self._hold_caption()
        if extra:
            self.wait(extra)

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def title_card(self, num, title, hold=1.5):
        a = tx(num, 30, DIM)
        b = tx(title, 50, bold=True)
        if b.width > 12.5:
            b.scale_to_fit_width(12.5)
        g = VGroup(a, b).arrange(DOWN, buff=0.3)
        self.add(g)
        self.wait(hold)
        self.remove(g)

    def clear_all(self, keep=(), run_time=0.7):
        self.uncaption()
        keep = set(keep) | {self.hdr, self.clock}
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def run(self, tracker, target, t, rate=linear):
        self.play(tracker.animate.set_value(target), run_time=t, rate_func=rate)


def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m, placed on a common baseline so labels in a row line up."""
    t = tx(s, size, color)
    plain = re.sub(r"<[^>]+>", "", s)
    desc = any(c in "gjpqy(),;[]{}|µφμβγηρχψξζ" for c in plain) or 'rise="-' in s
    dd, asc = 0.0027 * size, 0.0098 * size        # CMU Serif descender / ascender depth per font-size unit
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


def toy_note(s):
    return tx(s, 16, DIM)


# ------------------------------------------------------------------ geometry mobjects
GRID_CS = np.arange(-6.25, 6.2501, 0.25)
GRID_T = np.linspace(-6.4, 6.4, 640)


def grid_segments(kappa):
    segs = []
    for c in GRID_CS:
        for wx, wy in ((np.full_like(GRID_T, c), GRID_T), (GRID_T, np.full_like(GRID_T, c))):
            qx, qy = world_to_image(wx, wy, kappa)
            ins = (np.abs(qx) <= 1.0005) & (np.abs(qy) <= 1.0005)
            idx = np.flatnonzero(np.diff(np.r_[0, ins.astype(int), 0]))
            for a, b in zip(idx[::2], idx[1::2]):
                if b - a >= 2:
                    segs.append(np.c_[np.clip(qx[a:b], -1, 1), np.clip(qy[a:b], -1, 1)])
    return segs


def set_paths(mob, segs, center, half):
    mob.clear_points()
    for s in segs:
        p = np.c_[center[0] + s[:, 0] * half, center[1] + s[:, 1] * half, np.zeros(len(s))]
        mob.start_new_path(p[0])
        mob.add_points_as_corners(p[1:])
    return mob


def world_polyline(wx, wy, kappa, center, half):
    qx, qy = world_to_image(wx, wy, kappa)
    ins = (np.abs(qx) <= 1) & (np.abs(qy) <= 1)
    return [np.c_[qx[ins], qy[ins]]] if ins.sum() > 1 else []


def flow_field(fn, n, center, size, gain=1.0, color=AMBER, sw=2.2, dots=True):
    """Arrow grid of a flow fn(u, v) → (fu, fv) in normalised coords, drawn in a square of side `size`."""
    t = (np.arange(n) + 0.5) / n * 2 - 1
    g = VGroup()
    h = size / 2
    for v in t[::-1]:
        for u in t:
            fu, fv = fn(np.array(u), np.array(v))
            a = pt(center[0] + u * h, center[1] + v * h)
            b = a + pt(float(fu), float(fv)) * h * gain
            ln = np.linalg.norm(b - a)
            if ln < 0.03:
                if dots:
                    g.add(Dot(a, radius=0.018, color=color))
                continue
            g.add(Arrow(a, b, buff=0, color=color, stroke_width=sw, tip_length=min(0.1, 0.45 * ln),
                        max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=30))
    return g


def hourglass(center, w=1.9, h=1.6, label_="U", tag=None):
    """Encoder–decoder with skip connections (flow-prediction module U)."""
    c = np.array(center)
    blocks = VGroup()
    hs = [1.0, 0.72, 0.46, 0.3, 0.46, 0.72, 1.0]
    xs = np.linspace(-w / 2, w / 2, len(hs))
    for x, hh in zip(xs, hs):
        blocks.add(Rectangle(width=w / len(hs) * 0.72, height=h * hh, stroke_color="#90CAF9", stroke_width=1.5,
                             fill_color="#274056", fill_opacity=1).move_to(c + RIGHT * x))
    skips = VGroup()
    for i in range(3):
        a = blocks[i].get_top()
        b = blocks[len(hs) - 1 - i].get_top()
        skips.add(ArcBetweenPoints(a + UP * 0.05, b + UP * 0.05, angle=-PI / 3, color="#90CAF9",
                                   stroke_width=1.2, stroke_opacity=0.6))
    g = VGroup(skips, blocks, tx(I(label_), 30).move_to(c + DOWN * (h / 2 + 0.3)))
    if tag:
        t = tx(tag, 16, AMBER).move_to(c + DOWN * (h / 2 + 0.62))
        g.add(t)
    return g


def trapezoid(center, w=1.2, h=1.5, color="#B0BEC5", fill="#37474F"):
    s, t = h / 2, h / 4
    p = Polygon([-w / 2, s, 0], [w / 2, t, 0], [w / 2, -t, 0], [-w / 2, -s, 0], color=color, stroke_width=2,
                fill_color=fill, fill_opacity=1)
    return p.move_to(center)


def rbox(center, w, h, text, size=26, fill=BOX, stroke="#90A4AE"):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, fill_color=fill, fill_opacity=1,
                         stroke_color=stroke, stroke_width=2).move_to(center)
    return VGroup(r, tx(text, size).move_to(center))


def pulse(scene, path_pts, color=AMBER, t=0.8):
    d = Dot(path_pts[0], radius=0.07, color=color)
    path = VMobject().set_points_as_corners([np.array(p) for p in path_pts])
    scene.add(d)
    scene.play(MoveAlongPath(d, path), run_time=t, rate_func=smooth)
    scene.remove(d)


# ================================================================== T1 Hook
def gif_frames():
    f0 = np.asarray(Image.open(os.path.join(DATA, "gif_frames/frame_0.png")).convert("RGB"))
    f1 = np.asarray(Image.open(os.path.join(DATA, "gif_frames/frame_1.png")).convert("RGB"))
    return f0, f1


class T1_Hook(Base):
    def construct(self):
        t1 = tx("Self-Supervised Deep Learning", 50, bold=True).move_to([0, 3.0, 0])
        t2 = tx("for Fisheye Image Rectification", 36).move_to([0, 2.35, 0])
        f0, f1 = gif_frames()
        t0 = self.renderer.time
        state = lambda: int((self.renderer.time - t0) / 0.8) % 2
        g = arr_img(f0, 3.0).move_to([0, -0.05, 0])
        live(g, lambda: f1 if state() else f0)
        lg = always_redraw(lambda: label(g, "rectified by FE-GAN" if state() else "fisheye input", 22,
                                         GREEN if state() else FISH, d=DOWN))
        auth = tx("Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang", 22, DIM).move_to([0, -2.3, 0])
        aff = tx("National Taiwan University  ·  ASUS Intelligent Cloud Services  ·  IEEE ICASSP 2020", 20, DIM
                 ).move_to([0, -2.7, 0])
        self.add(t1, t2, g, lg, auth, aff)
        self.caption("Fisheye lenses see wide, but they bend every straight line.")
        self.hold(1.6)
        self.clear_all(run_time=0.5)
        # pinhole → fisheye morph (toy)
        stack = np.load(gen("morph.npy"))
        kt = ValueTracker(1.0)
        C, H = pt(-3.0, 0.25), 2.5
        im = arr_img(stack[0], 2 * H).move_to(C)
        live(im, lambda: interp(stack, (1 - kt.get_value()) * (NK - 1)))
        gridm = VMobject().set_stroke(WHITE, 1.2, 0.5)
        gridm.add_updater(lambda m: set_paths(m, grid_segments(kt.get_value()), C, H))
        x0, x1, ys = 1.2, 5.8, 1.3
        track = Line(pt(x0, ys), pt(x1, ys), color=DIM, stroke_width=3)
        knob = always_redraw(lambda: Dot(pt(x0 + (x1 - x0) * (1 - kt.get_value()), ys), radius=0.1,
                                         color=interpolate_color(ManimColor(PIN), ManimColor(FISH), 1 - kt.get_value())))
        lp = tx("pinhole", 24, PIN).next_to(track.get_start(), DOWN, buff=0.22)
        lf = tx("fisheye", 24, FISH).next_to(track.get_end(), DOWN, buff=0.22)
        r1 = always_redraw(lambda: tx(f"ray angle at the image corner: {np.degrees(inv_theta(np.sqrt(2), kt.get_value())):.0f}°",
                                      24).move_to([3.5, 0.0, 0]))
        note = toy_note("toy scene; same focal length, pinhole r = f tan θ → equidistant fisheye r = f θ").move_to([0, -2.6, 0])
        tt = np.linspace(-6.3, 6.3, 900)
        top_y, gnd_y = ((-0.95 + 0.95 + 0.74 * 2 + 0.13) - 0.7) * 0.5, (-0.95 - 0.7) * 0.5
        hl = VMobject().set_stroke(AMBER, 4)
        hl.add_updater(lambda m: set_paths(m, world_polyline(tt, np.full_like(tt, top_y), kt.get_value(), C, H) +
                                           world_polyline(tt, np.full_like(tt, gnd_y), kt.get_value(), C, H), C, H))
        self.play(FadeIn(im), run_time=0.5)
        self.add(framed(im, DIM, 1.2), gridm, hl, track, knob, lp, lf, r1, note)
        self.caption("Same scene, same focal length: a pinhole view becomes a fisheye view.")
        self.run(kt, 0.0, 4.0, rate=smooth)
        self.wait(0.5)
        self.caption("Undoing it usually needs calibration, or paired ground truth that is hard to get.")
        self.hold(1.0)
        self.clear_all(run_time=0.5)


# ================================================================== T2 Method
class T2_Method(Base):
    def construct(self):
        rect_a = np.asarray(Image.open(gen("rect.png")))
        S = 3.2
        cA, cM, cB = pt(-4.4, 0.8), pt(0, 0.8), pt(4.4, 0.8)
        A = img(gen("fisheye.png"), S).move_to(cA)
        fl = img(gen("flow_rgb.png"), S).move_to(cM).set_opacity(0.55)
        field = flow_field(flow_true_uv, 11, cM, S, gain=0.5, color=WHITE, sw=2.0)
        outbox = Square(S, color=PIN, stroke_width=1.5).move_to(cB)
        self.add(A, framed(A, FISH, 1.5), outbox, label(A, f"fisheye {X_}", 24, FISH), label(outbox, f"rectified {XT}", 24, PIN))
        a1 = arrow(A.get_right(), fl.get_left(), buff=0.1)
        a2 = arrow(fl.get_right(), outbox.get_left(), buff=0.1)
        self.add(a1, tx(U_, 24).next_to(a1, UP, buff=0.08), a2, tx(W_, 24).next_to(a2, UP, buff=0.08))
        self.play(FadeIn(fl), LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else FadeIn(a) for a in field],
                                           lag_ratio=0.01), run_time=1.2)
        self.add(label(fl, f"predicted flow {F_} = {U_}({X_})", 24, AMBER), framed(fl, AMBER, 1.2))
        eq = tx(EQ_WARP, 34).move_to([0, -1.45, 0])
        self.add(eq, toy_note("toy example: exact flow of a toy lens, arrows at half length").move_to([0, -2.05, 0]))
        rv = ValueTracker(0.0)
        out = arr_img(np.zeros((NS, NS, 3), np.uint8), S).move_to(cB)
        rows = np.arange(NS)[:, None, None]
        live(out, lambda: np.where(rows < rv.get_value() * NS, rect_a, 0).astype(np.uint8))
        self.add(out, outbox)
        self.caption("FE-GAN predicts a pixel flow, then warps the fisheye image through it.")
        self.run(rv, 1.0, 2.0)
        out.clear_updaters()
        self.hold()
        self.caption("A GAN trained on unpaired images: its discriminator asks, distorted or distortion-free?")
        self.clear_all(run_time=0.5)
        # self-supervision from geometry
        cL, SL = pt(-3.4, 0.55), 3.0
        fr = flow_field(flow_true_uv, 9, cL, SL, gain=0.7, color=AMBER, sw=3.0)
        ghost = fr.copy().set_color(WHITE).set_opacity(0.3)
        bxL = Square(SL, color=DIM, stroke_width=1.2).move_to(cL)
        self.add(bxL, ghost, fr, label(bxL, "cross-rotation consistency", 24, CYAN))
        cR, SR = pt(3.4, 0.55), 3.0
        h = SR / 2
        bxR = Square(SR, color=DIM, stroke_width=1.2).move_to(cR)
        faint = flow_field(flow_true_uv, 9, cR, SR, gain=0.5, color=WHITE, sw=1.6).set_opacity(0.4)
        ring = VGroup()
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            u, v = 0.7 * np.cos(ang), 0.7 * np.sin(ang)
            fu, fv = flow_true_uv(np.array(u), np.array(v))
            p0 = cR + pt(u, v) * h
            ring.add(Arrow(p0, p0 + pt(float(fu), float(fv)) * h * 1.6, buff=0, color=GREEN, stroke_width=3.5,
                           tip_length=0.09, max_tip_length_to_length_ratio=0.5))
        circ = Circle(radius=0.7 * h, color=GREEN, stroke_width=2.5).move_to(cR)
        self.add(bxR, faint, label(bxR, "intra-image warping consistency", 24, GREEN))
        self.add(tx(f"{RT}({G_}({X_})) = {G_}({RT}({X_}))", 26).next_to(bxL, DOWN, buff=0.25),
                 tx("radial and smooth flow", 26).next_to(bxR, DOWN, buff=0.25),
                 toy_note("toy flow fields").move_to([0, -2.55, 0]))
        self.caption("No pairs needed. Rotate the input by 90°, and the output must simply rotate;")
        self.play(Rotate(fr, PI / 2, about_point=cL), run_time=1.8)
        self.caption("and a fisheye flow must point along the radius, with one length per circle.")
        self.play(Create(circ), LaggedStart(*[GrowArrow(a) for a in ring], lag_ratio=0.05), run_time=1.2)
        self.hold()
        self.clear_all(run_time=0.5)


# ================================================================== T3 Payoff
class T3_Payoff(Base):
    def construct(self):
        cols = [("fisheye", "fisheye input", FISH), ("cyclegan", "CycleGAN", RED), ("unit", "UNIT", RED),
                ("aleman", "Aleman-Flores", RED), ("ours", "FE-GAN", GREEN)]
        xs = [-5.0, -2.5, 0, 2.5, 5.0]
        for r, (k, y) in enumerate(zip(["ch0", "mc0"], [1.5, -0.7])):
            for x, (ck, nm, c) in zip(xs, cols):
                m = img(f"fig2_{ck}_{k}.png", 2.05).move_to([x, y, 0])
                self.add(m)
                if r == 0:
                    self.add(label(m, nm, 24, c))
        self.add(tx("paper Fig. 2", 16, DIM).move_to([6.3, 3.3, 0]))
        head = tx("+2.3 dB PSNR over the best baseline (MCindoor, paper Table 1)", 30, AMBER).move_to([0, -2.4, 0])
        self.add(head)
        self.caption("Straighter results than CycleGAN, UNIT and a classical method.")
        self.caption("Best PSNR, SSIM and RMSE on MCindoor, Wireframe and LSUN Churches.")
        self.hold(0.6)
        self.clear_all(run_time=0.5)
        t1 = tx("Self-Supervised Deep Learning", 48, bold=True).move_to([0, 2.1, 0])
        t1b = tx("for Fisheye Image Rectification", 36).move_to([0, 1.4, 0])
        t2 = tx("IEEE ICASSP 2020", 28).move_to([0, 0.5, 0])
        t3 = tx("code: github.com/ByronHsu/FEGAN", 22, DIM).move_to([0, 0.0, 0])
        t4 = tx("Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang", 22, DIM).move_to([0, -0.8, 0])
        t5 = tx("National Taiwan University  ·  ASUS Intelligent Cloud Services", 22, DIM).move_to([0, -1.25, 0])
        t6 = tx("* equal contribution", 18, DIM).move_to([0, -1.65, 0])
        self.add(t1, t1b, t2, t3, t4, t5, t6)
        self.wait(4.5)
