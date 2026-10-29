"""Self-Supervised Deep Learning for Fisheye Image Rectification (FE-GAN) — teaser animation, v2.

Source: Chao*, Hsu*, Lee, Wang, "Self-Supervised Deep Learning for Fisheye Image Rectification", IEEE ICASSP 2020
(pp. 2248-2252). Paper figures in data/ are crops of the paper (fisheye-rectification/pdfs/, via scratch/crop_figs.py);
data/icassp2020-result.gif is the website teaser (bchao1.github.io), frames in data/gif_frames/.
All geometry visuals are toy computations done here with numpy (a procedural street scene, an equidistant fisheye lens,
the exact rectifying flow, bilinear warps, rotation checks and flow-loss values) and cached in data/generated/
(delete data/generated/done_v1 / done_v2 to rebuild; v2 adds the blurred-error strength family).

Render (see render.sh):
    manim -ql --media_dir media animation_fisheye_v2.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
import matplotlib
import fcntl
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
ASSET_FLAG = "done_v1"

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


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(ASSET_FLAG)):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(ASSET_FLAG)):
            return
        rng = np.random.default_rng(2020)
        cv = draw_scene()
        # pinhole → fisheye morph (κ = 1 → 0, same focal length)
        kap = np.linspace(1, 0, NK)
        stack = np.stack([u8(render_view(cv, k)) for k in kap])
        np.save(gen("morph.npy"), stack)
        pin, fish = stack[0].astype(np.float32) / 255, stack[-1].astype(np.float32) / 255
        Image.fromarray(stack[0]).save(gen("pinhole.png"))
        Image.fromarray(stack[-1]).save(gen("fisheye.png"))
        # rectification with the exact flow
        rect = warp_fn(fish, flow_true_uv)
        Image.fromarray(u8(rect)).save(gen("rect.png"))
        u, v = grid(NS)
        fu, fv = flow_true_uv(u, v)
        fmax = float(np.hypot(fu, fv).max())
        Image.fromarray(flow_rgb(fu, fv, fmax)).save(gen("flow_rgb.png"))
        stats = {"rmse_rect": float(np.sqrt(((rect - pin) ** 2).mean()) * 255), "fmax": fmax}
        # cross-rotation check (Eq. 3): radial rectifier vs x-only rectifier, R = 90° counter-clockwise
        rot = lambda a: np.ascontiguousarray(np.rot90(a, 1))
        for name, fn in (("rad", flow_true_uv), ("xonly", flow_xonly_uv)):
            g = warp_fn(fish, fn)
            g_r = warp_fn(rot(fish), fn)
            dmap = np.abs(rot(g) - g_r).mean(-1)
            stats[f"geo_{name}"] = float(dmap.mean() * 255)
            Image.fromarray(u8(g)).save(gen(f"G_{name}.png"))
            Image.fromarray(u8(g_r)).save(gen(f"G_rot_{name}.png"))
            Image.fromarray(u8(rot(g))).save(gen(f"rot_G_{name}.png"))
            Image.fromarray(cmap_rgb(dmap / 0.35)).save(gen(f"geo_diff_{name}.png"))
        Image.fromarray(u8(rot(fish))).save(gen("fisheye_rot.png"))
        # candidate flows for the intra-image warping terms (S07)
        cands = {"noisy": (fu + rng.normal(0, 0.02, fu.shape), fv + rng.normal(0, 0.02, fv.shape)),
                 "swirl": (-fv, fu), "radial": (fu, fv)}
        for name, (a, b) in cands.items():
            stats[f"loss_{name}"] = flow_losses(a, b)
            Image.fromarray(u8(down2(warp(fish, *[np.kron(c, np.ones((2, 2))) for c in (a, b)], 2 * NS))))\
                .save(gen(f"cand_{name}.png"))
        # radial family s · f_true, s ∈ [0, S_MAX]: all satisfy the constraints; error vs the true pinhole view
        ss = np.linspace(0, S_MAX, NSTR)
        fam, err = [], []
        for s_ in ss:
            w = warp_fn(fish, lambda a, b, s_=s_: tuple(s_ * c for c in flow_true_uv(a, b)))
            fam.append(u8(w))
            err.append(float(np.sqrt(((w - pin) ** 2).mean()) * 255))
        np.save(gen("family.npy"), np.stack(fam))
        np.save(gen("family_err.npy"), np.array(err))
        stats["family_err_min_s"] = float(ss[int(np.argmin(err))])
        # unpaired distortion-free set Y (toy): other facades, pinhole views
        for k, (wall, trim, fl) in enumerate((((80, 110, 150), (220, 220, 225), 3), ((196, 170, 96), (245, 240, 225), 2),
                                              ((96, 140, 110), (230, 225, 205), 1))):
            c2 = draw_scene(Lc=1.1, ppu=160, wall=wall, trim=trim, floors=fl, sc=(0.38, 0.46, 0.5)[k],
                             voff=(1.05, 0.7, 0.45)[k])
            Image.fromarray(u8(sample(c2, *grid(256), 1.1))).save(gen(f"yset_{k}.png"))
        with open(gen("stats.txt"), "w") as fh:
            for k_, v_ in stats.items():
                fh.write(f"{k_} {v_}\n")
        open(gen(ASSET_FLAG), "w").write("ok")


def read_stats():
    out = {}
    for line in open(gen("stats.txt")):
        k, *v = line.split(" ", 1)
        out[k] = eval(v[0])
    return out


NSTR2 = 81            # v2: finer flow-strength family
S_MAX2 = 1.5


def build_assets_v2():
    """v2: strength family s · f_true on a finer grid, error = RMSE after a 3-px Gaussian blur (graded, not a spike)."""
    from scipy.ndimage import gaussian_filter
    if os.path.exists(gen("done_v2")):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen("done_v2")):
            return
        fish = np.asarray(Image.open(gen("fisheye.png"))).astype(np.float32) / 255
        pin = np.asarray(Image.open(gen("pinhole.png"))).astype(np.float32) / 255
        bl = lambda a: gaussian_filter(a, (3, 3, 0))
        pb = bl(pin)
        fam, err = [], []
        for s_ in np.linspace(0, S_MAX2, NSTR2):
            w = warp_fn(fish, lambda a, b, s_=s_: tuple(s_ * c for c in flow_true_uv(a, b)))
            fam.append(u8(w))
            err.append(float(np.sqrt(((bl(w) - pb) ** 2).mean()) * 255))
        np.save(gen("family2.npy"), np.stack(fam))
        np.save(gen("family2_err.npy"), np.array(err))
        open(gen("done_v2"), "w").write("ok")


build_assets()
build_assets_v2()


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


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        t1 = tx("Self-Supervised Deep Learning", 54, bold=True).move_to([0, 3.0, 0])
        t2 = tx("for Fisheye Image Rectification", 40).move_to([0, 2.28, 0])
        a = img("gif_frames/frame_0.png", 3.0).move_to([-2.0, -0.3, 0])
        b = img("gif_frames/frame_1.png", 3.0).move_to([2.0, -0.3, 0])
        la = label(a, "fisheye input", 22, FISH)
        lb = label(b, "rectified by FE-GAN", 22, GREEN)
        ar = arrow([-0.35, -0.3, 0], [0.35, -0.3, 0], buff=0)
        auth = tx("Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang", 22, DIM).move_to([0, -2.5, 0])
        aff = tx("National Taiwan University  ·  ASUS Intelligent Cloud Services  ·  IEEE ICASSP 2020", 22, DIM
                 ).move_to([0, -2.95, 0])
        eq = tx("* equal contribution", 18, DIM).move_to([0, -3.35, 0])
        self.add(t1, t2, auth, aff, eq)
        self.play(FadeIn(a), Create(framed(a, FISH, 1.5)), run_time=0.7)
        self.add(la)
        self.wait(1.2)
        self.play(GrowArrow(ar), FadeIn(b), Create(framed(b, GREEN, 1.5)), run_time=0.8)
        self.add(lb)
        self.wait(4.0)


# ================================================================== S01 Hook
class S01_Hook(Base):
    def construct(self):
        self.title_card("Part 1", "Why fisheye images look bent")
        self.header("1 · Why fisheye images look bent")
        stack = np.load(gen("morph.npy"))
        kt = ValueTracker(1.0)
        C, H = pt(-3.0, 0.2), 2.45
        im = arr_img(stack[0], 2 * H).move_to(C)
        live(im, lambda: interp(stack, (1 - kt.get_value()) * (NK - 1)))
        fr = framed(im, DIM, 1.2)
        gridm = VMobject().set_stroke(WHITE, 1.2, 0.5)
        gridm.add_updater(lambda m: set_paths(m, grid_segments(kt.get_value()), C, H))
        lab = label(im, "one toy street scene, one focal length", 22, DIM)
        # slider + readouts
        x0, x1, ys = 1.2, 5.8, 1.6
        track = Line(pt(x0, ys), pt(x1, ys), color=DIM, stroke_width=3)
        knob = always_redraw(lambda: Dot(pt(x0 + (x1 - x0) * (1 - kt.get_value()), ys), radius=0.1,
                                         color=interpolate_color(ManimColor(PIN), ManimColor(FISH), 1 - kt.get_value())))
        lp = tx("pinhole", 24, PIN).next_to(track.get_start(), DOWN, buff=0.22)
        lf = tx("fisheye", 24, FISH).next_to(track.get_end(), DOWN, buff=0.22)
        ttl = tx("lens projection", 24, DIM).move_to([(x0 + x1) / 2, ys + 0.5, 0])

        def edge_deg():
            return np.degrees(inv_theta(1.0, kt.get_value()))

        def corner_deg():
            return np.degrees(inv_theta(np.sqrt(2), kt.get_value()))
        r1 = always_redraw(lambda: tx(f"ray angle at the image edge: {edge_deg():.0f}°", 24).move_to([3.5, 0.3, 0]))
        r2 = always_redraw(lambda: tx(f"ray angle at the image corner: {corner_deg():.0f}°", 24).move_to([3.5, -0.25, 0]))
        note = toy_note("toy: same focal length f = half the image width; pinhole r = f tan θ → equidistant fisheye r = f θ"
                        ).move_to([0, -2.72, 0])
        self.play(FadeIn(im), run_time=0.7)
        self.add(fr, gridm, lab, track, knob, lp, lf, ttl, r1, r2, note)
        self.caption("An ordinary (pinhole) camera keeps straight lines in the world straight in the image.")
        self.hold()
        self.caption("A fisheye lens with the same focal length squeezes a much wider view into the frame.")
        self.run(kt, 0.0, 4.0, rate=smooth)
        self.hold()
        # highlight a few world lines
        top_y = ((-0.95 + 0.95 + 0.74 * 2 + 0.13) - 0.7) * 0.5
        gnd_y = (-0.95 - 0.7) * 0.5
        tt = np.linspace(-6.3, 6.3, 900)
        hl = VMobject().set_stroke(AMBER, 4)

        def hl_upd(m):
            k = kt.get_value()
            segs = world_polyline(tt, np.full_like(tt, top_y), k, C, H) + \
                world_polyline(tt, np.full_like(tt, gnd_y), k, C, H) + \
                world_polyline(np.full_like(tt, 0.825), np.linspace(gnd_y, top_y, 900), k, C, H)
            set_paths(m, segs, C, H)
        hl.add_updater(hl_upd)
        self.add(hl)
        self.caption("The price: straight lines bend, and the bending grows toward the edges.")
        self.run(kt, 1.0, 2.2, rate=smooth)
        self.run(kt, 0.0, 2.2, rate=smooth)
        self.caption("Bent lines break the pinhole assumption that recognition and segmentation models rely on.")
        self.caption("The goal: undo the distortion from a single fisheye image.")
        self.clear_all()


# ================================================================== S02 Projection
def curve_fns():
    return [("f tan θ", PIN, lambda t: np.tan(t), "pinhole"),
            ("f θ", FISH, lambda t: t, "equidistant fisheye"),
            ("2f sin(θ/2)", BLUE, lambda t: 2 * np.sin(t / 2), "equisolid"),
            ("2f tan(θ/2)", PURPLE, lambda t: 2 * np.tan(t / 2), "stereographic"),
            ("f sin θ", "#B0BEC5", lambda t: np.sin(t), "orthographic")]


class S02_Projection(Base):
    def construct(self):
        self.title_card("Part 2", "Pinhole vs fisheye projection")
        self.header("2 · Pinhole vs fisheye projection")
        O = pt(-5.6, -2.15)
        FD = 1.55
        xp = O[0] + FD
        th = ValueTracker(np.radians(25))
        quarter = Arc(radius=FD, start_angle=0, angle=PI / 2, arc_center=O, color=DIM, stroke_width=1.5)
        axis = DashedLine(O, O + RIGHT * 2.3, color=DIM, stroke_width=1.5)
        plane = Line(pt(xp, O[1]), pt(xp, 3.3), color=WHITE, stroke_width=2)
        lplane = tx("image plane", 18, DIM).move_to([xp + 0.2, O[1] - 0.3, 0])
        lf = tx(I("f"), 22, DIM).move_to(O + RIGHT * FD / 2 + DOWN * 0.25)
        cam = Dot(O, radius=0.06, color=WHITE)
        def ray_end():
            t = th.get_value()
            ln = min((FD + 0.45) / np.cos(t), (3.3 - O[1]) / max(np.sin(t), 1e-6))
            return O + ln * pt(np.cos(t), np.sin(t))
        ray = always_redraw(lambda: Line(O, ray_end(), color=WHITE, stroke_width=2, stroke_opacity=0.8))
        ang = always_redraw(lambda: Arc(radius=0.42, start_angle=0, angle=th.get_value(), arc_center=O,
                                        color=WHITE, stroke_width=2))
        angl = always_redraw(lambda: tx(TH, 22).move_to(O + 0.68 * pt(np.cos(th.get_value() / 2), np.sin(th.get_value() / 2))))
        pin_seg = always_redraw(lambda: Line(pt(xp - 0.07, O[1]), pt(xp - 0.07, O[1] + FD * np.tan(th.get_value())),
                                             color=PIN, stroke_width=6))
        pin_dot = always_redraw(lambda: Dot(pt(xp - 0.07, O[1] + FD * np.tan(th.get_value())), color=PIN, radius=0.07))
        fe_arc = always_redraw(lambda: Arc(radius=FD, start_angle=0, angle=th.get_value(), arc_center=O,
                                           color=FISH, stroke_width=6))
        fe_seg = always_redraw(lambda: Line(pt(xp + 0.07, O[1]), pt(xp + 0.07, O[1] + FD * th.get_value()),
                                            color=FISH, stroke_width=6))
        fe_dot = always_redraw(lambda: Dot(pt(xp + 0.07, O[1] + FD * th.get_value()), color=FISH, radius=0.07))
        pin_l = always_redraw(lambda: tx(f"{I('r')} = {I('f')} tan {TH}", 22, PIN).next_to(
            pt(xp - 0.1, min(O[1] + FD * np.tan(th.get_value()), 2.9)), LEFT, buff=0.15))
        fe_l = always_redraw(lambda: tx(f"{I('r')} = {I('f')} {TH}", 22, FISH).next_to(
            pt(xp + 0.1, O[1] + FD * th.get_value()), RIGHT, buff=0.15))
        # r(θ) plot
        ax = Axes(x_range=[0, 90, 15], y_range=[0, 4, 1], x_length=6.4, y_length=4.4,
                  axis_config={"include_numbers": False, "include_tip": False, "color": DIM, "stroke_width": 2}
                  ).move_to([3.35, 0.35, 0])
        ticks = VGroup(*[tx(f"{d}°", 18, DIM).next_to(ax.c2p(d, 0), DOWN, buff=0.12) for d in range(0, 91, 15)],
                       *[tx(str(k), 18, DIM).next_to(ax.c2p(0, k), LEFT, buff=0.12) for k in range(1, 5)])
        xl = tx(f"ray angle {TH}", 20, DIM).next_to(ax.c2p(45, 0), DOWN, buff=0.5)
        yl = tx(f"image radius {I('r')} / {I('f')}", 20, DIM).rotate(PI / 2).next_to(ax.c2p(0, 2), LEFT, buff=0.45)
        fns = curve_fns()
        curves = []
        for k, (nm, c, fn, _) in enumerate(fns):
            xmax = np.degrees(np.arctan(4)) if k == 0 else 90
            curves.append(ax.plot(lambda d, fn=fn: fn(np.radians(d)), x_range=[0, xmax, 0.5], color=c,
                                  stroke_width=4 if k < 2 else 3))
        f_, t_ = I("f"), TH
        leg_txt = [f"pinhole   {f_} tan {t_}", f"equidistant fisheye   {f_} {t_}", f"equisolid   2{f_} sin({t_}/2)",
                   f"stereographic   2{f_} tan({t_}/2)", f"orthographic   {f_} sin {t_}"]
        leg = []
        for k, (nm, c, fn, name) in enumerate(fns):
            sw = Line(ORIGIN, RIGHT * 0.35, color=c, stroke_width=4)
            t = tx(leg_txt[k], 20, c)
            g = VGroup(sw, t).arrange(RIGHT, buff=0.15)
            g.move_to(ax.c2p(3, 3.75 - 0.42 * k), aligned_edge=LEFT)
            leg.append(g)
        v90 = DashedLine(ax.c2p(90, 0), ax.c2p(90, 4), color=DIM, stroke_width=1.5)
        d_pin = always_redraw(lambda: Dot(ax.c2p(np.degrees(th.get_value()), min(np.tan(th.get_value()), 4)), color=PIN,
                                          radius=0.07).set_opacity(1 if np.tan(th.get_value()) <= 4 else 0))
        d_fe = always_redraw(lambda: Dot(ax.c2p(np.degrees(th.get_value()), th.get_value()), color=FISH, radius=0.07))
        self.add(quarter, axis, plane, lplane, lf, cam, ray, ang, angl)
        self.add(ax, ticks, xl, yl, v90)
        self.play(Create(pin_seg), run_time=0.4)
        self.add(pin_dot, pin_l, curves[0], leg[0], d_pin)
        self.caption(f"A pinhole camera puts a ray at angle {TH} at radius {I('r')} = {I('f')} tan {TH} on the image plane.")
        self.run(th, np.radians(62), 3.0, rate=smooth)
        self.hold()
        inf = tx("tan θ → ∞ at 90°", 22, PIN).move_to(ax.c2p(76, 4) + UP * 0.35 + LEFT * 0.6)
        self.caption(f"Near 90°, tan {TH} explodes: no flat pinhole image can hold a 180° view.")
        self.run(th, np.radians(84), 2.5, rate=smooth)
        self.add(inf)
        self.hold()
        self.add(fe_arc, fe_seg, fe_dot, fe_l, curves[1], leg[1], d_fe)
        self.caption(f"An equidistant fisheye unrolls the arc instead: {I('r')} = {I('f')} {TH}, finite all the way to 90°.")
        self.run(th, np.radians(40), 3.0, rate=smooth)
        self.run(th, np.radians(80), 2.0, rate=smooth)
        self.hold()
        self.play(*[Create(curves[k]) for k in (2, 3, 4)], run_time=1.0)
        self.add(*leg[2:])
        self.caption("Other designs follow other curves and real lenses vary; FE-GAN assumes none of these models.")
        self.clear_all()

        # --- the same map applied to a whole plane (manim nonlinear transform)
        C, S = pt(0, 0.05), 1.65
        pin = img(gen("pinhole.png"), 2 * S).move_to(C)
        fish = img(gen("fisheye.png"), 2 * S).move_to(C)
        step = 0.25 * S
        lines = VGroup()
        for c in np.arange(-16 * step, 16 * step + 1e-6, step):
            lines.add(Line(C + pt(c, -9), C + pt(c, 9)).insert_n_curves(90))
            lines.add(Line(C + pt(-9, c), C + pt(9, c)).insert_n_curves(90))
        lines.set_stroke(PIN, 1.3, 0.55)

        def fmap(p):
            d = p - C
            r = np.linalg.norm(d) / S
            return p if r < 1e-9 else C + d * (np.arctan(r) / r)

        def finv(p):
            d = p - C
            r = np.linalg.norm(d) / S
            return p if r < 1e-9 else C + d * (np.tan(min(r, 1.52)) / r)
        circ = DashedCircle(radius=S * np.pi / 2, color=AMBER, stroke_width=2).move_to(C) if "DashedCircle" in globals() \
            else DashedVMobject(Circle(radius=S * np.pi / 2, color=AMBER, stroke_width=2).move_to(C), num_dashes=60)
        circl = tx(f"{TH} = 90°:  {I('r')} = {I('f')} π/2", 22, AMBER).move_to(C + pt(4.3, 1.9))
        self.add(pin, lines)
        self.caption("Apply the fisheye map to the whole image plane: every point slides along its own radius.")
        self.play(ApplyPointwiseFunction(fmap, lines), FadeIn(fish), run_time=4.0, rate_func=smooth)
        self.remove(pin)
        self.play(Create(circ), run_time=0.8)
        self.add(circl)
        self.caption(f"The entire infinite plane lands inside a circle of radius {I('f')} π/2.")
        self.hold()
        self.caption(f"Rectification is the inverse, {I('r')} ↦ {I('f')} tan({I('r')}/{I('f')}); the rim stretches to infinity.")
        self.play(ApplyPointwiseFunction(finv, lines), FadeOut(fish), FadeIn(pin), FadeOut(circ), FadeOut(circl),
                  run_time=3.5, rate_func=smooth)
        self.clear_all()


# ================================================================== S03 Flow
class S03_Flow(Base):
    def construct(self):
        self.title_card("Part 3", "Rectification as a pixel flow")
        self.header("3 · Rectification as a pixel flow")
        fish_a = np.asarray(Image.open(gen("fisheye.png"))).astype(np.float32) / 255
        rect_a = np.asarray(Image.open(gen("rect.png")))
        S = 3.6
        h = S / 2
        cA, cM, cB = pt(-4.4, 0.45), pt(0, 0.45), pt(4.4, 0.45)
        A = img(gen("fisheye.png"), S).move_to(cA)
        fA = framed(A, FISH, 1.5)
        outbox = Square(S, color=PIN, stroke_width=1.5).move_to(cB)
        lA = label(A, f"fisheye image {X_} (toy)", 22, FISH)
        lB = label(outbox, f"rectified {XT}", 22, PIN)
        self.play(FadeIn(A), Create(fA), Create(outbox), run_time=0.7)
        self.add(lA, lB)
        self.caption("Rectifying is resampling: each output pixel fetches its colour from somewhere in the fisheye.")
        pts = [(0.5, 0.42), (0.86, 0.72), (-0.84, -0.66)]
        swatches = VGroup()
        for (u, v) in pts:
            fu, fv = flow_true_uv(np.array(u), np.array(v))
            fu, fv = float(fu), float(fv)
            p_out = cB + pt(u, v) * h
            p_ghost = cA + pt(u, v) * h
            q = cA + pt(u + fu, v + fv) * h
            rr = int(np.clip((1 - (v + fv)) / 2 * NS, 0, NS - 1))
            cc = int(np.clip((u + fu + 1) / 2 * NS, 0, NS - 1))
            colr = '#%02x%02x%02x' % tuple(int(round(255 * c_)) for c_ in fish_a[rr, cc])
            d_out = Dot(p_out, radius=0.07, color=WHITE)
            ghost = Circle(radius=0.08, color=WHITE, stroke_width=2).move_to(p_ghost)
            link = DashedLine(p_out, p_ghost, color=DIM, stroke_width=1.2, dash_length=0.08)
            ar = Arrow(p_ghost, q, buff=0, color=AMBER, stroke_width=4, tip_length=0.14,
                       max_tip_length_to_length_ratio=0.5) if np.hypot(fu, fv) * h > 0.05 else Dot(q, radius=0.03, color=AMBER)
            qd = Dot(q, radius=0.08, color=AMBER)
            sw = Square(0.26, stroke_color=WHITE, stroke_width=1.5, fill_color=colr, fill_opacity=1).move_to(p_out)
            self.play(FadeIn(d_out), Create(link), FadeIn(ghost), run_time=0.45)
            self.play(GrowArrow(ar) if isinstance(ar, Arrow) else FadeIn(ar), FadeIn(qd), run_time=0.55)
            self.play(FadeIn(sw), FadeOut(d_out), run_time=0.35)
            self.play(FadeOut(link), run_time=0.2)
            swatches.add(VGroup(ghost, ar, qd, sw))
        eq = tx(EQ_WARP, 32).move_to([0, -1.95, 0])
        self.add(eq)
        lp = tx(f"{FP} : the flow at pixel {P_}", 22, AMBER).next_to(eq, DOWN, buff=0.18)
        self.add(lp)
        self.caption(f"The flow {FP} says where to look; bilinear interpolation reads the colour there.")
        self.hold()
        # the whole flow field
        fl = img(gen("flow_rgb.png"), S).move_to(cM).set_opacity(0.55)
        field = flow_field(flow_true_uv, 11, cM, S, gain=0.5, color=WHITE, sw=2.0)
        lM = label(fl, f"distortion flow {F_}", 22, AMBER)
        self.play(FadeIn(fl), LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else FadeIn(a) for a in field],
                                           lag_ratio=0.01), run_time=1.6)
        self.add(lM, framed(fl, AMBER, 1.2))
        note = toy_note("toy: exact flow of the toy lens; arrows drawn at half length; colour = direction, brightness = length"
                        ).move_to([0, 2.95, 0])
        self.add(note)
        self.caption(f"Every pixel gets one: the flow field {F_} is an H × W × 2 map of 2D offsets.")
        self.hold()
        # reveal the rectified image row by row
        rv = ValueTracker(0.0)
        out = arr_img(np.zeros((NS, NS, 3), np.uint8), S).move_to(cB)
        rows = np.arange(NS)[:, None, None]

        def reveal():
            k = rv.get_value() * NS
            return np.where(rows < k, rect_a, 0).astype(np.uint8)
        live(out, reveal)
        self.add(out, outbox, lB)
        self.remove(swatches)
        self.caption("Resample the whole image through it, and the lines come out straight.")
        self.run(rv, 1.0, 2.6, rate=linear)
        out.clear_updaters()
        self.hold()
        self.caption("FE-GAN's network predicts this flow directly, instead of the parameters of one lens model.")
        self.clear_all()


# ================================================================== S04 Data
class S04_Data(Base):
    def construct(self):
        self.title_card("Part 4", "Why not just supervise?")
        self.header("4 · Why not just supervise?")
        # left: supervised pairs
        sep = Line(pt(-0.2, 2.9), pt(-0.2, -2.6), color=DIM, stroke_width=1.5)
        hl = tx("prior deep methods: supervised", 26, RED).move_to([-3.75, 2.75, 0])
        a = img(gen("fisheye.png"), 2.0).move_to([-5.35, 0.9, 0])
        b = img(gen("pinhole.png"), 2.0).move_to([-2.15, 0.9, 0])
        lab_a = label(a, "fisheye", 20, FISH)
        lab_b = label(b, "ground truth", 20, PIN)
        link = DoubleArrow(a.get_right(), b.get_left(), buff=0.1, color=WHITE, stroke_width=3, tip_length=0.14)
        ll = tx("paired", 18, DIM).next_to(link, UP, buff=0.08)
        needs = VGroup(*[tx(s, 20, DIM) for s in (
            "each needs one of: distortion-free ground truth,",
            "distortion parameters, line segments,",
            "segmentation maps, ground-truth flow, or image pairs")]).arrange(DOWN, buff=0.1).move_to([-3.75, -0.75, 0])
        cite = tx("(paper refs. [8, 9, 10]: FishEyeRecNet, Xue et al., Li et al.)", 16, DIM).next_to(needs, DOWN, buff=0.18)
        self.add(sep, hl)
        self.play(FadeIn(a), FadeIn(b), GrowFromCenter(link), run_time=0.7)
        self.add(lab_a, lab_b, ll, needs, cite)
        self.caption("Supervised networks need the answer for every training image: its true pinhole view, or labels.")
        self.caption("Shooting every scene through both lenses from the same spot is impractical, and lenses vary.")
        # right: unpaired sets
        hr = tx("FE-GAN: unpaired", 26, GREEN).move_to([3.55, 2.75, 0])
        self.add(hr)
        rng = np.random.default_rng(3)
        xs = ["fig2_fisheye_mc0.png", "fig2_fisheye_wf1.png", "fig2_fisheye_ch0.png", "fig2_fisheye_wf0.png"]
        ys = ["fig1_y.png", gen("yset_0.png"), gen("yset_1.png"), gen("yset_2.png")]
        pos = [(-0.62, 0.55), (0.62, 0.55), (-0.62, -0.62), (0.62, -0.62)]
        Xg, Yg = Group(), Group()
        for k, (f, (dx, dy)) in enumerate(zip(xs, pos)):
            m = img(f, 1.12).move_to([1.95 + dx, 0.95 + dy, 0]).rotate(rng.uniform(-0.12, 0.12))
            Xg.add(m)
        for k, (f, (dx, dy)) in enumerate(zip(ys, pos)):
            m = img(f, 1.12).move_to([5.2 + dx, 0.95 + dy, 0]).rotate(rng.uniform(-0.12, 0.12))
            Yg.add(m)
        bx = RoundedRectangle(width=2.9, height=2.75, corner_radius=0.3, color=FISH, stroke_width=2).move_to([1.95, 0.95, 0])
        by = RoundedRectangle(width=2.9, height=2.75, corner_radius=0.3, color=PIN, stroke_width=2).move_to([5.2, 0.95, 0])
        lx = tx(f"set {X_}: fisheye images", 20, FISH).next_to(bx, DOWN, buff=0.15)
        ly = tx(f"set {Y_}: distortion-free images", 20, PIN).next_to(by, DOWN, buff=0.15)
        self.play(LaggedStart(*[FadeIn(m) for m in Xg], lag_ratio=0.2), Create(bx), run_time=1.0)
        self.add(lx)
        self.play(LaggedStart(*[FadeIn(m) for m in Yg], lag_ratio=0.2), Create(by), run_time=1.0)
        self.add(ly)
        srcn = VGroup(tx(f"{X_}: fisheye inputs from paper Fig. 2", 16, DIM),
                      tx(f"{Y_}: paper Fig. 1 + three toy scenes", 16, DIM)).arrange(DOWN, buff=0.06).move_to([3.55, -1.3, 0])
        self.add(srcn)
        nop = tx("no correspondences, no camera parameters", 22, GREEN).move_to([3.55, -2.05, 0])
        self.add(nop)
        self.caption("FE-GAN learns from two unpaired sets: fisheye images X and unrelated distortion-free images Y.")
        self.clear_all()


# ================================================================== S05 GAN
class S05_GAN(Base):
    def construct(self):
        self.title_card("Part 5", "Fisheye GAN: generator and discriminator")
        self.header("5 · Fisheye GAN (FE-GAN)")
        yr = 1.35
        x_im = img("fig1_x.png", 1.6).move_to([-6.15, yr, 0])
        Ub = hourglass([-3.65, yr, 0], w=1.9, h=1.55, tag="self-attention")
        f_im = img("fig1_flow.png", 1.5).move_to([-1.45, yr, 0])
        Wb = rbox([0.35, yr, 0], 0.95, 0.95, I("W"), 30, fill="#4E3B24", stroke=ORANGE)
        xt_im = img("fig1_xtilde.png", 1.6).move_to([2.2, yr, 0])
        lx = label(x_im, X_, 26, FISH)
        lf = label(f_im, f"{F_} = {U_}({X_})", 22, AMBER)
        lxt = label(xt_im, f"{XT} = {G_}({X_})", 22, PIN)
        a1 = arrow(x_im.get_right(), Ub[1].get_left() + LEFT * 0.02, buff=0.08)
        a2 = arrow(Ub[1].get_right(), f_im.get_left(), buff=0.08)
        a3 = arrow(f_im.get_right(), Wb.get_left(), buff=0.08)
        a4 = arrow(Wb.get_right(), xt_im.get_left(), buff=0.08)
        skip = CurvedArrow(x_im.get_bottom() + DOWN * 0.05, Wb.get_bottom() + DOWN * 0.05, angle=PI / 5, color=FISH,
                           stroke_width=2.5, tip_length=0.14)
        Gbox = DashedVMobject(RoundedRectangle(width=6.1, height=2.9, corner_radius=0.2, stroke_width=1.5, color=DIM)
                              .move_to([-1.6, yr - 0.05, 0]), num_dashes=70)
        Gl = tx(f"generator {G_} = ({U_}, {W_})", 20, DIM).move_to([-1.6, yr + 1.6, 0])
        Wl = tx("warp (STN)", 18, ORANGE).next_to(Wb, UP, buff=0.12)
        self.play(FadeIn(x_im), run_time=0.4)
        self.add(lx)
        self.play(GrowArrow(a1), FadeIn(Ub), run_time=0.6)
        self.play(GrowArrow(a2), FadeIn(f_im), run_time=0.5)
        self.add(lf)
        self.play(GrowArrow(a3), FadeIn(Wb), Create(skip), run_time=0.6)
        self.add(Wl)
        self.play(GrowArrow(a4), FadeIn(xt_im), run_time=0.5)
        self.add(lxt, Gbox, Gl)
        src = tx("images: paper Fig. 1 (the flow panel is the paper's schematic)", 16, DIM).move_to([-1.6, -0.55, 0])
        self.add(src)
        self.caption(f"The generator {G_} = ({U_}, {W_}): {U_} predicts the flow {F_} = {U_}({X_}), {W_} warps {X_} with it.")
        pulse(self, [x_im.get_right(), Ub[1].get_left(), Ub[1].get_right(), f_im.get_left(), f_im.get_right(),
                     Wb.get_left(), Wb.get_right(), xt_im.get_left()], AMBER, 1.6)
        self.caption(f"{U_}: an encoder–decoder with self-attention. {W_}: a spatial transformer, a differentiable resampler.")
        # discriminator
        Dt = trapezoid([4.2, yr, 0], 1.15, 1.5)
        Dl = tx(I("D"), 30).move_to(Dt)
        a5 = arrow(xt_im.get_right(), Dt.get_left(), buff=0.08)
        y_im = img("fig1_y.png", 1.25).move_to([4.2, -0.5, 0])
        ly = VGroup(tx(f"{Y_}: real", 18, PIN), tx("distortion-free", 18, PIN)).arrange(DOWN, buff=0.05).next_to(y_im, RIGHT, buff=0.15).shift(DOWN * 0.3)
        a6 = arrow(y_im.get_top(), Dt.get_bottom(), buff=0.08)
        hA = rbox([6.05, yr + 0.72, 0], 1.05, 0.55, DADV, 22)
        hC = rbox([6.05, yr - 0.72, 0], 1.05, 0.55, DCLS, 22)
        aA = arrow(Dt.get_right() + UP * 0.2, hA.get_left(), buff=0.05, sw=2.5)
        aC = arrow(Dt.get_right() + DOWN * 0.2, hC.get_left(), buff=0.05, sw=2.5)
        lA = tx("real / fake", 18, DIM).next_to(hA, UP, buff=0.1)
        lC = VGroup(tx("distorted /", 18, DIM), tx("distortion-free", 18, DIM)).arrange(DOWN, buff=0.04).next_to(hC, DOWN, buff=0.1)
        self.play(GrowArrow(a5), FadeIn(Dt), FadeIn(Dl), run_time=0.6)
        self.play(FadeIn(y_im), GrowArrow(a6), run_time=0.5)
        self.add(ly)
        self.play(GrowArrow(aA), GrowArrow(aC), FadeIn(hA), FadeIn(hC), run_time=0.6)
        self.add(lA, lC)
        self.caption(f"The discriminator {D_} = ({DADV}, {DCLS}) shares its first layers and has two heads.")
        e1 = tx(EQ_ADV, 26).move_to([0, -1.75, 0])
        n1 = tx("(1)", 20, DIM).move_to([6.6, -1.75, 0])
        self.add(e1, n1)
        self.remove(src)
        self.caption(f"{DADV} tells real distortion-free images {Y_} apart from generated outputs {XT}.")
        e2 = tx(EQ_CLS, 26).move_to([0, -2.45, 0])
        if e2.width > 12.4:
            e2.scale_to_fit_width(12.4)
        e2.set_x(-0.15)
        n2 = tx("(2)", 20, DIM).move_to([6.6, -2.45, 0])
        self.add(e2, n2)
        self.caption(f"{DCLS} labels images distorted or distortion-free; {G_} wins when its outputs pass as distortion-free.")
        self.caption("Adversarial losses judge appearance. What pins down the geometry?")
        self.clear_all()


# ================================================================== S06 Rotation
class S06_Rotation(Base):
    def construct(self):
        self.title_card("Part 6", "Self-supervision I: cross-rotation consistency")
        self.header("6 · Cross-rotation consistency")
        st = read_stats()
        s = 1.9
        TL, TR, BL, BR = pt(-5.35, 1.25), pt(-2.0, 1.25), pt(-5.35, -1.35), pt(-2.0, -1.35)
        x = img(gen("fisheye.png"), s).move_to(TL)
        gx = img(gen("G_rad.png"), s).move_to(TR)
        rx = img(gen("fisheye_rot.png"), s).move_to(BL)
        grx = img(gen("G_rot_rad.png"), s).move_to(BR)
        lx = label(x, X_, 22, FISH)
        lgx = label(gx, f"{G_}({X_})", 22, PIN)
        lrx = label(rx, f"{RT}({X_})", 22, FISH, d=DOWN)
        lgrx = label(grx, f"{G_}({RT}({X_}))  =  {RT}({G_}({X_}))  ?", 20, AMBER, d=DOWN)
        aG1 = arrow(x.get_right(), gx.get_left(), buff=0.12)
        aR1 = arrow(x.get_bottom(), rx.get_top(), buff=0.12)
        aG2 = arrow(rx.get_right(), grx.get_left(), buff=0.12)
        aR2 = arrow(gx.get_bottom(), grx.get_top(), buff=0.12, color=AMBER)
        tG1 = tx(G_, 24).next_to(aG1, UP, buff=0.08)
        tR1 = tx(RT, 24).next_to(aR1, LEFT, buff=0.1)
        tG2 = tx(G_, 24).next_to(aG2, UP, buff=0.08)
        tR2 = tx(RT, 24, AMBER).next_to(aR2, RIGHT, buff=0.1)
        self.play(FadeIn(x), run_time=0.4)
        self.add(lx)
        th_set = tx(f"{TH} ∈ {{90°, 180°, 270°}}", 22, DIM).move_to([3.55, 1.45, 0])
        self.caption(f"Borrowed from GcGAN: rotate the input by {TH} ∈ {{90°, 180°, 270°}}; the output should just rotate.")
        cp = x.copy()
        self.play(cp.animate.move_to(BL).rotate(PI / 2), GrowArrow(aR1), run_time=1.2)
        self.add(rx, tR1, lrx)
        self.remove(cp)
        self.play(GrowArrow(aG1), FadeIn(gx), run_time=0.6)
        self.add(tG1, lgx)
        self.play(GrowArrow(aG2), FadeIn(grx), run_time=0.6)
        self.add(tG2)
        cp2 = gx.copy()
        self.play(cp2.animate.move_to(BR).rotate(PI / 2), GrowArrow(aR2), run_time=1.2)
        self.remove(cp2)
        self.add(tR2, lgrx)
        e1 = tx(EQ_GEO1, 24).move_to([3.55, 2.6, 0])
        e2 = tx(EQ_GEO2, 24).next_to(e1, DOWN, buff=0.15).align_to(e1, LEFT).shift(RIGHT * 0.6)
        n3 = tx("(3)", 18, DIM).next_to(e2, RIGHT, buff=0.25)
        self.add(e1, e2, n3, th_set)
        self.caption("Rectify then rotate, or rotate then rectify: both paths must give the same image (Eq. 3).")
        self.hold()
        # toy check
        dA = img(gen("geo_diff_rad.png"), 1.75).move_to([2.2, -0.45, 0])
        dB = img(gen("geo_diff_xonly.png"), 1.75).move_to([5.0, -0.45, 0])
        self.play(FadeIn(dA), FadeIn(dB), Create(framed(dA, GREEN, 1.5)), Create(framed(dB, RED, 1.5)), run_time=0.5)
        self.add(label(dA, "radial rectifier", 20, GREEN), label(dB, "x-only rectifier", 20, RED),
                 tx(f"mean = {st['geo_rad']:.2f}", 20, GREEN).next_to(dA, DOWN, buff=0.12),
                 tx(f"mean = {st['geo_xonly']:.1f}", 20, RED).next_to(dB, DOWN, buff=0.12),
                 toy_note(f"toy: |{RT}({G_}({X_})) − {G_}({RT}({X_}))| per pixel, {TH} = 90°, 0–255 scale").move_to([3.6, -2.05, 0]))
        self.caption("Toy check: a radial rectifier agrees exactly; one that only fixes horizontal bending does not.")
        self.clear_all()

        # flow invariance (Eq. 4)
        e4 = tx(EQ_FLOWC, 28).move_to([0, 2.6, 0])
        n4 = tx("(4)", 18, DIM).next_to(e4, RIGHT, buff=0.3)
        self.add(e4, n4)
        cL, cR, S = pt(-3.3, 0.05), pt(3.3, 0.05), 3.1
        fr = flow_field(flow_true_uv, 9, cL, S, gain=0.7, color=AMBER, sw=3.0)
        fx = flow_field(flow_xonly_uv, 9, cR, S, gain=0.7, color=RED, sw=3.0)
        ghost_r = fr.copy().set_color(WHITE).set_opacity(0.3)
        ghost_x = fx.copy().set_color(WHITE).set_opacity(0.3)
        bxL = Square(S, color=DIM, stroke_width=1.2).move_to(cL)
        bxR = Square(S, color=DIM, stroke_width=1.2).move_to(cR)
        self.play(Create(bxL), Create(bxR), FadeIn(fr), FadeIn(fx), run_time=0.8)
        self.add(label(bxL, "radial flow (the toy lens)", 22, AMBER), label(bxR, "x-only flow", 22, RED))
        self.add(ghost_r, ghost_x, toy_note("toy flows, arrows ×0.7; the faint copy stays behind as the turn happens").move_to([0, -2.35, 0]))
        self.bring_to_front(fr, fx)
        self.caption(f"Now turn each flow field by 90°.")
        self.play(Rotate(fr, PI / 2, about_point=cL), Rotate(fx, PI / 2, about_point=cR), run_time=2.0)
        self.add(tx("lands on itself ✓", 22, GREEN).next_to(bxL, DOWN, buff=0.15),
                 tx("becomes a y-only field ✗", 22, RED).next_to(bxR, DOWN, buff=0.15))
        self.caption(f"A fisheye flow is radially symmetric, so it is unchanged: {U_}({X_}) and {U_}({RT}({X_})) must match.")
        self.caption(f"The x-only flow changes, so it cannot be right for both {X_} and {RT}({X_}).")
        self.clear_all()


# ================================================================== S07 Radial
class S07_Radial(Base):
    def construct(self):
        self.title_card("Part 7", "Self-supervision II: intra-image warping consistency")
        self.header("7 · Intra-image warping consistency")
        st = read_stats()
        c0, S = pt(-3.7, 0.2), 3.5
        h = S / 2
        box = Square(S, color=DIM, stroke_width=1.2).move_to(c0)
        field = flow_field(flow_true_uv, 11, c0, S, gain=0.5, color=WHITE, sw=1.8).set_opacity(0.5)
        circs = VGroup(*[Circle(radius=r * h, color=DIM, stroke_width=1.2).move_to(c0) for r in (0.35, 0.7)])
        hi = Circle(radius=0.7 * h, color=AMBER, stroke_width=2.5).move_to(c0)
        ring = VGroup()
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            u, v = 0.7 * np.cos(a), 0.7 * np.sin(a)
            fu, fv = flow_true_uv(np.array(u), np.array(v))
            p0 = c0 + pt(u, v) * h
            ring.add(Arrow(p0, p0 + pt(float(fu), float(fv)) * h * 1.6, buff=0, color=AMBER, stroke_width=3.5,
                           tip_length=0.09, max_tip_length_to_length_ratio=0.5))
        cdot = Dot(c0, radius=0.05, color=WHITE)
        self.play(Create(box), FadeIn(field), Create(circs), run_time=0.8)
        self.add(cdot, label(box, "toy flow of the fisheye lens", 22, DIM))
        self.add(toy_note("amber: flow on one circle, drawn ×0.8; grey: ×0.5").next_to(box, DOWN, buff=0.12))
        self.caption("If the optical axis passes through the image centre, fisheye flow has two simple properties.")
        self.play(Create(hi), LaggedStart(*[GrowArrow(a) for a in ring], lag_ratio=0.05), run_time=1.2)
        x0 = 0.1
        rows = [
            (tx("Radial-symmetric", 26, AMBER), 2.45),
            (tx("points straight toward or away from the centre,", 20, DIM), 2.0),
            (tx("with one length on each circle", 20, DIM), 1.66),
            (tx(f"direction:  {EQ_DIR}  → 1", 24), 1.1),
            (tx(f"circles:  {EQ_VAR}  → 0", 24), 0.5),
        ]
        for m, y in rows:
            m.move_to([0, y, 0]).align_to(pt(x0, 0), LEFT)
            self.add(m)
        n6 = tx("(6)", 18, DIM).move_to([6.7, 0.8, 0])
        self.add(n6)
        self.caption("Radially symmetric: each arrow lies along its radius, and arrows on one circle share a length.")
        self.hold()
        rows2 = [
            (tx("Smooth", 26, AMBER), -0.2),
            (tx("neighbouring pixels have similar flow", 20, DIM), -0.62),
            (tx(EQ_SMOOTH, 24), -1.2),
        ]
        for m, y in rows2:
            m.move_to([0, y, 0]).align_to(pt(x0, 0), LEFT)
            self.add(m)
        n5 = tx("(5)", 18, DIM).move_to([6.7, -1.2, 0])
        self.add(n5, tx("assumption: optical axis through the image centre", 18, DIM).move_to([0, -1.85, 0]).align_to(pt(x0, 0), LEFT))
        self.caption("And smooth. The paper turns both into losses on the predicted flow (Eqs. 5 and 6).")
        self.clear_all()

        # three candidate flows
        xs = [-4.5, 0, 4.5]
        names = [("noisy flow", "noisy"), ("swirl (tangential)", "swirl"), ("radial (true)", "radial")]
        u, v = grid(NS)
        fu, fv = flow_true_uv(u, v)
        rng = np.random.default_rng(5)
        fns = {"noisy": lambda a, b: (flow_true_uv(a, b)[0] + rng.normal(0, 0.02, np.shape(a)),
                                      flow_true_uv(a, b)[1] + rng.normal(0, 0.02, np.shape(b))),
               "swirl": lambda a, b: (-flow_true_uv(a, b)[1], flow_true_uv(a, b)[0]),
               "radial": flow_true_uv}
        good = {"noisy": (False, None, False), "swirl": (True, False, True), "radial": (True, True, True)}
        for x, (nm, key) in zip(xs, names):
            c = pt(x, 1.5)
            bx = Square(2.0, color=DIM, stroke_width=1.2).move_to(c)
            ff = flow_field(fns[key], 9, c, 2.0, gain=0.5, color=AMBER, sw=1.8)
            out = img(gen(f"cand_{key}.png"), 2.0).move_to([x, -0.62, 0])
            self.play(Create(bx), FadeIn(ff), FadeIn(out), run_time=0.5)
            self.add(label(bx, nm, 22, GREEN if key == "radial" else WHITE))
            sm, al, var = st[f"loss_{key}"]
            gs = good[key]
            colr = lambda ok: GREEN if ok else (AMBER if ok is None else RED)
            vals = VGroup(tx(f"smoothness  {sm:.2f}", 19, colr(gs[0])),
                          tx(f"direction |cos|  {al:.2f}", 19, colr(gs[1])),
                          tx(f"circle variance  {var:.2f}", 19, colr(gs[2]))).arrange(DOWN, buff=0.07).move_to([x, -2.2, 0])
            self.add(vals)
        self.add(toy_note("toy: top = flow (×0.5), middle = the fisheye warped by it; loss terms in pixels of a 512 × 512 image"
                          ).move_to([0, -2.85, 0]))
        self.caption("Toy check with three candidate flows: noise breaks smoothness, a swirl breaks the direction term.")
        self.caption("Only the smooth, radial flow passes both tests, and it rectifies the image.")
        self.clear_all()

        # strength family: the constraints fix the shape, the adversarial losses fix the strength
        fam = np.load(gen("family2.npy"))
        err = np.load(gen("family2_err.npy"))
        sv = ValueTracker(0.0)
        ssx = np.linspace(0, S_MAX2, NSTR2)
        out = arr_img(fam[0], 3.4).move_to([-3.7, 0.35, 0])
        live(out, lambda: interp(fam, sv.get_value() / S_MAX2 * (NSTR2 - 1)))
        lo = always_redraw(lambda: tx(f"flow = {sv.get_value():.2f} × true flow", 24, AMBER).next_to(out, DOWN, buff=0.18))
        ymax = float(np.ceil(err.max() / 10) * 10)
        ax = Axes(x_range=[0, S_MAX2, 0.25], y_range=[0, ymax, 10], x_length=5.6, y_length=3.6,
                  axis_config={"include_numbers": False, "include_tip": False, "color": DIM}).move_to([3.3, 0.45, 0])
        ticks = VGroup(*[tx(f"{t:.1f}", 18, DIM).next_to(ax.c2p(t, 0), DOWN, buff=0.1) for t in (0, 0.5, 1.0, 1.5)],
                       *[tx(str(int(k)), 18, DIM).next_to(ax.c2p(0, k), LEFT, buff=0.1) for k in np.arange(10, ymax + 1, 10)])
        xl = tx("flow strength (× true flow)", 20, DIM).next_to(ax, DOWN, buff=0.45)
        yl = tx("error vs the true pinhole view", 20, DIM).next_to(ax, UP, buff=0.15)
        curve = VMobject(color=AMBER, stroke_width=3).set_points_as_corners([ax.c2p(a_, b_) for a_, b_ in zip(ssx, err)])
        dot = always_redraw(lambda: Dot(ax.c2p(sv.get_value(), float(np.interp(sv.get_value(), ssx, err))),
                                        color=WHITE, radius=0.08))
        self.play(FadeIn(out), run_time=0.5)
        self.add(lo, ax, ticks, xl, yl)
        self.add(toy_note("toy: RMSE (0–255) after a 3-pixel blur, against the pinhole view; known only because the scene is synthetic"
                          ).move_to([0, -2.5, 0]))
        self.caption("But every radial flow passes these tests, even zero flow. Scale the true flow and watch:")
        self.play(Create(curve), run_time=0.1)
        self.add(dot)
        self.run(sv, S_MAX2, 4.5, rate=linear)
        self.caption("Too weak stays bent, too strong over-corrects. The adversarial losses pick the strength.")
        self.run(sv, 1.0, 2.0, rate=smooth)
        self.clear_all()

        # the full objective
        items = [("(1)", f"ℒ{sub('adv')}", "outputs look like real distortion-free images"),
                 ("(2)", f"ℒ{sub('cls')}", "outputs are classified as distortion-free"),
                 ("(3)", f"ℒ{sub('geo')}", "rotate then rectify = rectify then rotate"),
                 ("(4)", f"ℒ{sub('flow')}", "the same flow for rotated inputs"),
                 ("(5)", f"ℒ{sub('smooth')}", "the flow is smooth"),
                 ("(6)", f"ℒ{sub('radial')}", "the flow is radial, one length per circle")]
        g = VGroup()
        for k, (n, l, d) in enumerate(items):
            y = 2.45 - 0.55 * k
            c = [AMBER, AMBER, CYAN, CYAN, GREEN, GREEN][k]
            g.add(tx(n, 22, DIM).move_to([-5.4, y, 0]), tx(l, 28, c).move_to([-4.1, y, 0]),
                  tx(d, 24).move_to([0, y, 0]).align_to(pt(-3.0, 0), LEFT))
        tags = VGroup(tx("adversarial", 18, AMBER).move_to([5.3, 2.17, 0]),
                      tx("cross-rotation", 18, CYAN).move_to([5.3, 1.07, 0]),
                      tx("intra-image warping", 18, GREEN).move_to([5.3, -0.03, 0]))
        for m in g:
            self.add(m)
        self.add(tags)
        full = tx(EQ_FULL, 28).move_to([0, -1.25, 0])
        self.add(full)
        self.caption("Six terms, no camera parameters, no image pairs: the supervision comes from geometry.")
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results")
        self.header("8 · Results")
        # datasets
        ex = [("MCindoor20000", "mc0"), ("Wireframe", "wf1"), ("LSUN Churches", "ch0")]
        for x, (nm, k) in zip([-4.4, 0, 4.4], ex):
            m = img(f"fig2_fisheye_{k}.png", 2.6).move_to([x, 0.8, 0])
            self.add(m, label(m, nm, 24))
        self.add(tx("fisheye transform of Basu &amp; Licardie (1995), random parameters: 1000 fisheye images per dataset",
                    22).move_to([0, -1.1, 0]),
                 tx("ground-truth images are used only to score the outputs: PSNR, SSIM, RMSE", 22, DIM).move_to([0, -1.6, 0]),
                 tx("fisheye inputs from paper Fig. 2", 16, DIM).move_to([0, -2.1, 0]))
        self.caption("Three datasets (indoor, wireframe and church scenes), each turned into 1000 synthetic fisheye images.")
        self.clear_all()
        # Fig. 2 comparisons
        cols = [("fisheye", "fisheye input", FISH), ("cyclegan", "CycleGAN", RED), ("unit", "UNIT", RED),
                ("aleman", "Aleman-Flores", RED), ("ours", "FE-GAN (ours)", GREEN)]
        xs = [-4.9, -2.45, 0, 2.45, 4.9]
        for page, keys in enumerate((["mc0", "wf1", "ch0"], ["mc2", "wf2", "ch2"])):
            grp = Group()
            for r, (k, y) in enumerate(zip(keys, [1.75, -0.1, -1.95])):
                for x, (ck, nm, c) in zip(xs, cols):
                    m = img(f"fig2_{ck}_{k}.png", 1.8).move_to([x, y, 0])
                    grp.add(m)
                    if r == 0:
                        grp.add(label(m, nm, 22, c))
            for y, nm in zip([1.75, -0.1, -1.95], ["MCindoor", "Wireframe", "LSUN Churches"]):
                grp.add(tx(nm, 16, DIM).rotate(PI / 2).move_to([-6.2, y, 0]))
            grp.add(tx("paper Fig. 2", 16, DIM).move_to([6.3, 3.62, 0]))
            self.play(FadeIn(grp), run_time=0.6)
            if page == 0:
                self.caption("CycleGAN and UNIT change colour and texture, not geometry; UNIT even hallucinates content.")
                self.caption("Aleman-Flores fits one distortion model and under-corrects. FE-GAN straightens the lines.")
            else:
                self.caption("More examples: the same pattern on all three datasets.")
            self.clear_all()
        # website teaser (GIF, 2 frames × 800 ms)
        f0 = np.asarray(Image.open(os.path.join(DATA, "gif_frames/frame_0.png")).convert("RGB"))
        f1 = np.asarray(Image.open(os.path.join(DATA, "gif_frames/frame_1.png")).convert("RGB"))
        t0 = self.renderer.time
        g = arr_img(f0, 4.4).move_to([0, 0.4, 0])
        state = lambda: int(((self.renderer.time - t0) / 0.8)) % 2
        live(g, lambda: f1 if state() else f0)
        lg = always_redraw(lambda: label(g, "rectified by FE-GAN" if state() else "fisheye input", 24,
                                         GREEN if state() else FISH, d=DOWN))
        self.add(g, lg, tx("project-page teaser (bchao1.github.io)", 16, DIM).move_to([0, 2.95, 0]))
        self.caption("The project-page teaser: the same church, before and after FE-GAN.")
        self.wait(0.8)
        self.clear_all()
        # Table 1
        rows = [("", "MCindoor", "Wireframe", "LSUN Churches"),
                ("CycleGAN", "11.404 / 0.335 / 71.902", "11.505 / 0.257 / 68.549", "10.982 / 0.193 / 72.683"),
                ("UNIT", "12.959 / 0.44 / 61.387", "8.965 / 0.215 / 93.236", "10.336 / 0.235 / 79.812"),
                ("FE-GAN (ours)", "15.215 / 0.498 / 45.672", "12.846 / 0.352 / 59.887", "12.781 / 0.377 / 59.492")]
        xs = [-5.3, -2.0, 1.55, 5.1]
        for r, row in enumerate(rows):
            cl = DIM if r == 0 else (GREEN if r == 3 else WHITE)
            for s, x in zip(row, xs):
                if s:
                    self.add(tx(s, 22, cl).move_to([x, 2.3 - 0.6 * r, 0]))
        self.add(Line([-6.6, 2.0, 0], [6.6, 2.0, 0], color=DIM, stroke_width=1.5),
                 tx("paper Table 1: PSNR ↑ / SSIM ↑ / RMSE ↓ against the ground-truth distortion-free images", 18, DIM
                    ).move_to([0, 0.05, 0]))
        self.caption("Paper Table 1: FE-GAN beats both unpaired translators on every metric, on every dataset.")
        self.hold()
        gains = ["+2.3 dB", "+1.3 dB", "+1.8 dB"]
        self.add(tx("PSNR gain", 22, AMBER).move_to([xs[0], -0.72, 0]),
                 tx("over best baseline", 18, AMBER).move_to([xs[0], -1.05, 0]))
        for x, gtxt, ref in zip(xs[1:], gains, ["vs 12.959 (UNIT)", "vs 11.505 (CycleGAN)", "vs 10.982 (CycleGAN)"]):
            self.add(tx(gtxt, 34, AMBER).move_to([x, -0.72, 0]), tx(ref, 16, DIM).move_to([x, -1.12, 0]))
        self.caption("1.3 to 2.3 dB more PSNR than the best baseline, without a single paired training image.")
        self.clear_all()
        # Table 2 ablation
        ab = [("w/o self-attention", 12.388, 0.361, 62.184, RED),
              ("w/o cross-rotation consistency", 11.021, 0.324, 72.889, RED),
              ("w/o intra-image warping", 12.38, 0.303, 62.319, RED),
              ("full FE-GAN", 15.215, 0.498, 45.672, GREEN)]
        x0, sc = -2.1, 0.40
        bars = []
        for k, (nm, ps, ss_, rm, c) in enumerate(ab):
            y = 1.9 - 0.95 * k
            bar = Rectangle(width=ps * sc, height=0.5, fill_color=c, fill_opacity=0.85, stroke_width=0)
            bar.move_to([x0 + ps * sc / 2, y, 0])
            bars.append(bar)
            self.add(tx(nm, 22).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT))
        self.play(*[GrowFromEdge(b, LEFT) for b in bars], run_time=0.8)
        for k, (nm, ps, ss_, rm, c) in enumerate(ab):
            y = 1.9 - 0.95 * k
            self.add(tx(f"{ps:g} dB", 22, c).next_to(bars[k], RIGHT, buff=0.12),
                     tx(f"SSIM {ss_:g} · RMSE {rm:g}", 16, DIM).next_to(bars[k], RIGHT, buff=0.12).shift(DOWN * 0.36))
        self.add(tx("paper Table 2: ablation on MCindoor, PSNR (bars)", 18, DIM).move_to([0, -2.1, 0]))
        self.caption("Ablation: every part helps. Dropping cross-rotation consistency costs the most, 4.2 dB.")
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Takeaways")
        self.header("9 · Takeaways")
        summ = [
            "1   Among the first self-supervised deep models for fisheye rectification",
            "2   Predicts a pixel flow, not the parameters of one lens model",
            "3   Rotation and radial-flow consistency replace paired supervision",
        ]
        sg = VGroup(*[tx(s, 28) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 1.5, 0])
        for m in sg:
            self.add(m)
            self.wait(0.9)
        res = tx("best PSNR, SSIM and RMSE on all three datasets vs CycleGAN and UNIT", 24, AMBER).move_to([0, -0.15, 0])
        asm_ = tx("keep in mind: assumes the optical axis through the image centre; evaluated on synthetic fisheye images",
                  20, DIM).move_to([0, -0.8, 0])
        self.add(res, asm_)
        self.caption("Straight lines back, learned from unpaired photos and a little geometry.")
        self.clear_all()
        self.remove(self.hdr)
        t1 = tx("Self-Supervised Deep Learning", 48, bold=True).move_to([0, 2.2, 0])
        t1b = tx("for Fisheye Image Rectification", 36).move_to([0, 1.5, 0])
        t2 = tx("IEEE ICASSP 2020  ·  pp. 2248–2252", 26).move_to([0, 0.6, 0])
        t3 = tx("code: github.com/ByronHsu/FEGAN", 22, DIM).move_to([0, 0.1, 0])
        t4 = tx("Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang", 22, DIM).move_to([0, -0.7, 0])
        t5 = tx("National Taiwan University  ·  ASUS Intelligent Cloud Services", 22, DIM).move_to([0, -1.15, 0])
        t6 = tx("* equal contribution", 18, DIM).move_to([0, -1.55, 0])
        self.add(t1, t1b, t2, t3, t4, t5, t6)
        self.wait(4.0)
