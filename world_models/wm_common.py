"""Shared pieces for the 10 s World Models clip styles (styles/animation_wm_sNN_*.py).

Everything here is style-agnostic: paper/segment data, a playing-video mobject, control icons + chips, 3b1b-style
concept glyphs, captions, figure stacks, and a scene base class that keeps an absolute timeline.
"""
import glob
import os

import numpy as np
from PIL import Image
from manim import *

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(ROOT, "data", "clips")
CROPS = os.path.join(ROOT, "data", "figcrops")

# ---------------------------------------------------------------- palette / type
FONT = "Lato"
BG = "#0E1016"
FG = "#ECEFF4"
MUTED = "#8A93A3"
DIM = "#3A404C"
DIM_TXT = "#5F6776"
C_CAM = "#58C4DD"   # 3b1b blue
C_HAND = "#F0AC5F"  # 3b1b gold
C_TIME = "#5CD0B3"  # 3b1b teal
C_MEM = "#B79CF6"   # soft purple
FPS_SRC = 30        # clip frames were extracted at 30 fps
FW, FH = 14.222, 8.0
BOUNDS = [0.0, 2.5, 5.0, 7.5, 10.0]  # paper segment boundaries (s)

SEGS = [
    dict(key="camctrl2", speed=1.30, color=C_CAM, control="Camera", verb="steer it",
         lines=("Camera-controlled", "World Generation"), title="Camera-controlled World Generation", venue="ICLR 2025, ICCV 2025", paper="CameraCtrl  /  CameraCtrl II",
         figs=[("cc1_row", "CameraCtrl  ·  Fig. 1"), ("cc2_rows", "CameraCtrl II  ·  Fig. 1")]),
    dict(key="genreal", speed=0.82, color=C_HAND, control="Hands", verb="touch it",
         lines=("Interactive World Generation", "with Hand-object Interaction"), title="Interactive World Generation with Hand-object Interaction", venue="CVPR 2026",
         paper="Generated Reality", figs=[("gr_inputs", "Generated Reality  ·  Fig. 1")]),
    dict(key="bullettime", speed=1.0, color=C_TIME, control="Time", verb="freeze time",
         lines=("Space-time Controlled", "World Generation"), title="Space-time Controlled World Generation", venue="CVPR 2026", paper="BulletTime",
         figs=[("bt_fig1", "BulletTime  ·  Fig. 1"), ("bt_bullet", "BulletTime  ·  Fig. 5")]),
    dict(key="spmem", speed=1.0, color=C_MEM, control="Memory", verb="remember it",
         lines=("World Models", "with Spatial Memory"), title="World Models with Spatial Memory", venue="NeurIPS 2025", paper="Spatial Memory",
         figs=[("sm_teaser", "Spatial Memory  ·  Fig. 1")]),
]


def T(s, size=24, color=FG, weight=NORMAL, font=FONT, **kw):
    return Text(s, font=font, font_size=size, color=color, weight=weight, **kw)


def px_per_unit():
    return config.pixel_width / config.frame_width


# ---------------------------------------------------------------- playing video
class VideoCard:
    """A demo clip as an ImageMobject whose pixels advance with scene time.

    Position / size / opacity / darkening are ValueTrackers (x, y, w, alpha, dim) applied by an updater, so the clip
    keeps playing while it moves or fades (manim's FadeIn on an ImageMobject would freeze the pixel array).
    fit: box (w, h) in units; the clip is contained in it. Frames are resized once to the largest width they will
    be shown at (max_w), then manim scales the mobject.
    """

    def __init__(self, key, speed=1.0, box=(9.4, 5.3), center=ORIGIN, radius=0.13, max_w=None, loop=False,
                 border=MUTED, border_w=1.4, border_op=0.55, cover=False):
        self.paths = sorted(glob.glob(os.path.join(CLIPS, key, "*.jpg")))
        self.speed, self.loop, self.cache, self.clock, self.playing, self.last = speed, loop, {}, 0.0, False, None
        W0, H0 = Image.open(self.paths[0]).size
        aspect = W0 / H0
        if cover:  # fill the box, cropping overflow (crop is done in pixel space)
            self.w_u, self.h_u = box
        else:
            s = min(box[0] / W0, box[1] / H0)
            self.w_u, self.h_u = W0 * s, H0 * s
        self.cover_aspect = self.w_u / self.h_u if cover else None
        mw = max_w or self.w_u
        self.pw = int(round(min(mw * px_per_unit(), 1920 if not cover else 1e9)))
        self.pw = max(self.pw, 16)
        self.ph = int(round(self.pw / (self.w_u / self.h_u)))
        self.src_aspect = aspect
        # rounded-corner alpha mask
        r = radius * self.pw / self.w_u
        yy, xx = np.mgrid[0:self.ph, 0:self.pw]
        if r > 0.5:
            dx = np.maximum(np.maximum(r - xx, xx - (self.pw - 1 - r)), 0)
            dy = np.maximum(np.maximum(r - yy, yy - (self.ph - 1 - r)), 0)
            self.mask = np.clip(r - np.sqrt(dx ** 2 + dy ** 2) + 0.5, 0, 1)
        else:
            self.mask = np.ones((self.ph, self.pw))
        c = np.array(center, dtype=float)
        self.x, self.y = ValueTracker(c[0]), ValueTracker(c[1])
        self.w, self.alpha, self.dim = ValueTracker(self.w_u), ValueTracker(0.0), ValueTracker(0.0)
        self.img = ImageMobject(np.zeros((self.ph, self.pw, 4), dtype=np.uint8))
        self.img.set_width(self.w_u)
        self.border_op = border_op
        self.border = RoundedRectangle(width=self.w_u, height=self.h_u, corner_radius=max(radius, 1e-3))
        self.border.set_stroke(border, border_w, opacity=0).set_fill(opacity=0)
        if border is None or border_w == 0:
            self.border.set_stroke(width=0)
        self.radius = radius
        # videos always render beneath overlays (captions, bands, figures), whatever order they are added in
        self.img.set_z_index(-2)
        self.border.set_z_index(-1)
        self.img.add_updater(self._tick)
        self.border.add_updater(self._tick_border)
        self._tick(self.img, 0)

    # frame i as RGBA uint8 at (ph, pw)
    def frame(self, i):
        if i not in self.cache:
            im = Image.open(self.paths[i]).convert("RGB")
            if self.cover_aspect is not None:
                W0, H0 = im.size
                if W0 / H0 > self.cover_aspect:
                    nw = int(H0 * self.cover_aspect)
                    im = im.crop(((W0 - nw) // 2, 0, (W0 - nw) // 2 + nw, H0))
                else:
                    nh = int(W0 / self.cover_aspect)
                    im = im.crop((0, (H0 - nh) // 2, W0, (H0 - nh) // 2 + nh))
            im = im.resize((self.pw, self.ph), Image.LANCZOS)
            self.cache[i] = np.array(im)
        return self.cache[i]

    def index(self):
        n = len(self.paths)
        i = int(self.clock * FPS_SRC)
        return i % n if self.loop else min(i, n - 1)

    def pos(self):
        return np.array([self.x.get_value(), self.y.get_value(), 0])

    def _tick(self, m, dt):
        if self.playing:
            self.clock += dt * self.speed
        i = self.index()
        a, d = round(self.alpha.get_value(), 3), round(self.dim.get_value(), 3)
        if (i, a, d) != self.last:
            rgb = self.frame(i) if d == 0 else (self.frame(i) * (1 - d)).astype(np.uint8)
            arr = np.dstack([rgb, (255 * self.mask * a).astype(np.uint8)])
            m.pixel_array = arr
            self.last = (i, a, d)
        w = self.w.get_value()
        if abs(m.width - w) > 1e-4:
            m.set_width(w)
        m.move_to(self.pos())

    def _tick_border(self, m):
        k = self.w.get_value() / self.w_u
        if abs(m.width - self.w_u * k) > 1e-4:
            m.set_width(self.w_u * k)
        m.move_to(self.pos()).set_stroke(opacity=self.border_op * self.alpha.get_value())

    def height_now(self):
        return self.h_u * self.w.get_value() / self.w_u

    def mobs(self):
        return [self.img, self.border]

    def start(self, scene, alpha=None):
        self.playing = True
        if alpha is not None:
            self.alpha.set_value(alpha)
        scene.add(*self.mobs())
        return self


# ---------------------------------------------------------------- icons (small, stroke-only)
def icon_camera(s=1.0):
    body = RoundedRectangle(width=0.27, height=0.19, corner_radius=0.04)
    lens = Polygon([0.135, 0.035, 0], [0.24, 0.085, 0], [0.24, -0.085, 0], [0.135, -0.035, 0])
    dot = Circle(radius=0.035).move_to(body)
    return VGroup(body, lens, dot).center().scale(s)


def icon_hand(s=1.0):
    palm = RoundedRectangle(width=0.2, height=0.17, corner_radius=0.06)
    top = palm.get_top()[1] - 0.01
    fingers = VGroup(*[Line([x, top, 0], [x, top + L, 0]) for x, L in
                       zip([-0.075, -0.025, 0.025, 0.075], [0.11, 0.14, 0.13, 0.1])])
    thumb = Line(palm.get_left() + UP * 0.02, palm.get_left() + np.array([-0.08, 0.09, 0]))
    return VGroup(palm, fingers, thumb).center().scale(s)


def icon_clock(s=1.0):
    return VGroup(Circle(radius=0.15), Line(ORIGIN, UP * 0.11), Line(ORIGIN, RIGHT * 0.075)).scale(s, about_point=ORIGIN)


def icon_memory(s=1.0):
    dots = VGroup()
    for r, (y, sx) in enumerate([(-0.1, 0.1), (0.0, 0.075), (0.09, 0.055)]):
        for c in range(4):
            dots.add(Dot([(-1.5 + c) * sx, y, 0], radius=0.026 - 0.004 * r))
    return dots.center().scale(s)


ICONS = [icon_camera, icon_hand, icon_clock, icon_memory]


def styled_icon(k, color, t=0.0, live=True, s=1.0, width=2.6, opacity=1.0):
    """Icon k, animated by local time t when live (camera pans, hand waves, clock spins then freezes, cloud fills)."""
    fn = ICONS[k]
    base = fn(s)
    if k == 0 and live:
        base.rotate(0.35 * np.sin(2 * PI * t / 1.3))
    elif k == 1 and live:
        base.rotate(0.3 * np.sin(2 * PI * t / 0.7), about_point=base.get_bottom())
    elif k == 2 and live:
        ang = -2 * PI * 1.6 * min(t, 1.0)
        base[1].rotate(ang, about_point=base[0].get_center())
        base[2].rotate(ang / 12, about_point=base[0].get_center())
    if k == 3:
        n = int(min(t / 0.13, len(base))) if live else len(base)
        for j, d in enumerate(base):
            d.set_fill(color, opacity=(1.0 if j < max(n, 1) else 0.18) * opacity).set_stroke(width=0)
    else:
        base.set_stroke(color, width * s ** 0.5, opacity=opacity).set_fill(opacity=0)
        if k == 0:
            base[2].set_fill(color, opacity=opacity).set_stroke(width=0)
    return base


class LiveIcon(VMobject):
    """An icon that animates itself while self.live is True."""

    def __init__(self, k, color, s=1.0, width=2.6, live=False):
        super().__init__()
        self.k, self.color_, self.s, self.width_, self.live, self.t, self.op = k, color, s, width, live, 0.0, 1.0
        self.add(styled_icon(k, color, 0, False, s, width))
        self.add_updater(self._up)

    def _up(self, m, dt):
        self.t += dt
        c = self.get_center()
        new = styled_icon(self.k, self.color_, self.t, self.live, self.s, self.width_, self.op).move_to(c)
        m.submobjects = [new]

    def activate(self):
        self.live, self.t = True, 0.0
        return self


class Chip(VGroup):
    """Rounded control chip: icon + label. States: off / on / done / all."""

    def __init__(self, k, label=None, size=22, height=0.52, off_color=DIM, off_txt=DIM_TXT, on_txt=FG, done_txt="#B4BBC7",
                 fill_op=0.16):
        super().__init__()
        seg = SEGS[k]
        self.k, self.color_ = k, seg["color"]
        self.off_color, self.off_txt, self.on_txt, self.done_txt, self.fill_op = off_color, off_txt, on_txt, done_txt, fill_op
        self.icon = LiveIcon(k, off_txt)
        self.label = T(label or seg["control"], size)
        inner = VGroup(self.icon, self.label).arrange(RIGHT, buff=0.14)
        self.rect = RoundedRectangle(width=inner.width + 0.36, height=height, corner_radius=height * 0.27).move_to(inner)
        self.add(self.rect, self.icon, self.label)
        self.style("off", instant=True)

    def _c(self, state):
        if state in ("on", "all"):
            return self.color_, 1.0, self.fill_op if state == "on" else self.fill_op * 0.7, self.color_, self.on_txt
        if state == "done":
            return self.color_, 0.5, 0.0, self.color_, self.done_txt
        return self.off_color, 1.0, 0.0, self.off_txt, self.off_txt

    def style(self, state, instant=False):
        stroke, s_op, f_op, icon_c, lab_c = self._c(state)
        self.icon.color_ = icon_c
        self.icon.op = 0.65 if state == "done" else 1.0
        if state == "on":
            self.icon.activate()
        elif state == "all":
            self.icon.live = True
        else:
            self.icon.live = False
        if instant:
            self.rect.set_stroke(stroke, 1.8, opacity=s_op).set_fill(stroke, f_op)
            self.label.set_color(lab_c)
            return []
        return [self.rect.animate.set_stroke(stroke, 1.8, opacity=s_op).set_fill(stroke, f_op),
                self.label.animate.set_color(lab_c)]


# ---------------------------------------------------------------- larger concept glyphs (3b1b flourishes)
def glyph(k, size=2.0, color=None, prog=None, center=ORIGIN):
    """Big animated concept glyph for paper k driven by ValueTracker prog in [0, 1]. Returns (mobject, prog)."""
    color = color or SEGS[k]["color"]
    prog = prog or ValueTracker(0.0)
    s = size / 2.0
    if k == 0:  # camera frustum flying along a path over a ground grid
        grid = VGroup(*[Line([-1, y, 0], [1, y, 0]) for y in np.linspace(-0.8, 0.8, 5)],
                      *[Line([x, -0.8, 0], [x, 0.8, 0]) for x in np.linspace(-1, 1, 6)])
        grid.set_stroke(MUTED, 1, opacity=0.35).apply_function(lambda p: np.array([p[0] * (1 - 0.25 * p[1]), p[1] * 0.6, 0]))
        path = VMobject().set_points_smoothly([np.array(p) for p in [(-0.8, -0.45, 0), (-0.3, -0.2, 0), (0.1, 0.05, 0),
                                                                       (0.55, 0.3, 0), (0.8, 0.2, 0)]])
        path.set_stroke(color, 4)

        def draw():
            a = max(prog.get_value(), 1e-3)
            p = path.point_from_proportion(a)
            q = path.point_from_proportion(min(a + 0.02, 1)) - path.point_from_proportion(max(a - 0.02, 0))
            ang = np.arctan2(q[1], q[0])
            fr = Polygon(ORIGIN, [0.32, 0.17, 0], [0.32, -0.17, 0]).set_stroke(FG, 2.5).set_fill(color, 0.45)
            fr.rotate(ang, about_point=ORIGIN).shift(p)
            trail = path.copy().pointwise_become_partial(path, 0, a)
            ghosts = VGroup(*[Polygon(ORIGIN, [0.2, 0.1, 0], [0.2, -0.1, 0]).set_stroke(color, 1.5, opacity=0.5)
                              .rotate(np.arctan2(*(path.point_from_proportion(min(b + .02, 1)) -
                                                   path.point_from_proportion(max(b - .02, 0)))[1::-1]), about_point=ORIGIN)
                              .shift(path.point_from_proportion(b)) for b in np.arange(0.0, a, 0.2)])
            return VGroup(grid.copy(), trail, ghosts, fr).scale(s, about_point=ORIGIN)
    elif k == 1:  # hand skeleton reaching and closing on an object
        def draw():
            a = prog.get_value()
            close = 0.5 - 0.5 * np.cos(2 * PI * min(a * 1.5, 1.0))
            wrist = np.array([-0.55, -0.55, 0]) + np.array([0.25, 0.15, 0]) * min(a * 2, 1)
            palm = wrist + np.array([0.3, 0.3, 0])
            bones, joints = VGroup(), VGroup(Dot(wrist, radius=0.045), Dot(palm, radius=0.045))
            bones.add(Line(wrist, palm))
            for j, base_ang in enumerate([1.25, 0.95, 0.7, 0.45, 0.2]):
                ang = base_ang - (0.0 if j == 0 else 0.9 * close)
                p0 = palm + 0.12 * np.array([np.cos(base_ang + 0.6), np.sin(base_ang + 0.6), 0]) if j == 0 else palm
                pts = [p0]
                L = [0.22, 0.26, 0.28, 0.26, 0.22][j]
                cur = ang
                for seg in range(3):
                    pts.append(pts[-1] + L * (0.5 - 0.1 * seg) * np.array([np.cos(cur), np.sin(cur), 0]))
                    cur -= 0.7 * close
                for p, q in zip(pts[:-1], pts[1:]):
                    bones.add(Line(p, q))
                joints.add(*[Dot(p, radius=0.032) for p in pts[1:]])
            bones.set_stroke(color, 4)
            joints.set_fill(FG, 1)
            ball = Circle(radius=0.2).set_stroke(FG, 2).set_fill(color, 0.25).move_to([0.45, 0.35, 0])
            ball.scale(1 - 0.12 * close)
            return VGroup(ball, bones, joints).scale(s, about_point=ORIGIN)
    elif k == 2:  # time vs camera: world-time axis x camera axis, dot moves along camera with time frozen
        def draw():
            a = prog.get_value()
            ax = VGroup(Arrow([-0.9, -0.8, 0], [1.0, -0.8, 0], buff=0, stroke_width=3, tip_length=0.12),
                        Arrow([-0.9, -0.8, 0], [-0.9, 0.95, 0], buff=0, stroke_width=3, tip_length=0.12)).set_color(MUTED)
            lt = T("world time", 16, MUTED).next_to(ax[0], DOWN, buff=0.06)
            lc = T("camera", 16, MUTED).rotate(PI / 2).next_to(ax[1], LEFT, buff=0.06)
            diag = DashedLine([-0.9, -0.8, 0], [0.8, 0.8, 0], dash_length=0.06).set_stroke(MUTED, 2, opacity=0.6)
            t_now = -0.9 + 1.7 * min(a / 0.4, 1.0) * 0.45
            c_now = -0.8 + 1.6 * np.clip((a - 0.25) / 0.7, 0, 1)
            pathl = VMobject().set_points_as_corners([[-0.9, -0.8, 0], [t_now, -0.8 + (t_now + 0.9) * 0.94, 0]])
            v = Line([t_now, -0.8 + (t_now + 0.9) * 0.94, 0], [t_now, max(c_now, -0.8 + (t_now + 0.9) * 0.94), 0])
            trail = VGroup(pathl, v).set_stroke(color, 5)
            dot = Dot(v.get_end(), radius=0.07, color=FG)
            return VGroup(ax, lt, lc, diag, trail, dot).scale(s, about_point=ORIGIN)
    else:  # point cloud accumulating along a loop; revisiting reuses it
        rng = np.random.default_rng(3)
        pts = rng.uniform([-0.95, -0.7, 0], [0.95, 0.7, 0], (160, 3)) * np.array([1, 1, 0])
        loop = Circle(radius=0.5).stretch(1.4, 0)

        def draw():
            a = prog.get_value()
            p = loop.point_from_proportion((0.25 + a) % 1)
            n = int(len(pts) * min(a * 1.4, 1))
            cloud = VGroup(*[Dot(q, radius=0.022).set_fill(color, 0.9) for q in pts[:n]])
            trail = loop.copy().pointwise_become_partial(loop, 0, 1).set_stroke(MUTED, 2, opacity=0.4)
            cam = Dot(p, radius=0.08, color=FG)
            return VGroup(cloud, trail, cam).scale(s, about_point=ORIGIN)
    c = np.array(center, dtype=float)
    return always_redraw(lambda: draw().shift(c)), prog


# ---------------------------------------------------------------- text + figures
def caption(seg, size=36, venue_size=26, bar=True, color=FG, font=FONT, weight=BOLD, max_w=None, two_lines=False,
            align=LEFT):
    """Title (bold) over venue (accent), with an accent bar on the aligned side. two_lines uses seg['lines']."""
    if two_lines:
        t = VGroup(*[T(l, size, color, weight=weight, font=font) for l in seg["lines"]]).arrange(DOWN, buff=0.08, aligned_edge=align)
    else:
        t = T(seg["title"], size, color, weight=weight, font=font)
    if max_w and t.width > max_w:
        t.set_width(max_w)
    v = T(seg["venue"], venue_size, seg["color"], font=font)
    txt = VGroup(t, v).arrange(DOWN, buff=0.14, aligned_edge=align)
    if not bar:
        return txt
    b = Line(UP, DOWN).set_stroke(seg["color"], 5).set_height(txt.height + 0.06)
    return VGroup(b, txt).arrange(RIGHT, buff=0.2) if align is LEFT else VGroup(txt, b).arrange(RIGHT, buff=0.2)


def fig_image(png, width=None, height=None):
    img = ImageMobject(os.path.join(CROPS, png + ".png"))
    if width:
        img.set_width(width)
    if height and img.height > height:
        img.set_height(height)
    return img


def fig_stack(seg, width=3.55, max_h=4.1, label_size=17, label_color=MUTED, buff=0.22):
    items = Group()
    for png, label in seg["figs"]:
        img = fig_image(png, width, max_h)
        lab = T(label, label_size, label_color)
        items.add(Group(lab, img).arrange(DOWN, buff=0.09, aligned_edge=LEFT))
    return items.arrange(DOWN, buff=buff, aligned_edge=LEFT)


def fade_band(width=FW, height=2.2, y_bottom=-FH / 2, color=BLACK, max_op=0.8, n=24, up=True, power=1.6):
    """Vertical opacity ramp (cairo has no alpha gradients): n stacked strips, opaque at the bottom."""
    strips = VGroup()
    for i in range(n):
        f = i / (n - 1)
        r = Rectangle(width=width, height=height / n).set_stroke(width=0)
        r.set_fill(color, max_op * (1 - f) ** power)
        r.move_to([0, y_bottom + (i + 0.5) * height / n if up else y_bottom - (i + 0.5) * height / n, 0])
        strips.add(r)
    return strips


# ---------------------------------------------------------------- timeline scene
class TimedScene(Scene):
    """Scene with an absolute clock: goto(t) waits until t, play_t(..., run_time) advances it."""

    def setup(self):
        self.t = 0.0

    def goto(self, t):
        if t > self.t + 1e-6:
            self.wait(t - self.t)
            self.t = t

    def play_t(self, *anims, run_time, **kw):
        self.play(*anims, run_time=run_time, **kw)
        self.t += run_time

    def play_until(self, t, *anims, **kw):
        self.play_t(*anims, run_time=t - self.t, **kw)
