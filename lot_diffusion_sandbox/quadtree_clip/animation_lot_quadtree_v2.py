"""Level-of-Token Diffusion — 10-s X clip: quadtree in graphics → LoT token layout → diffusion, v2 (v1 + black quadtree fill, no white flash in the morph).

Tweet: "Given a layout and a token budget, LoT derives a quadtree-like multi-resolution token grid. A single model
generates coherent images across a wide range of budgets."

Running example: the deer of paper Fig. 10 (semantic-mask layout). Real assets from George's budget sweep
(read-only), copied by ../budget_slider on import: ../budget_slider/data/generated/mask/27778182_deer/B<budget>.{json,png}
and the SAM3 mask at SWEEP/sweep_mask/27778182_deer/semantic_mask.png.

Beats: (1) a region quadtree subdivides the mask, level by level, as in graphics; (2) its cells fly left and become
the LoT layout at B = 1,024 (shaded tokens), read by DiT + LoRA into the image; (3) the same layout is re-tokenized at
1,792 → 4,096 tokens and the image updates at each budget.

Render (see render.sh):
    manim -qh --media_dir media animation_lot_quadtree_v2.py LoTQuadtree
"""
from manim import *
import numpy as np
from PIL import Image
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SWEEP = "/home/brianchc/LoT-george-branch/results/2026-09-04_001859_budget-sweep-100-nopeople-mask-vrs/outputs"
ITEM = "27778182_deer"
SRC = os.path.join(ROOT, "budget_slider", "data", "generated", "mask", ITEM)
GEN = os.path.join(HERE, "data", "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style (as ../teaser_5min v3)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
HR = "#448AFF"
LR = "#00C853"
LR3 = "#1B7F3B"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
AMBER = "#FFCA28"
LORA = "#FF7043"
BOX = "#1E2A33"
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓⊤")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


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
BUDGETS = [1024, 1792, 2560, 3328, 4096]
N = 64
QT_MIN = 2                               # graphics quadtree stops at 2 × 2 cells (5 levels below the root)


def gen(name):
    return os.path.join(GEN, name)


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if not os.path.exists(gen("mask64.npy")):
        m = np.asarray(Image.open(os.path.join(SWEEP, "sweep_mask", ITEM, "semantic_mask.png")).convert("L"))
        m64 = np.asarray(Image.fromarray(m).resize((N, N), Image.BOX)).astype(float) / 255
        np.save(gen("mask64.npy"), m64)
        rgba = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
        rgba[m > 127] = (110, 110, 110, 255)                        # dim silhouette, black elsewhere
        rgba[m <= 127, 3] = 255
        Image.fromarray(rgba).resize((512, 512), Image.BOX).save(gen("mask_dim.png"))
    for B in BUDGETS:
        if not os.path.exists(gen(f"B{B}_cells.npy")):
            c = np.asarray(Image.open(os.path.join(SRC, f"B{B}.png")).convert("RGB").resize((N, N), Image.BOX))
            np.save(gen(f"B{B}_cells.npy"), c.astype(float) / 255)


build_assets()
MASK = np.load(gen("mask64.npy"))


def quadtree_levels():
    """Region quadtree of the mask: split a block while it is mixed (part object, part background).
    Returns the blocks created at each level (level 0 = the root) and the final leaves (r0, r1, c0, c1)."""
    levels, leaves = [[(0, N, 0, N)]], []
    frontier = [(0, N, 0, N)]
    while frontier:
        nxt = []
        for r0, r1, c0, c1 in frontier:
            v = MASK[r0:r1, c0:c1]
            if r1 - r0 > QT_MIN and v.min() < 0.5 <= v.max():
                h = (r1 - r0) // 2
                nxt += [(r0, r0 + h, c0, c0 + h), (r0, r0 + h, c0 + h, c1),
                        (r0 + h, r1, c0, c0 + h), (r0 + h, r1, c0 + h, c1)]
            else:
                leaves.append((r0, r1, c0, c1))
        if nxt:
            levels.append(nxt)
        frontier = nxt
    return levels, leaves


# ------------------------------------------------------------------ token layouts (rectangular extents)
def _mix(a, b, t):
    return interpolate_color(ManimColor(a), ManimColor(b), t)


AREA_COL = {1: ManimColor(HR), 2: _mix(HR, LR, 0.5), 4: ManimColor(LR), 8: _mix(LR, LR3, 0.5), 16: ManimColor(LR3)}


class Layout:
    """Leaves drawn on an N × N grid of side `size`. cells=None: graphics quadtree cells (white outlines, no fill);
    else LoT tokens: fill = mean image colour over the patch, outline = extent colour, 10% cell margin."""

    def __init__(self, leaves, cells, size, center):
        self.place(size, center)
        self.leaves = list(leaves)
        self.toks = [self._make(l, cells) for l in self.leaves]

    def place(self, size, center):
        self.size, self.c = size, size / N
        self.center = np.array([center[0], center[1], 0.0])

    def _make(self, l, cells):
        r0, r1, c0, c1 = l
        h, w = r1 - r0, c1 - c0
        pos = self.center + np.array([(c0 + c1) / 2 * self.c - self.size / 2, self.size / 2 - (r0 + r1) / 2 * self.c, 0])
        if cells is None:
            return Rectangle(width=w * self.c, height=h * self.c, stroke_color=WHITE, stroke_width=1.6,
                             fill_color=BLACK, fill_opacity=0).move_to(pos)   # black fill: no white flash in the morph
        fill = rgb_to_color(cells[r0:r1, c0:c1].reshape(-1, 3).mean(0))
        sw = (0.3 + 0.43 * (np.sqrt(h * w) - 1)) * self.size / 4.6
        kw = dict(fill_color=fill, fill_opacity=1, stroke_color=AREA_COL.get(h * w, ManimColor(LR3)), stroke_width=sw)
        W, H = w * self.c - 0.2 * self.c, h * self.c - 0.2 * self.c
        if min(W, H) < 0.1:
            m = Rectangle(width=W, height=H, **kw)
        else:
            m = RoundedRectangle(width=W, height=H, corner_radius=0.1 * min(W, H), **kw)
        return m.move_to(pos)

    def group(self):
        return VGroup(*self.toks)

    def morph(self, leaves, cells):
        """One ReplacementTransform per connected group of overlapping old / new tokens."""
        n_old = len(self.leaves)
        own_old = np.zeros((N, N), int)
        own_new = np.zeros((N, N), int)
        for k, (r0, r1, c0, c1) in enumerate(self.leaves):
            own_old[r0:r1, c0:c1] = k
        for k, (r0, r1, c0, c1) in enumerate(leaves):
            own_new[r0:r1, c0:c1] = k
        par = list(range(n_old + len(leaves)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in zip(own_old.ravel(), own_new.ravel()):
            ra, rb = find(a), find(n_old + b)
            if ra != rb:
                par[ra] = rb
        comps = {}
        for k in range(n_old + len(leaves)):
            comps.setdefault(find(k), ([], []))[k >= n_old].append(k if k < n_old else k - n_old)
        new_toks = [self._make(l, cells) for l in leaves]
        anims = []
        for olds, news in comps.values():
            src = self.toks[olds[0]] if len(olds) == 1 else VGroup(*[self.toks[k] for k in olds])
            dst = new_toks[news[0]] if len(news) == 1 else VGroup(*[new_toks[k] for k in news])
            anims.append(ReplacementTransform(src, dst))
        self.leaves, self.toks = list(leaves), new_toks
        return anims


# ------------------------------------------------------------------ the clip
S0, Y0 = 4.6, 0.25                       # beat 1: centred quadtree panel
S1, X1 = 4.0, 4.4                        # beats 2–3: layout at −X1, image at +X1
LAB_DY = 0.32


class LoTQuadtree(Scene):
    def cap(self, s):
        if getattr(self, "_cap", None) is not None:
            self.remove(self._cap)
        self._cap = tx(s, 28).move_to([0, CAPTION_Y, 0])
        self.add(self._cap)

    def construct(self):
        leaves = {B: [tuple(l) for l in json.load(open(os.path.join(SRC, f"B{B}.json")))] for B in BUDGETS}
        cells = {B: np.load(gen(f"B{B}_cells.npy")) for B in BUDGETS}

        # ---- beat 1 (0–3.0 s): graphics quadtree over the layout
        mask = ImageMobject(gen("mask_dim.png")).set(height=S0).move_to([0, Y0, 0])
        lab = tx("quadtree subdivision, as in graphics", 26, DIM).move_to([0, Y0 + S0 / 2 + LAB_DY, 0])
        self.add(mask, lab)
        self.cap("Graphics: a quadtree splits only where detail is.")
        levels, qleaves = quadtree_levels()
        qt = Layout([], None, S0, (0, Y0))
        lvl_mobs = [VGroup(*[qt._make(b, None) for b in lv]) for lv in levels]
        self.add(lvl_mobs[0])
        self.wait(0.2)
        for g in lvl_mobs[1:]:
            self.play(Create(g, lag_ratio=0.0), run_time=1.5 / (len(lvl_mobs) - 1), rate_func=linear)
        self.wait(3.0 - self.renderer.time)

        # ---- beat 2 (3.0–6.5 s): quadtree cells → LoT tokens at B = 1,024 → DiT → image
        self.remove(*lvl_mobs)
        qt = Layout(qleaves, None, S0, (0, Y0))
        self.add(qt.group())
        lot = qt
        lot.place(S1, (-X1, Y0))
        lab2 = tx("LoT token layout", 26, DIM).move_to([-X1, Y0 + S1 / 2 + LAB_DY, 0])
        self.cap("LoT: layout + budget → a quadtree-like token grid.")
        mask_to = mask.copy().set(height=S1).move_to([-X1, Y0, 0]).set_opacity(0)
        self.play(*lot.morph(leaves[1024], cells[1024]), Transform(mask, mask_to),
                  FadeOut(lab), run_time=0.9, rate_func=smooth)
        self.remove(mask)
        self.add(lab2)
        bt = ValueTracker(1024)
        self.add(always_redraw(lambda: tx(fmt(int(round(bt.get_value()))) + " tokens", 28, AMBER, bold=True)
                               .move_to([-X1, Y0 - S1 / 2 - 0.35, 0])))

        dit = dit_box([0, Y0, 0], lora=True)
        a1 = arrow([-X1 + S1 / 2 + 0.1, Y0, 0], [-0.85, Y0, 0])
        a2 = arrow([0.85, Y0, 0], [X1 - S1 / 2 - 0.1, Y0, 0])
        pic = ImageMobject(os.path.join(SRC, "B1024.png")).set(height=S1).move_to([X1, Y0, 0])
        frame = framed(pic, GHOST)
        lab3 = tx("generated image", 26, DIM).move_to([X1, Y0 + S1 / 2 + LAB_DY, 0])
        self.play(GrowArrow(a1), FadeIn(dit), run_time=0.35)
        self.play(GrowArrow(a2), FadeIn(pic), FadeIn(frame), run_time=0.35)
        self.add(lab3)
        self.wait(6.5 - self.renderer.time)

        # ---- beat 3 (6.5–10 s): one model, the same layout at every budget
        self.cap("One model, coherent images at every budget.")
        for B in BUDGETS[1:]:
            new = ImageMobject(os.path.join(SRC, f"B{B}.png")).set(height=S1).move_to([X1, Y0, 0])
            self.play(bt.animate.set_value(B), *lot.morph(leaves[B], cells[B]), FadeIn(new),
                      run_time=0.35, rate_func=smooth)
            self.remove(pic)
            pic = new
            self.bring_to_front(frame)
            self.wait(0.35)
        self.wait(10.0 - self.renderer.time)
