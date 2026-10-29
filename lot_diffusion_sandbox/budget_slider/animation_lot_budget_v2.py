"""Level-of-Token Diffusion — token-budget slider clips, v2 (one 5-s clip per paper scene).

Source: George's budget sweep (paper App. A.7, Figs. 10–13), read-only:
    /home/brianchc/LoT-george-branch/results/2026-09-04_001859_budget-sweep-100-nopeople-mask-vrs/outputs
    sweep_<arm>/<item>/B<budget>/{grid.pt, out.png}   (LoT-FLUX.2 4B, 1024², seed 42, 30 steps)
The 16 scenes are the items of make_supp_figures.sh: supp_mask_1 = Fig. 10, supp_mask_2 = Fig. 11,
supp_vrs_1 = Fig. 12, supp_vrs_2 = Fig. 13 (4 prompts each).

Each clip: token layout (left, vector tokens drawn from grid.pt, colour-shaded by the generated image) and the
generated image (right); v2 drops the source / prompt-seed label; a slider under them steps B = 1,024 → 1,792 → 2,560 → 3,328 → 4,096, one second per budget.

Render (see render.sh):
    manim -qh --media_dir media animation_lot_budget_v2.py <Scene>
"""
from manim import *
import numpy as np
from PIL import Image
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = "/home/brianchc/LoT-george-branch/results/2026-09-04_001859_budget-sweep-100-nopeople-mask-vrs/outputs"
GEN = os.path.join(HERE, "data", "generated")                  # assets copied / built on import

config.background_color = BLACK

# ------------------------------------------------------------------ style (as ../teaser_5min v3)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
HR = "#448AFF"                           # 1 × 1 (full resolution)
LR = "#00C853"                           # 2 × 2
LR3 = "#1B7F3B"                          # 4 × 4
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
AMBER = "#FFCA28"
HEADER_Y = 3.62
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


def fmt(n):
    return f"{n:,}"


# ------------------------------------------------------------------ scenes of paper Figs. 10–13
BUDGETS = [1024, 1792, 2560, 3328, 4096]
N = 64                                   # FLUX.2 at 1024²: 64 × 64 latent tokens
FIGS = {                                 # fig: (arm, items top to bottom), from make_supp_figures.sh
    10: ("mask", ["27778182_deer", "19124730_mountain", "6367607_building", "29291198_mountain"]),
    11: ("mask", ["5ipoNYbDnpE_building", "29498261_mountain", "28293822_mountain", "13712077_car"]),
    12: ("vrs", ["09PQIp2Yqpc_mountain", "121828d33b59c694f49b4c5c689bf454_tree", "28293822_mountain",
                 "29291198_mountain"]),
    13: ("vrs", ["29498261_mountain", "29593109_mountain", "4w1j0dgvlvI_mountain", "7912886_flower"]),
}
ARM_NAME = {"mask": "semantic-mask layout", "vrs": "texture-variance layout"}
SCENES = [(f"F{f}_{k + 1}_{it.split('_')[-1].capitalize()}", f, k, arm, it)
          for f, (arm, items) in FIGS.items() for k, it in enumerate(items)]


def gen(*p):
    return os.path.join(GEN, *p)


def build_assets():
    """Copy out.png and convert grid.pt leaves + 64 × 64 box-mean colours, once per (arm, item, budget)."""
    for _, _, _, arm, it in SCENES:
        d = gen(arm, it)
        if os.path.exists(os.path.join(d, "done")):
            continue
        import torch
        os.makedirs(d, exist_ok=True)
        for B in BUDGETS:
            src = os.path.join(SWEEP, f"sweep_{arm}", it, f"B{B}")
            g = torch.load(os.path.join(src, "grid.pt"), map_location="cpu", weights_only=False)
            leaves = [tuple(int(v) for v in l) for l in g["leaves"]]          # (r0, r1, c0, c1) on the 64 × 64 grid
            assert len(leaves) == B, (arm, it, B, len(leaves))
            json.dump(leaves, open(os.path.join(d, f"B{B}.json"), "w"))
            shutil.copy(os.path.join(src, "out.png"), os.path.join(d, f"B{B}.png"))
            c = np.asarray(Image.open(os.path.join(src, "out.png")).convert("RGB").resize((N, N), Image.BOX))
            np.save(os.path.join(d, f"B{B}_cells.npy"), c.astype(float) / 255)
        json.dump(g["prompt"], open(os.path.join(d, "prompt.json"), "w"))
        open(os.path.join(d, "done"), "w").close()


build_assets()


# ------------------------------------------------------------------ token layout with rectangular extents
def _mix(a, b, t):
    return interpolate_color(ManimColor(a), ManimColor(b), t)


AREA_COL = {1: ManimColor(HR), 2: _mix(HR, LR, 0.5), 4: ManimColor(LR), 8: _mix(LR, LR3, 0.5), 16: ManimColor(LR3)}


class Layout:
    """Leaves (r0, r1, c0, c1) drawn as tokens on an n × n grid; fill = mean image colour over the patch,
    outline = extent colour (by area: 1 blue, 4 green, 16 dark green; 2 and 8 in between), 10% cell margin."""

    def __init__(self, leaves, cells, size, center):
        self.size, self.c = size, size / N
        self.center = np.array([center[0], center[1], 0.0])
        self.leaves = list(leaves)
        self.toks = [self._make(l, cells) for l in self.leaves]

    def _make(self, l, cells):
        r0, r1, c0, c1 = l
        h, w = r1 - r0, c1 - c0
        fill = rgb_to_color(cells[r0:r1, c0:c1].reshape(-1, 3).mean(0))
        sw = (0.3 + 0.43 * (np.sqrt(h * w) - 1)) * self.size / 4.6
        kw = dict(fill_color=fill, fill_opacity=1, stroke_color=AREA_COL[h * w], stroke_width=sw)
        W, H = w * self.c - 0.2 * self.c, h * self.c - 0.2 * self.c
        if min(W, H) < 0.1:
            m = Rectangle(width=W, height=H, **kw)
        else:
            m = RoundedRectangle(width=W, height=H, corner_radius=0.1 * min(W, H), **kw)
        return m.move_to(self.center + np.array([(c0 + c1) / 2 * self.c - self.size / 2,
                                                 self.size / 2 - (r0 + r1) / 2 * self.c, 0]))

    def group(self):
        return VGroup(*self.toks)

    def morph(self, leaves, cells):
        """One Transform per connected group of overlapping old / new tokens (layouts need not be nested)."""
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
S = 4.4                                  # panel side
PX = 3.4                                 # panel centres x = ±PX
PY = 0.6
SL_Y, SL_X0, SL_X1 = -2.3, -4.2, 4.2     # slider
HOLD_END, TRANS, HOLD = 0.8, 0.4, 0.6    # 0.8 + 4 × 0.4 + 3 × 0.6 + 0.8 = 5.0 s


def bx(B):
    return SL_X0 + (B - BUDGETS[0]) / (BUDGETS[-1] - BUDGETS[0]) * (SL_X1 - SL_X0)


class BudgetClip(Scene):
    FIG, ROW, ARM, ITEM = 10, 0, "mask", "27778182_deer"

    def construct(self):
        d = gen(self.ARM, self.ITEM)
        leaves = {B: [tuple(l) for l in json.load(open(os.path.join(d, f"B{B}.json")))] for B in BUDGETS}
        cells = {B: np.load(os.path.join(d, f"B{B}_cells.npy")) for B in BUDGETS}

        # header, source, panel labels
        self.add(tx("Token budget · " + ARM_NAME[self.ARM], 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y))
        self.add(tx("token layout", 26, DIM).move_to([-PX, PY + S / 2 + 0.3, 0]))
        self.add(tx("generated image", 26, DIM).move_to([PX, PY + S / 2 + 0.3, 0]))

        lay = Layout(leaves[BUDGETS[0]], cells[BUDGETS[0]], S, (-PX, PY))
        self.add(lay.group())
        pic = ImageMobject(os.path.join(d, f"B{BUDGETS[0]}.png")).set(height=S).move_to([PX, PY, 0])
        frame = framed(pic, GHOST)
        self.add(pic, frame)

        # live count between the panels
        bt = ValueTracker(BUDGETS[0])

        def count():
            b = int(round(bt.get_value()))
            return VGroup(tx(fmt(b), 44, AMBER, bold=True).move_to([0, PY + 0.75, 0]),
                          tx("tokens", 22, DIM).move_to([0, PY + 0.3, 0]),
                          tx(f"{100 * b / N ** 2:.0f}% of {fmt(N ** 2)}", 20, DIM).move_to([0, PY - 0.55, 0]))

        self.add(always_redraw(count), arrow([-1.05, PY - 0.12, 0], [1.05, PY - 0.12, 0]))

        # slider
        self.add(Line([SL_X0, SL_Y, 0], [SL_X1, SL_Y, 0], color=GHOST, stroke_width=4))
        ticks = VGroup(*[Line([bx(B), SL_Y - 0.09, 0], [bx(B), SL_Y + 0.09, 0], color=DIM, stroke_width=2)
                         for B in BUDGETS])
        tlab = [tx(fmt(B), 20, DIM).move_to([bx(B), SL_Y - 0.36, 0]) for B in BUDGETS]
        self.add(ticks, *tlab, tx("token budget <i>B</i>", 22, DIM).next_to([SL_X0, SL_Y, 0], LEFT, buff=0.35))
        self.add(always_redraw(lambda: Line([SL_X0, SL_Y, 0], [bx(bt.get_value()), SL_Y, 0],
                                            color=AMBER, stroke_width=5)))
        self.add(always_redraw(lambda: Dot([bx(bt.get_value()), SL_Y, 0], radius=0.12, color=WHITE)))
        self.remove(tlab[0])
        cur = tx(fmt(BUDGETS[0]), 20, WHITE).move_to(tlab[0])
        self.add(cur)

        self.add(tx("A larger token budget buys finer tokens and more detail.", 28)
                 .move_to([0, CAPTION_Y, 0]))
        self.wait(HOLD_END)
        for k, B in enumerate(BUDGETS[1:], 1):
            new = ImageMobject(os.path.join(d, f"B{B}.png")).set(height=S).move_to([PX, PY, 0])
            self.remove(cur)
            self.add(tlab[k - 1])
            self.play(bt.animate.set_value(B), *lay.morph(leaves[B], cells[B]), FadeIn(new),
                      run_time=TRANS, rate_func=smooth)
            self.remove(pic)
            pic = new
            self.bring_to_front(frame)
            self.remove(tlab[k])
            cur = tx(fmt(B), 20, WHITE).move_to(tlab[k])
            self.add(cur)
            self.wait(HOLD if k < len(BUDGETS) - 1 else HOLD_END)


for _name, _f, _k, _arm, _it in SCENES:
    globals()[_name] = type(_name, (BudgetClip,), dict(FIG=_f, ROW=_k, ARM=_arm, ITEM=_it))
