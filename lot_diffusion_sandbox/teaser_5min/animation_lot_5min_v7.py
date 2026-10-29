"""Level-of-Token Diffusion — ~5-minute teaser, v7 (click-through slide clips).

v7: S05_AsymFlow: the sampler row sits lower, so its tokens no longer overlap the corgi layout; the zoom lines end
above the "noisy patch" label instead of crossing it.

v6: token outlines are drawn inside the token, so edges align across extents; framed() draws nothing.

v5 (2026-10-14): the Fig. 4, Fig. 7a and Fig. 7b panels come from their true sources, not from crops of the
paper's figure rasters. The crops kept thin white or black gutter lines and were resampled twice. The panels
are now the files the paper's figure scripts placed (results/figure4/assets, results/paper_figures_456_portable/
assets in the LoT repo). Two Fig. 7a panels (DDiT, Foveated) exist only as rasters embedded in the PDF. v5
extracts those rasters directly. The old crops are kept in data/generated/_pdfcrop_backup_2026-10-14/.
Everything else is as v4.

v4 (2026-10-14): for a reveal.js deck. CAPTIONS = False turns off captions, "k · title" headers and part cards;
every scene is cut into sections (`self.next_section`, render with --save_sections), one visual step each. A section
always ends on its finished visual (0.3-s hold; the slide freezes on the last frame); fade-outs of the previous
content open the next section; scenes no longer fade to black at the end. S03 drops the attention-pairs line
(no quadratic-cost claims anywhere on screen).

Source: "Level-of-Token Diffusion", anonymous ICLR 2027 submission 7014
(../7014_Level_of_Token_Diffusion.pdf, supplementary site mirrored in ../web/).
Running example: the site's painted demo "A girl riding a corgi" (LoT-FLUX.2); its 64 × 64 layout is rebuilt
from the site's token grid (quadtree of A.3, Eq. 23, reproduces the site's 1,312 / 140 / 79 / 15 leaves).
Tokens are colour-shaded: fill = mean colour of the image over the token's patch, outline = extent colour.

Render (see render.sh):
    manim -ql --save_sections --media_dir media animation_lot_5min_v5.py <Scene>
Scenes, in order: S00_Title, S01_Cost, S02_Known, S03_Layout, S04_Naive, S05_AsymFlow, S06_DiT,
S07_Sources, S08_Results, S09_Outlook.
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

# ------------------------------------------------------------------ style (as the 1-min v4 / foveated 5-min v2)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × ⊙ − → ≈ …
HR = "#448AFF"                           # 1 × 1 (full resolution)
LR = "#00C853"                           # 2 × 2
LR3 = "#1B7F3B"                          # 4 × 4
LR4 = "#0E5A2A"                          # 8 × 8
EXT_COL = {1: HR, 2: LR, 4: LR3, 8: LR4}
EXT_TXT = {1: HR, 2: LR, 4: "#43A047", 8: "#2E7D32"}    # readable text versions of the extent colours
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
RED = "#FF5252"
AMBER = "#FFCA28"
PURPLE = "#E040FB"
LORA = "#FF7043"
BOX = "#1E2A33"
FOV_C = "#B0BEC5"                        # Foveated Diffusion in plots
PAINT = {"0": "#000000", "0.3": "#FF8A00", "0.6": "#FFD400", "1": "#FFFFFF"}   # brush colours of the site's canvas

CAPTIONS = False                         # v4: False = no captions, no "k · title" headers, no part cards
END_HOLD = 0.3                           # hold on the finished visual at the end of every section
HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓⊤")


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
    # No visible frame: thin outlines around images read as white crop borders on the slides.
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw, stroke_opacity=0)


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


def small_box(label, center, w=1.1, h=0.62, stroke="#90A4AE", size=20):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.08, fill_color=BOX, fill_opacity=1,
                         stroke_color=stroke, stroke_width=2).move_to(center)
    return VGroup(r, tx(label, size).move_to(r))


def fmt(n):
    return f"{n:,}"


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


N = 64                                   # FLUX.2 at 1024²: 64 × 64 token grid
QT = {8: 0.02, 4: 0.44, 2: 0.74}         # fitted split thresholds (A.3, Eq. 23); reproduce the site's leaf counts
SITE_LEAVES = {1: 1312, 2: 140, 4: 79, 8: 15}
TOKGRID = os.path.join(WEB, "b72bc22236f8abc9ffd53f8b980b6c031bad9faae21a35438deb4389cb7474fb.webp")
RESULT = os.path.join(IMG, "paint0_result.png")
FIG7A_X = [109.1, 188.1, 267.1, 346.1, 425.1]     # panel left edges (pt) on PDF page 9, width 77.6
FIG7B_COLS = [(3, 262), (266, 524), (528, 786), (798, 1056), (1060, 1318), (1322, 1581)]   # px at 4× of (108, 186)
FIG7B_ROWS = [(70, 215), (258, 404)]
FIG4_X = [6, 432, 858, 1284, 1710, 2136, 2562, 2988]  # panel left edges (px) in data/img/fig4_full.png, width 420
# v5: true sources of the figure panels (read-only, LoT repo results; matched to the v4 crops by MSE)
LOT_RES = "/home/brianchc/LoT-george-branch/results"
FIG4_SRC = os.path.join(LOT_RES, "figure4", "assets")        # placed by scripts/figure4_layout_sources.py
FIG4_FILES = [f"aesthetic_003534_{c}_{t}.png" for c in ("sam3_box", "sam3_mask", "vrs", "dof")
              for t in ("overlay", "generation")]               # deer row: (layout, generation) per cue
FIG7_SRC = os.path.join(LOT_RES, "paper_figures_456_portable", "assets")   # placed by scripts/figure6_baselines.py
FIG7A_FILES = ["baseline_image_1_0.png", "baseline_image_1_1.png", 493, 494, "baseline_image_1_4.png"]
# (ints: PDF xrefs of the 488 x 488 DDiT / Foveated rasters pasted over page 9; no other copy exists on disk)
FIG7B_FRAMES = (1, 38, 75)                                      # video7_frames/method{m+1}_frame{NN}.png
FIG4_SIZE, FIG7A_SIZE, FIG7B_SIZE = (414, 414), (617, 617), (512, 288)   # v4 crop heights, true aspect ratios
BACKUP = os.path.join(GEN, "_pdfcrop_backup_2026-10-14")
XT_T = [1.0, 0.75, 0.5, 0.25, 0.0]


def _detail_field():
    """Smoothed detail field D on the 64 × 64 lattice, read off the site's token-grid image (14 px per cell)."""
    a = np.asarray(Image.open(TOKGRID).convert("RGB")).astype(float) / 255
    return np.array([[a[i * 14 + 3:i * 14 + 11, j * 14 + 3:j * 14 + 11].mean() for j in range(N)] for i in range(N)])


def _layout(D):
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


def _save(im, name):
    tmp = gen(f"{name}.{os.getpid()}.png")
    im.save(tmp)
    os.replace(tmp, gen(name))


def _true_source_panels():
    """v5: replace the v4 figure crops with their true sources (LANCZOS to the v4 size). Back up the crops once."""
    import io
    import shutil
    import fitz
    jobs = {f"fig4_{k}.png": (os.path.join(FIG4_SRC, f), FIG4_SIZE) for k, f in enumerate(FIG4_FILES)}
    doc = fitz.open(PDF)
    for k, f in enumerate(FIG7A_FILES):
        jobs[f"fig7a_{k}.png"] = ((f"{PDF}#xref{f}" if isinstance(f, int) else os.path.join(FIG7_SRC, f)), FIG7A_SIZE)
    for m in range(4):
        for k, fr in enumerate(FIG7B_FRAMES):
            jobs[f"fig7b_{m}_{k}.png"] = (os.path.join(FIG7_SRC, "video7_frames", f"method{m + 1}_frame{fr:02d}.png"),
                                          FIG7B_SIZE)
    os.makedirs(BACKUP, exist_ok=True)
    for name, (src, size) in jobs.items():
        if os.path.exists(gen(name)) and not os.path.exists(os.path.join(BACKUP, name)):
            shutil.copy2(gen(name), os.path.join(BACKUP, name))
        if "#xref" in src:
            im = Image.open(io.BytesIO(doc.extract_image(int(src.split("#xref")[1]))["image"]))
        else:
            im = Image.open(src)
        _save(im.convert("RGB").resize(size, Image.LANCZOS), name)
    tmp = gen(f"true_sources.json.{os.getpid()}")
    json.dump({n: s for n, (s, _) in jobs.items()}, open(tmp, "w"), indent=1)
    os.replace(tmp, gen("true_sources.json"))


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if not os.path.exists(gen("corgi_layout.json")):
        D = _detail_field()
        np.save(gen("detail_D.npy"), D)
        tmp = gen("corgi_layout.json") + f".{os.getpid()}"
        json.dump(_layout(D), open(tmp, "w"))
        os.replace(tmp, gen("corgi_layout.json"))
    if not os.path.exists(gen("detail_D.png")):
        D = np.load(gen("detail_D.npy"))
        _save(Image.fromarray((D * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST), "detail_D.png")
    if not os.path.exists(gen("cells.npy")):
        c = np.asarray(Image.open(RESULT).convert("RGB").resize((N, N), Image.BOX)).astype(float) / 255
        np.save(gen("cells.npy"), c)
    if not os.path.exists(gen("fig7a_4.png")) or not os.path.exists(gen("fig7b_3_2.png")):
        import fitz
        page = fitz.open(PDF)[8]
        for k, x0 in enumerate(FIG7A_X):
            pix = page.get_pixmap(clip=fitz.Rect(x0 + 0.6, 97.4, x0 + 77.0, 174.5), matrix=fitz.Matrix(8, 8))
            tmp = gen(f"fig7a_{k}.{os.getpid()}.png")
            pix.save(tmp)
            os.replace(tmp, gen(f"fig7a_{k}.png"))
        # Fig. 7b: 4 methods (Full, ToMe-SD | Foveated, Ours) × 3 frames
        for r, (r0, r1) in enumerate(FIG7B_ROWS):
            for c, (c0, c1) in enumerate(FIG7B_COLS):
                m = 2 * r + c // 3
                rect = fitz.Rect(108 + c0 / 4 + 0.4, 186 + r0 / 4 + 0.4, 108 + c1 / 4 - 0.4, 186 + r1 / 4 - 0.4)
                pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(8, 8))
                tmp = gen(f"fig7b_{m}_{c % 3}.{os.getpid()}.png")
                pix.save(tmp)
                os.replace(tmp, gen(f"fig7b_{m}_{c % 3}.png"))
    if not os.path.exists(gen("fig4_7.png")):
        f4 = Image.open(os.path.join(IMG, "fig4_full.png")).convert("RGB")
        for k, x0 in enumerate(FIG4_X):
            _save(f4.crop((x0 + 3, 159, x0 + 417, 573)), f"fig4_{k}.png")
    if not os.path.exists(gen("true_sources.json")):
        _true_source_panels()
    if not os.path.exists(gen("blocky.png")):
        cells = np.load(gen("cells.npy"))
        lv = json.load(open(gen("corgi_layout.json")))
        out = np.zeros_like(cells)
        for i, j, b in lv:
            out[i:i + b, j:j + b] = cells[i:i + b, j:j + b].reshape(-1, 3).mean(0)
        _save(Image.fromarray((out * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST), "blocky.png")


build_assets()
LEAVES = [tuple(l) for l in json.load(open(gen("corgi_layout.json")))]
CELLS = np.load(gen("cells.npy"))                  # 64 × 64 × 3 mean colours of the result image


def block_color(i, j, b):
    return rgb_to_color(CELLS[i:i + b, j:j + b].reshape(-1, 3).mean(0))


# ------------------------------------------------------------------ Level-of-Token field
SW = {1: 0.3, 2: 0.9, 4: 1.6, 8: 2.2}               # outline widths of shaded tokens (at panel size 4.6)


def tok(side, color, stroke=None, sw=0.0):
    """Token of the given side. 1 × 1 tokens are plain squares (their 10% corner radius is sub-pixel)."""
    kw = dict(fill_color=color, fill_opacity=1, stroke_color=stroke if stroke else color, stroke_width=sw)
    side = side - 0.01 * sw          # outline inside the edge (manim stroke_width 1 = 0.01 units)
    if side < 0.1:
        return Square(side_length=side, **kw)
    return RoundedRectangle(width=side, height=side, corner_radius=0.1 * side, **kw)


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
    """Token layout drawn on an n × n grid of size `size`; `morph` merges / splits blocks in place.
    shade=True: fill = mean image colour of the token's patch, outline = extent colour."""

    def __init__(self, blocks, size, center, n=N, shade=False):
        self.n, self.size, self.c, self.shade = n, size, size / n, shade
        self.center = np.array([center[0], center[1], 0.0])
        self.toks = {b: self._make(b) for b in blocks}

    def pos(self, i, j, b):
        return self.center + np.array([(j + b / 2) * self.c - self.size / 2, self.size / 2 - (i + b / 2) * self.c, 0])

    def cell_rect(self, i, j, b, color, sw=3):
        return Square(side_length=b * self.c, color=color, stroke_width=sw).move_to(self.pos(i, j, b))

    def _make(self, blk):
        i, j, b = blk
        side = b * self.c - 0.2 * self.c
        if self.shade:
            return tok(side, block_color(i, j, b), EXT_COL[b], SW[b] * self.size / 4.6).move_to(self.pos(i, j, b))
        return tok(side, EXT_COL[b]).move_to(self.pos(i, j, b))

    def group(self):
        return VGroup(*self.toks.values())

    def count(self):
        return len(self.toks)

    def relocate(self, size, center):
        self.size, self.c = size, size / self.n
        self.center = np.array([center[0], center[1], 0.0])

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


def lag(anims, total=0.6):
    return LaggedStart(*anims, lag_ratio=total / max(len(anims), 1))


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None

    def section(self, name):
        """Start a new slide section (v4)."""
        self.next_section(name)

    def end_section(self, hold=END_HOLD):
        """Hold the finished visual briefly so the section's last frame shows everything added (v4)."""
        self.wait(hold)

    def header(self, s):
        if not CAPTIONS:
            return
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

    # Captions stay up long enough to read: ~16 characters per second + 0.8 s (teaser pacing).
    def _hold_caption(self):
        if not CAPTIONS:
            return
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
        if not CAPTIONS:                       # v4: draws nothing and holds nothing
            return None
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

    def title_card(self, num, title, hold=1.6):
        if not CAPTIONS:                       # v4: part cards skipped
            return
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
        keep = set(keep) | ({self.hdr} if self.hdr is not None else set())
        mobs = [m for m in self.mobjects if m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)


# 4 × 4-token patches on the girl's face and on empty sky: (row, col, extent) on the 64 × 64 grid.
FACE = (11, 30, 4)
SKY = (4, 4, 4)
FOCUS = (0.47, 0.5)          # (u, v) of the subject, for LaggedStart order
GRASS = min([l for l in LEAVES if l[2] == 4 and l[0] >= 52], key=lambda l: abs(l[1] - 10))   # one 4 × 4 grass token
S = 4.6
PL, PR = np.array([-3.4, 0.35, 0]), np.array([3.4, 0.35, 0])


def leaf_of(i, j):
    for (a, b, e) in LEAVES:
        if a <= i < a + e and b <= j < b + e:
            return (a, b, e)


def count_line(n, extra="", size=28):
    return tx(f"<b>{fmt(n)}</b> tokens{extra}", size)


def load_clip(name, h):
    fs = sorted(os.listdir(os.path.join(CLIPS, name)))
    arr = [np.asarray(Image.open(os.path.join(CLIPS, name, f)).convert("RGBA").resize((640, 360))) for f in fs]
    m = ImageMobject(arr[0].copy())
    m.set(height=h)
    return m, arr


def play_clip(frames, fps=15):
    clock = {"t": 0.0}

    def upd(mob, dt):
        clock["t"] += dt
        mob.pixel_array = frames[min(int(clock["t"] * fps), len(frames) - 1)]
    return upd


def extent_key(y, x=0.0, counts=True, size=20):
    el = VGroup()
    for b, n in SITE_LEAVES.items():
        sw = tok(0.26, EXT_COL[b])
        el.add(VGroup(sw, tx(f"{b} × {b}" + (f": {fmt(n)}" if counts else ""), size).next_to(sw, RIGHT, buff=0.08)))
    return el.arrange(RIGHT, buff=0.28).move_to([x, y, 0])


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("Level-of-Token Diffusion", 72, bold=True).move_to([0, 2.75, 0])
        subt = fit_width(tx("adaptive, efficient generation from multiresolution token layouts", 32), 12.8)
        subt.move_to([0, 1.95, 0])
        who = tx("Anonymous ICLR 2027 submission 7014", 22, DIM).move_to([0, -3.3, 0])
        f0 = LoTField(state_min(8), 3.8, (0, -0.6), shade=True)
        self.add(title, subt, who)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in f0.toks.values()], lag_ratio=0.02), run_time=1.0)
        for mb in (4, 2, 1):
            self.play(lag(f0.morph(state_min(mb), FOCUS), 0.5), run_time=0.8)
            self.wait(0.3)
        self.wait(1.4)
        self.play(FadeOut(f0.group()), FadeOut(title), FadeOut(subt), FadeOut(who), run_time=0.6)


# ================================================================== S01 Cost
class S01_Cost(Base):
    def construct(self):
        self.title_card("Part 1 · The problem", "Every token costs the same")
        self.header("1 · Every token costs the same")
        pic = img(RESULT, S).move_to(PL)
        self.play(FadeIn(pic), run_time=0.5)
        self.caption("A diffusion transformer generates an image as a grid of latent tokens.")
        c = S / N
        lines = VGroup(*[Line(PL + [-S / 2 + k * c, -S / 2, 0], PL + [-S / 2 + k * c, S / 2, 0]) for k in range(N + 1)],
                       *[Line(PL + [-S / 2, -S / 2 + k * c, 0], PL + [S / 2, -S / 2 + k * c, 0]) for k in range(N + 1)]
                       ).set_stroke(WHITE, 0.6, 0.5)
        self.play(Create(lines, lag_ratio=0.01), run_time=1.0)
        field = LoTField(state_max(1), S, PL, shade=True)
        self.play(LaggedStart(*[FadeIn(t) for t in field.toks.values()], lag_ratio=0.0003), run_time=1.2)
        self.play(FadeOut(pic), FadeOut(lines), run_time=0.5)
        cnt = tx(f"<i>h</i> × <i>w</i> = 64 × 64 = {col('4,096', HR)} tokens", 28).next_to(field.group(), DOWN, buff=0.22)
        self.add(cnt)
        self.caption("Each token is the latent of a 16 × 16-pixel patch of the 1024² image.")

        # face vs sky
        bf, bs = field.cell_rect(*FACE, AMBER), field.cell_rect(*SKY, PURPLE)
        kf = Square(0.3).set_stroke(AMBER, 3).move_to([1.2, 1.3, 0])
        ks = Square(0.3).set_stroke(PURPLE, 3).move_to([1.2, 0.6, 0])
        lf = tx("16 tokens on the girl's face", 26).next_to(kf, RIGHT, buff=0.25)
        ls = tx("16 tokens of empty sky", 26).next_to(ks, RIGHT, buff=0.25)
        self.play(Create(bf), Create(bs), Create(kf), Create(ks), run_time=0.5)
        self.add(lf, ls)
        self.caption("Both cost exactly the same compute, though only one carries detail.")
        self.uncaption()
        self.play(FadeOut(VGroup(bf, bs, kf, ks)), run_time=0.3)
        self.remove(lf, ls)

        # attention
        A = 3.6
        att = Square(A).set_fill("#10233F", 1).set_stroke(HR, 2).move_to([3.5, 0.5, 0])
        rng = np.random.default_rng(3)
        speck = VGroup(*[Square(A / 32 * 0.8).set_fill(HR, float(o)).set_stroke(width=0).move_to(
            att.get_corner(UL) + [(q + 0.5) * A / 32, -(p + 0.5) * A / 32, 0])
            for p in range(32) for q in range(32) for o in [rng.uniform(0.05, 0.6)]])
        lab = tx("4,096 × 4,096 ≈ 16.8 M query–key pairs", 26).next_to(att, DOWN, buff=0.22)
        note = tx("schematic", 18, DIM).next_to(att, UP, buff=0.12)
        self.play(FadeIn(att), FadeIn(speck), run_time=0.6)
        self.add(lab, note)
        self.caption("Every denoising step, self-attention compares every token with every other token.")
        q = (40, 20)
        hq = field.cell_rect(q[0], q[1], 1, WHITE, sw=3)
        rowy = att.get_top()[1] - (q[0] * 64 + q[1] + 0.5) / 4096 * A
        hr_ = Rectangle(width=A + 0.1, height=0.06, color=WHITE, stroke_width=2).move_to([att.get_x(), rowy, 0])
        self.play(Create(hq), Create(hr_), run_time=0.4)
        for qn in [(12, 31), (58, 50)]:
            yn = att.get_top()[1] - (qn[0] * 64 + qn[1] + 0.5) / 4096 * A
            self.play(hq.animate.move_to(field.pos(qn[0], qn[1], 1)), hr_.animate.set_y(yn), run_time=0.6)
        self.uncaption()
        self.play(FadeOut(VGroup(att, speck, hq, hr_)), run_time=0.4)
        self.remove(lab, note)

        # scaling
        s1 = Square(0.8).set_fill(HR, 0.6).set_stroke(HR, 2).move_to([1.6, 1.0, 0])
        s2 = Square(3.2).set_fill(HR, 0.6).set_stroke(HR, 2).move_to([4.6, 0.3, 0])
        l1a = tx("<i>N</i> tokens", 24).next_to(s1, UP, buff=0.2)
        l1b = tx("<i>N</i>" + sup("2") + " pairs", 24).next_to(s1, DOWN, buff=0.2)
        l2a = tx("2× resolution: 4<i>N</i> tokens", 24).next_to(s2, UP, buff=0.2)
        l2b = tx("16 <i>N</i>" + sup("2") + " pairs", 24).next_to(s2, DOWN, buff=0.2)
        self.play(FadeIn(s1), run_time=0.4)
        self.add(l1a, l1b)
        self.play(GrowFromCenter(s2), run_time=0.9)
        self.add(l2a, l2b)
        self.caption("Double the resolution: 4× the tokens, 16× the attention. Video multiplies it by the frames.")
        self.clear_all(run_time=0.6)
        self.wait(0.1)


# ================================================================== S02 Known
PROMPT = ("“A whimsical cinematic storybook photograph of a little girl riding an enormous friendly "
          "Pembroke Welsh corgi … " + col("expressive faces, natural hands, detailed soft fur", AMBER) + " … "
          + col("Simple softly defocused pale blue sky", PURPLE) + " and distant green meadow background …”")


class S02_Known(Base):
    def construct(self):
        self.header("2 · Detail is known before the pixels")
        self.section("prompt")
        # v4: hand-wrapped lines (the 44-character wrap ran the box off the left edge)
        lines = ["“A whimsical cinematic storybook photograph", "of a little girl riding an enormous friendly",
                 "Pembroke Welsh corgi …", col("expressive faces, natural hands, detailed soft fur", AMBER),
                 "… " + col("Simple softly defocused pale blue sky", PURPLE),
                 "and distant green meadow background …”"]
        ptxt = fit_width(VGroup(*[tx(l, 24) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.14), 6.0)
        ptxt.move_to([-3.75, 0.45, 0])
        pbox = SurroundingRectangle(ptxt, buff=0.3, corner_radius=0.12, color=GHOST, stroke_width=2)
        plab = tx("the prompt (site demo, excerpt)", 24, DIM).next_to(pbox, UP, buff=0.2)
        self.play(FadeIn(pbox), run_time=0.4)
        self.add(ptxt, plab)
        self.caption("The prompt already says where detail matters, and where it does not.")
        self.end_section()

        self.section("detail_map")
        cmap = img(os.path.join(IMG, "paint0_canvas.png"), 4.2).move_to([3.6, 0.45, 0])
        edge = framed(cmap, GHOST)
        mlab = tx("detail map, painted by an agent", 24, DIM).next_to(cmap, UP, buff=0.2)
        leg = VGroup()
        for v, c in PAINT.items():
            sq = Square(side_length=0.26, fill_color=c, fill_opacity=1, stroke_color=DIM, stroke_width=1)
            leg.add(VGroup(sq, tx(v, 22).next_to(sq, RIGHT, buff=0.1)))
        leg.arrange(RIGHT, buff=0.35)
        leg = VGroup(tx("detail", 22, DIM), leg).arrange(RIGHT, buff=0.3).next_to(cmap, DOWN, buff=0.25)
        a = arrow(pbox.get_right(), cmap.get_left(), sw=2)
        self.play(GrowArrow(a), FadeIn(cmap), FadeIn(edge), run_time=0.7)
        self.add(mlab, leg)
        self.caption("Here an agent turned it into a detail map: white = full detail, black = none.")
        self.end_section()

        self.section("other_sources")
        src = tx("the paper also derives layouts from semantic masks · boxes · texture variance · depth of field",
                 22, DIM).move_to([0, -2.6, 0])
        self.add(fit_width(src, 12.8))
        self.caption("Masks, boxes, texture or depth give the same kind of map.")
        self.end_section()

        self.section("question")
        self.clear_all(run_time=0.5)
        if self.hdr is not None:
            self.remove(self.hdr)
        q = tx("Can a diffusion model spend its tokens where the plan says?", 40, bold=True)
        self.add(fit_width(q, 12.5))
        self.end_section()


# ================================================================== S03 Layout
class S03_Layout(Base):
    def construct(self):
        self.title_card("Part 2 · The method", "Level-of-Token layout")
        self.header("3 · Level-of-Token layout")
        self.section("detail_field")
        dmap = img(gen("detail_D.png"), S).move_to(PL)
        dlab = tx("detail field <i>D</i> on the 64 × 64 token grid", 24, DIM).next_to(dmap, UP, buff=0.2)
        field = LoTField(state_min(8), S, PR, shade=True)
        flab = tx("Level-of-Token layout", 24, DIM).move_to([PR[0], dlab.get_y(), 0])
        self.play(FadeIn(dmap), run_time=0.5)
        dfr = framed(dmap, GHOST)
        self.add(dlab, dfr)
        self.caption("The painted map, averaged onto the token grid and smoothed, gives a detail field D.")
        self.end_section()

        self.section("coarse_tokens")
        self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in field.toks.values()], lag_ratio=0.02), run_time=0.8)
        self.add(flab)
        cc = [PR[0], -2.3, 0]
        cnt = count_line(64, "  (all 8 × 8)").move_to(cc)
        self.add(cnt)
        self.caption("Start with 8 × 8 tokens. Split a block into four wherever D is high enough.")
        self.end_section()

        self.section("split")
        for k, mb in enumerate((4, 2, 1)):
            anims = field.morph(state_min(mb), FOCUS)
            self.play(lag(anims, 0.6), run_time=1.2)
            self.remove(cnt)
            cnt = count_line(field.count()).move_to(cc)
            self.add(cnt)
            if k < 2:
                self.wait(0.5)                    # sub-steps: 211 → 562 → 1,546
        self.caption("Fine tokens land on the girl and the corgi; sky and meadow stay coarse.")
        key = extent_key(-2.78, PR[0])
        self.add(fit_width(key, 6.2))
        self.end_section()

        # what a token is
        self.section("extents")
        self.uncaption()
        self.play(FadeOut(dmap), FadeOut(dfr), run_time=0.5)
        self.remove(dlab)
        U = 0.17
        shapes = VGroup()
        for (eh, ew), cl in [((1, 1), HR), ((2, 2), LR), ((4, 4), LR3), ((8, 8), LR4), ((2, 4), "#26A69A")]:
            r = RoundedRectangle(width=ew * U - 0.03, height=eh * U - 0.03, corner_radius=0.1 * min(eh, ew) * U,
                                 fill_color=cl, fill_opacity=1, stroke_width=0)
            lab = tx(f"({eh}, {ew})", 22).next_to(r, DOWN, buff=0.12)
            shapes.add(VGroup(r, lab))
        shapes.arrange(RIGHT, buff=0.4, aligned_edge=DOWN).move_to([-3.4, 1.0, 0])
        slab = tx("token extent <i>e</i> = (<i>e</i>" + sub("h") + ", <i>e</i>" + sub("w") + ")", 26).next_to(
            shapes, UP, buff=0.35)
        self.play(LaggedStart(*[FadeIn(s_, shift=UP * 0.2) for s_ in shapes], lag_ratio=0.15), run_time=0.8)
        self.add(slab)
        self.caption("Each token is a rectangle of the grid; FLUX.2 uses sides of 1, 2, 4 or 8, not only squares.")
        self.end_section()

        # v4: sequence length and compression only (no attention-pairs / quadratic line)
        self.section("compression")
        eqs = VGroup(
            tx("sequence length <i>L</i> = 1,546", 28),
            tx("compression <i>HW</i> / <i>L</i> = 4,096 / 1,546 = 2.65×", 28),
        )
        for e in eqs:
            fit_width(e, 6.0)
        eqs.arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([-3.4, -1.1, 0])
        self.add(eqs)
        self.caption("The DiT now sees 1,546 tokens instead of 4,096: 2.65× fewer.")
        self.end_section()


# ================================================================== S04 Naive
class S04_Naive(Base):
    def construct(self):
        self.header("4 · Just feed big tokens to the DiT?")
        self.section("one_vector")
        field = LoTField(LEAVES, S, PR, shade=True)
        flab = tx("one vector per token", 24, DIM).next_to(field.group(), UP, buff=0.2)
        self.play(FadeIn(field.group()), run_time=0.6)
        self.add(flab)
        self.caption("A coarse token is a single vector for its whole patch.")
        self.end_section()

        # decode as is: every token fills its patch with one value
        self.section("blocky")
        c = S / N
        flat = [Square(side_length=b * c, fill_color=block_color(i, j, b), fill_opacity=1, stroke_width=0)
                .move_to(field.pos(i, j, b)) for (i, j, b) in field.toks]
        self.caption("Decode it as is, and each patch gets one value: blocky sky, blocky grass.")
        self.play(*[Transform(m, f) for m, f in zip(field.toks.values(), flat)], run_time=1.2)
        self.remove(flab)
        flab = tx("toy: one value per token", 24, DIM).next_to(field.group(), UP, buff=0.2)
        self.add(flab)
        bad = tx("blocky", 24, RED).next_to(field.group(), DOWN, buff=0.2)
        self.add(bad)
        self.end_section()

        self.section("lot_output")
        pic = img(RESULT, S).move_to(PL)
        plab = tx("LoT output, same layout", 24, DIM).next_to(pic, UP, buff=0.2)
        self.play(FadeIn(pic), run_time=0.6)
        self.add(plab)
        self.caption("The VAE decoder needs every fine position, and a pretrained DiT only predicts flow for the tokens it gets.")
        self.caption("LoT keeps the short sequence but predicts full-resolution flow for every patch.")
        self.end_section()


# ================================================================== S05 Asymmetric flow (toy)
def _toy():
    """Toy patch-wise asymmetric flow on the 4 × 4 grass token, per colour channel.
    A = patch mean (16 → 1, orthonormal), P = A Aᵀ; oracle head û_A = P ε − x₀ (Eq. 8); Eq. 9 recovers û."""
    i, j, b = GRASS
    x = CELLS[i:i + b, j:j + b].reshape(16, 3)
    x0 = (x - x.mean(0)) / (np.abs(x - x.mean(0)).max() + 1e-6) * 0.8 + (2 * x.mean(0) - 1)   # contrast-stretched
    rng = np.random.default_rng(11)
    eps = rng.standard_normal((16, 3)) * 0.8
    A = np.full((16, 1), 0.25)
    P = A @ A.T
    I = np.eye(16)

    def flow(xt, t):
        uA = P @ eps - x0                                  # oracle asymmetric target
        return P @ uA + (I - P) @ (xt + uA) / max(t, 1e-6)

    return x0, eps, A, P, flow


def disp(v):
    return rgb_to_color(np.clip((v + 1) / 2, 0, 1))


def tiles(vals, side, center, n=4, stroke=HR, sw=1.5):
    g = VGroup()
    center = np.array([center[0], center[1], 0.0])
    for k in range(n * n):
        r, c_ = divmod(k, n)
        g.add(tok(side * 0.9, disp(vals[k]), stroke, sw).move_to(
            center + [(c_ - (n - 1) / 2) * side, ((n - 1) / 2 - r) * side, 0]))
    return g


class S05_AsymFlow(Base):
    def construct(self):
        self.header("5 · Patch-wise asymmetric flow")
        x0, eps, A, P, flow = _toy()
        self.section("zoom_patch")
        field = LoTField(LEAVES, 2.6, (-5.0, 1.55), shade=True)
        hl = field.cell_rect(*GRASS, AMBER, sw=3).set_z_index(10)
        self.play(FadeIn(field.group()), run_time=0.5)
        self.play(Create(hl), run_time=0.3)
        self.caption("Zoom into one 4 × 4 token of grass: 16 fine positions, one token.")

        T = 0.6
        xt = (1 - T) * x0 + T * eps
        ts = 0.34
        y1 = -1.0
        p_xt = tiles(xt, ts, (-5.0, y1))
        l_xt = tx(f"noisy patch <i>x</i>" + sub("t") + f",  <i>t</i> = {T}", 22).next_to(p_xt, UP, buff=0.15)
        ztop = l_xt.get_top()[1] + 0.06
        zoom = VGroup(DashedLine(hl.get_corner(DL), [p_xt.get_left()[0], ztop, 0], color=AMBER, stroke_width=1.5),
                      DashedLine(hl.get_corner(DR), [p_xt.get_right()[0], ztop, 0], color=AMBER, stroke_width=1.5))
        self.play(Create(zoom), LaggedStart(*[FadeIn(t) for t in p_xt], lag_ratio=0.04), run_time=0.8)
        self.add(l_xt)
        self.end_section()

        # in: the token is Aᵀ x_t, one vector → DiT + LoRA → head
        self.section("token_dit_head")
        tokv = (A.T @ xt)[0] / 4.0                              # = patch mean (display scale)
        t_in = tok(0.62, disp(tokv), LR3, 2.5).move_to([-2.55, y1, 0])
        a1 = arrow(p_xt.get_right(), t_in.get_left(), sw=2)
        l_a1 = tx("<i>A</i>" + sub("e") + sup("⊤"), 24).next_to(a1, UP, buff=0.08)
        l_in = VGroup(tx("one token", 22), tx("(toy: <i>A</i>" + sub("e") + " = patch mean)", 18, DIM)
                      ).arrange(DOWN, buff=0.06).next_to(t_in, DOWN, buff=0.2)
        self.caption("The DiT sees the noisy patch only through its token: one vector, "
                     "<i>A</i>" + sub("e") + sup("⊤") + " <i>x</i>" + sub("t") + ".")
        self.play(GrowArrow(a1), FadeIn(t_in), run_time=0.6)
        self.add(l_a1, l_in)

        dit = dit_box([-0.55, y1, 0], w=1.3, h=1.6, lora=True)
        a2 = arrow(t_in.get_right(), dit.get_left(), sw=2)
        head = RoundedRectangle(width=1.1, height=0.78, corner_radius=0.08, fill_color=BOX, fill_opacity=1,
                                stroke_color=AMBER, stroke_width=2).move_to([1.35, y1, 0])
        head = VGroup(head, VGroup(tx("head", 20), tx("4 × 4", 20)).arrange(DOWN, buff=0.04).move_to(head))
        a3 = arrow(dit.get_right(), head.get_left(), sw=2)
        self.play(GrowArrow(a2), FadeIn(dit), run_time=0.5)
        self.play(GrowArrow(a3), FadeIn(head), run_time=0.4)
        self.end_section()

        # the equation
        self.section("eq9")
        eq = tx("<i>û</i> = " + col("<i>P</i>" + sub("e") + " <i>û</i>" + sub("A"), AMBER) + "  +  "
                + col("(<i>I</i> − <i>P</i>" + sub("e") + ") (<i>x</i>" + sub("t") + " + <i>û</i>" + sub("A")
                      + ") / σ" + sub("t"), HR), 34).move_to([2.9, 2.2, 0])
        e1 = tx("from the head", 20, AMBER)
        e2 = tx("completed from <i>x</i>" + sub("t"), 20, HR)
        self.caption("The head predicts an asymmetric flow <i>û</i>" + sub("A")
                     + ": low-rank noise, but full-rank clean data.")
        self.add(eq)
        tag = tx("Eq. 9", 20, DIM).next_to(eq, UP, buff=0.15)
        self.add(tag)
        self.end_section()

        self.section("eq9_terms")
        e1.next_to(eq, DOWN, buff=0.15).shift(LEFT * 1.1)
        e2.next_to(eq, DOWN, buff=0.15).shift(RIGHT * 1.35)
        self.add(e1, e2)
        self.caption("Eq. 9 fills in the rest of the flow from <i>x</i>" + sub("t") + " itself, patch by patch.")
        self.end_section()

        self.section("clean_estimate")
        u = flow(xt, T)
        x0hat = xt - T * u
        err = np.abs(u - (eps - x0)).max()
        p_out = tiles(x0hat, ts, (3.95, y1))
        a4 = arrow(head.get_right(), p_out.get_left(), sw=2)
        l_out = tx("clean estimate <i>x</i>" + sub("t") + " − <i>t û</i>", 22).next_to(p_out, UP, buff=0.15)
        mant, ex = f"{max(err, 1e-300):.0e}".split("e")
        l_err = tx("toy check: max |<i>û</i> − (ε − <i>x</i>" + sub("0") + f")| = {mant} × 10"
                   + sup(str(int(ex)).replace("-", "−")), 20, DIM).next_to(p_out, DOWN, buff=0.2)
        self.play(GrowArrow(a4), LaggedStart(*[FadeIn(t) for t in p_out], lag_ratio=0.04), run_time=0.9)
        self.add(l_out, l_err)
        self.caption("With a perfect head the flow is exact: all 16 positions get their own detail back.")
        self.end_section()

        # the sampler: from noise to the full-resolution patch
        self.section("sampler")
        self.uncaption()
        gone = [p_xt, l_xt, zoom, a1, l_a1, l_in, t_in, a2, a3, a4, dit, head, p_out, l_out, l_err]
        self.play(*[FadeOut(m) for m in gone], run_time=0.5)
        steps = [1.0, 0.75, 0.5, 0.25, 0.0]
        xs = eps.copy()
        tgrid = np.linspace(1, 0, 41)
        snaps = {1.0: xs.copy()}
        for a_, b_ in zip(tgrid[:-1], tgrid[1:]):
            xs = xs + (b_ - a_) * flow(xs, a_)
            for s_ in steps:
                if abs(b_ - s_) < 1e-9:
                    snaps[s_] = xs.copy()
        row = VGroup()
        for k, s_ in enumerate(steps):
            g_ = tiles(snaps[s_], 0.3, (-4.6 + k * 2.3, -1.45))
            g_.add(tx(f"<i>t</i> = {s_:g}", 22).next_to(g_, DOWN, buff=0.18))
            tk = tok(0.3, disp((A.T @ snaps[s_])[0] / 4.0), LR3, 2).next_to(g_[:16], UP, buff=0.3)
            g_.add(tk)
            row.add(g_)
        lrow = tx("toy sampler, 40 Euler steps: the 4 × 4 patch (bottom) and its token (top)", 22, DIM).move_to([0, -2.7, 0])
        self.add(lrow)
        self.caption("Run the sampler: every step reads one token and writes 16 positions.")
        for g_ in row:
            self.play(FadeIn(g_), run_time=0.45)
        self.caption("So the short sequence still produces a full-resolution image, not a blocky one.")
        foot = tx("toy: 4 × 4 patch per colour channel, <i>A</i> = patch mean, oracle head (the paper fits <i>A</i>"
                  + sub("e") + " to VAE tokens and trains the head)", 18, DIM).move_to([0, -3.08, 0])
        self.add(fit_width(foot, 12.8))
        self.end_section()


# ================================================================== S06 DiT
class S06_DiT(Base):
    def construct(self):
        self.header("6 · Telling the DiT what each token is")
        self.section("inputs")
        field = LoTField(LEAVES, 2.4, (-5.4, 0.6), shade=True)
        self.play(FadeIn(field.group()), run_time=0.5)
        # input heads per extent → DiT → output heads per extent
        ins = VGroup(*[small_box(f"in {b}×{b}", [-2.95, 1.65 - k * 0.7, 0], w=1.05, h=0.5, stroke=EXT_COL[b], size=18)
                       for k, b in enumerate((1, 2, 4, 8))])
        dit = dit_box([-0.4, 0.6, 0], w=1.9, h=2.6, lora=True, label="DiT")
        nb = tx("× <i>N</i> blocks", 20, DIM).next_to(dit, DOWN, buff=0.12)
        outs = VGroup(*[small_box(f"<i>h</i>" + sub(f"({b},{b})"), [2.15, 1.65 - k * 0.7, 0], w=1.05, h=0.5,
                                  stroke=EXT_COL[b], size=20) for k, b in enumerate((1, 2, 4, 8))])
        a_in = VGroup(*[arrow([field.group().get_right()[0] + 0.05, 0.6, 0], m.get_left(), sw=1.5, buff=0.05) for m in ins])
        a_mid = VGroup(*[arrow(m.get_right(), dit.get_left(), sw=1.5, buff=0.05) for m in ins])
        a_out = VGroup(*[arrow(dit.get_right(), m.get_left(), sw=1.5, buff=0.05) for m in outs])
        grid = Square(1.9).set_fill(HR, 0.25).set_stroke(HR, 2).move_to([4.9, 0.6, 0])
        glab = VGroup(tx("full-resolution", 20, DIM), tx("flow, 64 × 64", 20, DIM)).arrange(DOWN, buff=0.05).next_to(
            grid, DOWN, buff=0.15)
        a_end = VGroup(*[arrow(m.get_right(), grid.get_left(), sw=1.5, buff=0.05) for m in outs])
        self.caption("Inputs: each extent gets its own projection, the pretrained one composed with <i>A</i>"
                     + sub("e") + sup("⊤") + ".")
        self.play(LaggedStart(*[GrowArrow(a) for a in a_in], lag_ratio=0.1), FadeIn(ins), run_time=0.7)
        self.play(FadeIn(dit), LaggedStart(*[GrowArrow(a) for a in a_mid], lag_ratio=0.1), run_time=0.6)
        self.add(nb)
        self.end_section()

        self.section("outputs")
        self.caption("Outputs: one head per extent, initialised from the pretrained head as <i>W</i>" + sub("e")
                     + " = <i>A</i>" + sub("e") + " <i>W</i>" + sub("out") + ".")
        self.play(LaggedStart(*[GrowArrow(a) for a in a_out], lag_ratio=0.1), FadeIn(outs), run_time=0.7)
        self.play(LaggedStart(*[GrowArrow(a) for a in a_end], lag_ratio=0.1), FadeIn(grid), run_time=0.6)
        self.add(glab)
        self.end_section()

        # shape embedding
        self.section("shape_embedding")
        se = tx("<i>s</i>(<i>e</i>) = (log" + sub("2") + " <i>e</i>" + sub("h") + ",  log" + sub("2") + " <i>e</i>"
                + sub("w") + ",  log" + sub("2") + " <i>e</i>" + sub("h") + "<i>e</i>" + sub("w") + ",  log" + sub("2")
                + " <i>e</i>" + sub("h") + " / <i>e</i>" + sub("w") + ")", 28).move_to([0.2, 2.75, 0])
        sel = tx("patch-shape embedding, zero-initialised MLP, added to every input token", 20, DIM).next_to(
            se, DOWN, buff=0.1)
        self.caption("A shape embedding tells every token its size and aspect; it starts at zero, so nothing breaks.")
        self.add(se, sel)
        self.end_section()

        # RoPE at the patch centre
        self.section("rope")
        U = 0.3
        rp = VGroup()
        base = np.array([-3.2, -2.1, 0])
        cells = VGroup(*[Square(U).set_stroke(GHOST, 1).move_to(base + [k * U, 0, 0]) for k in range(-5, 9)],
                       *[Square(U).set_stroke(GHOST, 1).move_to(base + [k * U, -U, 0]) for k in range(-5, 9)],
                       *[Square(U).set_stroke(GHOST, 1).move_to(base + [k * U, U, 0]) for k in range(-5, 9)],
                       *[Square(U).set_stroke(GHOST, 1).move_to(base + [k * U, 2 * U, 0]) for k in range(-5, 9)])
        t1 = tok(U * 0.9, HR).move_to(base + [-4 * U, 0, 0])
        t2 = tok(2 * U - 0.03, LR).move_to(base + [-1.5 * U, 0.5 * U, 0])
        t4 = tok(4 * U - 0.03, LR3).move_to(base + [2.5 * U, 0.5 * U, 0])
        dots = VGroup(*[Dot(t.get_center(), radius=0.05, color=WHITE) for t in (t1, t2, t4)])
        rp.add(cells, t1, t2, t4, dots)
        rl = VGroup(tx("RoPE index at the patch centre", 22),
                    tx("<i>c</i> = (<i>u</i> + (<i>e</i>" + sub("h") + " − 1)/2,  <i>v</i> + (<i>e</i>" + sub("w")
                       + " − 1)/2)", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(cells, RIGHT, buff=0.4)
        self.caption("Positions: RoPE at each patch centre on the fine grid; a 1 × 1 token keeps its pretrained position.")
        self.play(FadeIn(rp), run_time=0.5)
        self.add(rl)
        # (the training caption had no visual of its own; with CAPTIONS = False it adds no section)
        self.caption("Trained: LoRA (rank 256), the heads and the shape MLP, on FLUX.2 klein 4B and Wan2.1 14B.")
        self.end_section()


# ================================================================== S07 Sources
class S07_Sources(Base):
    def construct(self):
        self.title_card("Part 3 · Results", "Layouts from anywhere")
        self.header("7 · Layouts from anywhere")
        self.section("fig4_deer")
        s, gap, pitch = 1.5, 0.06, 3.28
        names = [("bounding box", "1.73× · 1.54×"), ("semantic mask", "2.18× · 1.81×"),
                 ("texture variance", "1.36× · 1.28×"), ("depth", "3.11× · 2.29×")]
        grp = Group()
        for k, (n, v) in enumerate(names):
            cx = (k - 1.5) * pitch
            lay = img(gen(f"fig4_{2 * k}.png"), s).move_to([cx - (s + gap) / 2, 0.7, 0])
            out = img(gen(f"fig4_{2 * k + 1}.png"), s).move_to([cx + (s + gap) / 2, 0.7, 0])
            grp.add(lay, out)
            self.add(tx(n, 24).move_to([cx, 1.75, 0]))
            self.add(tx(v, 22, AMBER).move_to([cx, -0.3, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in grp], lag_ratio=0.08), run_time=1.0)
        self.add(tx("layout | generation, LoT-FLUX.2  (paper Fig. 4)", 20, DIM).move_to([0, 2.35, 0]))
        self.add(tx("compression · speedup", 20, DIM).move_to([0, -0.75, 0]))
        self.caption("Four layouts, four deer: fine where the layout is fine, smooth or defocused where it is coarse.")
        self.caption("The budget sets the speed: 3.11× fewer tokens from depth gives a 2.29× speedup.")
        self.end_section()

        # video: three site clips, layout over video
        self.section("video_clips")
        self.clear_all(run_time=0.5)
        h = 1.5
        specs = [("show03", "driving · semantic mask", "1.66×"), ("show02", "robotics · semantic mask", "1.50×"),
                 ("show14", "game · bounding box", "2.02×")]
        mobs = []
        for k, (nm, lab, cmp) in enumerate(specs):
            x = (k - 1) * 4.3
            lay, lf = load_clip(f"{nm}_lay", h)
            vid, vf = load_clip(f"{nm}_gen", h)
            lay.move_to([x, 1.3, 0])
            vid.move_to([x, -0.35, 0])
            mobs.append((lay, lf, vid, vf))
            self.add(tx(lab, 22).next_to(lay, UP, buff=0.15))
            self.add(tx(f"{cmp} fewer tokens", 20, AMBER).next_to(vid, DOWN, buff=0.15))
        self.play(*[FadeIn(m[0]) for m in mobs], *[FadeIn(m[2]) for m in mobs], run_time=0.5)
        for lay, lf, vid, vf in mobs:
            lay.add_updater(play_clip(lf))
            vid.add_updater(play_clip(vf))
        self.add(tx("layout (top) and generated video (bottom), LoT-Wan2.1 14B, 81 frames  (supplementary website)",
                    20, DIM).move_to([0, -1.85, 0]))
        self.caption("Video works the same way: fine tokens on the actors, coarse sky, road and walls.")
        n_fr = min(len(m[1]) for m in mobs)
        self.wait(n_fr / 15 + 0.1)              # play the 81-frame clips through; the section ends on their last frame
        for lay, lf, vid, vf in mobs:
            lay.clear_updaters()
            vid.clear_updaters()
        self.end_section()


# ================================================================== S08 Results
TAB1 = {   # method: [(speedup, HPSv3)] at compression 1.893×, 2.406×, 2.935× (paper Tab. 1)
    "ToMe-SD": [(1.538, 5.3569), (1.793, 2.9150), (2.003, 0.9253)],
    "DDiT": [(1.340, 9.0052), (1.521, 8.0549), (1.321, 7.4075)],
    "Foveated Diffusion": [(1.040, 9.0092), (1.100, 8.2674), (1.180, 7.8433)],
    "Ours (LoT)": [(1.560, 10.1211), (1.818, 9.7533), (2.040, 9.5522)],
}
TAB1_COL = {"ToMe-SD": RED, "DDiT": PURPLE, "Foveated Diffusion": FOV_C, "Ours (LoT)": HR}


class S08_Results(Base):
    def construct(self):
        self.header("8 · Results")
        self.section("tab1_plot")
        ax = Axes(x_range=[1.0, 2.1, 0.25], y_range=[6.5, 11, 1], x_length=6.2, y_length=4.3, tips=False,
                  axis_config={"include_numbers": False, "color": DIM, "stroke_width": 2}).move_to([-2.6, 0.35, 0])
        xt_ = VGroup(*[tx(f"{v:g}×", 20, DIM).next_to(ax.c2p(v, 6.5), DOWN, buff=0.12) for v in (1.0, 1.25, 1.5, 1.75, 2.0)])
        yt_ = VGroup(*[tx(f"{v}", 20, DIM).next_to(ax.c2p(1.0, v), LEFT, buff=0.12) for v in (7, 8, 9, 10, 11)])
        xl = tx("speedup vs full-resolution FLUX.2 4B", 22, DIM).next_to(ax, DOWN, buff=0.5)
        yl = tx("HPSv3 ↑", 22, DIM).next_to(ax.c2p(1.0, 11), UP, buff=0.4)
        self.play(Create(ax), run_time=0.5)
        self.add(xt_, yt_, xl, yl)
        full = Dot(ax.c2p(1.001, 10.4299), radius=0.08, color=WHITE)
        fl = tx("full resolution", 20).next_to(full, RIGHT, buff=0.12)
        self.add(full, fl)
        legend = VGroup()
        plots = []
        for m, pts in TAB1.items():
            c = TAB1_COL[m]
            vis = [(x, y) for x, y in pts if y >= 6.5]
            dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.07, color=c) for x, y in vis])
            line = VMobject(color=c, stroke_width=3).set_points_as_corners([ax.c2p(x, y) for x, y in vis]) \
                if len(vis) > 1 else VGroup()
            plots.append(VGroup(line, dots))
            legend.add(VGroup(Dot(radius=0.07, color=c), tx(m, 22, c, bold=(m == "Ours (LoT)"))).arrange(RIGHT, buff=0.12))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        legend.move_to([1.1, 1.6, 0], aligned_edge=LEFT)
        tome = tx("ToMe-SD: HPSv3 0.9–5.4, below the axis", 20, RED).next_to(legend, DOWN, buff=0.3).align_to(legend, LEFT)
        self.caption("Images, three token budgets: LoT is the fastest and scores closest to full resolution.")
        self.play(*[Create(p) for p in plots], run_time=1.2)
        self.add(legend, tome)
        budg = VGroup(tx("points: 1.89×, 2.41×, 2.94× compression", 20, DIM),
                      tx("same budget for every method (paper Tab. 1)", 20, DIM)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.06).next_to(tome, DOWN, buff=0.3).align_to(legend, LEFT)
        self.add(budg)
        self.end_section()

        self.section("tab2_video")
        vid = VGroup(tx("video, Wan2.1 14B, 3× budget (Tab. 2)", 22),
                     tx("speedup: " + col("ours 3.53×", HR) + " · ToMe-SD 3.31× · Foveated 3.28×", 20, DIM),
                     tx("imaging: full 0.647 · " + col("ours 0.565", HR) + " · Foveated 0.495 · ToMe-SD 0.421", 20, DIM))
        for v in vid:
            fit_width(v, 5.7)
        vid.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(budg, DOWN, buff=0.4).align_to(legend, LEFT)
        self.add(vid)
        self.caption("Video, same story: 3.53× faster, with the best imaging quality of the compressed methods.")
        self.end_section()

        # Fig. 7a (image) and Fig. 7b (video) at the same budget
        self.section("fig7a_image")
        self.clear_all(run_time=0.5)
        s, pitch, y = 2.3, 2.52, 0.55
        names = [("Full res. FLUX.2", "1.00×", ""), ("ToMe-SD", "2.10×", "noisy low-res regions"),
                 ("DDiT", "1.90×", "noisy low-res regions"), ("Foveated Diffusion", "1.63×", "distorted at token borders"),
                 ("Ours (LoT)", "2.29×", "")]
        head = tx("image, same budget: 1,269 tokens  (paper Fig. 7a)", 22, DIM).move_to([0, 2.55, 0])
        self.add(head)
        ims = [img(gen(f"fig7a_{k}.png"), s).move_to([(k - 2) * pitch, y, 0]) for k in range(5)]
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.12), run_time=0.8)
        for k, (n, sp, bad) in enumerate(names):
            c = HR if k == 4 else WHITE
            self.add(fit_width(tx(n, 22, c, bold=(k == 4)), s + 0.1).next_to(ims[k], UP, buff=0.14))
            self.add(tx(f"{sp} speedup", 22, c if k == 4 else DIM).next_to(ims[k], DOWN, buff=0.14))
            if bad:
                self.add(fit_width(tx(bad, 20, RED), s).next_to(ims[k], DOWN, buff=0.5))
        self.add(framed(ims[4], HR, 3))
        self.caption("Same token budget: merging and pruning blur or break; LoT keeps the detail.")
        self.end_section()

        self.section("fig7b_video")
        self.clear_all(run_time=0.5)
        h = 1.05
        vnames = [("Full res. Wan2.1 14B", "1.00×"), ("ToMe-SD", "1.60×"), ("Foveated Diffusion", "1.59×"),
                  ("Ours (LoT)", "1.62×")]
        head = tx("video, same budget: 50,134 tokens  (paper Fig. 7b)", 22, DIM).move_to([0, 2.55, 0])
        self.add(head)
        rows = Group()
        for m, (n, sp) in enumerate(vnames):
            yy = 1.6 - m * 1.25
            fr = Group(*[img(gen(f"fig7b_{m}_{k}.png"), h) for k in range(3)]).arrange(RIGHT, buff=0.06).move_to([1.3, yy, 0])
            rows.add(fr)
            c = HR if m == 3 else WHITE
            lab = VGroup(tx(n, 22, c, bold=(m == 3)), tx(f"{sp} speedup", 20, c if m == 3 else DIM)).arrange(
                DOWN, aligned_edge=RIGHT, buff=0.06).next_to(fr, LEFT, buff=0.3)
            self.add(lab)
            if m == 3:
                self.add(framed(fr, HR, 3))
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.15), run_time=0.8)
        self.add(tx("noisy", 20, RED).next_to(rows[1], RIGHT, buff=0.2))
        self.caption("Video at the same budget: the baselines turn noisy; LoT stays clean and is the fastest.")
        self.end_section()

        # ablation (Tab. 3, HPSv3, ≈ 1.52× compression)
        self.section("ablation")
        self.clear_all(run_time=0.5)
        abl = [("direct HR prediction (no Eq. 9)", 7.0595), ("binary grid (1 × 1 and 2 × 2 only)", 8.1955),
               ("no size embedding", 10.0284), ("full LoT", 10.3477)]
        x0, W = -1.2, 6.0
        self.add(tx("ablation, HPSv3 ↑ at ≈ 1.52× compression  (paper Tab. 3)", 22, DIM).move_to([0, 2.3, 0]))
        bars = []
        for k, (n, v) in enumerate(abl):
            yy = 1.3 - k * 0.95
            c = HR if k == 3 else GHOST
            self.add(tx(n, 24, HR if k == 3 else WHITE).next_to([x0 - 0.2, yy, 0], LEFT, buff=0))
            b = Rectangle(width=W * v / 11, height=0.5, stroke_width=0).set_fill(c if k == 3 else "#5A5A5A", 1)
            b.move_to([x0 + W * v / 22, yy, 0])
            bars.append((b, v, yy))
        self.play(*[GrowFromEdge(b, LEFT) for b, _, _ in bars], run_time=0.8)
        for b, v, yy in bars:
            self.add(tx(f"{v:.2f}", 22).next_to(b, RIGHT, buff=0.15))
        self.caption("Take out the asymmetric flow, the multi-level grid or the size embedding, and quality drops.")
        self.end_section()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.header("9 · Limitations and outlook")
        s = 2.6
        ims = [img(os.path.join(IMG, f"fig22_{k}.png"), s) for k in range(4)]
        xs = [-4.6, -1.9, 1.9, 4.6]
        for m, x in zip(ims, xs):
            m.move_to([x, 0.5, 0])
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.1), run_time=0.8)
        self.add(tx("higher budget: 3,189 tokens (1.28×)", 22).move_to([-3.25, 2.1, 0]))
        self.add(tx("lower budget: 1,466 tokens (2.79×)", 22).move_to([3.25, 2.1, 0]))
        self.add(tx("broken staff lines", 22, RED).next_to(ims[3], DOWN, buff=0.15))
        self.add(tx("layout | generation, same prompt and seed  (paper Fig. 22)", 20, DIM).move_to([0, -1.75, 0]))
        self.caption("Detail everywhere leaves little to save; forcing a low budget breaks fine structure.")
        self.clear_all(run_time=0.5)

        # outlook: the layout as an editable plan
        cm = img(os.path.join(IMG, "paint0_canvas.png"), 2.6).move_to([-4.4, 0.5, 0])
        field = LoTField(LEAVES, 2.6, (0, 0.5), shade=True)
        out = img(RESULT, 2.6).move_to([4.4, 0.5, 0])
        a1 = arrow(cm.get_right(), field.group().get_left(), sw=2)
        a2 = arrow(field.group().get_right(), out.get_left(), sw=2)
        self.add(tx("plan", 22, DIM).next_to(cm, UP, buff=0.15), tx("compute", 22, DIM).next_to(field.group(), UP, buff=0.15),
                 tx("image", 22, DIM).next_to(out, UP, buff=0.15))
        self.play(FadeIn(cm), FadeIn(framed(cm, GHOST)), run_time=0.4)
        self.play(GrowArrow(a1), FadeIn(field.group()), run_time=0.6)
        self.play(GrowArrow(a2), FadeIn(out), run_time=0.6)
        self.caption("Layouts could become an editable plan: an agent or a user decides where the compute goes.")
        self.clear_all(run_time=0.5)

        t = tx("Level-of-Token Diffusion in three ideas", 40, bold=True).move_to([0, 2.4, 0])
        ideas = VGroup(tx("1.  A layout of variable-size tokens: fine where detail matters, coarse elsewhere", 26),
                       tx("2.  Patch-wise asymmetric flow: full-resolution flow from a short sequence", 26),
                       tx("3.  An extent-aware DiT: per-size heads, shape embedding, patch-centre RoPE", 26)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.5, 0])
        for i_ in ideas:
            fit_width(i_, 12.6)
        self.add(t)
        for i_ in ideas:
            self.add(i_)
            self.wait(1.8)
        big = tx("up to 2.04× faster images  ·  3.53× faster video", 32, HR).move_to([0, -1.5, 0])
        self.add(big)
        self.wait(2.5)
        self.clear_all(run_time=0.5)
        self.remove(self.hdr)

        emb = LoTField(LEAVES, 1.9, (0, 2.3), shade=True)
        t = tx("Level-of-Token Diffusion", 64, bold=True).move_to([0, 0.75, 0])
        tag = fit_width(tx("fine tokens where detail matters  ·  up to 2.04× faster images  ·  3.53× faster video",
                           30, HR), 12.8).move_to([0, -0.15, 0])
        a = tx("Anonymous ICLR 2027 submission 7014", 24).move_to([0, -1.2, 0])
        u = tx("iclr-submission7014.github.io/supplementary", 26, DIM).move_to([0, -1.85, 0])
        self.add(emb.group(), t, tag, a, u)
        self.wait(4.5)
