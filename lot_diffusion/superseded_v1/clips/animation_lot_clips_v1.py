"""Level-of-Token Diffusion — 10-second thread clips, v1 (3b1b style, one Scene per clip).

Every clip is exactly 10.0 s, starts on an informative first frame (X autoplay/thumbnail) and carries a small
"Level-of-Token Diffusion" tag. Numbers on screen come from the paper (7014_Level_of_Token_Diffusion.pdf) or the
supplementary site manifests mirrored in ../web/. Storyboard: clips/storyboard.md.

Concept clips  C01_EqualCompute  C02_Lineage  C03_AnySize  C04_Sources  C05_AsymFlow  C06_ShapeRoPE  C07_Budget
               C08_Numbers
Result clips   R01_PaintGUI  R02_Agent  R03_Scene3D  R04_Games  R05_Driving  R06_Robotics  R07_DepthOfField
               R08_MoveTheBox  R09_Typography  R10_Boxes  R11_VideoBaselines
Render: ./render.sh l|h [script] [Scene ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lot_common import *  # noqa: E402,F403

CLIP_LEN = 10.0
HEAD_Y = 3.42


# ------------------------------------------------------------------ helpers shared by the clips
class Clip(Base):
    def setup(self):
        super().setup()
        self.head = None
        self.add(brand())

    def headline(self, s, size=34, y=HEAD_Y):
        if self.head is not None:
            self.remove(self.head)
        self.head = tx(s, size)
        if self.head.width > 13.4:
            self.head.scale_to_fit_width(13.4)
        self.head.move_to([0, y, 0])
        self.add(self.head)
        return self.head

    def finish(self):
        # t0 lets the teasers replay a clip body mid-scene
        self.pad_to(getattr(self, "t0", 0.0) + CLIP_LEN + getattr(self, "extra", 0.0))


def runs(mask, min_len=40):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask)))
    return out


def panel_boxes(path, thr=245, frac=0.3, min_len=60):
    """split a figure raster into panels by near-white gutters"""
    a = np.asarray(Image.open(path).convert("L")).astype(int)
    cols = runs((a < thr).mean(0) > frac, min_len)
    rows = runs((a < thr).mean(1) > frac, min_len)
    return rows, cols


def crop_panel(path, r, c, name):
    out = os.path.join(GEN, name)
    if not os.path.exists(out):
        rows, cols = panel_boxes(path)
        (y0, y1), (x0, x1) = rows[r], cols[c]
        Image.open(path).convert("RGB").crop((x0 + 2, y0 + 2, x1 - 2, y1 - 2)).save(out)
    return out


def cell_center(center, width, H, W, i, j):
    cw = width / W
    return np.array([center[0] - width / 2 + (j + 0.5) * cw, center[1] + cw * H / 2 - (i + 0.5) * cw, 0])


def rect_mob(center, width, H, W, r, color=WHITE, sw=3):
    """outline around lattice rect r = (y0, y1, x0, x1) of a layout drawn at `center` with `width`"""
    cw = width / W
    y0, y1, x0, x1 = r
    m = Rectangle(width=(x1 - x0) * cw, height=(y1 - y0) * cw).set_stroke(color, sw).set_fill(opacity=0)
    m.move_to([center[0] - width / 2 + (x0 + x1) / 2 * cw, center[1] + cw * H / 2 - (y0 + y1) / 2 * cw, 0])
    return m


class LayoutVideo:
    """layout that changes with the video's latent time slices (Wan: slice 0 = frame 0, then 4 frames/slice)"""

    def __init__(self, layouts, card, mode="level", gap=0.1):
        self.layouts, self.card = layouts, card
        px = int(min(card.width * px_per_unit(), 1600))
        self.frames = {}
        self.px, self.mode, self.gap = px, mode, gap
        self.img = ImageMobject(self._arr(0))
        self.img.set_width(card.width).move_to(card.pos())
        self.last = None
        self.img.add_updater(self._tick)

    def _arr(self, k):
        if k not in self.frames:
            self.frames[k] = draw_layout(self.layouts[k], self.px, mode=self.mode, gap=self.gap, bg=(12, 12, 12, 255))
        return self.frames[k]

    def _tick(self, m, dt):
        i = self.card.index()
        k = min(len(self.layouts) - 1, (i + 3) // 4)
        if k != self.last:
            m.pixel_array = self._arr(k)
            self.last = k
        m.move_to(self.card.pos())


def class_legend(items, size=18, per_row=3):
    chips = []
    for itx in items:
        c = "#%02X%02X%02X" % tuple(itx["color_rgb"])
        sq = Square(0.2).set_fill(c, 1).set_stroke(DIM, 0.8)
        chips.append(VGroup(sq, tx(f"{itx['label']} · {itx['token_level']}", size, DIM)).arrange(RIGHT, buff=0.1))
    rows = [VGroup(*chips[k:k + per_row]).arrange(RIGHT, buff=0.35) for k in range(0, len(chips), per_row)]
    return VGroup(*rows).arrange(DOWN, buff=0.12, aligned_edge=LEFT)


def showcase_meta(key):
    for r in json.load(open(os.path.join(CLIPS, "showcase_index.json"))):
        if r["key"] == key:
            return r


def yellow_box(path):
    """fractional bbox of the yellow 1x1-token box drawn in a Fig. 19 grid panel"""
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    m = (a[..., 0] > 200) & (a[..., 1] > 150) & (a[..., 2] < 110)
    ys, xs = np.nonzero(m)
    H, W = m.shape
    return ys.min() / H, ys.max() / H, xs.min() / W, xs.max() / W


def box_layout(fb, n=64):
    """Fig. 19 layouts: 1x1 tokens inside the box, 2x2 elsewhere (box snapped to the 2x2 lattice)"""
    y0, y1, x0, x1 = [int(round(v * n / 2)) * 2 for v in fb]
    rects = []
    for y in range(0, n, 2):
        for x in range(0, n, 2):
            if y0 <= y < y1 and x0 <= x < x1:
                rects += [(y + dy, y + dy + 1, x + dx, x + dx + 1) for dy in (0, 1) for dx in (0, 1)]
            else:
                rects.append((y, y + 2, x, x + 2))
    return Layout(rects, n, n), (y0, y1, x0, x1)


# ================================================================== concept clips
class C01_EqualCompute(Clip):
    """Uniform 64x64 tokens -> the paper's semantic-mask LoT layout for the same dog image (1,351 tokens)."""

    def construct(self):
        src = os.path.join(IMG, "img_000000000042_sam3_mask_generated.png")
        L0, L1 = Layout.uniform(64, 64), load_layout("img_000000000042_sam3_mask")
        W_, C_L, C_R = 5.3, np.array([-3.35, 0.0, 0]), np.array([3.35, 0.0, 0])
        pic = img(src, h=W_).move_to(C_L)
        M = LayoutMorph(L0, L1, W_, center=C_R, mode="image", image=src, gap=0.1)
        self.headline("A diffusion transformer pays the same for every patch")
        v = ValueTracker(4096)
        cnt = self.counter(v, lambda x: f"{fmt(x)} tokens", 32, pos=[C_R[0], -3.05, 0])
        self.add(pic, framed(pic), M.img, cnt)
        self.wait(0.9)
        # one background token vs one zebra token: same cost
        fine = np.array([r for r in L1.rects if (r[1] - r[0]) == 1])
        cy, cx = np.median(fine[:, 0]), np.median(fine[:, 2])
        fy, _, fx, _ = fine[np.argmin((fine[:, 0] - cy) ** 2 + (fine[:, 2] - cx) ** 2)]
        marks = VGroup()
        for c in (C_L, C_R):
            marks.add(rect_mob(c, W_, 64, 64, (fy, fy + 1, fx, fx + 1), AMBER, 3).scale(2.2))
            marks.add(rect_mob(c, W_, 64, 64, (4, 5, 4, 5), TEAL, 3).scale(2.2))
        self.play(Create(marks), run_time=0.5)
        self.caption(f"{col('empty background', TEAL)} costs exactly as much as {col('the dog’s fur', AMBER)}",
                     hold=False)
        self.wait(1.4)
        self.remove(marks)
        self.uncaption(hold=False)
        self.headline("Level-of-Token Diffusion: coarse tokens where detail is not needed", 32)
        self.play(M.t.animate.set_value(1), v.animate.set_value(L1.L), run_time=3.0)
        lev = layout_mobject(L1, W_, mode="level", gap=0.1).move_to(C_R)
        leg = legend_levels(size=20).move_to([C_L[0], -3.05, 0])
        self.play(FadeIn(lev), FadeIn(leg), run_time=0.7)
        self.caption(f"{fmt(L1.L)} tokens instead of 4,096: {L1.compression:.1f}× fewer, same 1024 × 1024 image",
                     hold=False, y=3.42 - 7.0)
        self.remove(cnt)
        self.finish()


class C02_Lineage(Clip):
    """LOD / variable-rate shading / compression adapt to existing content; generation can plan detail."""

    def construct(self):
        self.headline("Graphics has long spent detail only where it is needed")
        xs = [-4.6, 0.0, 4.6]
        y = 0.45
        # 1) level of detail: dense near mesh, sparse far mesh
        def mesh(r, n, c):
            pts = [c + r * np.array([np.cos(2 * PI * k / n), np.sin(2 * PI * k / n), 0]) for k in range(n)]
            g = VGroup(Polygon(*pts).set_stroke(TEAL, 2))
            inner = [c + 0.55 * (p - c) for p in pts[::2]]
            g.add(Polygon(*inner).set_stroke(TEAL, 1.2, 0.8))
            for k, p in enumerate(pts):
                g.add(Line(c, p).set_stroke(TEAL, 1, 0.6))
            return g
        p1 = VGroup(mesh(1.05, 28, np.array([xs[0] - 0.55, y + 0.05, 0])),
                    mesh(0.42, 7, np.array([xs[0] + 1.25, y + 0.55, 0])))
        t1 = tx("Level of detail · 1976", 24, DIM).move_to([xs[0], y - 1.75, 0])
        # 2) variable-rate shading: shading-rate tiles around a detailed centre
        yy, xx = np.mgrid[0:24, 0:32]
        det = np.exp(-(((yy - 12) / 7.0) ** 2 + ((xx - 15) / 9.0) ** 2))
        vrs = quadtree_from_detail(det, thresholds=(0.12, 0.35, 0.7), top=8)
        p2 = layout_mobject(vrs, 3.7, gap=0.12).move_to([xs[1], y, 0])
        t2 = tx("Variable-rate shading · 2014", 24, DIM).move_to([xs[1], y - 1.75, 0])
        # 3) image compression: quadtree over an image
        ipath = os.path.join(IMG, "f10_alps_ref_img.png")
        a = np.asarray(Image.open(ipath).convert("L").resize((128, 128), Image.BOX)).astype(float) / 255
        blk = a.reshape(32, 4, 32, 4).std(axis=(1, 3))
        qt = quadtree_from_detail(np.clip(blk * 9, 0, 1), thresholds=(0.25, 0.45, 0.7), top=8)
        pim = img(ipath, h=2.75).move_to([xs[2], y, 0])
        p3 = Group(pim, layout_mobject(qt, 2.75, mode="outline", line="#FFFFFF", gap=0.0).move_to(pim))
        t3 = tx("Image compression · 1993", 24, DIM).move_to([xs[2], y - 1.75, 0])
        self.add(p1, t1)
        self.play(FadeIn(p2), FadeIn(t2), run_time=0.5)
        self.wait(0.3)
        self.play(FadeIn(p3), FadeIn(t3), run_time=0.5)
        self.caption("…but each of them adapts to content that already exists", hold=False)
        self.wait(1.9)
        # generation can plan detail before the first pixel
        top = Group(p1, t1, p2, t2, p3, t3)
        self.uncaption(hold=False)
        self.play(top.animate.scale(0.5).move_to([0, 2.05, 0]), FadeOut(self.head), run_time=0.8)
        self.head = None
        names = ["bbox_living-room_boxes.png", "bbox_living-room_layout.png", "bbox_living-room_generated.png"]
        labs = ["a plan (boxes + detail)", "LoT token layout", "generated image"]
        row = Group()
        for k, (n, l) in enumerate(zip(names, labs)):
            m = img(n, w=3.5).move_to([-4.45 + 4.45 * k, -0.95, 0])
            row.add(Group(m, framed(m), tx(l, 22, DIM).next_to(m, DOWN, buff=0.12)))
        arrs = VGroup(arrow(row[0][0].get_right(), row[1][0].get_left(), buff=0.12),
                      arrow(row[1][0].get_right(), row[2][0].get_left(), buff=0.12))
        self.play(FadeIn(row[0]), run_time=0.4)
        self.play(GrowArrow(arrs[0]), FadeIn(row[1]), run_time=0.5)
        self.play(GrowArrow(arrs[1]), FadeIn(row[2]), run_time=0.5)
        self.caption("Generation can plan detail <i>before</i> the first pixel exists: turn the plan into compute",
                     hold=False, size=27)
        self.finish()


class C03_AnySize(Clip):
    """Uniform 45x80 -> a real texture-variance video layout (show22, slice 0) with anisotropic tokens."""

    def construct(self):
        L1 = load_layout("show22", 0)
        L0 = Layout.uniform(L1.H, L1.W)
        W_ = 8.0
        C = np.array([-2.55, 0.25, 0])
        M = LayoutMorph(L0, L1, W_, center=C, mode="level", gap=0.1)
        self.headline("Level-of-Token layouts: tokens of any size and shape")
        v = ValueTracker(L0.L)
        cnt = self.counter(v, lambda x: f"L = {fmt(x)} tokens", 30, pos=[C[0], C[1] - W_ * L1.H / L1.W / 2 - 0.4, 0])
        self.add(M.img, cnt)
        self.wait(0.5)
        self.play(M.t.animate.set_value(1), v.animate.set_value(L1.L), run_time=2.6)
        # zoom window around an anisotropic token
        pick = None
        for r in L1.rects:
            eh, ew = r[1] - r[0], r[3] - r[2]
            if (eh, ew) in ((2, 4), (4, 2), (4, 1), (1, 4)) and 3 <= r[0] <= L1.H - 8 and 4 <= r[2] <= L1.W - 12:
                pick = r
                break
        wy0, wx0 = max(pick[0] - 2, 0), max(pick[2] - 3, 0)
        win = (wy0, wy0 + 7, wx0, wx0 + 10)
        wl = Layout([(max(a, win[0]) - win[0], min(b, win[1]) - win[0], max(c, win[2]) - win[2],
                       min(d, win[3]) - win[2])
                      for (a, b, c, d) in L1.rects if a < win[1] and b > win[0] and c < win[3] and d > win[2]],
                     7, 10)
        ZW, ZC = 4.0, np.array([4.75, 1.2, 0])
        zoom = layout_mobject(wl, ZW, gap=0.06).move_to(ZC)
        wr = rect_mob(C, W_, L1.H, L1.W, win, WHITE, 2.5)
        zr = Rectangle(width=zoom.width, height=zoom.height).move_to(ZC).set_stroke(WHITE, 2.5)
        links = VGroup(DashedLine(wr.get_corner(UR), zr.get_corner(UL), dash_length=0.08).set_stroke(DIM, 1.5),
                       DashedLine(wr.get_corner(DR), zr.get_corner(DL), dash_length=0.08).set_stroke(DIM, 1.5))
        self.play(Create(wr), run_time=0.3)
        self.play(FadeIn(zoom), Create(zr), Create(links), run_time=0.6)
        # extent labels on the picked token
        pr = (pick[0] - win[0], pick[1] - win[0], pick[2] - win[2], pick[3] - win[2])
        hl = rect_mob(ZC, ZW, 7, 10, pr, WHITE, 4)
        eh, ew = pick[1] - pick[0], pick[3] - pick[2]
        lh = tx(f"{it('e')}{sub('h')} = {eh}", 22).next_to(zr, RIGHT, buff=0.1).set_y(hl.get_y()).shift(LEFT * 0.0)
        lh.next_to(zr, DOWN, buff=0.1).set_x(hl.get_x())
        lw = tx(f"{it('e')}{sub('w')} = {ew}", 22).next_to(zr, UP, buff=0.08).set_x(hl.get_x())
        lh = tx(f"{it('e')}{sub('h')} = {eh}, {it('e')}{sub('w')} = {ew}", 22, AMBER).next_to(zr, DOWN, buff=0.1)
        lw = VGroup()
        hl.set_stroke(AMBER, 4)
        self.play(Create(hl), FadeIn(lh), run_time=0.5)
        eq1 = tx(f"{it('R')}{sub('i')} = [{it('u')}{sub('i')}, {it('u')}{sub('i')} + {it('e')}{sub('h,i')}) × "
                 f"[{it('v')}{sub('i')}, {it('v')}{sub('i')} + {it('e')}{sub('w,i')})", 28)
        eq1.scale_to_fit_width(min(eq1.width, 4.6)).move_to([4.75, -1.35, 0])
        eq2 = tx(f"{it('e')}{sub('h')}, {it('e')}{sub('w')} ∈ {{1, 2, 4, 8}} images · {{1, 2, 4}} video", 24,
                 DIM).next_to(eq1, DOWN, buff=0.22)
        eq2.scale_to_fit_width(min(eq2.width, 4.6))
        self.add(eq1, eq2)
        self.wait(1.3)
        comp = tx(f"compression {it('HW')} / {it('L')} = {fmt(L1.dense)} / {fmt(L1.L)} = {L1.compression:.2f}×", 28)
        comp.move_to([0, -3.35, 0])
        self.remove(cnt)
        self.add(comp)
        self.finish()


class C04_Sources(Clip):
    """Paper Fig. 4, deer row: bounding box / semantic mask / texture variance / depth -> layout -> generation."""

    def construct(self):
        fig = os.path.join(IMG, "fig4_full.png")
        names = ["Bounding box", "Semantic mask", "Texture variance", "Depth"]
        stats = [("1.73", "1.54"), ("2.18", "1.81"), ("1.36", "1.28"), ("3.11", "2.29")]
        self.headline("Where should detail go? Any spatial cue can say · one model, same prompt", 30)
        xs = [-5.3, -1.77, 1.77, 5.3]
        cols = Group()
        for k in range(4):
            lay = img(crop_panel(fig, 0, 2 * k, f"fig4_r0_c{2 * k}.png"), w=2.8).move_to([xs[k], 1.2, 0])
            gen = img(crop_panel(fig, 0, 2 * k + 1, f"fig4_r0_c{2 * k + 1}.png"), w=2.8).move_to([xs[k], -1.75, 0])
            title = tx(names[k], 28, bold=True).next_to(lay, UP, buff=0.14)
            ar = arrow(lay.get_bottom(), gen.get_top(), buff=0.04, sw=2.5, tip=0.12)
            st = VGroup(tx(f"{stats[k][0]}× fewer tokens", 20, FINE), tx(f"≈ {stats[k][1]}× faster", 20, DIM))
            st.arrange(DOWN, buff=0.06).next_to(gen, DOWN, buff=0.12)
            cols.add(Group(title, lay, framed(lay), ar, gen, framed(gen), st))
        self.add(cols[0][:3])
        self.wait(0.3)
        self.play(FadeIn(Group(*cols[0][3:])), run_time=0.5)
        for k in range(1, 4):
            self.play(FadeIn(Group(*cols[k][:3])), run_time=0.45)
            self.play(FadeIn(Group(*cols[k][3:])), run_time=0.45)
        self.finish()


class C05_AsymFlow(Clip):
    """Patch-wise asymmetric flow: big patch -> one token -> DiT -> extent head -> dense full-rank velocity."""

    def construct(self):
        self.headline("Fewer tokens in, full-resolution flow out")
        rng = np.random.default_rng(3)
        # latent x_t on an 8x8 lattice with a small LoT layout
        lay = Layout([(0, 4, 0, 4), (0, 2, 4, 6), (0, 2, 6, 8), (2, 4, 4, 6), (2, 4, 6, 8), (4, 8, 0, 2), (4, 6, 2, 4),
                      (6, 8, 2, 4)] + [(y, y + 1, x, x + 1) for y in range(4, 8) for x in range(4, 8)], 8, 8)
        g = 2.5
        cw = g / 8
        C0 = np.array([-5.6, 0.55, 0])
        cells = VGroup()
        for i in range(8):
            for j in range(8):
                v = np.clip(0.45 + 0.18 * rng.standard_normal(), 0.1, 0.9)
                cells.add(Square(cw * 0.92).set_fill(rgb_to_color([v, v, v]), 1).set_stroke(width=0)
                          .move_to(cell_center(C0, g, 8, 8, i, j)))
        outl = VGroup(*[rect_mob(C0, g, 8, 8, r, level_hex(r[1] - r[0], r[3] - r[2]), 2.5) for r in lay.rects])
        lx = tx(f"noisy latent {it('x')}{sub(it('t'))}", 22, DIM).next_to(cells, DOWN, buff=0.15)
        self.add(cells, outl, lx)
        self.wait(0.4)
        # take the 2x2 token (rows 0-2, cols 4-6): 4 dense sites -> A_e^T -> one token
        pr = (0, 2, 4, 6)
        sel = rect_mob(C0, g, 8, 8, pr, WHITE, 4)
        self.play(Create(sel), run_time=0.3)
        col4 = VGroup(*[Square(0.28).set_fill(cells[i * 8 + j].get_fill_color(), 1).set_stroke(WHITE, 1)
                        for i in (0, 1) for j in (4, 5)]).arrange(DOWN, buff=0.0).move_to([-3.55, 1.35, 0])
        l4 = tx(f"{it('x')}{sup('(i)')}{sub(it('t'))} ∈ ℝ{sup('4' + it('D'))}", 20).next_to(col4, UP, buff=0.1)
        proj = rbox(0.95, 0.62, AMBER).move_to([-2.45, 1.35, 0])
        pl = tx(f"{it('A')}{sub(it('e'))}{sup('⊤')}", 26).move_to(proj)
        tok = Square(0.34).set_fill(level_hex(2, 2), 1).set_stroke(WHITE, 1).move_to([-1.5, 1.35, 0])
        lt = tx(f"ℝ{sup(it('D'))}", 20).next_to(tok, UP, buff=0.1)
        self.play(TransformFromCopy(sel, col4), FadeIn(l4), run_time=0.6)
        self.play(GrowArrow(a1 := arrow(col4.get_right(), proj.get_left(), buff=0.06, sw=2.5, tip=0.12)),
                  FadeIn(proj), FadeIn(pl), run_time=0.35)
        self.play(GrowArrow(a2 := arrow(proj.get_right(), tok.get_left(), buff=0.06, sw=2.5, tip=0.12)),
                  FadeIn(tok), FadeIn(lt), run_time=0.35)
        # the whole reduced sequence goes through the pretrained DiT
        seq = VGroup(*[Square(0.2).set_fill(level_hex(r[1] - r[0], r[3] - r[2]), 1).set_stroke(width=0)
                       for r in lay.rects]).arrange(RIGHT, buff=0.05).move_to([-0.2, -0.35, 0])
        sl = tx(f"{it('L')} = 24 tokens instead of 64", 20, DIM).next_to(seq, DOWN, buff=0.12)
        dit = rbox(1.5, 1.35, WHITE).move_to([0.85, 1.35, 0])
        dl = VGroup(tx("DiT", 30, bold=True), tx("FLUX.2 / Wan2.1", 16, DIM)).arrange(DOWN, buff=0.08).move_to(dit)
        self.play(FadeIn(seq), FadeIn(sl), FadeIn(dit), FadeIn(dl),
                  GrowArrow(a3 := arrow(tok.get_right(), dit.get_left(), buff=0.06, sw=2.5, tip=0.12)), run_time=0.5)
        self.play(seq.animate.scale(0.35).move_to(dit.get_center()).set_opacity(0), FadeOut(sl), run_time=0.5)
        # extent head h_(2,2) -> dense asymmetric velocity (4 sites)
        head = rbox(1.0, 0.62, MID).move_to([2.55, 1.35, 0])
        hl = tx(f"{it('h')}{sub('(2,2)')}", 24).move_to(head)
        ua = VGroup(*[Square(0.28).set_fill(rgb_to_color([0.35 + 0.1 * k, 0.5, 0.75]), 1).set_stroke(WHITE, 1)
                      for k in range(4)]).arrange(DOWN, buff=0.0).move_to([3.65, 1.35, 0])
        ul = tx(f"û{sub('A')}{sup('(i)')}", 22).next_to(ua, UP, buff=0.1)
        self.play(GrowArrow(a4 := arrow(dit.get_right(), head.get_left(), buff=0.06, sw=2.5, tip=0.12)),
                  FadeIn(head), FadeIn(hl), run_time=0.35)
        self.play(GrowArrow(a5 := arrow(head.get_right(), ua.get_left(), buff=0.06, sw=2.5, tip=0.12)),
                  FadeIn(ua), FadeIn(ul), run_time=0.35)
        eq = tx(f"û{sup('(i)')} = {it('P')}{sub(it('e'))} û{sub('A')} + ({it('I')} − {it('P')}{sub(it('e'))}) "
                f"({it('x')}{sub(it('t'))} + û{sub('A')}) / σ{sub(it('t'))}", 36)
        eq.move_to([1.2, -1.35, 0])
        eqn = tx("asymmetric velocity → full-rank velocity, patch by patch", 22, DIM).next_to(eq, DOWN, buff=0.2)
        self.add(eq, eqn)
        # dense full-resolution velocity grid lights up
        C1 = np.array([5.65, 1.35, 0])
        vel = VGroup(*[Square(0.23).set_fill(rgb_to_color([0.25 + 0.05 * ((i + j) % 5), 0.55, 0.85]), 1)
                       .set_stroke(width=0).move_to(C1 + np.array([(j - 3.5) * 0.25, (3.5 - i) * 0.25, 0]))
                       for i in range(8) for j in range(8)])
        vl = tx("full-res velocity", 20, DIM).next_to(vel, DOWN, buff=0.12)
        self.play(GrowArrow(arrow(ua.get_right(), vel.get_left(), buff=0.08, sw=2.5, tip=0.12)),
                  LaggedStart(*[FadeIn(s) for s in vel], lag_ratio=0.02), FadeIn(vl), run_time=0.9)
        self.caption("Patch-wise asymmetric flow: every step predicts dense flow from a reduced token sequence",
                     hold=False, size=27)
        self.finish()


class C06_ShapeRoPE(Clip):
    """Finest-grid-aligned RoPE at the patch centre + patch-shape embedding s(e)."""

    def construct(self):
        self.headline("Telling a pretrained DiT where each token is, and how big")
        n, g = 8, 3.6
        cw = g / n
        C0 = np.array([-3.7, 0.5, 0])
        grid = VGroup(*[Square(cw).set_stroke(GHOST, 1.2).set_fill(opacity=0).move_to(cell_center(C0, g, n, n, i, j))
                        for i in range(n) for j in range(n)])
        idx = VGroup(*[tx(str(k), 16, DIM).move_to(cell_center(C0, g, n, n, -0.75, k)) for k in range(n)] +
                     [tx(str(k), 16, DIM).move_to(cell_center(C0, g, n, n, k, -0.75)) for k in range(n)])
        r = (2, 4, 1, 5)  # e = (2, 4) at (u, v) = (2, 1)
        big = rect_mob(C0, g, n, n, r, WHITE, 0).set_fill(level_hex(2, 4), 0.85)
        one = rect_mob(C0, g, n, n, (6, 7, 6, 7), WHITE, 0).set_fill(level_hex(1, 1), 0.85)
        self.add(grid, idx, big, one)
        p = Dot(cell_center(C0, g, n, n, 2, 1), radius=0.08, color=AMBER)
        pl = tx(f"top-left ({it('u')}, {it('v')}) = (2, 1)", 22, AMBER).move_to([C0[0], -1.75, 0])
        self.play(FadeIn(p), FadeIn(pl), run_time=0.4)
        self.wait(0.5)
        c = cell_center(C0, g, n, n, 2.5, 2.5)
        cl = tx(f"RoPE index = patch centre ({it('u')} + ({it('e')}{sub('h')} − 1)/2, "
                f"{it('v')} + ({it('e')}{sub('w')} − 1)/2) = (2.5, 2.5)", 22)
        cl.scale_to_fit_width(min(cl.width, 6.3)).move_to([-3.75, -2.2, 0])
        self.play(p.animate.move_to(c), run_time=0.7)
        self.add(cl)
        d1 = Dot(cell_center(C0, g, n, n, 6, 6), radius=0.07, color=AMBER)
        n1 = tx("1×1: exactly the pretrained RoPE", 20, DIM).move_to([-3.75, -2.62, 0])
        self.play(FadeIn(d1), run_time=0.3)
        self.add(n1)
        self.wait(1.2)
        # shape embedding
        X = 3.5
        s_eq = tx(f"{it('s')}({it('e')}) = (log{sub('2')} {it('e')}{sub('h')}, log{sub('2')} {it('e')}{sub('w')}, "
                  f"log{sub('2')} {it('e')}{sub('h')}{it('e')}{sub('w')}, log{sub('2')} {it('e')}{sub('h')}/{it('e')}{sub('w')})",
                  24)
        s_eq.scale_to_fit_width(min(s_eq.width, 6.6)).move_to([X, 2.45, 0])
        vals = [1, 2, 3, -1]
        names = ["height", "width", "area", "aspect"]
        bars = VGroup()
        for k, (val, nm) in enumerate(zip(vals, names)):
            x = X - 1.8 + 1.2 * k
            b = Rectangle(width=0.55, height=abs(val) * 0.42).set_fill(level_hex(2, 4), 0.9).set_stroke(width=0)
            b.move_to([x, 0.3 + val * 0.21, 0])
            bars.add(VGroup(b, tx(f"{val:+d}".replace("+", "") if val < 0 else str(val), 22).next_to(b, UP if val > 0 else DOWN, buff=0.06),
                            tx(nm, 18, DIM).move_to([x, -0.1 if val > 0 else 0.75, 0])))
        axis = Line([X - 2.4, 0.3, 0], [X + 2.4, 0.3, 0]).set_stroke(DIM, 1.5)
        self.add(s_eq)
        self.play(Create(axis), LaggedStart(*[GrowFromEdge(b[0], DOWN if vals[k] > 0 else UP) for k, b in enumerate(bars)],
                                            lag_ratio=0.15), run_time=0.8)
        self.add(*[VGroup(b[1], b[2]) for b in bars])
        mlp = rbox(2.3, 0.62, PURPLE).move_to([X, -1.0, 0])
        ml = tx(f"MLP {it('g')}{sub('φ')} (zero-init)", 20).move_to(mlp)
        emb = tx(f"{it('h')}{sup('0')}{sub(it('i'))} = {it('W')}{sub('in,' + it('e'))} {it('x')}{sup('(i)')}{sub(it('t'))} + "
                 f"{it('g')}{sub('φ')}({it('s')}({it('e')}{sub(it('i'))}))", 26).move_to([X, -2.0, 0])
        self.play(FadeIn(mlp), FadeIn(ml), GrowArrow(arrow([X, -0.45, 0], mlp.get_top(), buff=0.05, sw=2.5, tip=0.12)),
                  run_time=0.5)
        self.add(emb)
        self.caption("Zero-initialised, so training starts from the pretrained model's behaviour", hold=False, size=26)
        self.finish()


class C07_Budget(Clip):
    """Paper Fig. 10, deer: token budget B = 1,024 ... 4,096 (semantic-mask guided)."""

    def construct(self):
        tags = ["B1024", "B1792", "B2560", "B3328", "B4096"]
        Bs = [1024, 1792, 2560, 3328, 4096]
        self.headline("One knob: the token budget")
        H_ = 4.5
        CI, CG = np.array([-2.7, 0.3, 0]), np.array([2.7, 0.3, 0])
        ims = [img(f"f10_deer_{t}_img.png", h=H_).move_to(CI) for t in tags]
        grs = [img(f"f10_deer_{t}_grid.png", h=H_).move_to(CG) for t in tags]
        fi, fg = framed(ims[0]), framed(grs[0])
        li = tx("generation", 22, DIM).next_to(fi, UP, buff=0.1)
        lg = tx("token layout (semantic-mask guided)", 22, DIM).next_to(fg, UP, buff=0.1)
        # slider
        x0, x1, ys = -4.2, 4.2, -2.75
        track = Line([x0, ys, 0], [x1, ys, 0]).set_stroke(DIM, 3)
        ticks = VGroup(*[VGroup(Line([x0 + (x1 - x0) * k / 4, ys - 0.1, 0], [x0 + (x1 - x0) * k / 4, ys + 0.1, 0])
                                .set_stroke(DIM, 2), tx(fmt(B), 18, DIM).move_to([x0 + (x1 - x0) * k / 4, ys - 0.32, 0]))
                         for k, B in enumerate(Bs)])
        knob = Dot([x0, ys, 0], radius=0.12, color=AMBER)
        cur = {"k": 0}
        lab = tx(f"{it('B')} = 1,024 tokens (of 4,096)", 28).move_to([0, ys + 0.45, 0])
        self.add(ims[0], grs[0], fi, fg, li, lg, track, ticks, knob, lab)
        self.wait(0.9)
        for k in range(1, 5):
            new_lab = tx(f"{it('B')} = {fmt(Bs[k])} tokens" + (" (of 4,096)" if k < 4 else " = full resolution"), 28)
            new_lab.move_to(lab)
            self.play(knob.animate.move_to([x0 + (x1 - x0) * k / 4, ys, 0]), FadeIn(ims[k]), FadeIn(grs[k]),
                      FadeOut(ims[k - 1]), FadeOut(grs[k - 1]), run_time=0.8)
            self.remove(lab)
            lab = new_lab
            self.add(lab)
            self.wait(0.75)
        self.caption("Lower budgets simplify textures but keep the subject and the scene", hold=False, size=26,
                     y=-3.55)
        self.finish()


class C08_Numbers(Clip):
    """Quality vs speed at matched budgets: Tab. 1 (2.935× compression) and Tab. 2 (3× budget)."""

    def construct(self):
        self.headline("Faster than the baselines, and it keeps the quality")
        img_pts = [("Full res. FLUX.2 4B", 1.00, 10.4299, "Full res."), ("ToMe-SD", 2.003, 0.9253, "ToMe-SD"),
                   ("DDiT", 1.321, 7.4075, "DDiT"), ("Foveated Diffusion", 1.180, 7.8433, "Foveated Diffusion"),
                   ("LoT (ours)", 2.040, 9.5522, "Ours")]
        vid_pts = [("Full res. Wan2.1 14B", 1.00, 0.6465, "Full res."), ("ToMe-SD", 3.305, 0.4208, "ToMe-SD"),
                   ("Foveated Diffusion", 3.276, 0.4952, "Foveated Diffusion"), ("LoT (ours)", 3.530, 0.5650, "Ours")]

        def chart(center, xr, yr, pts, title, ylab, xt, yt):
            ax = Axes(x_range=xr, y_range=yr, x_length=5.2, y_length=4.1, tips=False,
                      axis_config={"include_numbers": False, "stroke_color": DIM, "stroke_width": 2,
                                   "include_ticks": True, "tick_size": 0.05}).move_to(center)
            lbl = VGroup(tx(title, 26, bold=True).next_to(ax, UP, buff=0.2),
                         tx("speed-up →", 20, DIM).next_to(ax.x_axis, DOWN, buff=0.4),
                         tx(ylab, 20, DIM).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45))
            nums = VGroup(*[tx(f"{v:g}×", 16, DIM).next_to(ax.c2p(v, yr[0]), DOWN, buff=0.12) for v in xt] +
                          [tx(f"{v:g}", 16, DIM).next_to(ax.c2p(xr[0], v), LEFT, buff=0.1) for v in yt])
            dots = []
            for name, x, y, key in pts:
                c = METHOD_HEX[key]
                d = Dot(ax.c2p(x, y), radius=0.11 if key == "Ours" else 0.085, color=c)
                t = tx(name, 18, c).next_to(d, UP if key != "ToMe-SD" else UP, buff=0.1)
                if key in ("DDiT",):
                    t.next_to(d, DOWN, buff=0.1)
                if key == "Foveated Diffusion":
                    t.next_to(d, UP, buff=0.1).shift(RIGHT * 0.25)
                if key == "Full res.":
                    t.next_to(d, RIGHT, buff=0.12)
                if key == "Ours":
                    t = tx(name, 20, c, bold=True).next_to(d, LEFT, buff=0.12)
                dots.append(VGroup(d, t))
            return ax, lbl, nums, dots

        a1 = chart([-3.45, -0.15, 0], [0.8, 2.3, 0.5], [0, 11.5, 2.5], img_pts, "Images · FLUX.2 4B",
                   "HPSv3 (human pref.) →", [1, 1.5, 2], [0, 5, 10])
        a2 = chart([3.55, -0.15, 0], [0.8, 3.8, 1.0], [0.38, 0.68, 0.1], vid_pts, "Video · Wan2.1 14B",
                   "VBench imaging quality →", [1, 2, 3], [0.4, 0.5, 0.6])
        self.add(a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
        self.wait(0.4)
        for k in range(4):
            self.play(FadeIn(a1[3][k], scale=1.3), *( [FadeIn(a2[3][k], scale=1.3)] if k < 3 else []), run_time=0.45)
        self.wait(0.3)
        rings = VGroup(Circle(0.22).set_stroke(FINE, 3).move_to(a1[3][4][0]), Circle(0.22).set_stroke(FINE, 3).move_to(a2[3][3][0]))
        self.play(FadeIn(a1[3][4], scale=1.5), FadeIn(a2[3][3], scale=1.5), Create(rings), run_time=0.6)
        self.caption("Same token budget: 2.04× faster images, 3.53× faster video, best quality among compressed methods",
                     hold=False, size=25)
        self.finish()


# ================================================================== result clips
class R01_PaintGUI(Clip):
    """Manual GUI recording (57 s) at 5.8×: paint a detail map -> token layout -> generate."""

    def construct(self):
        self.headline("Paint where the detail goes")
        card = VideoCard("rec3", box=(8.8, 6.0), center=[-2.45, -0.3, 0], speed=5.8, loop=False, cover=True,
                         radius=0.1)
        card.start(self)
        steps = [("1", "paint a detail map", "white = fine · black = coarse"),
                 ("2", "it becomes a token layout", "quadtree: 8×8 down to 1×1"),
                 ("3", "generate with LoT-FLUX.2", "edit, regenerate, compare")]
        items = VGroup()
        for k, (n, a, b) in enumerate(steps):
            num = Circle(0.24).set_stroke(AMBER, 2.5).set_fill(BLACK, 1)
            nt = tx(n, 22, AMBER).move_to(num)
            t = VGroup(tx(a, 23), tx(b, 18, DIM)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
            g = VGroup(VGroup(num, nt), t).arrange(RIGHT, buff=0.2)
            items.add(g.move_to([2.35 + g.width / 2, 1.4 - 1.3 * k, 0]))
        self.wait(0.3)
        for k, m in enumerate(items):
            self.play(FadeIn(m, shift=LEFT * 0.2), run_time=0.4)
            self.wait(1.6 if k < 2 else 0.4)
        self.caption("Recorded GUI session, sped up 5.8×", hold=False, size=22, y=-3.62)
        self.finish()


class R02_Agent(Clip):
    """Agent-assisted painting (rec0, 45 s) -> painted map -> token layout -> generated image (paint0)."""

    def construct(self):
        self.headline("…or let an agent paint the plan from a sentence")
        # rec0 canvas region: x 13.5–49.5 %, y 15–83 % of the 1600x900 recording
        X = [-4.55, 0.0, 4.55]
        S = 3.9
        card = VideoCard("rec0", box=(S, S), center=[X[0], 0.25, 0], speed=6.0, loop=False, radius=0.06,
                         crop=(0.135, 0.15, 0.495, 0.83))
        card.start(self)
        prompt = VGroup(tx("“a little girl riding an enormous", 20, DIM, italic=True),
                        tx("friendly Pembroke Welsh corgi…”", 20, DIM, italic=True)).arrange(DOWN, buff=0.06)
        prompt.move_to([X[0], -2.25, 0])
        self.add(prompt, tx("agent paints the detail map", 22, DIM).move_to([X[0], 2.5, 0]))
        self.wait(4.3)
        card.playing = False  # the recording scrolls to the settings panel after the painting is done
        L1 = load_layout("paint0")
        C = np.array([X[1], 0.25, 0])
        M = LayoutMorph(Layout.uniform(64, 64), L1, S, center=C, mode="level", gap=0.1)
        a1 = arrow([X[0] + S / 2 + 0.05, 0.25, 0], [X[1] - S / 2 - 0.05, 0.25, 0], buff=0, sw=3, tip=0.14)
        self.play(GrowArrow(a1), FadeIn(M.img), run_time=0.4)
        self.add(tx("token layout", 22, DIM).move_to([X[1], 2.5, 0]))
        v = ValueTracker(4096)
        cnt = self.counter(v, lambda x: f"{fmt(x)} tokens", 26, pos=[X[1], -2.25, 0])
        self.add(cnt)
        self.play(M.t.animate.set_value(1), v.animate.set_value(L1.L), run_time=1.8)
        res = img("paint0_result.png", h=S).move_to([X[2], 0.25, 0])
        a2 = arrow([X[1] + S / 2 + 0.05, 0.25, 0], [X[2] - S / 2 - 0.05, 0.25, 0], buff=0, sw=3, tip=0.14)
        self.play(GrowArrow(a2), FadeIn(res), FadeIn(framed(res)), run_time=0.5)
        self.add(tx("LoT-FLUX.2", 22, DIM).move_to([X[2], 2.5, 0]),
                 tx(f"{4096 / L1.L:.1f}× fewer tokens", 24, FINE).move_to([X[2], -2.25, 0]))
        self.caption("Agent-assisted session (sped up 6×): scene description → painted detail map → LoT-FLUX.2",
                     hold=False, size=22, y=-3.62)
        self.finish()


def three_cards(scene, keys, labels, big_first=True):
    """generated video big on the left, detail map + token grid stacked on the right"""
    big = VideoCard(keys[0], box=(8.4, 4.73), center=[-2.35, -0.05, 0], speed=0.6, loop=True)
    s1 = VideoCard(keys[1], box=(4.1, 2.31), center=[4.35, 1.2, 0], speed=0.6, loop=True)
    s2 = VideoCard(keys[2], box=(4.1, 2.31), center=[4.35, -1.45, 0], speed=0.6, loop=True)
    for c in (big, s1, s2):
        c.start(scene)
    labs = VGroup(tx(labels[0], 22, DIM).next_to(big.rect(), DOWN, buff=0.12),
                  tx(labels[1], 20, DIM).next_to(s1.rect(), UP, buff=0.08),
                  tx(labels[2], 20, DIM).next_to(s2.rect(), DOWN, buff=0.08))
    scene.add(labs)
    return big, s1, s2, labs


class R03_Scene3D(Clip):
    """Blender toy-train scene -> per-frame detail map -> temporal token layout -> LoT-Wan2.1 video."""

    def construct(self):
        self.headline("An animated 3D scene decides where the tokens go, frame by frame")
        three_cards(self, ["vd_train_gen", "vd_train_map", "vd_train_grid"],
                    ["LoT-Wan2.1 14B · 1280 × 720 · 81 frames", "detail map rasterised from the Blender scene",
                     "temporal token layout"])
        self.wait(2.0)
        self.caption("train, houses, trees 1×1 · ground and tracks 2×2 · sky 4×4 → 50,832 of 75,600 tokens (1.49× fewer)",
                     hold=False, size=23)
        self.finish()


class R04_Games(Clip):
    """Three game-like scenes: platformer (animated boxes), monkey warrior (bbox), desert coupe (texture)."""

    def construct(self):
        self.headline("Games: fine tokens on the player, coarse ones on sand and sky")
        xs = [-4.72, 0.0, 4.72]
        W_ = 4.5
        specs = [("vd_platformer_gen", "vd_platformer_grid", None, "platformer · animated boxes", "2.18× fewer tokens"),
                 ("show14_gen", "show14_lay", None, "monkey warrior · bounding box", "2.02× fewer tokens"),
                 ("show26_gen", None, "show26", "desert coupe · texture variance", "2.16× fewer tokens")]
        for k, (gk, lk, lay, name, stat) in enumerate(specs):
            box, cg, cl = (W_, W_ * 9 / 16), [xs[k], 1.05, 0], [xs[k], -1.65, 0]
            VideoCard(gk, box=box, center=cg, speed=0.6).start(self)
            if lk:
                VideoCard(lk, box=box, center=cl, speed=0.6).start(self)
            else:
                drv = hidden_driver(self, gk, box, cl, speed=0.6)
                self.add(LayoutVideo(load_video_layouts(lay), drv).img)
            self.add(tx(name, 20, DIM).move_to([xs[k], 2.6, 0]), tx(stat, 20, FINE).move_to([xs[k], -3.1, 0]))
        self.wait(0.5)
        self.finish()


def pair_rows(scene, specs, W_=5.2, y0=1.55, dy=-3.1):
    """rows of (layout video | generated video) pairs, centred"""
    cards = []
    for k, (gen, lay, lay_key) in enumerate(specs):
        y = y0 + k * dy
        box = (W_, W_ * 9 / 16)
        g = VideoCard(gen, box=box, center=[W_ / 2 + 0.08, y, 0], speed=0.6)
        g.start(scene)
        if lay:
            l = VideoCard(lay, box=box, center=[-W_ / 2 - 0.08, y, 0], speed=0.6)
            l.start(scene)
        else:
            l = hidden_driver(scene, gen, box, [-W_ / 2 - 0.08, y, 0], speed=0.6)
            scene.add(LayoutVideo(load_video_layouts(lay_key), l).img)
        cards.append((l, g))
    return cards


def side_stat(card, lines, side=RIGHT):
    g = VGroup(*[tx(t, s, c) for (t, s, c) in lines]).arrange(DOWN, buff=0.06)
    return g.next_to(card.rect(), side, buff=0.18)


class R05_Driving(Clip):
    """Driving scenes with semantic-mask layouts (show03 day intersection, show04 rainy night)."""

    def construct(self):
        self.headline("Driving: coarse sky and asphalt, fine road users", 30, y=3.62)
        cards = pair_rows(self, [("show03_gen", "show03_lay", None), ("show04_gen", "show04_lay", None)])
        for k, key in enumerate(["show03", "show04"]):
            m = showcase_meta(key)
            self.add(side_stat(cards[k][1], [(f"{m['compression']:.2f}×", 26, FINE), ("fewer", 20, DIM), ("tokens", 20, DIM)]))
        self.add(side_stat(cards[0][0], [("day", 20, DIM)], LEFT), side_stat(cards[1][0], [("rainy", 20, DIM), ("night", 20, DIM)], LEFT))
        leg = class_legend(showcase_meta("show04")["legend"]["items"][:6], size=15, per_row=6)
        leg.scale_to_fit_width(min(leg.width, 13.0)).move_to([0, -3.5, 0])
        self.add(leg)
        self.finish()


class R06_Robotics(Clip):
    """Robot manipulation: arm + gripper + object at 1×1 (show02 mask, show10 box)."""

    def construct(self):
        self.headline("Robotics: detail on the arm, the gripper and the object", 30, y=3.62)
        cards = pair_rows(self, [("show02_gen", "show02_lay", None), ("show10_gen", "show10_lay", None)])
        for k, (key, src) in enumerate([("show02", ("semantic", "mask")), ("show10", ("bounding", "boxes"))]):
            m = showcase_meta(key)
            self.add(side_stat(cards[k][1], [(f"{m['compression']:.2f}×", 26, FINE), ("fewer", 20, DIM), ("tokens", 20, DIM)]))
            self.add(side_stat(cards[k][0], [(src[0], 20, DIM), (src[1], 20, DIM)], LEFT))
        leg = class_legend(showcase_meta("show02")["legend"]["items"], size=15, per_row=3)
        leg.scale_to_fit_width(min(leg.width, 13.0)).move_to([0, -3.5, 0])
        self.add(leg)
        self.finish()


class R07_DepthOfField(Clip):
    """Depth-of-field layouts: blur radius from depth decides token size (show30, show31, show39)."""

    def construct(self):
        self.headline("Depth of field: out-of-focus regions get big tokens", 32)
        xs = [-4.72, 0.0, 4.72]
        W_ = 4.5
        for k, key in enumerate(["show30", "show31", "show39"]):
            g = VideoCard(key + "_gen", box=(W_, W_ * 9 / 16), center=[xs[k], 1.05, 0], speed=0.6)
            g.start(self)
            l = hidden_driver(self, key + "_gen", (W_, W_ * 9 / 16), [xs[k], -1.6, 0], speed=0.6)
            self.add(LayoutVideo(load_video_layouts(key), l).img)
            m = showcase_meta(key)
            self.add(tx(f"{m['compression']:.2f}× fewer tokens", 20, FINE).move_to([xs[k], -3.0, 0]))
        leg = legend_levels(((1, 1), (2, 2), (4, 4)), size=18).move_to([0, 2.72, 0])
        self.add(leg)
        eq = tx(f"blur radius {it('r')} = {it('K')} |1/{it('d')} − 1/{it('d')}{sub(it('f'))}| : "
                f"small {it('r')} → fine tokens, large {it('r')} → coarse", 24)
        eq.move_to([0, -3.55, 0])
        self.add(eq)
        self.finish()


class R08_MoveTheBox(Clip):
    """Fig. 19: same prompt, noise and token count; moving the 1×1 box moves the subject (FLUX.2 9B)."""

    def construct(self):
        self.headline("Move the fine-token box, and the subject follows")
        rows = [("a", "fire hydrant", "1,636 tokens · 2.50×"), ("c", "bird", "1,222 tokens · 3.35×"),
                ("i", "cat", "1,519 tokens · 2.70×")]
        S = 3.9
        CL, CR, CD = np.array([-4.55, 0.25, 0]), np.array([0.05, 0.25, 0]), np.array([4.85, 0.25, 0])
        tl = VGroup(tx("token layout", 22, DIM).move_to(CL + UP * (S / 2 + 0.25)),
                    tx("LoT-FLUX.2 9B", 22, DIM).move_to(CR + UP * (S / 2 + 0.25)),
                    tx("FLUX.2 9B, no layout", 22, DIM).move_to(CD + UP * (S / 2 + 0.25)))
        self.add(tl)
        prev = None
        for k, (r, name, stat) in enumerate(rows):
            fa, fb = yellow_box(os.path.join(IMG, f"f19_{r}_gridA.png")), yellow_box(os.path.join(IMG, f"f19_{r}_gridB.png"))
            LA, ra = box_layout(fa)
            LB, rb = box_layout(fb)
            la = layout_mobject(LA, S, gap=0.1).move_to(CL)
            lb = layout_mobject(LB, S, gap=0.1).move_to(CL)
            ia, ib = img(f"f19_{r}_oursA.png", h=S).move_to(CR), img(f"f19_{r}_oursB.png", h=S).move_to(CR)
            dn = img(f"f19_{r}_dense.png", h=S).move_to(CD)
            boxA = rect_mob(CL, S, 64, 64, ra, AMBER, 3)
            boxB = rect_mob(CL, S, 64, 64, rb, AMBER, 3)
            lab = tx(f"“{name}” · {stat} fewer", 24).move_to([CR[0] - 2.3, -2.3, 0])
            grp = Group(la, ia, dn, framed(ia), framed(dn), boxA, lab)
            if prev is None:
                self.add(grp)
            else:
                self.play(FadeOut(prev), FadeIn(grp), run_time=0.35)
            self.wait(0.55)
            self.play(Transform(boxA, boxB), FadeIn(lb), FadeOut(la), FadeIn(ib), FadeOut(ia), run_time=0.9)
            self.wait(0.9)
            prev = Group(grp, lb, ib)
        self.caption("Subject inside the fine region: 44.9% (FLUX.2 9B) → 66.8% (LoT-FLUX.2 9B) over 2,096 COCO prompts",
                     hold=False, size=24)
        self.finish()


class R09_Typography(Clip):
    """Typography / curve layouts at 2048²: layout -> LoT image, dense FLUX.2 for reference (site Sec. 4.4)."""

    def construct(self):
        self.headline("Draw with tokens: letters, spirals, Hilbert curves")
        tj = json.load(open(os.path.join(IMG, "typography_index.json")))
        smp = {s["number"]: s for s in tj["samples"]}
        xs = [-4.7, 0.0, 4.7]
        S = 3.8
        items = []
        for k, n in enumerate([82, 40, 86]):
            s = smp[n]
            g = img(f"typo{n:03d}_grid.png", h=S).move_to([xs[k], 0.35, 0])
            o = img(f"typo{n:03d}_ours.png", h=S).move_to([xs[k], 0.35, 0])
            st = VGroup(tx(f"{fmt(s['tokens'])} of {fmt(s['dense_tokens'])} tokens", 18, DIM),
                        tx(f"{s['seconds']['ours']:.0f} s vs {s['seconds']['dense']:.0f} s dense · "
                           f"{s['seconds']['speedup']:.2f}× faster", 18, FINE)).arrange(DOWN, buff=0.06)
            st.move_to([xs[k], -1.95, 0])
            items.append((g, o, st))
        self.add(*[Group(g, framed(g)) for g, _, _ in items])
        top_lab = tx("token layout", 22, DIM).move_to([0, 2.6, 0])
        self.add(top_lab)
        self.wait(1.0)
        for k, (g, o, st) in enumerate(items):
            self.play(FadeIn(o), run_time=0.6)
            self.add(st)
            self.wait(0.25)
        self.remove(top_lab)
        self.add(tx("LoT-FLUX.2 9B · 2048 × 2048", 22, DIM).move_to([0, 2.6, 0]))
        self.wait(1.2)
        dn = img("typo040_dense.png", h=1.55).move_to([-4.9, -2.95, 0])
        self.play(FadeIn(dn), run_time=0.4)
        self.add(tx("dense FLUX.2, same prompt, no layout: the curve is not followed", 20, DIM)
                 .next_to(dn, RIGHT, buff=0.25))
        self.finish()


class R10_Boxes(Clip):
    """Ideogram-style labelled boxes with per-box detail level -> LoT layout -> image (FLUX.2 9B, 1536×1024)."""

    def construct(self):
        self.headline("Layouts from labelled boxes, with a detail level per box")
        bj = json.load(open(os.path.join(IMG, "bbox_index.json")))
        tok = {e["id"]: e["tokens"] for e in bj["examples"]}
        W_ = 4.35
        xs = [-4.65, 0.0, 4.65]
        prev = None
        for k, ex in enumerate(["reading-corner", "coastal", "still-life"]):
            a = img(f"bbox_{ex}_box_overlay.png", w=W_).move_to([xs[0], 0.2, 0])
            b = img(f"bbox_{ex}_layout.png", w=W_).move_to([xs[1], 0.2, 0])
            c = img(f"bbox_{ex}_generated.png", w=W_).move_to([xs[2], 0.2, 0])
            st = tx(f"{fmt(tok[ex])} of 6,144 tokens · {6144 / tok[ex]:.2f}× fewer", 22, FINE).move_to([0, -1.75, 0])
            grp = Group(a, b, framed(b), c, framed(c), st)
            if prev is None:
                self.add(a)
                self.add(tx("boxes: label · level", 22, DIM).move_to([xs[0], 1.95, 0]),
                         tx("LoT layout", 22, DIM).move_to([xs[1], 1.95, 0]),
                         tx("generation", 22, DIM).move_to([xs[2], 1.95, 0]),
                         arrow([xs[0] + W_ / 2 + 0.02, 0.2, 0], [xs[1] - W_ / 2 - 0.02, 0.2, 0], buff=0, sw=2.5, tip=0.12),
                         arrow([xs[1] + W_ / 2 + 0.02, 0.2, 0], [xs[2] - W_ / 2 - 0.02, 0.2, 0], buff=0, sw=2.5, tip=0.12))
                self.wait(0.4)
                self.play(FadeIn(Group(b, framed(b))), run_time=0.4)
                self.play(FadeIn(Group(c, framed(c))), FadeIn(st), run_time=0.4)
                prev = grp
                self.wait(1.4)
            else:
                self.play(FadeOut(prev), FadeIn(grp), run_time=0.5)
                prev = grp
                self.wait(2.0)
        self.caption("Like Ideogram / Reve layouts, but the layout now also decides the compute", hold=False, size=26,
                     y=-2.9)
        self.finish()


class R11_VideoBaselines(Clip):
    """Site Sec. 5, example 0: same VRS token layout (2.44× compression) for ToMe-SD, Foveated Diffusion and LoT."""

    def construct(self):
        self.headline("Same token budget, four models", 32, y=3.6)
        ci = json.load(open(os.path.join(CLIPS, "comparisons_index.json")))[0]
        st = ci["stats"]
        W_ = 5.0
        pos = {"Dense Wan2.1": [-2.65, 1.55], "ToMe-SD": [2.65, 1.55], "Foveated Diffusion": [-2.65, -1.72],
               "Ours": [2.65, -1.72]}
        names = {"Dense Wan2.1": "Full-res. Wan2.1 14B", "ToMe-SD": "ToMe-SD", "Foveated Diffusion": "Foveated Diffusion",
                 "Ours": "LoT-Wan2.1 (ours)"}
        colors = {"Dense Wan2.1": "Full res.", "ToMe-SD": "ToMe-SD", "Foveated Diffusion": "Foveated Diffusion", "Ours": "Ours"}
        for meth, p in pos.items():
            c = VideoCard(ci["videos"][meth], box=(W_, W_ * 9 / 16), center=[p[0], p[1], 0], speed=0.6,
                          border=METHOD_HEX[colors[meth]] if meth == "Ours" else DIM, border_w=3 if meth == "Ours" else 1.2)
            c.start(self)
            secs = st[meth]["generation_seconds"]
            lab = tx(f"{names[meth]} · {secs:.0f} s", 20, METHOD_HEX[colors[meth]], bold=(meth == "Ours"))
            lab.next_to(c.rect(), UP, buff=0.06).align_to(c.rect(), LEFT)
            self.add(lab)
        self.add(VGroup(tx(f"{st['Ours']['compression']:.2f}×", 30, FINE), tx("fewer tokens", 20, DIM),
                        tx("for all three", 20, DIM), tx("compressed", 20, DIM), tx("models", 20, DIM))
                 .arrange(DOWN, buff=0.06).move_to([6.2, 0, 0]))
        self.finish()
