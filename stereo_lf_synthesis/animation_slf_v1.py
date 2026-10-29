"""Robust Light Field Synthesis From Stereo Images With Left-Right Geometric Consistency — 5-min teaser, v1.

Source: Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen, IEEE ICIP 2021, pp. 1844-1848, National Taiwan University.
Numbers on screen: stereo-lf-synthesis/summaries/chao2021robust.md. Style, helpers and caches: slf_common.py.
Real computation shown:
  * ground-truth HCI origami light field (9 x 9 views): parallax, shift-and-add refocusing, epipolar-plane image, PSAS shuffle
  * the paper's synthesized origami light field (website GIF, 81 views) refocused by shift-and-add
  * our short re-run of the authors' code (DisparityNet + PSAS, paper losses) on the paper's HCI split:
    disparity maps, warped light fields F_l / F_r, per-view error maps, blending, refinement, training snapshots

Render (see render.sh):  manim -ql --media_dir media animation_slf_v1.py <Scene>
Scenes, in order: S00_Title ... S10_Outlook.
"""
from slf_common import *

build_assets()
build_repro_assets()


def mini_grid(center, cell, inputs, color=AMBER, empty="#2A2D33"):
    g = ViewGrid(center, cell)
    return Group(g.mobs, g.fill(inputs, color)), g


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = fitw(tx("Robust Light Field Synthesis From Stereo Images", 48, bold=True)).move_to([0, 3.1, 0])
        subt = tx("with Left-Right Geometric Consistency", 34).move_to([0, 2.45, 0])
        gif = G("gif336.npy")
        card = arr_img(gif[4, 4], 3.3).move_to([-1.2, -0.15, 0])
        live(card, lambda: gif[view_path(self.clock.get_value())])
        fr = framed(card, DIM, 1.2)
        grid = ViewGrid([2.75, -0.15, 0], 0.2)
        inp = grid.fill([(ROW, JL), (ROW, JR)], AMBER)
        cur = always_redraw(lambda: grid.box(*view_path(self.clock.get_value()), color=WHITE, sw=2.5))
        gl1 = tx("2 input views", 20, AMBER).next_to(grid.mobs, UP, buff=0.42)
        gl2 = tx("79 synthesized", 20, GREEN).next_to(gl1, DOWN, buff=0.08)
        nt = note("the paper's synthesized 9 × 9 light field of HCI origami (website teaser)").next_to(card, DOWN, buff=0.15)
        auth = tx("Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen", 24, DIM).move_to([0, -2.55, 0])
        aff = tx("National Taiwan University  ·  IEEE ICIP 2021", 22, DIM).move_to([0, -2.98, 0])
        self.add(title, subt, auth, aff)
        self.play(FadeIn(card), Create(fr), FadeIn(grid.mobs), run_time=0.6)
        self.add(inp, cur, gl1, gl2, nt)
        self.wait(7.5)


# ================================================================== S01 Light fields
class S01_LightField(Base):
    def construct(self):
        self.title_card("Part 1", "What is a light field?")
        self.header("1 · Light fields")
        th = G("gt_thumbs.npy")
        v336 = G("gt336.npy")
        grid = ViewGrid([-3.7, 0.25, 0], 0.37, th)
        self.play(LaggedStart(*[FadeIn(m) for m in grid.mobs], lag_ratio=0.012), run_time=1.6)
        gl = label(grid.mobs, f"angular coordinates ({I('i')}, {I('j')}): 9 × 9 views", 22, DIM)
        gn = note("HCI 4D light field benchmark, scene “origami” (ground truth)").next_to(grid.mobs, DOWN, buff=0.15)
        self.add(gl, gn)
        sel = [4, 4]
        big = arr_img(v336[4, 4], 3.6).move_to([2.9, 0.25, 0])
        live(big, lambda: v336[sel[0], sel[1]])
        bl = always_redraw(lambda: label(big, f"view {V_(sel[0], sel[1])}:  {I('H')} × {I('W')} pixels", 22))
        box = always_redraw(lambda: VGroup(grid.box(sel[0], sel[1], BLACK, 8), grid.box(sel[0], sel[1], AMBER, 4)))
        self.add(big, framed(big, DIM, 1), bl, box)
        self.caption("A light field is a grid of photos of one scene, each from a slightly shifted viewpoint.")
        t = ValueTracker(0)

        def path(v):          # middle row left → right, then around the rim
            if v < 1:
                return 4, int(round(8 * v))
            ph = (v - 1) * 2 * np.pi
            return int(np.clip(round(4 - 4 * np.sin(ph)), 0, 8)), int(np.clip(round(4 + 4 * np.cos(ph)), 0, 8))

        def upd(m):
            sel[0], sel[1] = path(t.get_value())
        box.add_updater(lambda m: upd(m))
        self.hold()
        self.caption("Move across the grid and the scene shifts: the near crane more than the far wall.")
        self.run(t, 1, 2.2)
        self.run(t, 2, 3.0)
        self.hold()
        box.clear_updaters()
        self.remove(box, bl)
        # refocusing
        rf = G("refocus_gt.npy")
        a = ValueTracker(0.0)
        self.remove(big)
        big2 = arr_img(interp(rf, a_index(0.0)), 3.6).move_to([2.9, 0.25, 0])
        live(big2, lambda: interp(rf, a_index(a.get_value())))
        self.add(big2, label(big2, "refocused image", 22, AMBER))
        eq = tx(f"refocus = average over ({I('i')}, {I('j')}) of {V_(I('i'), I('j'))} shifted by "
                f"{I('a')}·({I('i')} − 4, {I('j')} − 4)", 22).move_to([0, -2.8, 0])
        sl, dot = focus_slider([2.9, -2.2, 0], 3.2, a.get_value)
        self.add(eq, sl, dot)
        self.caption("Shift each view in proportion to its position, then average: you refocus after the shot.")
        self.run(a, A_WALL, 2.2, smooth)
        self.run(a, A_CRANE, 3.2, smooth)
        self.run(a, 0.0, 1.6, smooth)
        self.clear_all()
        # capture is hard
        # plenoptic camera sketch
        lens = Ellipse(width=0.35, height=2.2, color=WHITE, fill_color="#B3E5FC", fill_opacity=0.3, stroke_width=2).move_to([-5.2, 0.6, 0])
        mla = VGroup(*[Circle(0.09, color=CYAN, stroke_width=2).move_to([-3.6, 0.6 + 0.2 * k, 0]) for k in range(-5, 6)])
        sensor = Line([-3.3, -0.55, 0], [-3.3, 1.75, 0], color=ORANGE, stroke_width=5)
        rays = VGroup(*[Line([-6.4, 0.6 + 0.6 * s_, 0], [-5.2, 0.6 + 0.35 * s_, 0], color=AMBER, stroke_width=1.5) for s_ in (-1, 0, 1)])
        pl = VGroup(lens, mla, sensor, rays, tx("main lens", 16, DIM).next_to(lens, UP, buff=0.1),
                    tx("microlenses", 16, CYAN).next_to(mla, UP, buff=0.1).shift(LEFT * 0.2),
                    tx("sensor", 16, ORANGE).next_to(sensor, RIGHT, buff=0.1))
        pt = VGroup(tx("plenoptic camera", 26), tx("one sensor shared by all views:", 20, DIM),
                    tx("spatial vs angular resolution", 20, DIM)).arrange(DOWN, buff=0.1).move_to([-4.6, -1.4, 0])
        cams = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.04, width=0.34, height=0.26, color=WHITE, stroke_width=1.5),
                               Circle(0.07, color=BLUE, stroke_width=2)).move_to([2.6 + 0.48 * u, 0.6 + 0.42 * v, 0])
                        for u in range(-2, 3) for v in range(-2, 3)])
        ct = VGroup(tx("camera array", 26), tx("one camera per view:", 20, DIM),
                    tx("bulky and costly", 20, DIM)).arrange(DOWN, buff=0.1).move_to([2.6, -1.4, 0])
        self.play(FadeIn(pl), FadeIn(cams), run_time=0.8)
        self.add(pt, ct)
        self.caption("But capturing one is hard: plenoptic cameras trade resolution, camera arrays are bulky.")
        self.hold(0.5)
        self.clear_all()


# ================================================================== S02 Sparse inputs
class S02_Sparse(Base):
    def construct(self):
        self.title_card("Part 2", "How few photos are enough?")
        self.header("2 · Synthesis from sparse views")
        xs = [-5.1, -1.7, 1.7, 5.1]
        cell = 0.18
        cfg = [
            ([(i, j) for i in (1, 4, 7) for j in (1, 4, 7)], "3 × 3 grid of views", "angular super-resolution", BLUE),
            ([(0, 0), (0, 8), (8, 0), (8, 8)], "4 corner views", "Kalantari et al. 2016", PURPLE),
            ([(4, 4)], "1 centre view", "Srinivasan et al. 2017", BLUE),
            ([(ROW, JL), (ROW, JR)], "2 views: a stereo pair", "this paper", GREEN),
        ]
        grids = []
        for x, (cells, t1, t2, c) in zip(xs, cfg):
            g = ViewGrid([x, 0.9, 0], cell)
            f = g.fill(cells, c)
            l1 = tx(t1, 22).move_to([x, -0.35, 0])
            l2 = tx(t2, 19, c if c == GREEN else DIM).move_to([x, -0.7, 0])
            grids.append((g, f, l1, l2))
        caps = ["Angular super-resolution fills in a grid of views, but you need a grid to begin with.",
                "Kalantari et al. start from the 4 corner views; Srinivasan et al. from a single view.",
                None,
                "This paper: a stereo pair, the left- and right-most views of the middle row."]
        for k, (g, f, l1, l2) in enumerate(grids):
            self.play(FadeIn(g.mobs), run_time=0.35)
            self.add(f, l1, l2)
            if k == 1:
                self.caption(caps[1])
            elif k == 2:
                x = tx("✗", 30, RED).move_to([xs[2] + 0.75, 1.75, 0])
                self.add(x, tx("absolute depth from one image", 17, RED).move_to([xs[2], -1.0, 0]),
                         tx("is ill-posed", 17, RED).move_to([xs[2], -1.25, 0]))
                self.hold()
                self.caption("From one image, absolute depth is ill-posed, which is a problem for AR.")
            elif k == 0:
                self.caption(caps[0])
            else:
                self.add(SurroundingRectangle(Group(g.mobs, l2), color=GREEN, buff=0.15, stroke_width=2))
                self.caption(caps[3])
        self.hold()
        pros = VGroup(tx("✓ easy to capture: a stereo camera, or slide one phone camera sideways", 21, GREEN),
                      tx("✓ stereo gives absolute depth: useful for AR headsets", 21, GREEN),
                      tx("prior stereo → light field:  Zhang et al. 2015, iterative, 514.84 s on a CPU, warps only the left view", 19, DIM),
                      tx("Stereo Magnification (multiplane images) needs camera poses at training and test time", 19, DIM)
                      ).arrange(DOWN, buff=0.14, aligned_edge=LEFT).move_to([0, -2.1, 0])
        fitw(pros, 13)
        self.add(pros)
        self.caption("Stereo is easy to capture and gives absolute depth; earlier stereo methods are slow or need poses.")
        self.hold(0.6)
        self.clear_all()


# ================================================================== S03 Disparity and warping
class S03_Disparity(Base):
    def construct(self):
        self.title_card("Part 3", "Disparity and warping")
        self.header("3 · Disparity and warping")
        sl = G("sl504.npy")
        sr = G("sr504.npy")
        h = 2.55
        L_ = arr_img(sl, h).move_to([-5.0, 1.35, 0])
        R_ = arr_img(sr, h).move_to([-2.25, 1.35, 0])
        self.add(L_, framed(L_, LEFTC, 2), R_, framed(R_, RIGHTC, 2),
                 label(L_, f"{SL} = {V_(4, 0)}", 22, LEFTC), label(R_, f"{SR} = {V_(4, 8)}", 22, RIGHTC))
        self.caption("Input: the left- and right-most views of the middle row, a stereo pair.")
        # scanline + EPI
        s2u = h / 504
        yline = L_.get_top()[1] - EPI_Y * s2u
        x0u = L_.get_left()[0] + EPI_X[0] * s2u
        x1u = L_.get_left()[0] + EPI_X[1] * s2u
        scan = VGroup(Line([L_.get_left()[0], yline, 0], [L_.get_right()[0], yline, 0], color=RED, stroke_width=1.5),
                      Line([x0u, yline, 0], [x1u, yline, 0], color=AMBER, stroke_width=4))
        scan2 = Line([R_.get_left()[0], yline, 0], [R_.get_right()[0], yline, 0], color=RED, stroke_width=1.5)
        self.hold()
        self.add(scan, scan2)
        epi = G("epi.npy")[:, EPI_X[0]:EPI_X[1]]
        RH = 14
        W_ = EPI_X[1] - EPI_X[0]
        full = np.repeat(epi, RH, 0)
        ew = 6.2
        eh = ew * full.shape[0] / W_
        ec = np.array([3.35, 1.35, 0])
        rows = []
        for j in range(N):
            r = arr_img(full[j * RH:(j + 1) * RH], nearest=True)
            r.stretch_to_fit_width(ew)
            r.stretch_to_fit_height(eh / N)
            r.move_to(ec + UP * (eh / 2 - (j + 0.5) * eh / N))
            rows.append(r)
        ef = Rectangle(width=ew, height=eh, color=DIM, stroke_width=1).move_to(ec)
        el = label(ef, "epipolar-plane image: this scanline in views (4, 0) … (4, 8)", 20, AMBER)
        jl = VGroup(*[tx(f"{I('j')} = {j}", 14, DIM).move_to([ec[0] - ew / 2 - 0.38, rows[j].get_center()[1], 0])
                      for j in (0, 8)])
        self.add(ef, el, jl)
        self.caption("Stack that scanline from all 9 views of the row: an epipolar-plane image.")
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * DOWN) for r in rows], lag_ratio=0.25), run_time=2.2)
        # slope lines from our re-run's disparity D_lr
        D = G("d_lr504.npy")
        lines = VGroup()
        for xs_px, c in ((150, AMBER), (238, CYAN), (330, GREEN)):
            d = float(np.median(D[EPI_Y - 2:EPI_Y + 3, xs_px - 2:xs_px + 3]))
            pts = []
            for j in (0, 8):
                xp = xs_px - d * j                                     # V(4, j)(x) = S_l(x + d j)
                pts.append([ec[0] - ew / 2 + (xp - EPI_X[0]) / W_ * ew, rows[j].get_center()[1], 0])
            lines.add(DashedLine(pts[0], pts[1], color=c, stroke_width=3, dash_length=0.08))
        self.hold()
        self.play(Create(lines), run_time=1.0)
        dl = tx(f"slope = disparity {I('D')}: pixels moved per view step", 22).next_to(ef, DOWN, buff=0.25)
        self.add(dl)
        self.caption("Every point traces a line; its slope is the disparity D, how far it moves per view.")
        self.hold()
        self.clear_all()
        # disparity map + warping of the whole row / grid
        dmap = arr_img(G("d_lr_rgb.npy"), 2.6).move_to([-5.0, 1.2, 0])
        lo, hi = G("d_range.npy")
        self.add(dmap, framed(dmap, LEFTC, 1.5), label(dmap, f"{DLR} = {I('g')}({SL}, {SR})", 22, LEFTC),
                 note(f"disparity network {I('g')}; our re-run").next_to(dmap, DOWN, buff=0.12),
                 note(f"{lo:+.1f} … {hi:+.1f} px per view (viridis)", 15).next_to(dmap, DOWN, buff=0.4))
        eqw = tx(EQW, 22).move_to([0, -2.45, 0])
        self.add(eqw)
        self.caption(f"A CNN {I('g')} predicts a disparity map from the pair; backward warping {SL} gives every view.")
        fl = G("Fl336.npy")
        fe = G("Fl_err336.npy")
        sel = [4, 0]
        V1 = arr_img(fl[4, 0], 2.6).move_to([-1.2, 1.2, 0])
        E1 = arr_img(fe[4, 0], 2.6).move_to([1.6, 1.2, 0])
        live(V1, lambda: fl[sel[0], sel[1]])
        live(E1, lambda: fe[sel[0], sel[1]])
        g = ViewGrid([4.9, 1.2, 0], 0.26)
        self.add(V1, framed(V1, LEFTC, 1.2), E1, framed(E1, DIM, 1.2), g.mobs, g.fill([(ROW, JL)], LEFTC),
                 always_redraw(lambda: label(V1, f"{FL} view {V_(sel[0], sel[1])}", 20, LEFTC)),
                 label(E1, f"error |{FL} − {FT}|", 20, RED),
                 always_redraw(lambda: g.box(sel[0], sel[1], WHITE, 2.5)),
                 note("our re-run, HCI origami (test scene)").next_to(E1, DOWN, buff=0.12))
        t = ValueTracker(0)

        def upd(m):
            v = t.get_value()
            if v <= 1:
                sel[0], sel[1] = 4, int(round(8 * v))
            else:
                sel[0], sel[1] = int(round(4 - 4 * (v - 1))), 8
        V1.add_updater(lambda m: upd(m))
        self.hold()
        self.caption(f"The further a view is from {SL}, the larger the error, and occluded regions cannot be filled.")
        self.run(t, 1, 3.5)
        self.run(t, 2, 1.6)
        self.hold(0.5)
        self.clear_all()


# ================================================================== S04 Left-right consistency
class S04_Consistency(Base):
    def construct(self):
        self.title_card("Part 4", "Left-right consistency")
        self.header("4 · Left-right consistency")
        sl = G("sl504.npy")
        sr = G("sr504.npy")
        th = {k: G(f"{k}_thumbs.npy") for k in ("Fl", "Fr")}
        gt = G("gt_thumbs.npy")

        def mosaic(t, cell=0.105):
            g = ViewGrid([0, 0, 0], cell, t, gap=0.008)
            return g.mobs

        rows = []
        for y, a, b, D, F, c, nm, dn in ((1.45, sl, sr, "d_lr_rgb.npy", "Fl", LEFTC, FL, DLR),
                                         (-1.05, sr, sl, "d_rl_rgb.npy", "Fr", RIGHTC, FR, DRL)):
            ia = arr_img(a, 1.05).move_to([-5.9, y + 0.12, 0])
            ib = arr_img(b, 1.05).move_to([-5.7, y - 0.12, 0])
            gbox = RoundedRectangle(corner_radius=0.12, width=1.3, height=1.0, color=AMBER, stroke_width=2.5).move_to([-3.9, y, 0])
            gt_ = tx(I("g"), 34, AMBER).move_to(gbox)
            dm = arr_img(G(D), 1.2).move_to([-1.85, y, 0])
            fm = mosaic(th[F]).move_to([1.05, y, 0])
            ar1 = arrow([-5.05, y, 0], gbox.get_left(), color=DIM)
            ar2 = arrow(gbox.get_right(), dm.get_left(), color=DIM)
            ar3 = arrow(dm.get_right(), fm.get_left(), color=c)
            wl = tx(f"warp {SL if F == 'Fl' else SR}", 17, c).next_to(ar3, UP, buff=0.05)
            grp = Group(ib, ia, framed(ia, c, 2), ar1, gbox, gt_, ar2, dm, label(dm, dn, 20, c), ar3, wl, fm,
                        label(fm, nm, 22, c))
            rows.append(grp)
        order = tx("inputs swapped", 17, RIGHTC).move_to([-5.8, -2.0, 0])
        same = tx("same weights", 17, AMBER).move_to([-3.9, 0.2, 0])
        self.play(FadeIn(rows[0]), run_time=0.6)
        self.caption(f"The network {I('g')} reads the pair as ({SL}, {SR}) and predicts {DLR}, which warps {SL} into {FL}.")
        self.hold()
        self.play(FadeIn(rows[1]), run_time=0.6)
        self.add(order, same, DashedLine([-3.9, 0.95, 0], [-3.9, -0.55, 0], color=AMBER, stroke_width=2))
        self.caption(f"The same {I('g')} runs again with the pair swapped: {DRL} warps {SR} into {FR}.")
        tg = mosaic(gt).move_to([4.6, 0.2, 0])
        self.add(tg, label(tg, f"target {FT}", 22, WHITE))
        a1 = CurvedArrow([1.95, 1.45, 0], [3.55, 0.6, 0], color=RED, angle=-0.4)
        a2 = CurvedArrow([1.95, -1.05, 0], [3.55, -0.2, 0], color=RED, angle=0.4)
        self.hold()
        self.play(Create(a1), Create(a2), run_time=0.6)
        eq1 = tx(EQ1, 24).move_to([0, -2.35, 0])
        self.add(eq1, tx(LC, 22, RED).move_to([2.75, 0.2, 0]))
        self.caption("Both light fields must match the target: the left-right consistency loss.")
        eq2 = fitw(tx(EQ2, 21, DIM), 12.5).move_to([0, -2.85, 0])
        self.hold()
        self.add(eq2)
        self.caption("A total-variation loss keeps both disparity maps smooth.")
        self.hold(0.6)
        self.clear_all()


# ================================================================== S05 Distance-weighted alpha blending
class S05_Blending(Base):
    def construct(self):
        self.title_card("Part 5", "Distance-weighted alpha blending")
        self.header("5 · Distance-weighted alpha blending")
        M = metrics()["mean"]
        H = {k: np.array(M["mse_" + k]) for k in ("Fl", "Fr", "Fc", "Fo")}
        vmax = max(H["Fl"].max(), H["Fr"].max())
        xs = [-4.4, 0, 4.4]
        hh = 2.3
        maps = []
        for x, k, c, nm in zip(xs, ("Fl", "Fr", "Fc"), (LEFTC, RIGHTC, GREEN),
                               (f"{FL}: warped from {SL}", f"{FR}: warped from {SR}", f"{FC}: blended")):
            m = heat_img(H[k], vmax, h=hh).move_to([x, 1.45, 0])
            maps.append(Group(m, framed(m, DIM, 1), label(m, nm, 22, c)))
        dots = VGroup(*[Dot(radius=0.07, color=WHITE).move_to(maps[q][0].get_center() + np.array([(jj - 4) * hh / 9, 0, 0]))
                        for q in range(3) for jj in (0, 8)])
        cb = colorbar(1.9, lo="low", hi="high").move_to([6.75, 1.45, 0])
        ntt = note("per-view mean squared error, our re-run, mean over the 4 HCI test scenes;  ● = input views"
                   ).move_to([0, 0.05, 0])
        self.add(maps[0], maps[1], cb, ntt, dots[:4])
        self.caption("Error grows with the distance from the photo each light field was warped from.")
        f2 = img("fig2_error.png", h=1.75).move_to([-0.9, -1.3, 0])
        f2n = note("paper Fig. 2: left, right, blended", 16).next_to(f2, RIGHT, buff=0.25)
        self.hold()
        self.add(f2, f2n)
        self.caption(f"The paper's Fig. 2 shows the same pattern: {FL} is good on the left, {FR} on the right.")
        self.hold()
        self.remove(f2, f2n)
        # distance maps + alpha
        IJ = np.stack(np.meshgrid(np.arange(9), np.arange(9), indexing="ij"), -1).astype(float)
        Kl = np.linalg.norm(IJ - [4, 0], axis=-1)
        Kr = np.linalg.norm(IJ - [4, 8], axis=-1)
        Al = Kr / (Kl + Kr)
        am = heat_img(Al, 1.0, "gray", h=1.9).move_to([0, -1.45, 0])
        nums = VGroup()
        for (i, j) in ((4, 0), (4, 4), (4, 8)):
            nums.add(tx(f"{Al[i, j]:.1f}", 18, RED if Al[i, j] > 0.5 else AMBER).move_to(
                am.get_center() + np.array([(j - 4) * 1.9 / 9, -(i - 4) * 1.9 / 9, 0])))
        eqs = VGroup(tx(EQ3, 21), tx(EQ4, 21)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([-4.4, -1.35, 0])
        eqn = note("distance of view (i, j) to each input", 16).next_to(eqs, DOWN, buff=0.1)
        eqa = VGroup(tx(EQA, 23, AMBER), tx(EQ5, 23)).arrange(DOWN, buff=0.14, aligned_edge=LEFT).move_to([4.3, -1.35, 0])
        aml = label(am, f"{AL}  (white = 1, black = 0)", 19, AMBER)
        self.remove(ntt)
        self.add(eqs, eqn)
        self.caption("Measure how far each view is from the left and the right input views.")
        self.hold()
        self.add(am, framed(am, DIM, 1), aml, nums, eqa)
        self.caption(f"Weight each view by its distance: close to {SL}, trust {FL}; close to {SR}, trust {FR}.")
        self.hold()
        self.add(maps[2], dots[4:])
        self.caption("The blended light field is accurate everywhere, not just near one input.")
        self.hold()
        self.play(*[FadeOut(m) for m in (eqs, eqn, am, aml, nums, eqa)], *[FadeOut(m) for m in self.mobjects
                                                                            if isinstance(m, SurroundingRectangle) and m.get_center()[1] < -0.5],
                  run_time=0.4)
        r = metrics()["mean"]
        ab = VGroup(tx(f"our re-run, HCI test (before refinement):  {FL} only {r['Fl']:.2f} dB  ·  "
                       f"average ({FL} + {FR})/2  {r['avg']:.2f} dB  ·  distance-weighted  {r['Fc']:.2f} dB", 21, WHITE),
                    tx("paper Table 3 (full model):  average blending 31.92 / 28.95 dB  →  distance-weighted 32.18 / 29.73 dB"
                       "  (INRIA / HCI)", 21, AMBER)).arrange(DOWN, buff=0.22)
        fitw(ab, 13.2).move_to([0, -1.2, 0])
        self.add(ab)
        self.caption("Paper ablation: distance-weighted beats plain averaging, +0.26 dB on INRIA and +0.78 dB on HCI.")
        self.hold(0.6)
        self.clear_all()


# ================================================================== S06 PSAS refinement
class S06_Refine(Base):
    def construct(self):
        self.title_card("Part 6", "Refinement: a shuffling 3D CNN")
        self.header("6 · Patch-wise spatial-angular shuffling (PSAS)")
        th = G("gt_thumbs.npy")
        grid = ViewGrid([-3.9, 0.35, 0], 0.37, th)
        p, q = PATCH
        ps = 0.37 / 9
        self.add(grid.mobs, label(grid.mobs, f"{FC}: 81 views (tensor 81 × {I('H')} × {I('W')} × 3)", 20, WHITE))
        marks = VGroup(*[Square(ps, color=AMBER, stroke_width=1.6).move_to(
            grid.pos(i, j) + np.array([(q - 4) * ps, -(p - 4) * ps, 0])) for i in range(N) for j in range(N)])
        self.caption("Warping cannot fix occlusions or shiny surfaces, so a 3D CNN refines the light field.")
        self.hold()
        self.play(Create(marks), run_time=0.8)
        self.caption("Shuffle: gather the same small patch from all 81 views and tile them into one image.")
        mos = arr_img(G("shuffle.npy"), 3.5).move_to([3.3, 0.35, 0])
        tiles = VGroup(*[Square(3.5 / 9, stroke_width=0, fill_color=AMBER, fill_opacity=0.35).move_to(
            mos.get_center() + np.array([(j - 4) * 3.5 / 9, -(i - 4) * 3.5 / 9, 0])) for i in range(N) for j in range(N)])
        flies = [marks[k].copy() for k in range(81)]
        self.play(LaggedStart(*[Transform(flies[k], tiles[k]) for k in range(81)], lag_ratio=0.02), run_time=2.4)
        self.add(mos, framed(mos, AMBER, 1.5), label(mos, "one patch, seen from all 81 viewpoints", 20, AMBER))
        self.remove(*flies)
        self.add(note("the authors' shuffle on the ground truth; tile = 56 × 56 px").next_to(mos, DOWN, buff=0.12))
        self.hold()
        blk = VGroup()
        names = ["3D conv", "shuffle", "3D conv"]
        for k, s in enumerate(names):
            b = RoundedRectangle(corner_radius=0.1, width=1.5, height=0.55, color=AMBER if "conv" in s else CYAN,
                                 stroke_width=2).move_to([-2.2 + 2.2 * k, -2.2, 0])
            blk.add(b, tx(s, 20).move_to(b))
        for k in range(2):
            blk.add(arrow([-2.2 + 2.2 * k + 0.78, -2.2, 0], [-2.2 + 2.2 * (k + 1) - 0.78, -2.2, 0], color=DIM))
        blk.add(tx("× 5 PSAS blocks", 20, DIM).move_to([4.3, -2.2, 0]))
        self.add(blk)
        self.caption("A PSAS block: 3D conv, shuffle, 3D conv. The same kernels now see views instead of pixels.")
        self.hold()
        self.clear_all()
        # what is left after blending (our re-run) + the paper's refinement gain
        i, j, by, bx, Z = [int(v) for v in G("zoom_box.npy")]
        xs = [-4.2, 0, 4.2]
        for x, k, nm, c in zip(xs, ("Fl", "Fc", "gt"), (f"{FL} only (warped {SL})", f"{FC} blended", f"{FT} ground truth"),
                               (LEFTC, GREEN, WHITE)):
            a = arr_img(G(f"zoom_{k}.npy"), 2.4).move_to([x, 1.5, 0])
            self.add(a, framed(a, c, 1.5), label(a, nm, 21, c))
            if k != "gt":
                e = arr_img(G(f"zoom_err_{k}.npy"), 1.3).move_to([x, -0.45, 0])
                self.add(e, framed(e, DIM, 1), tx("error", 17, RED).next_to(e, RIGHT, buff=0.12))
        self.add(note(f"our re-run, HCI origami, view {V_(i, j)}, zoom {Z} × {Z} px").move_to([0, -1.32, 0]))
        self.caption("Our re-run: blending removes most warping error; what is left sits on occlusion edges.")
        eq7 = fitw(tx(EQ7, 22), 13).move_to([0, -1.8, 0])
        self.hold()
        self.add(eq7, fitw(tx("paper Table 3: PSAS refiner 32.18 / 29.73 dB vs Srinivasan et al.'s 3D CNN 31.33 / 29.54 dB "
                              "(INRIA / HCI)", 19, AMBER), 13).move_to([0, -2.2, 0]),
                 note("our short re-run's refiner stayed close to identity (30.91 → 30.91 dB), so this gain is the paper's",
                      16).move_to([0, -2.58, 0]))
        self.caption("Trained end to end; in the paper, PSAS beats a standard 3D CNN refiner on both datasets.")
        self.hold(0.6)
        self.clear_all()


# ================================================================== S07 Our re-run
class S07_Rerun(Base):
    def construct(self):
        self.title_card("Part 7", "Re-running the authors' code")
        self.header("7 · A short re-run on HCI")
        snaps = G("snaps.npy")
        sd = G("snaps_d.npy")
        its = G("snap_its.npy")
        log = json.load(open(os.path.join(REPRO, "log.json")))
        k = ValueTracker(0)
        im = arr_img(snaps[0], 3.1).move_to([-4.6, 0.75, 0])
        dm = arr_img(sd[0], 3.1).move_to([-1.3, 0.75, 0])
        live(im, lambda: snaps[int(round(k.get_value()))])
        live(dm, lambda: sd[int(round(k.get_value()))])
        self.add(im, framed(im, GREEN, 1.5), dm, framed(dm, LEFTC, 1.5), label(im, f"{FO}, centre view (test scene)", 20, GREEN),
                 label(dm, f"{DLR}", 20, LEFTC))
        itl = always_redraw(lambda: tx(f"training step {int(its[int(round(k.get_value()))])}", 22, AMBER).move_to([-2.95, -1.15, 0]))
        self.add(itl)
        ax = Axes(x_range=[0, int(its[-1]), 2000], y_range=[20, 36, 4], x_length=4.6, y_length=3.0,
                  axis_config=dict(color=DIM, stroke_width=1.5, include_ticks=True), tips=False).move_to([4.2, 0.7, 0])
        xs = np.array([r["it"] for r in log])
        ys = np.array([r["psnr_o"] for r in log])
        k9 = 9
        ys_s = np.array([ys[max(0, n - k9 + 1):n + 1].mean() for n in range(len(ys))])      # trailing mean
        pts = [ax.c2p(x, float(np.clip(y, 20, 36))) for x, y in zip(xs, ys_s)]
        curve = VMobject(color=GREEN, stroke_width=2.5).set_points_as_corners(pts)
        self.add(ax, tx("training PSNR (135 × 135 crops)", 18, DIM).next_to(ax, UP, buff=0.12),
                 tx("dB", 16, DIM).next_to(ax, LEFT, buff=0.1).shift(UP * 1.2),
                 tx("step", 16, DIM).next_to(ax, DOWN, buff=0.1).shift(RIGHT * 2.0))
        for yv in (20, 28, 36):
            self.add(tx(str(yv), 14, DIM).next_to(ax.c2p(0, yv), LEFT, buff=0.08))
        self.caption("We re-ran the authors' networks with the paper's losses on its HCI split: 20 train, 4 test scenes.")
        self.play(k.animate.set_value(len(its) - 1), Create(curve), run_time=6.0, rate_func=lambda x: x ** 1.6)
        r = metrics()["mean"]
        nparams = "852,514 + 886,788 = 1.74M parameters (paper Table 2: 1.74M)"
        res = VGroup(tx(f"our short re-run ({int(its[-1])} steps): {r['Fo']:.2f} dB on the 4 HCI test scenes", 22, GREEN),
                     tx("paper: 29.73 dB (2000 epochs; own crops and evaluation, so not directly comparable)", 19, DIM),
                     tx(nparams, 19, DIM)).arrange(DOWN, buff=0.12).move_to([0, -2.4, 0])
        self.add(res)
        self.caption("Disparity net plus PSAS refiner: exactly the paper's 1.74M parameters.")
        self.hold(0.8)
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results")
        self.header("8 · Results")
        data = {"INRIA": (27.43, 27.08, 32.18), "HCI": (23.60, 23.18, 29.73), "Stanford": (17.75, 23.78, 25.85)}
        names = [("Srinivasan et al. (1 view)", BLUE), ("Kalantari et al. (4 views)", PURPLE), ("Ours (2 views)", GREEN)]
        ax = Axes(x_range=[0, 3, 1], y_range=[15, 35, 5], x_length=8.4, y_length=4.0,
                  axis_config=dict(color=DIM, stroke_width=1.5), tips=False).move_to([-1.3, 0.55, 0])
        self.add(ax)
        for yv in (15, 20, 25, 30, 35):
            self.add(tx(str(yv), 16, DIM).next_to(ax.c2p(0, yv), LEFT, buff=0.1))
        self.add(tx("PSNR (dB), full light field", 18, DIM).next_to(ax, UP, buff=0.1).shift(LEFT * 2.6))
        bars = []
        for gi, (ds, vals) in enumerate(data.items()):
            self.add(tx(ds, 22).move_to(ax.c2p(gi + 0.5, 15) + DOWN * 0.3))
            for bi, v in enumerate(vals):
                x = gi + 0.2 + 0.3 * bi
                b = Rectangle(width=0.8, height=ax.c2p(0, v)[1] - ax.c2p(0, 15)[1], stroke_width=0,
                              fill_color=names[bi][1], fill_opacity=0.9)
                b.move_to(ax.c2p(x, 15), aligned_edge=DOWN)
                t = tx(f"{v:.2f}", 15, names[bi][1]).next_to(b, UP, buff=0.06)
                bars.append((b, t))
        leg = VGroup(*[VGroup(Square(0.22, stroke_width=0, fill_color=c, fill_opacity=0.9), tx(n, 20, c)).arrange(RIGHT, buff=0.12)
                       for n, c in names]).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([4.6, 1.3, 0])
        self.add(leg)
        self.caption("Paper Table 1: on all three datasets, two views beat one view and four corner views.")
        self.play(*[GrowFromEdge(b, DOWN) for b, _ in bars], run_time=1.2)
        self.add(*[t for _, t in bars])
        head = VGroup(tx("+4.75 dB  INRIA", 24, AMBER), tx("+6.13 dB  HCI", 24, AMBER), tx("+2.07 dB  Stanford", 24, AMBER),
                      tx("over the best baseline", 18, DIM)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([4.6, -0.6, 0])
        self.hold()
        self.add(head)
        zl = tx("horizontal views only, vs Zhang et al. 2015:  34.18 vs 28.87 · 31.50 vs 23.06 · 29.06 vs 25.72 dB"
                "  (Zhang keeps the higher SSIM on Stanford)", 17, DIM).move_to([0, -2.5, 0])
        self.add(fitw(zl, 13.2))
        self.caption("That is +4.75, +6.13 and +2.07 dB over the best competitor, and far above Zhang et al.")
        self.hold(0.3)
        self.clear_all()
        # Fig. 3 crops
        for tag, sc, capt in (("r0R", "HCI origami", "Paper Fig. 3, origami: the baselines smear the crane and the flowers."),
                              ("r2R", "Stanford amethyst", "Stanford amethyst, large disparity: the single-view method breaks apart.")):
            xs = [-4.8, -1.6, 1.6, 4.8]
            for x, (m_, nm, c) in zip(xs, (("gt", "ground truth", WHITE), ("kal", "Kalantari et al.", PURPLE),
                                           ("sri", "Srinivasan et al.", BLUE), ("ours", "Ours", GREEN))):
                a = img(f"fig3/{tag}_{m_}0.png", h=2.3).move_to([x, 1.3, 0])
                b = img(f"fig3/{tag}_{m_}1.png", h=2.3).move_to([x, -1.08, 0])
                self.add(a, b, label(a, nm, 22, c))
            self.add(note(f"paper Fig. 3: {sc}, zoomed crops of synthesized corner views").move_to([0, -2.65, 0]))
            self.caption(capt)
            self.hold(0.8)
            self.clear_all(run_time=0.4)
        # Table 2: size and speed
        rows = [("Srinivasan et al.", "2.94M", 0.113, "GPU", BLUE), ("Kalantari et al.", "1.64M", 34.51, "GPU", PURPLE),
                ("Zhang et al.", "no learning", 514.84, "CPU", DIM), ("Ours", "1.74M", 0.235, "GPU", GREEN)]
        lx = lambda s: -1.2 + 1.55 * (np.log10(s) + 1)
        self.add(tx("parameters", 20, DIM).move_to([-3.0, 2.3, 0]), tx("time for one 7 × 7 light field (log scale)", 20, DIM
                                                                        ).move_to([2.2, 2.3, 0]))
        for r_, (n_, p_, t_, dv, c) in enumerate(rows):
            y = 1.5 - 0.8 * r_
            self.add(tx(n_, 24, c).move_to([-5.4, y, 0]), tx(p_, 24, c).move_to([-3.0, y, 0]))
            bar = Rectangle(width=max(lx(t_) + 1.2, 0.05), height=0.4, stroke_width=0, fill_color=c, fill_opacity=0.85)
            bar.move_to([-1.2, y, 0], aligned_edge=LEFT)
            self.play(GrowFromEdge(bar, LEFT), run_time=0.35)
            self.add(tx(f"{t_:g} s ({dv})", 20, c).next_to(bar, RIGHT, buff=0.12))
        self.add(note("paper Table 2 (INRIA)").move_to([0, -1.9, 0]))
        self.caption("The whole light field in one pass: 0.235 s, about 147 times faster than Kalantari et al.")
        self.hold(0.5)
        self.clear_all()
        # refocusing the paper's synthesized light field
        rf = G("refocus_gif.npy")
        a = ValueTracker(0.0)
        big = arr_img(interp(rf, a_index(0.0)), 3.9).move_to([-3.1, 0.55, 0])
        live(big, lambda: interp(rf, a_index(a.get_value())))
        sl, dot = focus_slider([-3.1, -2.05, 0], 3.6, a.get_value)
        self.add(big, framed(big, GREEN, 1.5), label(big, "refocusing the synthesized light field", 22, GREEN), sl, dot,
                 note("the paper's 9 × 9 output from 2 views (website teaser), shift-and-add", 16).move_to([-3.1, -2.6, 0]))
        f4 = img("fig4_refocus.png", h=3.9).move_to([3.5, 0.55, 0])
        self.caption("Refocus the light field synthesized from just two views: the wall, then the crane.")
        self.run(a, A_WALL, 2.0, smooth)
        self.run(a, A_CRANE, 3.0, smooth)
        self.add(f4, note("paper Fig. 4: phone stereo pair", 16).next_to(f4, DOWN, buff=0.08))
        self.caption("Trained on INRIA, it also works on stereo shot by sliding a phone sideways (paper Fig. 4).")
        self.run(a, A_WALL, 3.0, smooth)
        self.hold(0.4)
        self.clear_all()


# ================================================================== S09 Ablations
class S09_Ablations(Base):
    def construct(self):
        self.title_card("Part 9", "What matters")
        self.header("9 · Ablations (paper Table 3)")
        rows = [("F_l only (left view)", 30.67, 27.50, LEFTC), ("F_r only (right view)", 31.14, 27.41, RIGHTC),
                ("baseline 3D CNN refiner", 31.33, 29.54, DIM), ("average blending (F_l + F_r)/2", 31.92, 28.95, DIM),
                ("w/o left-right consistency", 32.07, 29.71, DIM), ("w/o disparity smoothing", 32.11, 29.48, DIM),
                ("full model", 32.18, 29.73, GREEN)]
        x0, x1 = 30.0, 32.4
        L = 6.0
        self.add(tx("INRIA PSNR (dB)", 20, DIM).move_to([0.9, 2.7, 0]), tx("HCI", 20, DIM).move_to([5.7, 2.7, 0]))
        for r, (n_, a, b, c) in enumerate(rows):
            y = 2.05 - 0.62 * r
            nm = n_.replace("F_l", FL).replace("F_r", FR)
            self.add(tx(nm, 21, c).move_to([-4.3, y, 0]))
            bar = Rectangle(width=(a - x0) / (x1 - x0) * L, height=0.38, stroke_width=0, fill_color=c, fill_opacity=0.85)
            bar.move_to([-2.1, y, 0], aligned_edge=LEFT)
            self.play(GrowFromEdge(bar, LEFT), run_time=0.25)
            self.add(tx(f"{a:.2f}", 18, c).next_to(bar, RIGHT, buff=0.1), tx(f"{b:.2f}", 18, c).move_to([5.7, y, 0]))
        self.add(note("bars start at 30 dB").move_to([0.9, -2.35, 0]))
        self.caption("Both views plus distance-weighted blending matter most: +1.51 dB over the left view alone.")
        self.caption("PSAS adds 0.85 dB over Srinivasan et al.'s 3D CNN refiner on INRIA; consistency and smoothing add a little.")
        self.hold(0.6)
        self.clear_all()


# ================================================================== S10 Outlook
class S10_Outlook(Base):
    def construct(self):
        self.header("10 · Summary")
        pts = VGroup(
            tx(f"1.  One disparity network, run both ways, warps each photo into a light field;", 24),
            tx(f"     the left-right consistency loss {LC} makes both match the target", 24),
            tx("2.  Distance-weighted alpha blending trusts whichever input view is closer", 24),
            tx("3.  A PSAS 3D CNN fixes occlusions: 1.74M parameters in total", 24),
            tx("+4.75 dB (INRIA)  ·  +6.13 dB (HCI)  ·  +2.07 dB (Stanford) over the best baseline", 24, AMBER),
            tx("limits: needs dense light fields for training · a horizontal stereo baseline only", 20, DIM),
            tx("follow-up: stereo → light field for near-eye AR light field displays (ISMAR / TVCG 2022)", 20, DIM),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        fitw(pts, 13).move_to([0, 0.5, 0])
        self.add(pts)
        self.caption("Two photos in, a full light field out: consistent, blended by distance, refined across views.")
        self.hold(0.5)
        self.clear_all()
        gif = G("gif336.npy")
        card = arr_img(gif[4, 4], 3.0).move_to([0, 0.8, 0])
        live(card, lambda: gif[view_path(self.clock.get_value())])
        self.add(card, framed(card, DIM, 1.2))
        self.add(fitw(tx("Robust Light Field Synthesis From Stereo Images with Left-Right Geometric Consistency", 30, bold=True)
                      ).move_to([0, -1.25, 0]),
                 tx("IEEE ICIP 2021, pp. 1844–1848  ·  code: github.com/bchao1/lf-flow", 22, DIM).move_to([0, -1.85, 0]),
                 tx("Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen  ·  National Taiwan University", 22, DIM
                    ).move_to([0, -2.3, 0]))
        self.wait(4.5)
