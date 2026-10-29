"""Robust Light Field Synthesis From Stereo Images With Left-Right Geometric Consistency — 1-minute teaser, v1.

Condensed from the 5-min teaser (../animation_slf_v*.py). Style, helpers and cached assets come from ../slf_common.py
(read-only here: the caches in ../data/generated are built by the parent render.sh).
Scenes: T1_Hook, T2_Method, T3_Payoff.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from slf_common import *  # noqa: E402


class T1_Hook(Base):
    def construct(self):
        title = fitw(tx("Robust Light Field Synthesis From Stereo Images", 46, bold=True)).move_to([0, 3.15, 0])
        subt = tx("with Left-Right Geometric Consistency", 32).move_to([0, 2.55, 0])
        gif = G("gif336.npy")
        card = arr_img(gif[4, 4], 3.3).move_to([-1.2, -0.05, 0])
        live(card, lambda: gif[view_path(self.clock.get_value())])
        grid = ViewGrid([2.75, -0.05, 0], 0.2)
        inp = grid.fill([(ROW, JL), (ROW, JR)], AMBER)
        cur = always_redraw(lambda: grid.box(*view_path(self.clock.get_value()), color=WHITE, sw=2.5))
        gl1 = tx("2 input views", 20, AMBER).next_to(grid.mobs, UP, buff=0.42)
        gl2 = tx("79 synthesized", 20, GREEN).next_to(gl1, DOWN, buff=0.08)
        auth = tx("Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen  ·  National Taiwan University  ·  IEEE ICIP 2021",
                  20, DIM).move_to([0, -2.45, 0])
        self.add(title, subt, auth)
        self.play(FadeIn(card), Create(framed(card, DIM, 1.2)), FadeIn(grid.mobs), run_time=0.5)
        self.add(inp, cur, gl1, gl2)
        self.caption("Two photos in, a full 9 × 9 light field out.")
        self.hold(2.0)
        self.clear_all()
        # the problem: 2 of 81 views
        th = G("gt_thumbs.npy")
        g = ViewGrid([-3.4, 0.3, 0], 0.37, th)
        for i in range(N):
            for j in range(N):
                if (i, j) not in ((ROW, JL), (ROW, JR)):
                    g.at(i, j).set_opacity(0.22)
        self.play(FadeIn(g.mobs), run_time=0.5)
        self.add(g.box(ROW, JL, LEFTC, 4), g.box(ROW, JR, RIGHTC, 4),
                 label(g.mobs, "a light field: 9 × 9 views", 22, DIM),
                 tx("stereo pair", 20, AMBER).next_to(g.mobs, DOWN, buff=0.2))
        self.caption("A light field is a grid of views. A stereo pair gives you only 2 of the 81.")
        self.hold()
        rf = G("refocus_gt.npy")
        a = ValueTracker(0.0)
        big = arr_img(interp(rf, a_index(0.0)), 3.5).move_to([2.9, 0.3, 0])
        live(big, lambda: interp(rf, a_index(a.get_value())))
        sl, dot = focus_slider([2.9, -2.1, 0], 3.2, a.get_value)
        self.add(big, framed(big, DIM, 1), label(big, "all 81 views: refocus after the shot", 22, AMBER), sl, dot)
        for i in range(N):
            for j in range(N):
                g.at(i, j).set_opacity(1)
        self.caption("With all 81 you can refocus after the shot, and change viewpoint.")
        self.run(a, A_WALL, 1.6, smooth)
        self.run(a, A_CRANE, 2.2, smooth)
        self.hold()
        self.clear_all(run_time=0.5)


class T2_Method(Base):
    def construct(self):
        sl = G("sl504.npy")
        L_ = arr_img(sl, 2.3).move_to([-5.3, 1.3, 0])
        dmap = arr_img(G("d_lr_rgb.npy"), 2.3).move_to([-5.3, -1.25, 0])
        self.add(L_, framed(L_, LEFTC, 2), label(L_, SL, 22, LEFTC), dmap, framed(dmap, LEFTC, 1.2),
                 label(dmap, f"disparity {DLR}", 20, LEFTC))
        fl = G("Fl336.npy")
        fe = G("Fl_err336.npy")
        sel = [4, 0]
        V1 = arr_img(fl[4, 0], 2.7).move_to([-1.9, 0.6, 0])
        E1 = arr_img(fe[4, 0], 2.7).move_to([1.0, 0.6, 0])
        live(V1, lambda: fl[sel[0], sel[1]])
        live(E1, lambda: fe[sel[0], sel[1]])
        g = ViewGrid([4.6, 0.6, 0], 0.27)
        self.add(V1, framed(V1, LEFTC, 1.2), E1, framed(E1, DIM, 1.2), g.mobs, g.fill([(ROW, JL)], LEFTC),
                 always_redraw(lambda: label(V1, f"warped view {V_(sel[0], sel[1])}", 20, LEFTC)),
                 label(E1, "error", 20, RED), always_redraw(lambda: g.box(sel[0], sel[1], WHITE, 2.5)),
                 note("our re-run of the authors' code, HCI origami").move_to([0, -1.2, 0]))
        t = ValueTracker(0)
        V1.add_updater(lambda m: sel.__setitem__(1, int(round(8 * t.get_value()))))
        self.caption("A CNN predicts disparity; warping the photo fills every view, but errors grow with distance.")
        self.run(t, 1, 3.2)
        self.hold()
        self.clear_all(run_time=0.5)
        # both ways + blend
        M = metrics()["mean"]
        H = {k: np.array(M["mse_" + k]) for k in ("Fl", "Fr", "Fc")}
        vmax = max(H["Fl"].max(), H["Fr"].max())
        xs = [-4.4, 0, 4.4]
        maps = []
        for x, k, c, nm in zip(xs, ("Fl", "Fr", "Fc"), (LEFTC, RIGHTC, GREEN),
                               (f"from {SL}", f"from {SR} (same network, inputs swapped)", "distance-weighted blend")):
            m = heat_img(H[k], vmax, h=2.6).move_to([x, 0.7, 0])
            maps.append(Group(m, framed(m, DIM, 1), label(m, nm, 21, c)))
        self.add(maps[0], maps[1], colorbar(2.0).move_to([6.75, 0.7, 0]),
                 note("per-view error, our re-run, 4 HCI test scenes").move_to([0, -1.05, 0]))
        eq = tx(f"{AL} = {KR} / ({KL} + {KR})      {FC} = {AL} · {FL} + (1 − {AL}) · {FR}", 24, AMBER).move_to([0, -1.9, 0])
        self.caption("Run it both ways and keep the two light fields consistent with each other.")
        self.hold()
        self.add(maps[2], eq)
        self.caption("Blend each view by its distance to the two photos: low error everywhere.")
        self.hold()
        self.clear_all(run_time=0.5)
        # PSAS shuffle + refinement
        th = G("gt_thumbs.npy")
        grid = ViewGrid([-4.4, 0.55, 0], 0.33, th)
        p, q = PATCH
        ps = 0.33 / 9
        marks = VGroup(*[Square(ps, color=AMBER, stroke_width=1.5).move_to(
            grid.pos(i, j) + np.array([(q - 4) * ps, -(p - 4) * ps, 0])) for i in range(N) for j in range(N)])
        mos = arr_img(G("shuffle.npy"), 3.0).move_to([-0.6, 0.55, 0])
        tiles = VGroup(*[Square(3.0 / 9, stroke_width=0, fill_color=AMBER, fill_opacity=0.35).move_to(
            mos.get_center() + np.array([(j - 4) * 3.0 / 9, -(i - 4) * 3.0 / 9, 0])) for i in range(N) for j in range(N)])
        self.add(grid.mobs, marks)
        flies = [marks[k].copy() for k in range(81)]
        self.caption("A 3D CNN shuffles patches across views, then refines the light field.")
        self.play(LaggedStart(*[Transform(flies[k], tiles[k]) for k in range(81)], lag_ratio=0.02), run_time=1.8)
        self.remove(*flies)
        self.add(mos, framed(mos, AMBER, 1.2), label(mos, "one patch from all 81 views", 20, AMBER))
        zc = arr_img(G("zoom_Fl.npy"), 1.9).move_to([3.2, 1.5, 0])
        zo = arr_img(G("zoom_Fc.npy"), 1.9).move_to([5.3, 1.5, 0])
        ec = arr_img(G("zoom_err_Fl.npy"), 1.9).move_to([3.2, -0.55, 0])
        eo = arr_img(G("zoom_err_Fc.npy"), 1.9).move_to([5.3, -0.55, 0])
        self.add(zc, zo, ec, eo, label(zc, "warped only", 20, LEFTC), label(zo, "blended", 20, GREEN),
                 note("error (our re-run)", 16).next_to(Group(ec, eo), DOWN, buff=0.1),
                 tx("PSAS refiner vs a standard 3D CNN: +0.85 dB (paper, INRIA)", 20, AMBER).move_to([0, -2.4, 0]))
        self.caption("Blending removes most warping error; the refiner handles occlusion edges.")
        self.hold(0.8)
        self.clear_all(run_time=0.5)


class T3_Payoff(Base):
    def construct(self):
        data = {"INRIA": (27.43, 27.08, 32.18), "HCI": (23.60, 23.18, 29.73), "Stanford": (17.75, 23.78, 25.85)}
        names = [("1 view (Srinivasan et al.)", BLUE), ("4 views (Kalantari et al.)", PURPLE), ("2 views (ours)", GREEN)]
        ax = Axes(x_range=[0, 3, 1], y_range=[15, 35, 5], x_length=7.6, y_length=3.8,
                  axis_config=dict(color=DIM, stroke_width=1.5), tips=False).move_to([-2.0, 0.6, 0])
        self.add(ax, tx("PSNR (dB), paper Table 1", 18, DIM).next_to(ax, UP, buff=0.1).shift(LEFT * 2.4))
        for yv in (15, 25, 35):
            self.add(tx(str(yv), 16, DIM).next_to(ax.c2p(0, yv), LEFT, buff=0.1))
        bars = []
        for gi, (ds, vals) in enumerate(data.items()):
            self.add(tx(ds, 22).move_to(ax.c2p(gi + 0.5, 15) + DOWN * 0.3))
            for bi, v in enumerate(vals):
                b = Rectangle(width=0.72, height=ax.c2p(0, v)[1] - ax.c2p(0, 15)[1], stroke_width=0,
                              fill_color=names[bi][1], fill_opacity=0.9).move_to(ax.c2p(gi + 0.2 + 0.3 * bi, 15), aligned_edge=DOWN)
                bars.append(b)
        leg = VGroup(*[VGroup(Square(0.22, stroke_width=0, fill_color=c, fill_opacity=0.9), tx(n, 20, c)).arrange(RIGHT, buff=0.12)
                       for n, c in names]).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([4.5, 1.6, 0])
        head = VGroup(tx("+4.75 dB  INRIA", 26, AMBER), tx("+6.13 dB  HCI", 26, AMBER), tx("+2.07 dB  Stanford", 26, AMBER),
                      tx("1.74M parameters", 22, DIM)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([4.5, -0.5, 0])
        self.add(leg)
        self.play(*[GrowFromEdge(b, DOWN) for b in bars], run_time=1.0)
        self.add(head)
        self.caption("Better than one view or four corner views on all three datasets, with 1.74M parameters.")
        self.hold()
        self.clear_all(run_time=0.5)
        rf = G("refocus_gif.npy")
        a = ValueTracker(0.0)
        big = arr_img(interp(rf, a_index(0.0)), 4.2).move_to([0, 0.55, 0])
        live(big, lambda: interp(rf, a_index(a.get_value())))
        sl, dot = focus_slider([0, -2.2, 0], 3.8, a.get_value)
        self.add(big, framed(big, GREEN, 1.5), label(big, "the paper's light field from 2 views, refocused", 22, GREEN), sl, dot)
        self.caption("Refocus a light field synthesized from just two photos.")
        self.run(a, A_WALL, 1.6, smooth)
        self.run(a, A_CRANE, 2.4, smooth)
        self.hold()
        self.clear_all(run_time=0.5)
        gif = G("gif336.npy")
        card = arr_img(gif[4, 4], 3.0).move_to([0, 0.8, 0])
        live(card, lambda: gif[view_path(self.clock.get_value())])
        self.add(card, framed(card, DIM, 1.2),
                 fitw(tx("Robust Light Field Synthesis From Stereo Images with Left-Right Geometric Consistency", 30, bold=True)
                      ).move_to([0, -1.25, 0]),
                 tx("IEEE ICIP 2021  ·  code: github.com/bchao1/lf-flow", 22, DIM).move_to([0, -1.85, 0]),
                 tx("Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen  ·  National Taiwan University", 22, DIM
                    ).move_to([0, -2.3, 0]))
        self.wait(4.5)
