def build_assets_v2():
    """v2 extras (numpy only): focus-weight maps from the cached ground-truth refocused images, shown with gamma 0.5."""
    if os.path.exists(gen("done_v2")):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen("done_v2")):
            return
        wimg = np.load(gen("wimg.npy")).astype(float) / 255.
        out = []
        for g in wimg:
            g = g.mean(-1)
            U = np.zeros_like(g)
            U[:, 1:] += np.abs(np.diff(g, axis=1))
            U[1:, :] += np.abs(np.diff(g, axis=0))
            Ub = (U - U.min()) / (U.max() - U.min() + 1e-8)
            W = np.exp(2.0 * Ub)
            out.append(cmap_rgb(((W - 1) / (np.e ** 2 - 1)) ** 0.5, "inferno"))
        np.save(gen("wmap2.npy"), np.stack(out))
        open(gen("done_v2"), "w").write("ok")


build_assets_v2()


# ================================================================== S01 Eye and VAC
class S01_Eye(Base):
    def construct(self):
        self.title_card("Part 1", "How your eyes sense depth")
        self.header("1 · How your eyes sense depth")
        eye = Eye(center=(3.3, 0.3), R=1.75, pupil=0.5)
        xo = ValueTracker(-5.6)
        f = ValueTracker(eye.f_for(-5.6))
        body = eye.body()
        lens = always_redraw(lambda: eye.lens(f.get_value()))
        rays = always_redraw(lambda: eye.rays(xo.get_value(), f.get_value())[0])
        src = always_redraw(lambda: star_shape(0.18).move_to([xo.get_value(), eye.c[1], 0]))

        def blurmark():
            lo, hi = eye.rays(xo.get_value(), f.get_value())[1]
            x = eye.c[0] + np.sqrt(max(eye.R ** 2 - ((lo + hi) / 2 - eye.c[1]) ** 2, 0))
            if hi - lo < 0.06:
                return VGroup(Dot([x, (lo + hi) / 2, 0], radius=0.08, color=GREEN),
                              tx("sharp", 22, GREEN).move_to([x + 0.75, (lo + hi) / 2, 0]))
            seg = Line([x + 0.14, lo, 0], [x + 0.14, hi, 0], color=RED, stroke_width=8)
            return VGroup(seg, tx("blur spot", 22, RED).next_to(seg, RIGHT, buff=0.12))
        bm = always_redraw(blurmark)
        lr = tx("retina", 22, ORANGE).move_to([eye.c[0] + 1.0, eye.c[1] - eye.R - 0.1, 0])
        ll = tx("lens", 22, WHITE).move_to([eye.xL, eye.c[1] + eye.R + 0.05, 0])
        self.play(Create(body), FadeIn(lens), run_time=0.8)
        self.add(lr, ll, rays, src, bm)
        self.caption("Light from a point enters the pupil; the lens focuses it onto the retina.")
        self.hold(0.2)
        self.run(xo, -1.2, 2.2, smooth)
        self.caption("Bring it closer: the rays now meet behind the retina, and the point blurs.")
        self.hold()
        self.caption("The eye reshapes its lens to refocus (accommodation). Blur is a depth cue.")
        self.run(f, eye.f_for(-1.2), 2.0, smooth)
        self.clear_all()

        # --- vergence and the conflict (top view)
        cx = -3.0
        eyes = [np.array([cx - 0.85, -2.25, 0]), np.array([cx + 0.85, -2.25, 0])]

        def top_eye(p):
            return VGroup(Circle(radius=0.3, color=WHITE, stroke_width=2).move_to(p),
                          Dot(p + UP * 0.22, radius=0.08, color=BLUE))
        E = VGroup(*[top_eye(p) for p in eyes])
        obj = star_shape(0.22).move_to([cx, 0.6, 0])
        verg = VGroup(*[Line(p + UP * 0.3, obj.get_center(), color=AMBER, stroke_width=3) for p in eyes])

        def focus_line(y):
            ln = DashedLine([cx - 1.7, y, 0], [cx + 1.7, y, 0], color=CYAN, stroke_width=4)
            return VGroup(ln, tx("focus", 20, CYAN).next_to(ln, LEFT, buff=0.12))
        acc = focus_line(0.6)
        leg1 = tx("vergence: where the two eyes point", 24, AMBER).move_to([3.2, 1.9, 0])
        leg2 = tx("accommodation: where each eye focuses", 24, CYAN).move_to([3.2, 1.35, 0])
        self.play(FadeIn(E), FadeIn(obj), run_time=0.5)
        self.add(leg1, leg2)
        self.play(Create(verg), Create(acc), run_time=0.6)
        ok = tx("real world: both at the same distance ✓", 24, GREEN).move_to([3.2, 0.55, 0])
        self.add(ok)
        self.caption("Both eyes also turn to meet at the object (vergence), right where they focus.")
        self.hold()
        # stereo display
        self.remove(ok, acc)
        scr_y = 1.9
        scr = Line([cx - 2.0, scr_y, 0], [cx + 2.0, scr_y, 0], color=WHITE, stroke_width=5)
        lscr = tx("screen", 20, DIM).next_to(scr, RIGHT, buff=0.15)
        virt = star_shape(0.22, color=AMBER).move_to([cx, -0.1, 0]).set_opacity(0.55)
        hits = []
        for p in eyes:
            d = virt.get_center() - p
            s = (scr_y - p[1]) / d[1]
            hits.append(p + s * d)
        pix = VGroup(Dot(hits[0], radius=0.09, color=RED), Dot(hits[1], radius=0.09, color=BLUE))
        rays = VGroup(*[DashedLine(p + UP * 0.3, h, color=DIM, stroke_width=2) for p, h in zip(eyes, hits)])
        self.play(FadeOut(obj), FadeOut(verg), FadeIn(scr), run_time=0.5)
        self.add(lscr, pix, rays)
        self.caption("A stereo display draws each eye's image on one screen, at a fixed distance.")
        self.hold()
        verg2 = VGroup(*[Line(p + UP * 0.3, virt.get_center(), color=AMBER, stroke_width=3) for p in eyes])
        acc2 = focus_line(scr_y - 0.22)
        self.add(virt, verg2, acc2)
        bx = cx + 2.35
        br = VGroup(Line([bx, -0.1, 0], [bx, scr_y - 0.22, 0], color=RED, stroke_width=4),
                    Line([bx - 0.15, -0.1, 0], [bx + 0.15, -0.1, 0], color=RED, stroke_width=4),
                    Line([bx - 0.15, scr_y - 0.22, 0], [bx + 0.15, scr_y - 0.22, 0], color=RED, stroke_width=4))
        bad = VGroup(tx("vergence-accommodation", 26, RED), tx("conflict (VAC)", 26, RED)).arrange(DOWN, buff=0.08
                                                                                                    ).move_to([3.2, 0.4, 0])
        self.add(br, bad)
        self.caption("The eyes converge on the virtual object but focus on the screen: the vergence-accommodation conflict.")
        harm = tx("eye strain · headache · even loss of depth perception", 22, DIM).move_to([3.2, -0.4, 0])
        self.add(harm)
        self.hold()
        fixes = VGroup(tx("varifocal / multifocal: need eye trackers or screen stacks", 20, DIM),
                       tx("holographic: needs high-resolution SLMs", 20, DIM),
                       tx("light field displays: none of these", 20, GREEN)).arrange(DOWN, aligned_edge=LEFT, buff=0.14
                                                                                    ).move_to([3.2, -1.5, 0])
        self.add(fixes)
        self.caption("VAC causes eye strain, headaches, even lost depth. Light field displays fix it with no extra parts.")
        self.clear_all()


# ================================================================== S02 Light fields
class S02_LightField(Base):
    def construct(self):
        self.title_card("Part 2", "Light fields and refocusing")
        self.header("2 · Light fields and refocusing")
        eye = Eye(center=(4.5, 0.2), R=1.35, pupil=0.62)
        vx = -5.0
        disp_x = 0.6
        vp = star_shape(0.18).move_to([vx, 0.2, 0])
        disp = Line([disp_x, -1.9, 0], [disp_x, 2.3, 0], color=WHITE, stroke_width=5)
        ld = tx("display", 20, DIM).next_to(disp, DOWN, buff=0.12)
        lv = tx("virtual point", 20, AMBER).next_to(vp, DOWN, buff=0.15)
        ys = [-0.45, 0.0, 0.45]
        cols = [RED, GREEN, BLUE]
        fans = VGroup()
        for y, c in zip(ys, cols):
            p_pupil = np.array([eye.xL, eye.c[1] + y, 0])
            d = p_pupil - vp.get_center()
            s = (disp_x - vp.get_center()[0]) / d[0]
            p_disp = vp.get_center() + s * d
            fans.add(DashedLine(vp.get_center(), p_disp, color=c, stroke_width=2, stroke_opacity=0.6),
                     Line(p_disp, p_pupil, color=c, stroke_width=3), Dot(p_disp, radius=0.07, color=c))
        f = eye.f_for(vx)
        inside = VGroup()
        for y, c in zip(ys, cols):
            pL = np.array([eye.xL, eye.c[1] + y])
            di = 1 / (1 / f - 1 / (eye.xL - vx))
            q = np.array([eye.xL + di, eye.c[1]])
            e = eye.hit(pL, q - pL)
            inside.add(Line([pL[0], pL[1], 0], [e[0], e[1], 0], color=c, stroke_width=3))
        self.play(Create(eye.body()), FadeIn(eye.lens(f)), FadeIn(disp), run_time=0.7)
        self.add(ld)
        self.play(Create(fans), run_time=0.8)
        self.add(vp, lv, inside)
        lab = tx("3 views inside one pupil", 22, CYAN).move_to([eye.xL - 0.3, 2.3, 0])
        self.add(lab)
        self.caption("A light field display sends several views into each pupil, closer together than the pupil.")
        self.caption("Their rays meet behind the display, so the eye can truly focus there.")
        self.clear_all()

        # --- the light field: 9 × 9 views, refocusing by shift-and-add
        views = np.load(gen("views.npy"))
        g = Group()
        cs = 0.39
        for iv in range(L):
            for iu in range(L):
                g.add(arr_img(views[iv, iu], cs * 0.94).move_to([-4.3 + (iu - 4) * cs, 0.55 - (iv - 4) * cs, 0]))
        lg = label(g, "toy light field: 9 × 9 views", 24)
        eqL = tx(f"{I('L')}({I('x')}, {I('y')}, {I('s')}, {I('t')})", 30).move_to([-4.3, -1.55, 0])
        eqd = tx(f"({I('x')}, {I('y')}): position in a view   ({I('s')}, {I('t')}): which view", 18, DIM).move_to([-4.3, -2.0, 0])
        self.play(LaggedStart(*[FadeIn(m) for m in g], lag_ratio=0.01), run_time=1.0)
        self.add(lg, eqL, eqd)
        self.caption(f"A light field {I('L')}({I('x')}, {I('y')}, {I('s')}, {I('t')}) is a grid of views from slightly different spots.")
        gt = np.load(gen("sweep_gt.npy"))
        a = ValueTracker(D_NEAR)
        R = arr_img(gt[0], 3.3).move_to([2.6, 0.75, 0])
        live(R, lambda: interp(gt, sweep_index(a.get_value())))
        self.hold()
        self.play(FadeIn(R), run_time=0.4)
        self.add(label(R, "refocused image (all 81 views)", 24), framed(R, DIM, 1.2))
        rl, dot = focus_ruler(np.array([2.6, -1.55, 0]), 4.4, a.get_value)
        self.add(rl, dot)
        self.add(toy_note("toy simulation: three layers at different depths, 9 × 9 views").move_to([2.6, -2.35, 0]))
        self.caption("Shift each view by its (s, t) offset and add: a photo focused at any depth.")
        self.run(a, D_FAR, 2.6, smooth)
        self.run(a, D_NEAR, 2.6, smooth)
        self.caption("The eye does exactly this, optically, with the views that enter its pupil.")
        self.clear_all()

        # --- sparse views alias
        base = np.load(gen("sweep_base.npy"))
        a = ValueTracker(D_NEAR)
        h = 3.3
        A = arr_img(gt[0], h).move_to([-2.2, 0.75, 0])
        Bm = arr_img(base[0], h).move_to([2.2, 0.75, 0])
        live(A, lambda: interp(gt, sweep_index(a.get_value())))
        live(Bm, lambda: interp(base, sweep_index(a.get_value())))
        ia = arr_img(aperture_rgb(np.ones((L, L)), px=10), 1.1, nearest=True).move_to([-5.4, 0.75, 0])
        cm = np.zeros((L, L))
        for v, u in CORN:
            cm[v, u] = 1
        ib = arr_img(aperture_rgb(cm, px=10), 1.1, nearest=True).move_to([5.4, 0.75, 0])
        self.play(FadeIn(A), FadeIn(Bm), run_time=0.4)
        self.add(label(A, "all 81 views", 26, GREEN), label(Bm, "only the 4 corner views", 26, RED), ia, ib,
                 label(ia, "views used", 18, DIM), label(ib, "views used", 18, DIM))
        rl, dot = focus_ruler(np.array([0, -1.55, 0]), 5.5, a.get_value)
        self.add(rl, dot)
        self.caption("Keep only the 4 corner views: in-focus parts stay sharp...")
        self.run(a, D_FAR, 2.4, smooth)
        self.caption("...but defocused parts break into ghost copies instead of blur: aliasing.")
        self.run(a, D_NEAR, 2.4, smooth)
        self.run(a, D_MID, 1.2, smooth)
        self.caption("Too few views cannot make natural blur, the very focus cue we need.")
        self.clear_all()


# ================================================================== S03 SDM vs TDM
class S03_Displays(Base):
    def construct(self):
        self.title_card("Part 3", "Space- vs time-division multiplexing")
        self.header("3 · Space- vs time-division multiplexing")
        views = np.load(gen("views.npy"))
        f1b = img("fig1b.png", w=5.6).move_to([-3.4, 0.55, 0])
        self.play(FadeIn(f1b), run_time=0.4)
        self.add(label(f1b, "SDM-LFD: microlens array (paper Fig. 1b)", 22, DIM))
        tiles = Group()
        ts = 0.95
        for i, iv in enumerate((0, 4, 8)):
            for j, iu in enumerate((0, 4, 8)):
                tiles.add(arr_img(views[iv, iu], ts * 0.96).move_to([3.4 + (j - 1) * ts, 0.55 - (i - 1) * ts, 0]))
        self.play(FadeIn(tiles), run_time=0.4)
        self.add(label(tiles, "display panel, 3 × 3 views", 22))
        self.add(tx("each view gets 1/9 of the panel", 24, RED).next_to(tiles, DOWN, buff=0.25))
        self.caption("Space division: a microlens array splits the panel, so in a 3 × 3 display each view gets one-ninth.")
        self.clear_all()

        f1c = img("fig1c.png", w=5.6).move_to([-3.4, 0.55, 0])
        self.play(FadeIn(f1c), run_time=0.4)
        self.add(label(f1c, "TDM-LFD: switchable shutter (paper Fig. 1c)", 22, DIM))
        t = ValueTracker(0.0)
        ti = lambda: int(t.get_value()) % 4
        cv = [views[v, u] for v, u in CORN]
        panel = arr_img(cv[0], 2.6).move_to([2.2, 0.75, 0])
        live(panel, lambda: cv[ti()])

        def apmask():
            m = np.zeros((L, L))
            v, u = CORN[ti()]
            m[v, u] = 1
            return aperture_rgb(m, px=10)
        ap = arr_img(apmask(), 1.5, nearest=True).move_to([5.3, 0.75, 0])
        live(ap, apmask)
        self.play(FadeIn(panel), FadeIn(ap), run_time=0.4)
        self.add(label(panel, "panel: one full view", 22), label(ap, "shutter", 22, CYAN))
        self.add(always_redraw(lambda: tx(f"time slot {ti() + 1} of 4", 22, AMBER).move_to([3.75, -0.95, 0])))
        self.caption("Time division: full-resolution views, one after another, while a shutter opens the matching window.")
        self.run(t, 12.0, 6.0)
        self.add(tx(EQ8, 30).move_to([2.2, -1.75, 0]))
        self.caption("The eye sums the frames into a refocused image (paper Eq. 8).")
        self.run(t, 20.0, 3.2)
        self.clear_all()

        fa = img("fig2_tdm.png", h=2.9).move_to([-4.6, 0.75, 0])
        fb = img("fig2_ctdm.png", h=2.9).move_to([-1.45, 0.75, 0])
        self.play(FadeIn(fa), FadeIn(fb), run_time=0.4)
        self.add(label(fa, "(a) TDM-LFD", 22, RED), label(fb, "(b) CTDM-LFD", 22, GREEN),
                 tx("refocused images, paper Fig. 2", 18, DIM).move_to([-3.0, -0.95, 0]))
        pros = VGroup(tx("✓ full panel resolution per view", 24, GREEN),
                      tx("✗ tiny window: little light", 24, RED),
                      tx("✗ sparse views: defocus aliasing", 24, RED)).arrange(DOWN, aligned_edge=LEFT, buff=0.3
                                                                                ).move_to([3.6, 0.75, 0])
        self.add(pros)
        self.caption("TDM keeps full resolution, but its tiny window passes little light, and sparse views alias (a).")
        self.caption("Look at the shelf behind the green object. Fixing (a) into (b) is what this paper is about.")
        self.clear_all()


# ================================================================== S04 Aperture = blur
class S04_Aperture(Base):
    def construct(self):
        self.title_card("Part 4", "Blur takes the shape of the aperture")
        self.header("4 · Blur takes the shape of the aperture")
        psfs = [np.load(gen(f"psf_{k}.npy")) for k in ("open", "corners", "coded")]
        ap9 = np.load(os.path.join(DATA, "fig9_apertures_9x9.npy")).astype(float)
        cm = np.zeros((L, L))
        for v, u in CORN:
            cm[v, u] = 1
        apert = [np.ones((L, L)), cm, ap9[0]]
        names = ["open aperture", "4 corner windows (TDM)", "coded aperture (paper Fig. 9)"]
        colsn = [WHITE, RED, GREEN]
        w = ValueTracker(0.0)
        wi = lambda: w.get_value() / (W20S[1] - W20S[0])
        xs = [-4.4, 0, 4.4]
        h = 2.8
        self.add(tx(EQ9s, 30).move_to([0, 2.85, 0]))
        for x, st, nm, c, P in zip(xs, psfs, names, colsn, apert):
            m = arr_img(st[0], h).move_to([x, 0.3, 0])
            live(m, (lambda s_: (lambda: interp(s_, wi())))(st))
            a = arr_img(aperture_rgb(P, px=10), 0.9, nearest=True).move_to([x, -1.7, 0])
            self.add(m, framed(m, DIM, 1.0), label(m, nm, 22, c), a)
        self.add(always_redraw(lambda: tx(f"defocus: {w.get_value():.0f} waves", 22, AMBER).move_to([0, -2.4, 0])))
        self.add(toy_note("toy Fourier optics: top, blur of one point (PSF); bottom, the 9 × 9 aperture; each panel normalised"
                          ).move_to([0, -2.8, 0]))
        self.caption(f"Fourier optics: a point's blur (PSF) is set by the aperture {I('P')} and the defocus {I('W')}.")
        self.run(w, 18.0, 4.5, smooth)
        self.caption("Out of focus, the blur is a scaled copy of the aperture itself.")
        self.caption("Open: a smooth blur. Four tiny windows: four dots, the ghost copies.")
        self.run(w, 6.0, 1.8, smooth)
        self.run(w, 18.0, 1.8, smooth)
        self.caption("A coded aperture spreads each point into a pattern we get to design.")
        self.clear_all()


# ================================================================== S05 CTDM pipeline
class S05_CTDM(Base):
    def construct(self):
        self.title_card("Part 5", "Coded time-division multiplexing")
        self.header("5 · Coded time-division multiplexing")
        cviews = np.load(gen("corner_views.npy"))
        enc = np.load(gen("enc_images.npy"))
        P = np.load(gen("P_final.npy"))
        cum = np.load(gen("cycle_cum.npy"))
        vs = Group(*[arr_img(cviews[i], 1.05).move_to([-6.0 + 0.16 * i, 0.55 - 0.16 * i, 0]) for i in range(4)])
        lv = label(vs, f"{I('n')} = 4 views", 22)
        box = Rectangle(width=0.8, height=3.4, color=AMBER, stroke_width=2.5).move_to([-3.95, 0.4, 0])
        bl = tx("encoding network  " + I("f"), 22, AMBER).rotate(PI / 2).move_to(box)
        a1 = arrow([-5.3, 0.4, 0], [-4.4, 0.4, 0])
        ys = [1.75, 0.65, -0.45, -1.55]
        encm = Group(*[arr_img(enc[i], 1.0).move_to([-2.3, y, 0]) for i, y in enumerate(ys)])
        apm = Group(*[arr_img(aperture_rgb(P[i], px=10), 0.95, nearest=True).move_to([-0.95, y, 0]) for i, y in enumerate(ys)])
        tl = VGroup(*[tx(f"{I('t')} = {i + 1}", 18, DIM).next_to(apm[i], RIGHT, buff=0.12) for i in range(4)])
        a2s = VGroup(*[arrow([-3.5, 0.4, 0], [-2.85, y, 0], buff=0.05, sw=2) for y in ys])
        le = tx("encoded", 20).move_to([-2.3, 2.55, 0])
        la = tx("apertures", 20, CYAN).move_to([-0.95, 2.55, 0])
        self.play(FadeIn(vs), run_time=0.4)
        self.add(lv)
        self.caption(f"CTDM-LFD: a network {I('f')} turns the {I('n')} sparse views into {I('k')} encoded images.")
        self.play(GrowArrow(a1), Create(box), run_time=0.4)
        self.add(bl, a2s)
        self.play(FadeIn(encm), run_time=0.4)
        self.add(le)
        self.hold()
        self.play(FadeIn(apm), run_time=0.4)
        self.add(la, tl)
        self.caption(f"Each encoded image is shown through its own coded aperture {PH}{sub(I('t'))}, one time slot after another.")
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, 3.999))
        R = arr_img(cum[0], 3.0).move_to([3.9, 0.4, 0])
        live(R, lambda: cum[ti()])
        a3 = arrow([0.4, 0.4, 0], [2.3, 0.4, 0])
        la3 = tx("lens + eye", 20, DIM).next_to(a3, UP, buff=0.1)
        hl = always_redraw(lambda: SurroundingRectangle(Group(encm[ti()], apm[ti()]), color=AMBER, buff=0.08, stroke_width=3))
        self.hold()
        self.play(GrowArrow(a3), FadeIn(R), run_time=0.4)
        self.add(la3, hl, framed(R, DIM, 1.0))
        self.add(always_redraw(lambda: label(R, f"retina, after {ti() + 1} of 4 frames", 22)))
        self.add(tx(EQ10, 30).move_to([2.6, -2.2, 0]))
        self.add(toy_note("images and apertures from the toy optimisation in Part 6; eye focused on the near star"
                          ).move_to([2.6, -2.75, 0]))
        self.caption("The eye adds up the frames: persistence of vision (paper Eq. 10).")
        self.run(t, 3.99, 4.0)
        self.caption("Encoded images are not views: they are whatever makes the sum right at every focus.")
        self.clear_all()
        f3 = img("fig3_pipeline.png", w=11.5).move_to([0, 0.45, 0])
        self.play(FadeIn(f3), run_time=0.5)
        self.add(tx("paper Fig. 3: the CTDM-LFD pipeline (n = 4, k = 4)", 20, DIM).next_to(f3, DOWN, buff=0.2))
        self.caption(f"In the paper, the network {I('f')} and the apertures {PH} are learned together, end to end.")
        self.clear_all()


# ================================================================== S06 Learning
class S06_Learning(Base):
    def construct(self):
        self.title_card("Part 6", "Learning the display end to end")
        self.header("6 · Learning the display end to end")
        e12 = tx(EQ12, 32).move_to([-3.2, 2.35, 0])
        d12 = VGroup(tx(f"{I('F')}{sub(I('c'))}: focal stack the display produces", 20, DIM),
                     tx(f"{I('F')}{sub(I('g'))}: focal stack of the full light field", 20, DIM)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(e12, DOWN, buff=0.25)
        e13 = tx(EQ13, 28).move_to([-3.2, -0.2, 0])
        e14 = tx(EQ14, 28).move_to([-3.2, -0.85, 0])
        self.add(e12, d12)
        self.caption("Train on focal stacks: what the eye would see at many depths, against the ground truth.")
        wimg = np.load(gen("wimg.npy"))
        wmap = np.load(gen("wmap2.npy"))
        hh = 1.9
        cells = []
        rowl = []
        for r, (nm, y) in enumerate((("near focus", 1.3), ("far focus", -1.0))):
            cells += [arr_img(wimg[r], hh).move_to([2.35, y, 0]), arr_img(wmap[r], hh).move_to([4.45, y, 0])]
            rowl.append(tx(nm, 20, DIM).rotate(PI / 2).move_to([1.1, y, 0]))
        self.hold()
        self.add(e13, e14)
        self.play(*[FadeIn(m) for m in cells], run_time=0.4)
        self.add(*rowl, label(cells[0], "ground truth", 20), label(cells[1], "weight 𝒲", 20, AMBER),
                 toy_note("toy scene, β = 2").next_to(cells[3], DOWN, buff=0.12))
        self.caption("A focus measure from image gradients weights each pixel, so in-focus detail counts more.")
        self.caption("With the paper's β = 2, the sharpest pixels count up to e² ≈ 7.4 times more.")
        self.clear_all()

        sP = np.load(gen("train_snapP.npy"))
        sI = np.load(gen("train_snapImg.npy"))
        sIt = np.load(gen("train_snapIt.npy"))
        hist = np.load(gen("train_hist.npy"))
        ps = np.load(gen("psnr.npy"))
        it = ValueTracker(0.0)

        def sidx():
            v = it.get_value()
            j = int(np.clip(np.searchsorted(sIt, v, side="right") - 1, 0, len(sIt) - 2))
            w = np.clip((v - sIt[j]) / max(sIt[j + 1] - sIt[j], 1), 0, 1)
            return j, w
        aps = []
        for i in range(4):
            m = arr_img(aperture_rgb(sP[0, i], px=10), 1.3, nearest=True).move_to([-5.55 + 1.5 * (i % 2), 1.0 - 1.5 * (i // 2), 0])
            live(m, (lambda k: (lambda: aperture_rgb((lambda j, w: (1 - w) * sP[j, k] + w * sP[j + 1, k])(*sidx()), px=10)))(i))
            aps.append(m)
        R = arr_img(sI[0], 3.0).move_to([-0.9, 0.35, 0])
        live(R, lambda: (lambda j, w: ((1 - w) * sI[j] + w * sI[j + 1]).astype(np.uint8))(*sidx()))
        ax = Axes(x_range=[0, NIT, 1000], y_range=[14, 34, 5], x_length=4.3, y_length=3.0, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([4.2, 0.35, 0])
        xt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(v, 14), DOWN, buff=0.1) for v in (0, 1000, 2000, 3000)])
        yt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (15, 20, 25, 30)])
        xl = tx("iteration", 18, DIM).next_to(ax, DOWN, buff=0.4)
        yl = tx("toy focal-stack PSNR (dB)", 18, DIM).next_to(ax, UP, buff=0.12)
        bl = DashedLine(ax.c2p(0, ps[0]), ax.c2p(NIT, ps[0]), color=RED, stroke_width=2)
        bll = tx("baseline TDM", 16, RED).next_to(ax.c2p(NIT, ps[0]), DOWN, buff=0.08).shift(LEFT * 0.6)

        def curve():
            n = max(2, int(np.searchsorted(hist[:, 0], it.get_value())) + 1)
            return ax.plot_line_graph(hist[:n, 0], np.clip(hist[:n, 1], 14, 34), add_vertex_dots=False,
                                      line_color=GREEN, stroke_width=3)
        self.play(*[FadeIn(m) for m in aps], FadeIn(R), Create(ax), run_time=0.6)
        self.add(tx("4 coded apertures", 20, CYAN).move_to([-4.8, 2.25, 0]), label(R, "eye focused on the near star", 20),
                 framed(R, DIM, 1.0), xt, yt, xl, yl, bl, bll, always_redraw(curve))
        self.add(always_redraw(lambda: tx(f"iteration {int(it.get_value())}", 22, AMBER).move_to([-4.8, -1.75, 0])))
        self.add(toy_note("toy optimisation: 4 images + 4 apertures fitted to one scene through the display model "
                          "(the paper trains a CNN f over a dataset)").move_to([0, -2.6, 0]))
        self.caption("Toy: start from random apertures and flat images, then follow the gradient through the display model.")
        self.run(it, 60, 2.5, rate=lambda x: x)
        self.run(it, NIT - 1, 6.5, rate=lambda x: x ** 2.2)
        for m in aps + [R]:
            m.clear_updaters()
        self.caption("The apertures grow into large blobs in different places: more light, and blur instead of ghosts.")
        self.clear_all()

        st = [np.load(gen(f"sweep_{k}.npy")) for k in ("base", "fP", "gt")]
        names = ["baseline TDM-LFD (4 windows)", f"learned images + apertures {PH}", "ground truth (81 views)"]
        colsn = [RED, GREEN, DIM]
        vals = [f"{ps[0]:.1f} dB", f"{ps[2]:.1f} dB", ""]
        a = ValueTracker(D_NEAR)
        xs = [-4.4, 0, 4.4]
        h = 3.4
        for x, s_, nm, c, v in zip(xs, st, names, colsn, vals):
            m = arr_img(s_[0], h).move_to([x, 0.75, 0])
            live(m, (lambda q: (lambda: interp(q, sweep_index(a.get_value()))))(s_))
            self.add(m, label(m, nm, 20, c))
            if v:
                self.add(tx("toy PSNR " + v, 18, c).next_to(m, DOWN, buff=0.12))
        rl, dot = focus_ruler(np.array([0, -1.85, 0]), 6.0, a.get_value)
        self.add(rl, dot)
        self.caption("Sweep the focus through the toy scene: the baseline breaks the blur into ghost copies.")
        self.run(a, D_FAR, 2.8, smooth)
        self.caption("The learned display stays sharp in focus and blurs smoothly, much closer to the truth.")
        self.run(a, D_NEAR, 2.8, smooth)
        self.run(a, D_FAR, 2.4, smooth)
        self.clear_all()


# ================================================================== S07 Results
class S07_Results(Base):
    def construct(self):
        self.title_card("Part 7", "Results")
        self.header("7 · Results")
        rows = [(f"TDM-LFD baseline, {I('n')} = 4", ("28.56", "0.8485 / 0.2626"), ("26.39", "0.8932 / 0.1434"), ("32.92", "0.9285 / 0.1199")),
                (f"CTDM-LFD, {I('n')} = 4, {I('k')} = 4", ("32.29", "0.9374 / 0.1284"), ("30.21", "0.9602 / 0.0736"), ("37.57", "0.9705 / 0.03119")),
                (f"TDM-LFD baseline, {I('n')} = 9", ("34.73", "0.9688 / 0.1385"), ("36.32", "0.9872 / 0.0363"), ("40.61", "0.9858 / 0.0397")),
                (f"CTDM-LFD, {I('n')} = 9, {I('k')} = 8", ("36.41", "0.9858 / 0.0632"), ("39.97", "0.9891 / 0.0186"), ("41.21", "0.9739 / 0.0107"))]
        xs = [-4.3, -0.4, 2.35, 5.1]
        tbl = VGroup()
        for c, (s, x) in enumerate(zip(["", "INRIA-syn", "INRIA-real", "SIGGRAPH16"], xs)):
            tbl.add(tx(s, 22, DIM).move_to([x, 2.55, 0]))
        for r, row in enumerate(rows):
            y = 1.85 - 0.72 * r
            cl = GREEN if "CTDM" in row[0] else RED
            tbl.add(tx(row[0], 22, cl).move_to([xs[0], y, 0]))
            for (p, rest), x in zip(row[1:], xs[1:]):
                tbl.add(tx(p + " dB", 24, cl).move_to([x, y + 0.1, 0]), tx(rest, 15, DIM).move_to([x, y - 0.2, 0]))
        hl = Line([-6.6, 2.28, 0], [6.6, 2.28, 0], color=DIM, stroke_width=1.5)
        self.add(tbl, hl)
        self.add(tx("paper Table V: focal stacks of the displayed light fields; PSNR ↑ (SSIM ↑ / LPIPS ↓ in grey)", 18, DIM
                    ).move_to([0, -1.1, 0]))
        self.caption("Paper Table V: on three light field datasets, CTDM-LFD beats the TDM baseline.")
        head = fitw(tx(f"{I('n')} = 4 views:  +3.7 dB (INRIA-syn)  ·  +3.8 dB (INRIA-real)  ·  +4.7 dB (SIGGRAPH16)", 26, AMBER)
                    ).move_to([0, -1.85, 0])
        self.hold()
        self.add(head)
        self.caption("About 4 dB more PSNR with 4 views, and better on every metric except one SSIM entry.")
        self.clear_all()

        w = 3.2
        xs = [-4.2, 0, 4.2]
        for x, m_, n_, c in zip(xs, ["base", "ours", "gt"], ["baseline TDM-LFD", "CTDM-LFD (ours)", "ground truth"], [RED, GREEN, DIM]):
            a = img(f"fig10_r0_{m_}.png", w=w).move_to([x, 1.15, 0])
            za = img(f"fig10_r0_{m_}_zA.png", w=1.5).move_to([x - 0.78, -1.3, 0])
            zb = img(f"fig10_r0_{m_}_zB.png", w=1.5).move_to([x + 0.78, -1.3, 0])
            self.add(a, za, zb, label(a, n_, 24, c))
        self.add(tx("INRIA-syn, n = 4 (paper Fig. 10); bottom: zoomed patches", 18, DIM).move_to([0, -2.35, 0]))
        self.caption("Synthetic scene: the baseline's defocused background shows aliasing; CTDM blurs it naturally.")
        self.clear_all()

        w = 2.55
        xs = [-3.0, 0, 3.0]
        for r, (nm, y) in enumerate((("mom_near", 1.25), ("mom_far", -1.35))):
            for x, m_ in zip(xs, ["base", "ours", "gt"]):
                a = img(f"fig11_{nm}_{m_}.png", w=w).move_to([x, y, 0])
                self.add(a)
                if r == 0:
                    self.add(label(a, {"base": "baseline TDM-LFD", "ours": "CTDM-LFD (ours)", "gt": "ground truth"}[m_], 22,
                                   {"base": RED, "ours": GREEN, "gt": DIM}[m_]))
        self.add(tx("near focus", 20, DIM).move_to([-5.3, 1.25, 0]), tx("far focus", 20, DIM).move_to([-5.3, -1.35, 0]),
                 tx("INRIA Lytro,", 18, DIM).move_to([5.45, 0.15, 0]), tx("n = 4, k = 4", 18, DIM).move_to([5.45, -0.15, 0]),
                 tx("(paper Fig. 11)", 18, DIM).move_to([5.45, -0.45, 0]))
        self.caption("Real Lytro light field, near focus: the baseline's far letters double up; CTDM's blur smoothly.")
        self.caption("Far focus: the letters are sharp in both, and CTDM's near blocks blur like the ground truth.")
        self.clear_all()

        rows17 = ["chairs", "mom", "poppy"]
        stacks = [[np.asarray(Image.open(os.path.join(DATA, f"fig17_{r}_{j}.png")).convert("RGB")) for j in range(9)] for r in rows17]
        j = ValueTracker(0.0)
        xs = [-4.3, 0, 4.3]
        for x, stck in zip(xs, stacks):
            m = arr_img(stck[0], 3.4).move_to([x, 0.6, 0])
            live(m, (lambda s_: (lambda: s_[int(np.clip(round(j.get_value()), 0, 8))]))(stck))
            self.add(m)
        self.add(always_redraw(lambda: tx(f"focal plane {int(np.clip(round(j.get_value()), 0, 8)) + 1} of 9", 24, AMBER
                                          ).move_to([0, -1.55, 0])),
                 tx("CTDM-LFD continuous focal stacks, n = 4, k = 4 (paper Fig. 17)", 18, DIM).move_to([0, -2.05, 0]))
        self.caption("Paper Fig. 17: sweeping the focus through CTDM-LFD light fields, front to back.")
        self.run(j, 8, 3.0)
        self.run(j, 0, 3.0)
        self.clear_all()

        lens = img("fig8_lens.png", w=4.6).move_to([-4.2, 1.05, 0])
        aps = Group(*[img(f"fig9_ap{i}.png", h=0.95) for i in range(4)]).arrange(RIGHT, buff=0.15).move_to([-4.2, -1.35, 0])
        self.add(lens, label(lens, "unit-magnification relay lens (Fig. 8)", 20, DIM), aps,
                 label(aps, "optimised 9 × 9 binary apertures (Fig. 9)", 20, DIM))
        n1 = img("fig13_near_ours.png", h=2.7).move_to([0.95, 0.5, 0])
        n2 = img("fig13_far_ours.png", h=2.7).move_to([4.35, 0.5, 0])
        self.add(n1, n2, label(n1, "near focus", 22, GREEN), label(n2, "far focus", 22, GREEN),
                 tx("Zemax optics simulation of CTDM-LFD (paper Fig. 13)", 18, DIM).move_to([2.65, -1.2, 0]))
        self.caption("A real optical design: spherical lenses, common glasses, binary coded aperture at the centre.")
        self.caption("Simulated in Zemax, it refocuses with natural bokeh.")
        self.clear_all()


# ================================================================== S08 Ablations
class S08_Ablations(Base):
    def construct(self):
        self.title_card("Part 8", "What matters")
        self.header("8 · What matters")
        ax = Axes(x_range=[0, 3, 1], y_range=[26, 33, 1], x_length=5.4, y_length=3.6, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([-3.4, 0.4, 0])
        yt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (26, 28, 30, 32)])
        vals = [(28.56, "baseline", RED), (29.85, f"learned {PH}", WHITE), (32.29, f"learned {I('f')} and {PH}", GREEN)]
        bars = VGroup()
        labs = VGroup()
        for i, (v, n, c) in enumerate(vals):
            p0, p1 = ax.c2p(i + 0.5, 26), ax.c2p(i + 0.5, v)
            b = Rectangle(width=1.1, height=p1[1] - p0[1], fill_color=c, fill_opacity=0.85, stroke_width=0)
            b.move_to((p0 + p1) / 2)
            bars.add(b)
            labs.add(tx(f"{v:.2f}", 20, c).next_to(b, UP, buff=0.08), tx(n, 18, c).next_to(p0, DOWN, buff=0.15))
        self.play(Create(ax), run_time=0.4)
        self.add(yt, tx("PSNR (dB)", 18, DIM).next_to(ax, UP, buff=0.12).shift(LEFT * 1.6),
                 tx("paper Table VI, INRIA-syn, n = 4, k = 4, l = 9", 18, DIM).next_to(ax, DOWN, buff=0.65))
        self.play(GrowFromEdge(bars[0], DOWN), run_time=0.35)
        self.add(labs[0], labs[1])
        self.play(GrowFromEdge(bars[1], DOWN), run_time=0.35)
        self.add(labs[2], labs[3])
        self.caption("Learning only the apertures gains 1.29 dB over the baseline.")
        self.hold()
        self.play(GrowFromEdge(bars[2], DOWN), run_time=0.35)
        self.add(labs[4], labs[5])
        self.caption(f"Learning the encoding network {I('f')} as well adds another 2.44 dB.")
        rt = [("binary vs continuous apertures", "31.24 vs 32.29 dB"),
              ("mirrored (symmetric) vs free apertures", "31.06 vs 32.29 dB"),
              ("loss weight β = 0 / 2 / 5", "31.25 / 32.29 / 32.07 dB"),
              (f"{I('k')} = 4 / 8 / 16 frames ({I('n')} = 4)", "32.29 / 33.12 / 32.43 dB")]
        g = VGroup()
        for a_, b_ in rt:
            g.add(VGroup(tx(a_, 22), tx(b_, 22, AMBER)).arrange(DOWN, aligned_edge=LEFT, buff=0.08))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([3.4, 0.4, 0])
        self.hold()
        self.add(g[0], g[1])
        self.caption("Binary apertures cost about 1 dB; forcing symmetric apertures costs a bit more.")
        self.hold()
        self.add(g[2], g[3])
        self.caption("β = 2 gives the best PSNR. More frames is not always better, and frame rate grows with k.")
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Summary")
        self.header("9 · Summary")
        summ = ["1   Coded time-division multiplexing: each frame through its own coded aperture",
                "2   Encoding network and apertures learned end to end with Fourier optics",
                "3   Focus-weighted focal-stack loss: sharp in focus, natural blur out of focus"]
        sg = VGroup(*[fitw(tx(s, 26)) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 1.6, 0])
        for s in sg:
            self.add(s)
            self.wait(0.7)
        lim = VGroup(tx("Trade-off: more frames k need a faster panel and shutter", 22, DIM),
                     tx("Validated with Fourier-optics and Zemax simulations", 22, DIM)).arrange(DOWN, buff=0.12
                                                                                              ).move_to([0, -0.2, 0])
        self.add(lim)
        self.add(tx("+3.7 dB over TDM-LFD with 4 views (INRIA-syn)  ·  natural defocus blur", 26, AMBER).move_to([0, -1.2, 0]))
        self.caption("Coded apertures plus learning: light field displays whose blur looks right.")
        self.clear_all()
        self.remove(self.hdr)
        frames = [np.asarray(Image.open(os.path.join(DATA, "gif_frames", f"{i:03d}.png")).convert("RGB")) for i in (0, 1)]
        gif = arr_img(frames[1], 2.4).move_to([0, 1.55, 0])
        t1 = fitw(tx("Time-Division Multiplexing Light Field Display", 40, bold=True)).move_to([0, -0.2, 0])
        t1b = tx("with Learned Coded Aperture", 30).move_to([0, -0.8, 0])
        t2 = tx("IEEE Transactions on Image Processing 32, 350–363 (2023)  ·  arXiv 2107.06205", 22).move_to([0, -1.45, 0])
        t4 = tx("Brian Chao*, Chang-Le Liu*, Homer H. Chen", 22, DIM).move_to([0, -2.05, 0])
        t5 = tx("National Taiwan University", 22, DIM).move_to([0, -2.47, 0])
        t6 = tx("* equal contribution", 18, DIM).move_to([0, -2.85, 0])
        self.add(gif, t1, t1b, t2, t4, t5, t6)
        self.wait(4.0)
