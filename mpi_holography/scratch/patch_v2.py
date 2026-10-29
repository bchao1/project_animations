"""v1 -> v2 patch (QA round 1): text fits, label overlaps, source notes top right, trims."""
p = "/home/brianchc/project_animations/mpi_holography/animation_mpi_v2.py"
s = open(p).read()


def R(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (n, a[:70])
    s = s.replace(a, b)


R("teaser animation, v1.", "teaser animation, v2.")
R("animation_mpi_v1.py <Scene>", "animation_mpi_v2.py <Scene>")
R('''def I(v):
    return f"<i>{v}</i>"''', '''def fit(m, w=13.0):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def src_note(s):
    """Source note, top right (header line)."""
    return tx(s, 16, DIM).to_edge(RIGHT, buff=0.35).set_y(HEADER_Y)


def I(v):
    return f"<i>{v}</i>"''')
R('''        title = tx("Fast Wave-optics Rendering of Multiplane Images", 50, bold=True).move_to([0, 3.0, 0])''',
  '''        title = fit(tx("Fast Wave-optics Rendering of Multiplane Images", 50, bold=True)).move_to([0, 3.0, 0])''')
R('''        auth = tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                  "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM).move_to([0, -2.6, 0])''',
  '''        auth = fit(tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                      "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM)).move_to([0, -2.6, 0])''')
R('''        self.caption("A holographic display rebuilds the light wave of a 3D scene.")
        self.caption("Your eye can focus at any depth, and parallax changes as it moves.")
        self.caption("But first a computer must turn 3D content into that wave: computer-generated holography.")''',
  '''        self.caption("A holographic display rebuilds the light wave of a 3D scene: you can refocus and look around.")
        self.caption("But a computer must first turn 3D content into that wave: computer-generated holography.")''')
R('''            self.wait(0.35)''', '''            self.wait(0.3)''')
R('''        self.caption("With smooth phase it rings but stays put. With random phase it spreads into every direction.")
        self.run(z, 0, 2.5, smooth)
        self.run(z, 20, 2.5, smooth)''', '''        self.caption("With smooth phase it rings but stays put. With random phase it spreads into every direction.")
        self.run(z, 10, 2.0, smooth)''')
R('''        h = 2.4
        xs = [-5.1, -1.7, 1.7, 5.1]''', '''        h = 2.3
        xs = [-5.1, -1.7, 1.7, 5.1]''')
R('''        arr_l = ["FFT", "× exp(" + I("i") + "2π" + I("f") + sub(I("z")) + I("z") + ")", "inverse FFT"]''',
  '''        arr_l = ["FFT", "× " + I("H"), "FFT" + sup("−1")]''')
R('''            self.add(labs[k + 1], tx(arr_l[k], 16, AMBER).next_to(ar, DOWN, buff=0.1))
            self.wait(0.6)''', '''            self.add(labs[k + 1], tx(arr_l[k], 20, AMBER).next_to(ar, UP, buff=0.1))
            self.wait(0.5)''')
R('''EQ_ASM = (f"𝒫({I('u')}; {I('z')}) = ℱ{sup('−1')}{{ ℱ{{{I('u')}}} · exp({I('i')}2π{I('f')}{sub(I('z'))}{I('z')}) }}")''',
  '''EQ_ASM = (f"𝒫({I('u')}; {I('z')}) = ℱ{sup('−1')}{{ ℱ{{{I('u')}}} · {I('H')} }},   "
          f"{I('H')} = exp({I('i')}2π{I('f')}{sub(I('z'))}{I('z')})")''')
R('''        names = ["layer 4 (back)", "layer 3", "layer 2", "layer 1 (front)"]
        for s, n in zip(sheets, names):
            self.add(tx(n, 18, DIM).next_to(s, DOWN, buff=0.12))
        self.add(tx(f"colour {I('c')}{sub(I('i'))}, opacity α{sub(I('i'))}, depth {I('d')}{sub(I('i'))}", 22
                    ).move_to([1.6, -1.9, 0]))''', '''        for s, n in zip(sheets, ["4", "3", "2", "1"]):
            self.add(tx(n, 20, DIM).move_to([s.get_x(), s.get_bottom()[1] - 0.2, 0]))
        self.add(tx(f"layer {I('i')} = 1 (front) … 4 (back):  colour {I('c')}{sub(I('i'))}, opacity α{sub(I('i'))}, "
                    f"depth {I('d')}{sub(I('i'))}", 22).move_to([1.6, -2.3, 0]))''')
R('''        self.run(s, -1, 2.0, smooth)
        self.run(s, 1, 3.5, smooth)
        self.caption("Near layers move more than far ones, so the view shows parallax and occlusion.")
        self.run(s, -1, 3.5, smooth)
        self.run(s, 0, 1.8, smooth)''', '''        self.run(s, -1, 1.8, smooth)
        self.run(s, 1, 3.0, smooth)
        self.caption("Near layers move more than far ones: parallax and occlusion, for free.")
        self.run(s, -1, 3.0, smooth)
        self.run(s, 0, 1.4, smooth)''')
R('''                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.6, 0.6, 0])''',
  '''                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.35, 0.6, 0])''')
R('''        g = VGroup(*[tx(("•  " if i < 3 else "") + s_, 24,''', '''        g = VGroup(*[tx(("•  " if i < 3 else "") + s_, 22,''')
R('''        self.caption("MPIs are cheap to get: networks predict them from one or two photos, or they are binned from splats.")
        self.caption("And they excel at front-facing views, exactly what a near-eye display's small eyebox needs.")''',
  '''        self.caption("MPIs are cheap to get, and they excel at front-facing views.")
        self.caption("That is exactly what a near-eye display's small eyebox needs.")''')
R('''        self.run(s, -S_MAX, 2.0, smooth)
        self.run(s, S_MAX, 3.5, smooth)
        self.caption("RGBD: dark gaps open along every occlusion edge. The MPI shows what was behind.")
        self.run(s, -S_MAX, 3.5, smooth)
        self.run(s, 0, 1.8, smooth)''', '''        self.run(s, -S_MAX, 1.8, smooth)
        self.run(s, S_MAX, 3.2, smooth)
        self.caption("RGBD: dark gaps open along every occlusion edge. The MPI shows what was behind.")
        self.run(s, -S_MAX, 3.2, smooth)
        self.run(s, 0, 1.4, smooth)''')
R('''        self.add(tx("defocused edge (focal slice)", 18, DIM).move_to([0, 2.55, 0]).set_opacity(0))
''', '')
R('''            else:
                self.wait(0.6)''', '''            else:
                self.wait(0.3)''')
R('''        self.caption("Random phases make the cross term average away: intensities obey the over operator exactly.")
        self.add(tx("random-phase alpha blending from Random-phase Wave Splatting (RPWS, Chao et al. 2025)", 20, DIM
                    ).move_to([0, -1.0, 0]))
        self.caption("That is RPWS's random-phase alpha blending, applied here to whole layers instead of single primitives.")''',
  '''        self.add(tx("random-phase alpha blending from Random-phase Wave Splatting (RPWS, Chao et al. 2025)", 20, DIM
                    ).move_to([0, -1.0, 0]))
        self.caption("Random phases make the cross term average away: intensities obey the over operator exactly.")''')
R('''        self.caption("The result is a random-phase hologram whose light fills the SLM's whole bandwidth: a wide eyebox.")''',
  '''        self.caption("The result: a random-phase hologram that fills the SLM's bandwidth, for a wide eyebox.")''')
R('''        e = tx(EQ_TM, 30).move_to([0, 2.75, 0])''', '''        e = tx(EQ_TM, 30).move_to([0, 2.85, 0])''')
R('''        h = 3.2
        a = arr_img(frames[0], h).move_to([-2.2, 0.35, 0])
        b = arr_img(cum[0], h).move_to([2.2, 0.35, 0])''', '''        h = 3.0
        a = arr_img(frames[0], h).move_to([-2.2, 0.2, 0])
        b = arr_img(cum[0], h).move_to([2.2, 0.2, 0])''')
R('''        src = tx("values read from paper Fig. 3 (Mip-NeRF 360 scenes)", 16, DIM).move_to([-1.6, -2.55, 0])
        self.add(src)''', '''        self.add(src_note("values read from paper Fig. 3 (Mip-NeRF 360 scenes)"))''')
R('''        self.add(tx("values read from paper Fig. 3 (Mip-NeRF 360 scenes)", 16, DIM).move_to([0, -2.1, 0]))''',
  '''        self.add(src_note("values read from paper Fig. 3 (Mip-NeRF 360 scenes)"))''')
R('''        self.add(v30, tx("≈ 30 layers", 18, AMBER).next_to(v30[0], UP, buff=0.05),
                 tx("≈ 30 layers", 18, AMBER).next_to(v30[1], UP, buff=0.05))''',
  '''        self.add(v30, tx("≈ 30 layers", 18, AMBER).next_to(pa.c2p(np.log2(30), 21.0), UR, buff=0.08),
                 tx("≈ 30 layers", 18, AMBER).next_to(sa.c2p(np.log2(30), 0.45), UR, buff=0.08))''')
R('''        w = 5.4
        pos = {"teaser_rgbd_front": (-2.85, 1.35), "teaser_rpws_front": (2.85, 1.35),
               "teaser_mpi_back": (-2.85, -1.3), "teaser_mpi_front": (2.85, -1.3)}''', '''        w = 4.3
        pos = {"teaser_rgbd_front": (-2.25, 1.6), "teaser_rpws_front": (2.25, 1.6),
               "teaser_mpi_back": (-2.25, -1.35), "teaser_mpi_front": (2.25, -1.35)}''')
R('''        self.add(tx("simulated, paper Fig. 1", 16, DIM).move_to([0, 0.02, 0]))''', '''        self.add(src_note("simulated, paper Fig. 1"))''')
R('''        xs = [-4.4, 0, 4.4]
        w = 4.2
        for k, x in enumerate(xs):
            a = img(f"sm{k}_back.png", w=w).move_to([x, 1.25, 0])
            b = img(f"sm{k}_front.png", w=w).move_to([x, -1.05, 0])''', '''        xs = [-4.3, 0, 4.3]
        w = 4.1
        for k, x in enumerate(xs):
            a = img(f"sm{k}_back.png", w=w).move_to([x, 1.35, 0])
            b = img(f"sm{k}_front.png", w=w).move_to([x, -1.0, 0])''')
R('''                 tx("MPIs predicted by Stereo Magnification (Zhou et al. 2018), turned into holograms (paper Fig. 6)",
                    16, DIM).move_to([0, -2.55, 0]))''',
  '''                 src_note("MPIs predicted by Stereo Magnification (Zhou et al. 2018) → holograms, paper Fig. 6"))''')
R('''        self.add(tx("back focus", 20, DIM).move_to([-6.75, 1.25, 0]).rotate(PI / 2),
                 tx("front focus", 20, DIM).move_to([-6.75, -1.05, 0]).rotate(PI / 2),''',
  '''        self.add(tx("back focus", 20, DIM).move_to([-6.7, 1.35, 0]).rotate(PI / 2),
                 tx("front focus", 20, DIM).move_to([-6.7, -1.0, 0]).rotate(PI / 2),''')
R('''        t1 = tx("Fast Wave-optics Rendering of Multiplane Images", 44, bold=True).move_to([0, 2.1, 0])''',
  '''        t1 = fit(tx("Fast Wave-optics Rendering of Multiplane Images", 44, bold=True)).move_to([0, 2.1, 0])''')
R('''        t4 = tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM).move_to([0, -0.6, 0])''',
  '''        t4 = fit(tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                    "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM)).move_to([0, -0.6, 0])''')
open(p, "w").write(s)
print("patched")
