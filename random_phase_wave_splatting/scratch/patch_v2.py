p = "/home/brianchc/project_animations/random_phase_wave_splatting/animation_rpws_v2.py"
s = open(p).read()
def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (n, a[:80])
    s = s.replace(a, b)

rep('"""Random-phase Wave Splatting — teaser animation, v1.', '"""Random-phase Wave Splatting — teaser animation, v2.')
rep("to rebuild).\n\nRender", "to rebuild; v2 adds data/generated/done_v2 extras).\n\nRender")
rep("manim -ql --media_dir media animation_rpws_v1.py", "manim -ql --media_dir media animation_rpws_v2.py")
# --- superscript with a subscript inside it
rep('''def col(x, c):''', '''def supsub(base, s_):
    """Superscript 'base' followed by a subscript that stays inside the superscript (Pango rise is absolute)."""
    return (f'<span rise="4500"><small><small>{base}</small></small></span>'
            f'<span rise="2600"><small><small><small>{s_}</small></small></small></span>')


def col(x, c):''')
rep('''EQ_CROSS = (f"|{W1} {I('e')}{sup(I('i') + PH1)} + {W2} {I('e')}{sup(I('i') + PH2)}|{SQ} = {W1}{SQ} + {W2}{SQ} + "''',
    '''EQ_CROSS = (f"|{W1} {I('e')}{supsub(I('i') + 'φ', '1')} + {W2} {I('e')}{supsub(I('i') + 'φ', '2')}|{SQ} = {W1}{SQ} + {W2}{SQ} + "''')
rep('''            f"√{I('c')}{sub(I('i'))} √({I('o')}{sub(I('i'))}|{I('u')}{sub(I('i'))}|) {I('e')}{sup(I('i') + '∠' + I('u') + sub(I('i')) + ' + ' + I('i') + 'φ' + sub(I('i')) + sup('(' + I('t') + ')'))} ; Δ{I('z')} )")''',
    '''            f"√{I('c')}{sub(I('i'))} √({I('o')}{sub(I('i'))}|{I('u')}{sub(I('i'))}|) "
            f"exp({I('i')}∠{I('u')}{sub(I('i'))} + {I('i')}φ{sub(I('i'))}{sup('(' + I('t') + ')')}) ; Δ{I('z')} )")''')
rep('''EQ_GWS = (f"{I('u')}{sub('SLM')} = ∑{sub(I('i'))} 𝒫( {I('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} |{I('u')}{sub(I('i'))}| "
          f"{I('T')}{sub(I('i'))} {I('e')}{sup(I('ikz') + sub(I('i')))} ; −{I('z')}{sub(I('i'))} )")''',
    '''EQ_GWS = (f"{I('u')}{sub('SLM')} = ∑{sub(I('i'))} 𝒫( {I('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} |{I('u')}{sub(I('i'))}| "
          f"{I('T')}{sub(I('i'))} {I('e')}{supsub(I('ikz'), I('i'))} ; −{I('z')}{sub(I('i'))} )")''')
# --- v2 asset extras: domain-coloured SLM fields, 7-decade spectra
rep('''build_assets()


def img(''', '''def build_assets_v2():
    """v2 extras: SLM fields in domain colouring and angular spectra over 7 decades (display capped for the flat one)."""
    if os.path.exists(gen("done_v2")):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen("done_v2")):
            return
        rng = np.random.default_rng(7)
        a_f, c_f, a_b, c_b = toy_scene()
        us = smooth_slm(a_f, c_f, a_b, c_b)
        ur = rpws_slm(a_f, c_f, a_b, c_b, rng)
        for name, u in (("field_smooth.png", us), ("field_rand.png", ur)):
            Image.fromarray(domain_rgb(u[CROP, CROP], 0.7)).save(gen(name))
        for name, u, cap in (("spec7_smooth.png", us, 1.0), ("spec7_rand.png", ur, 0.62)):
            L = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(u))) ** 2 + 1e-20)
            L = np.clip((L - (L.max() - 7)) / 7, 0, 1)
            if cap < 1:                      # flat spectrum: show its mean level, not its speckle peaks
                L = L * cap / np.percentile(L, 99)
            Image.fromarray(cmap_rgb(np.clip(L, 0, 1), "magma")).resize((360, 360), Image.BILINEAR).save(gen(name))
        open(gen("done_v2"), "w").write("ok")


build_assets()
build_assets_v2()


def img(''')
# --- hold(): wait until the current caption has been read before changing the visuals
rep('''    def uncaption(self):''', '''    def hold(self, extra=0.0):
        self._hold_caption()
        if extra:
            self.wait(extra)

    def uncaption(self):''')

# ================= S01
rep('''        la = tx("closed-form wavefronts", 20, DIM).next_to(ar, UP, buff=0.12)
        ph = img(gen("phase_smooth.png"), 2.7).move_to([0.55, 0.95, 0])''',
    '''        la = VGroup(tx("closed-form", 20, DIM), tx("wavefronts", 20, DIM)).arrange(DOWN, buff=0.06).next_to(ar, UP, buff=0.12)
        ph = img(gen("field_smooth.png"), 2.7).move_to([0.55, 0.95, 0])''')
rep('''        self.add(la, label(ph, "hologram phase at the SLM", 24))
        e1 = tx(EQ_GWS, 30).move_to([0, -1.25, 0])''', '''        self.add(la, label(ph, "SLM field (hue = phase)", 24))
        self.hold()
        e1 = tx(EQ_GWS, 30).move_to([0, -1.25, 0])''')
rep('''        self.caption(f"Every wave gets the phase {I('e')}{sup(I('ikz') + sub(I('i')))}, so all are in step at the SLM: "
                     "a smooth-phase hologram.")
        self.caption("Sharp and photoreal in focus. But this paper asks: what does that smooth phase cost us?")''',
    '''        self.caption(f"Each wave carries {I('e')}{supsub(I('ikz'), I('i'))}, so all arrive in step: one smooth phase (a single hue).")
        self.caption("Sharp and photoreal in focus. But what does that smooth phase cost?")''')
# ================= S02
rep('''        self.caption("A slit of light with smooth (constant) phase: the beam just keeps going straight.")
        self.play(FadeIn(r1), run_time=0.6)''', '''        self.caption("A slit of light with smooth (constant) phase: the beam just keeps going straight.")
        self.hold()
        self.play(FadeIn(r1), run_time=0.6)''')
rep('''        self.caption("Give every point a random phase: light scatters into every angle the pixels can produce.")
        self.play(FadeIn(r24), run_time=0.8)''', '''        self.caption("Give every point a random phase: light scatters into every angle the pixels can produce.")
        self.hold()
        self.play(FadeIn(r24), run_time=0.8)''')
rep('''        self.caption("Averaged over frames it spreads as a smooth cone, like light from a real, diffuse surface.")''',
    '''        self.caption("Averaged over frames: a smooth cone, like light from a real diffuse surface.")''')
rep('''        files = ["phase_smooth.png", "spec_smooth.png", "phase_rand.png", "spec_rand.png"]
        names = ["SLM phase", "angular spectrum", "SLM phase", "angular spectrum"]
        ims = []
        for x, f, nm in zip(xs, files, names):
            m = img(gen(f), h).move_to([x, 0.45, 0])
            ims.append(m)
            self.add(label(m, nm, 22, DIM))''', '''        files = ["field_smooth.png", "spec7_smooth.png", "field_rand.png", "spec7_rand.png"]
        names = ["SLM field (hue = phase)", "angular spectrum", "SLM field (hue = phase)", "angular spectrum"]
        ims = []
        labs = []
        for x, f, nm in zip(xs, files, names):
            m = img(gen(f), h).move_to([x, 0.45, 0])
            ims.append(m)
            labs.append(label(m, nm, 22, DIM))''')
rep('''        self.play(FadeIn(ims[0]), FadeIn(ims[1]), run_time=0.6)
        self.add(g1, sep)''', '''        self.play(FadeIn(ims[0]), FadeIn(ims[1]), run_time=0.6)
        self.add(g1, sep, labs[0], labs[1])''')
rep('''        self.caption("The hologram of our toy scene. Smooth phase puts nearly all energy at low spatial frequencies.")
        self.play(FadeIn(ims[2]), FadeIn(ims[3]), run_time=0.6)
        self.add(g2)''', '''        self.caption("The toy scene's hologram. Smooth phase puts nearly all energy at low spatial frequencies.")
        self.hold()
        self.play(FadeIn(ims[2]), FadeIn(ims[3]), run_time=0.6)
        self.add(g2, labs[2], labs[3])''')
rep('''        spn = tx("spectra: log scale, 5 decades; square = everything the 8 µm pixels can send out", 18, DIM''',
    '''        spn = tx("spectra: log scale, 7 decades; the square = every direction the 8 µm pixels can send light", 18, DIM''')
rep('''        self.caption("Random phase spreads energy evenly over the whole band: the SLM's space-bandwidth is used.")
        self.caption("Each spatial frequency is a direction. In a near-eye display, the spectrum becomes the eyebox.")''',
    '''        self.caption("Random phase spreads energy evenly over the band: the SLM's space-bandwidth product is used.")
        self.caption("Each frequency is a direction. In a near-eye display, the spectrum becomes the eyebox.")''')
# ================= S03
rep('''        spec_s = img(gen("spec_smooth.png"), h).move_to([xs[0], 0.5, 0])
        spec_r = img(gen("spec_rand.png"), h).move_to([xs[2], 0.5, 0])''', '''        spec_s = img(gen("spec7_smooth.png"), h).move_to([xs[0], 0.5, 0])
        spec_r = img(gen("spec7_rand.png"), h).move_to([xs[2], 0.5, 0])''')
rep('''            return always_redraw(lambda: Circle(radius=rad, color=CYAN, stroke_width=3).move_to(''',
    '''            return always_redraw(lambda: Circle(radius=rad, color=CYAN, stroke_width=5).move_to(''')
rep('''        self.caption("The eye's pupil samples one patch of the eyebox (cyan circle). Centred, both holograms look fine.")''',
    '''        self.caption("The eye's pupil catches one patch of the eyebox (cyan). Centred, both holograms look fine.")''')
rep('''        self.caption("Move the pupil: the smooth-phase hologram goes dark. Its light never reaches that part of the eyebox.")''',
    '''        self.caption("Move the pupil: the smooth-phase hologram goes dark. Its light never gets there.")''')
# ================= S04
rep('''        self.caption("Focus on the star (6 mm). Now watch the checkerboard, 12 mm behind it.")
        self.caption("Smooth phase: the far checkerboard barely blurs, it only rings and ghosts.")
        self.run(z, 18, 3.5, smooth)
        self.caption("Focus on the checkerboard: smooth phase shows the star sharp-edged and ringing, not blurred.")
        self.run(z, 6, 3.5, smooth)
        self.caption("Random phase blurs out-of-focus content like a real camera, but a single frame is full of speckle.")
        self.run(z, 18, 3.0, smooth)
        self.caption("With 24 frames the speckle averages out: clean, natural defocus blur, a depth cue for the eye.")
        self.run(z, 6, 3.0, smooth)
        self.run(z, 12, 1.5, smooth)''', '''        self.caption("Focus on the star, 6 mm. The checkerboard sits 12 mm behind it.")
        self.caption("Smooth phase: the far checkerboard barely blurs. It rings instead.")
        self.hold()
        self.run(z, 18, 2.5, smooth)
        self.caption("Focus on the checkerboard: the smooth-phase star stays sharp-edged and ringing.")
        self.caption("Random phase blurs out-of-focus content like a real camera, but one frame is full of speckle.")
        self.run(z, 6, 3.0, smooth)
        self.caption("With 24 frames the speckle averages out: clean, natural defocus blur, a real depth cue.")
        self.run(z, 18, 3.0, smooth)
        self.run(z, 12, 1.5, smooth)''')
# ================= S05
rep('''        self.caption("Speckle reduction is the classic reason to time-multiplex. RPWS finds a deeper one.")''',
    '''        self.caption("That is the classic reason to time-multiplex. RPWS finds a deeper one.")''')
# ================= S06
rep('''        self.caption(f"Two translucent primitives overlap. Blend their waves with weights {W1}, {W2}: the intensity has a cross term.")
        l1 = VGroup(''', '''        self.caption(f"Blend two overlapping waves with weights {W1}, {W2}: the intensity picks up a cross term.")
        l1 = VGroup(''')
rep('''        l2.move_to([3.4, 0.6, 0])
        self.add(l1)''', '''        l2.move_to([3.4, 0.6, 0])
        self.hold()
        self.add(l1)''')
rep('''        self.caption("Smooth phase: the waves are in step, amplitudes add. That is GWS's alpha wave blending.")
        self.add(l2)''', '''        self.caption("Smooth phase: waves in step, amplitudes add. That is GWS's alpha wave blending.")
        self.hold()
        self.add(l2)''')
rep('''        self.caption("Random phases: the cross term averages to zero over frames, so intensities add instead.")
        w_nrp =''', '''        self.caption("Random phases: the cross term averages to zero over frames, so intensities add.")
        self.hold()
        w_nrp =''')
rep('''        self.caption("So RPWS blends in intensity: square-root weights, from primitives fitted to intensity.")''',
    '''        self.caption("So RPWS blends in intensity: square-root weights, on primitives fitted to intensity.")''')
rep('''        self.add(label(mg, "ray-traced target", 24), label(mn, "GWS-NRP (naïve random phase)", 24, RED),
                 label(mr, "RPWS (square-root weights)", 24, GREEN))''', '''        self.add(label(mg, "ray-traced target", 24), label(mn, "GWS-NRP (naïve)", 24, RED),
                 label(mr, "RPWS (√ weights)", 24, GREEN))''')
rep('''        self.caption("Toy: orange in front of blue. One frame: both are speckled.")''',
    '''        self.caption("Toy: orange in front of blue. After one frame, both are speckled.")''')
rep('''        self.caption("Naïvely adding random phase to GWS (GWS-NRP): wrong colours wherever the Gaussians are translucent, however many frames.")
        self.caption("RPWS converges to the target: its alpha blending is exact in expectation.")
        self.caption("Time multiplexing is not a speckle trick here: it is what makes random-phase alpha blending correct.")''',
    '''        self.caption("GWS with random phase (GWS-NRP) keeps a ring of wrong colour, however many frames.")
        self.caption("RPWS converges to the target: exact alpha blending, in expectation over frames.")
        self.caption("So time multiplexing is not a speckle trick: it is what makes the blending correct.")''')
# ================= S07
rep('''        masks = VGroup(*[tx(f"× √(1 − α)  + own wave", 18, WHITE).next_to(planes[k], UP, buff=0.08) for k in (1, 2)])
        self.play(GrowArrow(arrs[0]), run_time=0.5)''', '''        masks = VGroup(*[VGroup(tx("× √(1 − α)", 18), tx("+ own wave", 18)).arrange(DOWN, buff=0.05
                                                                                         ).next_to(planes[k], UP, buff=0.08)
                         for k in (1, 2)])
        self.hold()
        self.play(GrowArrow(arrs[0]), run_time=0.5)''')
rep('''        self.caption("Carry its wave to the next primitive's plane, mask it by √(1 − α), add that primitive's random-phase wave.")
        self.play(GrowArrow(arrs[1]), run_time=0.5)''', '''        self.caption("Carry its wave to the next plane, mask by √(1 − α), add that primitive's random-phase wave.")
        self.hold()
        self.play(GrowArrow(arrs[1]), run_time=0.5)''')
rep('''        self.caption("Repeat until the SLM. The mask sits at the occluder's own depth, so it blocks light from every angle.")''',
    '''        self.caption("Repeat until the SLM. Each mask sits at the occluder's depth, blocking light from every angle.")''')
rep('''        self.caption("Toy check. Masking on the far plane, as GWS does, leaves dark halos beside the star in off-axis views.")''',
    '''        self.caption("Toy check: masking on the far plane, as GWS does, leaves dark halos in off-axis views.")''')
rep('''        self.caption("On real scenes (paper ablation): only the full RPWS, back to front and √ blending, gets colour and occlusion right.")''',
    '''        self.caption("Paper ablation: only full RPWS (back to front + √ blending) gets colour and occlusion right.")''')
# ================= S08
rep('''        self.caption("GWS at front focus: the background barely blurs and rings. RPWS at back focus: natural blur.")
        a2 =''', '''        self.caption("GWS at front focus: the background barely blurs and rings. RPWS at back focus: natural blur.")
        self.hold(0.5)
        a2 =''')
rep('''        self.caption("A simulated light field: GWS's light sits at the eyebox centre, so its right view is black.")
        self.caption("RPWS keeps every view: the energy is spread evenly across the eyebox.")''',
    '''        self.caption("Simulated light field: GWS's light sits at the eyebox centre, so its right view is black. RPWS keeps every view.")''')
rep('''        self.caption("Paper Table 1, on dense 10 × 10 light fields: RPWS with 24 frames is best on every metric.")
        head =''', '''        self.caption("Paper Table 1, dense 10 × 10 light fields: RPWS with 24 frames is best on every metric.")
        self.hold()
        head =''')
rep('''        self.caption("That is about 10.5 dB more light-field PSNR than GWS, and more with every extra frame.")''',
    '''        self.caption("About 10.5 dB more light-field PSNR than GWS; more frames, better quality.")''')
rep('''        self.caption("Captured on a real holographic display: GWS stays sharp at both focus settings, with ringing.")
        self.caption("RPWS shifts focus between the bulldozer and the table, with natural blur.")''',
    '''        self.caption("Captured on a real display: GWS looks alike at both focus settings; RPWS refocuses with natural blur.")''')
rep('''        self.caption("More frames, less speckle: at 24 frames the captured holograms are nearly speckle-free.")''',
    '''        self.caption("More frames, less speckle: at 24 frames the captures are nearly speckle-free.")''')
rep('''        self.caption("Not just Gaussians: the same framework turns soft-edged triangle splats into holograms.")''',
    '''        self.caption("Not just Gaussians: soft-edged triangle splats become holograms too.")''')
# ================= S09
rep('''        res = tx("full eyebox · parallax · natural defocus · +10.5 dB light-field PSNR over GWS", 26, AMBER''',
    '''        res = tx("full eyebox · parallax · natural defocus · +10.5 dB light-field PSNR over GWS (Blender, 24 frames)", 24, AMBER''')
open(p, "w").write(s)
print("patched")
