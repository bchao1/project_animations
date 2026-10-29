p = 'animation_gws_v3.py'
s = open(p).read()
def R(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:90])
    s = s.replace(old, new)
R('teaser animation, v2.', 'teaser animation, v3.')
R('animation_gws_v2.py <Scene>', 'animation_gws_v3.py <Scene>')
R('done_v2', 'done_v3', 4)
R('R = np.exp(1j * 2 * np.pi * 90e3 * X)', 'R = np.exp(1j * 2 * np.pi * 60e3 * X)')
# caption pacing
R('# Captions stay up long enough to read: ~15 characters per second + 1 s.', '# Captions stay up long enough to read: ~17 characters per second + 0.8 s.')
R('self.cap_need = len(plain) / 15.0 + 1.0', 'self.cap_need = len(plain) / 17.0 + 0.8')
# uniform label centres
R('''def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    return tx(s, size, color).next_to(m, d, buff=buff)''', '''def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m; centred at a fixed offset so labels in a row share one line."""
    t = tx(s, size, color).next_to(m, d, buff=buff)
    off = buff + 0.011 * size
    t.set_y(m.get_top()[1] + off if d is UP else m.get_bottom()[1] - off)
    return t''')
# higher-gamma LUT for the side view
R('''WLUT = wave_lut()''', '''WLUT = wave_lut()


def wave_lut_g(g):
    t = np.linspace(-1, 1, 256)[:, None]
    neg = np.array([0x29, 0x79, 0xFF]) / 255.0
    pos = np.array([0xFF, 0xCA, 0x28]) / 255.0
    rgb = np.where(t < 0, (-t) ** g * neg, t ** g * pos)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


WLUT2 = wave_lut_g(1.5)''')
R('''            a = wave_rgb(np.real(U * np.exp(-1j * 2 * PI * 0.8 * clk.get_value())), 1.2).astype(np.float32)''',
  '''            re = np.real(U * np.exp(-1j * 2 * PI * 0.8 * clk.get_value()))
            a = WLUT2[np.clip((re / 1.05 * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)].astype(np.float32)''')
# caption merges / accuracy
R('''        self.caption(f"A camera only records the intensity |{I('u')}|{SQ}. The phase is lost.")
        self.caption("Holography records the phase too, and plays it back.")''',
  '''        self.caption(f"A camera only records the intensity |{I('u')}|{SQ}: the phase is lost. Holography keeps it.")''')
R('''        self.caption("Real scenes need millions of points, and points know nothing about occlusion.")''',
  '''        self.caption("But detailed scenes need an excessive number of points, and occlusion is hard to model.")''')
R('''                "Meanwhile, neural rendering solved photoreal 3D with Gaussian splatting."]''',
  '''                "Meanwhile, neural rendering solved photoreal 3D with Gaussian splatting. Can holography use it?"]''')
R('''        self.add(src)
        self.caption("Can we turn those Gaussians straight into a hologram?")
        self.clear_all()''', '''        self.add(src)
        self.clear_all()''')
R('''        self.caption("The Gaussian Wave Splatting equation: the exact spectrum of any 2D Gaussian at the SLM.")
        self.caption("No per-primitive FFT, no resampling of tilted spectra: evaluate it on the frequency grid.")''',
  '''        self.caption("The Gaussian Wave Splatting equation: the exact spectrum of any 2D Gaussian at the SLM, "
                     "no per-primitive FFT or resampling needed.")''')
R('''        self.caption("On a real Gaussian scene: naïve blending oversaturates, silhouettes leave dark artifacts.")
        self.caption("Alpha wave blending reproduces the target rendering.")''',
  '''        self.caption("On a real scene, naïve blending oversaturates and silhouettes leave artifacts; "
                     "alpha wave blending matches the target.")''')
R('''        self.caption("Table 2: 6.84 s for fast GWS vs 180 s for exact, the paper's “nearly 30×” speedup,")
        self.caption("on par with prior point and polygon CGH, at far higher image quality.")''',
  '''        self.caption("Table 2: 6.84 s for fast GWS vs 180 s for exact (“nearly 30×”), on par with prior CGH "
                     "at far higher quality.")''')
open(p, 'w').write(s)
print('ok')
