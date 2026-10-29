p = "/home/brianchc/project_animations/random_phase_wave_splatting/animation_rpws_v3.py"
s = open(p).read()
def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (n, a[:80])
    s = s.replace(a, b)
rep('"""Random-phase Wave Splatting — teaser animation, v2.', '"""Random-phase Wave Splatting — teaser animation, v3.')
rep("manim -ql --media_dir media animation_rpws_v2.py", "manim -ql --media_dir media animation_rpws_v3.py")
# keep √(1 − α) on one caption line
rep('''    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        s = re.sub(r"(\\d) × (\\d)", "\\\\1\\u00a0×\\u00a0\\\\2", s)''', '''    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        s = re.sub(r"(\\d) × (\\d)", "\\\\1\\u00a0×\\u00a0\\\\2", s)
        s = s.replace("√(1 − α)", "√(1\\u00a0−\\u00a0α)")''')
rep('''        head = tx("+10.5 dB (Blender) and +10.7 dB (Mip-NeRF 360) light-field PSNR over GWS", 28, AMBER)''',
    '''        head = tx("+10.5 dB (Blender) and +10.7 dB (Mip-NeRF 360) light-field PSNR over GWS", 26, AMBER)''')
rep('''        w = 3.3
        cells = [("gws", "lego_front", -1.9, 1.15), ("rpws24", "lego_front", 1.9, 1.15),
                 ("gws", "lego_back", -1.9, -1.15), ("rpws24", "lego_back", 1.9, -1.15)]''',
    '''        w = 3.6
        cells = [("gws", "lego_front", -2.0, 1.2), ("rpws24", "lego_front", 2.0, 1.2),
                 ("gws", "lego_back", -2.0, -1.2), ("rpws24", "lego_back", 2.0, -1.2)]''')
rep('''        self.add(tx("experimentally", 18, DIM).move_to([5.3, 0.3, 0]), tx("captured", 18, DIM).move_to([5.3, 0.0, 0]),
                 tx("(paper Fig. 6)", 18, DIM).move_to([5.3, -0.3, 0]))''',
    '''        self.add(tx("experimentally", 18, DIM).move_to([5.5, 0.3, 0]), tx("captured", 18, DIM).move_to([5.5, 0.0, 0]),
                 tx("(paper Fig. 6)", 18, DIM).move_to([5.5, -0.3, 0]))''')
rep('''"Captured on a real display: GWS looks alike at both focus settings; RPWS refocuses with natural blur."''',
    '''"Captured on a real display: GWS barely changes with focus; RPWS refocuses with natural blur."''')
open(p, "w").write(s)
print("patched")
