p = 'animation_gws_v2.py'
s = open(p).read()
def R(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:80])
    s = s.replace(old, new)

R('done_v1', 'done_v2', 4)
# ---- assets
R('R = np.exp(1j * 2 * np.pi * 60e3 * X)', 'R = np.exp(1j * 2 * np.pi * 90e3 * X)')
R('''        N = 300
        for name, (cx, cy, su, sv, th, c, o) in {
            "g_red": (150, 150, 70, 30, 0.5, (255, 82, 82), 0.95),
            "g_amber": (185, 175, 60, 45, -0.3, (255, 202, 40), 0.9),
            "g_blue": (120, 190, 85, 40, 1.2, (68, 138, 255), 0.9),
            "g_one": (150, 150, 95, 40, 0.45, (77, 208, 225), 1.0),''', '''        N = 400
        for name, (cx, cy, su, sv, th, c, o) in {
            "g_red": (200, 200, 70, 30, 0.5, (255, 82, 82), 0.95),
            "g_amber": (235, 225, 60, 45, -0.3, (255, 202, 40), 0.9),
            "g_blue": (170, 240, 85, 40, 1.2, (68, 138, 255), 0.9),
            "g_one": (200, 200, 95, 40, 0.45, (77, 208, 225), 1.0),''')
R('''    U *= 1 / np.sqrt(1.0) * 0.55
    line_z = zc + X * np.tan(tilt)             # the disk's plane
    before = Z < line_z
    inc = np.exp(1j * 2 * np.pi * Z)
    U = np.where(before, inc, U)
    np.save(gen("tilted_U.npy"), U.astype(np.complex64))
    np.save(gen("tilted_mask.npy"), (np.abs(Z - line_z) < 0.25) & (np.abs(X) < 9 * np.cos(tilt)))''', '''    line_z = zc + X * np.tan(tilt)             # the disk's plane
    before = Z < line_z
    U = U / np.percentile(np.abs(U[~before]), 99)
    inc = np.exp(1j * 2 * np.pi * Z)
    U = np.where(before, inc, U)
    np.save(gen("tilted_U.npy"), U.astype(np.complex64))
    # disk drawn with brightness = Gaussian amplitude along the disk
    s_of = X / np.cos(tilt)
    disk = np.where(np.abs(Z - line_z) < 0.3, np.exp(-0.5 * (s_of / sig) ** 2), 0.0)
    np.save(gen("tilted_disk.npy"), disk.astype(np.float32))''')

# ---- continuous clock so waves keep moving during caption holds
R('''    def setup(self):
        self.cap = None
        self.hdr = None
''', '''    def setup(self):
        self.cap = None
        self.hdr = None
        self.clock = ValueTracker(0)
        self.clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(self.clock)
''')
R('''        keep = set(keep) | {self.hdr}''', '''        keep = set(keep) | {self.hdr, self.clock}''')
R('''    def title_card(self, num, title, hold=1.8):''', '''    def title_card(self, num, title, hold=1.5):''')
open(p, 'w').write(s)
print("ok")
