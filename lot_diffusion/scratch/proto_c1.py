import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lot_common import *


class C01_EqualCompute(Base):
    def construct(self):
        L0 = Layout.uniform(64, 64)
        L1 = load_layout("paint0")
        src = os.path.join(IMG, "paint0_result.png")
        pic = img(src, h=5.6).move_to([-3.45, 0.15, 0])
        fr = framed(pic)
        self.add(tx("A diffusion transformer pays the same for every patch", 30).move_to([0, 3.45, 0]))
        self.play(FadeIn(pic), FadeIn(fr), run_time=0.6)
        M = LayoutMorph(L0, L1, 5.6, center=[3.45, 0.15, 0], mode="image", image=src, gap=0.1)
        self.play(FadeIn(M.img), run_time=0.8)
        v = ValueTracker(4096)
        cnt = self.counter(v, lambda x: f"{fmt(x)} tokens", 32, pos=[3.45, -3.2, 0])
        self.add(cnt)
        self.wait(1.2)
        self.play(M.t.animate.set_value(1), v.animate.set_value(1546), run_time=3.0)
        self.wait(1.0)
        self.pad_to(10)
