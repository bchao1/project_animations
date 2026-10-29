from manim import *
import numpy as np
class T(Scene):
    def construct(self):
        a = np.zeros((60,120,4),np.uint8); a[...,3]=255
        m = ImageMobject(a).set(height=3)
        st = {"t":0.0}
        def up(mob, dt):
            st["t"] += dt
            b = np.zeros((60,120,4),np.uint8); b[...,3]=255
            b[..., 1] = (127+127*np.sin(np.arange(120)/5 - 6*st["t"]))[None,:].astype(np.uint8)
            mob.pixel_array = b
        m.add_updater(up)
        self.add(m)
        self.wait(1)
        self.play(m.animate.shift(RIGHT), run_time=1)
