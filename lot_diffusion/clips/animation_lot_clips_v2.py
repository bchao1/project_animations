"""Level-of-Token Diffusion — thread clips, v2: one beat of the 5-min teaser per clip (same code, STYLE.md §6).

Each clip runs a `Beats` method of ../animation_lot_v2.py in clip mode: header "Level-of-Token Diffusion", beats that
need a previous beat's object build it instantly, and the beat ends on a 1.6 s hold instead of a fade.
Render: ./render.sh l|h [Clip ...]  →  media/final/<res>/LoT_<Clip>.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animation_lot_v2 import *  # noqa: E402,F403


class K(Beats):
    clip = True
    CPS = 19.0                     # thread clips read a little faster (still ≥ the optics teasers' 17)

    def setup(self):
        super().setup()
        self.header(TITLE)
        self._first = True

    def play(self, *anims, **kw):
        # the first animation lands on frame 1, so the post's autoplay/thumbnail frame shows the beat's visual
        if self._first:
            self._first = False
            kw["run_time"] = 1 / config.frame_rate
        return super().play(*anims, **kw)


class K01_TokenCost(K):
    def construct(self):
        self.b_tokencost()


class K02_Attention(K):
    def construct(self):
        self.b_attention()


class K03_Plan(K):
    def construct(self):
        self.b_plan()


class K04_Merge(K):
    def construct(self):
        self.b_merge()


class K05_Shapes(K):
    def construct(self):
        self.b_shapes()


class K06_Sequence(K):
    def construct(self):
        self.b_sequence()


class K07_Naive(K):
    def construct(self):
        self.b_naive()


class K08_Patch(K):
    def construct(self):
        self.b_patch()


class K09_Flow(K):
    def construct(self):
        self.b_flow()


class K10_RoPE(K):
    def construct(self):
        self.b_rope()


class K11_Sources(K):
    def construct(self):
        self.b_sources()


class K12_Budget(K):
    def construct(self):
        self.b_budget()


class K13_MoveBox(K):
    def construct(self):
        self.b_movebox()


class K14_Driving(K):
    def construct(self):
        self.b_driving()


class K15_Speed(K):
    def construct(self):
        self.b_speed()


class K16_Quality(K):
    def construct(self):
        self.b_quality()


class K17_ImageCompare(K):
    def construct(self):
        self.b_imgcompare()


class K18_VideoCompare(K):
    def construct(self):
        self.b_vidcompare()
