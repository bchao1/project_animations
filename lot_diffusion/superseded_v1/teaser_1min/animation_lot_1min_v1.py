"""Level-of-Token Diffusion — 1-minute cut, v1: six 10-second clip bodies + end card (≈ 63 s).

Order: problem (C01) → any cue can drive the layout (C04) → method (C05) → 3D-scene-driven video (R03)
→ move the box (R08) → speed/quality (C08) → end card. Reads ../clips/ and ../data/ read-only.
Render: ./render.sh l|h
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clips"))
from animation_lot_clips_v1 import *  # noqa: E402,F403
import animation_lot_clips_v1 as CL  # noqa: E402


class M(CL.Clip):
    BODY = None

    def setup(self):
        Base.setup(self)
        self.head = None

    def construct(self):
        self.BODY.construct(self)


class M01_Problem(M):
    BODY = CL.C01_EqualCompute


class M02_Sources(M):
    BODY = CL.C04_Sources


class M03_Method(M):
    BODY = CL.C05_AsymFlow


class M04_Video(M):
    BODY = CL.R03_Scene3D


class M05_Control(M):
    BODY = CL.R08_MoveTheBox


class M06_Numbers(M):
    BODY = CL.C08_Numbers


class M07_End(M):
    def construct(self):
        t = tx("Level-of-Token Diffusion", 56, bold=True).move_to([0, 1.2, 0])
        s = tx("up to 2.04× faster images · up to 3.53× faster video · layout-adaptive", 28, DIM).next_to(t, DOWN, buff=0.35)
        g = img("fig1_image.png", h=2.3).next_to(s, DOWN, buff=0.4)
        self.add(t, s, g)
        self.wait(3.0)
