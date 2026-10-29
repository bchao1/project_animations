"""LoT website clip v3 (2026-10-06; v3 = 8 s, same look as v2)
v2 (2026-10-06; v2 = image fills the frame, text in boxes on top of it)
v1: one 5-s square (1024 × 1024) video of the giraffe example (V1 v7's top-left tile):
depth map in its token layout → noisy token layout → image → mean-colour layout → shrink → fluid settle → % + speedup.

Everything comes from ../social_launch/clip1_signal (signal_common_v7: item 0 = giraffe 025 / dof, its styled cue,
tokens, and tile 0 of physics_v6.npz — tiles never touch in that sim, so tile 0 alone is exact). No column header
("depth map") on purpose. The 60-fps physics is played PHYS_SPEED× faster so the settle fits the 5 s.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "social_launch", "clip1_signal"))
import signal_common_v7 as S           # noqa: E402

B = S.B
N, FPS, FULL = S.N, S.FPS, S.FULL
ITEM = 0                                # giraffe
GEN = os.path.join(HERE, "data", "generated")
FRAMES = os.path.join(GEN, "frames_v3")
PHYS = S.PHYS
speedup, ease, smooth = S.speedup, S.ease, S.smooth

# ------------------------------------------------------------------ geometry (1024² pixels; manim = px / 128)
W = 1024
TP = 992                                # tile side = 15.5 px per cell: the image fills the frame
X0 = Y0 = (W - TP) // 2                 # 16-px border
PAD = 18                                # px from the tile edge to the text boxes


def px_to_m(x, y):
    return x / 128.0 - 4.0, 4.0 - y / 128.0


# ------------------------------------------------------------------ timeline (s)
DURATION = 8.0
TO_NOISE = (1.20, 1.50)                 # 1.2-s hold on the tokenized depth map → noisy token layout
DENOISE = (1.50, 2.80)                  # noise → image inside the token boxes
OPEN = (2.50, 2.80)                     # gaps close: the clean image (held to 3.6)
TOKENIZE = (3.60, 4.20)                 # diagonal sweep: image → mean-colour layout
SHRINK = (4.20, 4.60)                   # every token → a 1 × 1 token
FALL0 = 4.70                            # physics starts (sim time = B.FALL[0] + PHYS_SPEED · (t − FALL0))
PHYS_SPEED = 1.5
COLLAPSE = (7.00, 7.40)                 # % on the level line + speedup fade in; hold to 8.0


def sim_time(t):
    """Seconds into the saved physics (0 = settle2d FALL[0])."""
    return PHYS_SPEED * (t - FALL0)
