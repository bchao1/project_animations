"""Left-panel physics: settle2d v3 (../../settle2d_clip/sim2d_v3.py, unchanged) on the corgi alone → physics_left_v1.npz.
settle2d_common_v3 is patched to one item (the corgi's exact leaves and image); its timeline stays local (fall at 4.6 s).
Usage: CUDA_VISIBLE_DEVICES=5 python sim_left_v1.py
"""
import os
import sys

import numpy as np
from PIL import Image

import q_common_v1 as Q

sys.path.insert(0, os.path.join(Q.SB, "settle2d_clip"))
os.environ["SIM_OUT"] = Q.PHYS_L
import settle2d_common_v3 as C    # noqa: E402

C.ITEMS = [("paint0", "painted", "corgi")]
C.NI = 1
C.load_item = lambda i: (Image.open(Q.CORGI_IMG).convert("RGB"), Q.corgi_rects())
import sim2d_v3                  # noqa: E402

os.makedirs(Q.GEN, exist_ok=True)
sim2d_v3.main()
