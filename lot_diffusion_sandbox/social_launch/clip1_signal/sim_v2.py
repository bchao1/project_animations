"""Physics for V1 v2: settle2d v3's sim2d_v3.py, unchanged, on the six v2 items → data/generated/physics_v2.npz.
Usage: CUDA_VISIBLE_DEVICES=0 python sim_v2.py
"""
import os

import signal_common_v2 as C

os.environ["SIM_OUT"] = C.PHYS
C.patch_settle()
os.makedirs(C.GEN, exist_ok=True)
import sim2d_v3      # noqa: E402

sim2d_v3.main()
