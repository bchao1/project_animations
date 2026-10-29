"""Physics for V1 v6: settle2d v3's sim2d_v3.py, unchanged, on the six v6 items → data/generated/physics_v6.npz.
Usage: CUDA_VISIBLE_DEVICES=0 python sim_v6.py
"""
import os

import signal_common_v6 as C

os.environ["SIM_OUT"] = C.PHYS
C.patch_settle()
os.makedirs(C.GEN, exist_ok=True)
import sim2d_v3      # noqa: E402

sim2d_v3.main()
