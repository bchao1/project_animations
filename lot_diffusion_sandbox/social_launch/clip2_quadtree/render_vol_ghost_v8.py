"""V2 v8: spacetime v16 colour volume (item 31) with the full-resolution wireframe + 100% tick fading in right after
the stacking (v16 t = GHOST_V8) instead of after the settle (8.9 – 9.5) → data/generated/vol11c_31_g40/NNNN.png.
Only frames whose wireframe fade differs are rendered (v16 t 4.0 – 9.5); the rest are symlinks to vol11c_31.
The shared spacetime_clip files are not modified (spacetime_common_v8 is patched in memory).
Usage: LOT_EGL_DEVICE=0 python render_vol_ghost_v8.py [first last]
"""
import os
import sys

os.environ["LOT_COLOR"] = "image"
os.environ["LOT_ITEM"] = "31"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "spacetime_clip"))
import spacetime_common_v8 as ST      # noqa: E402

GHOST_V8 = (4.0, 4.6)
SRC = ST.VOL
OUT = os.path.join(HERE, "data", "generated", "vol11c_31_g40")
ST.GHOST, ST.VOL = GHOST_V8, OUT
import render_volume_v11 as RV        # noqa: E402  (binds the patched GHOST / VOL)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b = int(round(GHOST_V8[0] * ST.FPS)), int(round(9.5 * ST.FPS))
    for i in range(int(round(ST.DURATION * ST.FPS))):
        p = os.path.join(OUT, f"{i:04d}.png")
        if not (a <= i <= b) and not os.path.lexists(p):
            try:
                os.symlink(os.path.join(SRC, f"{i:04d}.png"), p)
            except FileExistsError:                    # parallel chunks
                pass
    sys.argv = [sys.argv[0], *(sys.argv[1:] or [str(a), str(b)])]
    RV.main()
