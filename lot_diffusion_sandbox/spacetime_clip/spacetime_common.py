"""Shared data, timeline and camera for the space-time clip (used by render_volume.py and the manim scene).

Data: site showcase item 31 (VBench test prompt, depth-of-field layout, LoT-Wan2.1 14B, 81 frames → 21 latent
frames, 45 × 80 tokens per latent frame, 27,795 of 75,600 tokens = 2.72× fewer; the site's compression field). Leaves are exact, decoded from ../web/showcase.json
(token_grid: gzip-base64-uint8 (y0, y1, x0, x1) per latent frame). Tokens have extent (1, e_h, e_w): every token is
one latent frame thick; the stacked "tube" is a display of per-frame layouts, not 3D tokens.
"""
import base64
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ITEM = int(os.environ.get("LOT_ITEM", 31))
CLIP = os.path.join(ROOT, "data", "clips", f"show{ITEM:02d}_gen")      # 81 generated frames, 1280 × 720
GEN = os.path.join(HERE, "data", "generated")
VOL = os.path.join(GEN, f"vol_{ITEM}")                                        # rendered volume frames, 60 fps

H, W, T = 45, 80, 21
DURATION, FPS = 10.0, 60

# ------------------------------------------------------------------ timeline (s)
TOP_END = 1.0                     # top-down view of latent frame 0
TILT = (1.0, 2.6)                 # camera tilts from top-down to oblique
DROP0, DROP_DT, DROP_LEN = 1.3, 0.13, 0.45     # latent frame k ≥ 1 drops in at DROP0 + (k − 1) DROP_DT
SWEEP = (4.5, 9.4)                # a plane sweeps latent frames 0 → 20 while the video plays (≈ real time)
ORBIT = (2.6, 10.0)
CAPS = [(0.0, "One LoT layout per latent frame."),
        (2.8, "Stacked in time, the fine tokens follow her as she moves."),
        (6.8, "The far background stays coarse: 2.7× fewer tokens.")]


def decode():
    d = json.load(open(os.path.join(ROOT, "web", "showcase.json")))
    tg = d["items"][ITEM]["token_grid"]
    raw = np.frombuffer(gzip.decompress(base64.b64decode(tg["leaves"])), np.uint8).reshape(-1, 4).astype(int)
    off = tg["offsets"]
    assert (tg["height"], tg["width"], len(off) - 1) == (H, W, T)
    return [raw[off[t]:off[t + 1]] for t in range(T)]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease(t, a, b):
    return smooth((t - a) / (b - a))


def drop_start(k):
    return -1.0 if k == 0 else DROP0 + (k - 1) * DROP_DT


def sweep_tau(t):
    """Continuous latent-frame index of the sweep plane, or None before the sweep."""
    if t < SWEEP[0]:
        return None
    return (T - 1) * min((t - SWEEP[0]) / (SWEEP[1] - SWEEP[0]), 1.0)


def video_frame(t):
    """0-based index into the 81 generated frames shown next to the volume at clip time t."""
    tau = sweep_tau(t)
    if tau is None:
        landed = max([k for k in range(T) if t >= drop_start(k) + DROP_LEN] + [0])
        return 4 * landed
    return int(round(4 * tau))


def latent_of(f):
    """Latent frame of pixel frame f (0-based): latent 0 ← frame 0, latent i ≥ 1 ← frames 4i − 3 … 4i."""
    return 0 if f == 0 else (f + 3) // 4


# ------------------------------------------------------------------ geometry + camera
TS = 2.0                          # world units per latent frame along time (token cell = 1 unit)
THICK = 0.7                       # slab thickness as a fraction of TS (gaps show separate frames)
VOL_H = T * TS
REGION = (2.35, 0.45, 9.3, 6.0)   # manim (cx, cy, w, h) of the rendered volume
RES = (1256, 810)                 # output pixels of the region at 1080p
FOV = 28.0


def camera(t):
    """Eye, target at clip time t. Time runs up (+y); columns → +x; rows → +z (row 0 far)."""
    a = ease(t, *TILT)
    th = np.radians(89.0 + (24.0 - 89.0) * a)
    ph = np.radians(-40.0 * a + 80.0 * ease(t, *ORBIT) * a)
    d = 118.0 + (205.0 - 118.0) * a
    tgt = np.array([0.0, THICK * TS / 2 + (VOL_H / 2 - 2.0) * a, 0.0])
    eye = tgt + d * np.array([np.cos(th) * np.sin(ph), np.sin(th), np.cos(th) * np.cos(ph)])
    return eye, tgt


def matrices(t, aspect=RES[0] / RES[1]):
    eye, tgt = camera(t)
    f = tgt - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, [0.0, 1.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    n, fa = 5.0, 1000.0
    k = 1 / np.tan(np.radians(FOV) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1] = k / aspect, k
    proj[2, 2], proj[2, 3] = (fa + n) / (n - fa), 2 * fa * n / (n - fa)
    proj[3, 2] = -1
    return proj, view, eye


def project(t, p):
    """World point → manim coordinates (inside REGION)."""
    proj, view, _ = matrices(t)
    q = proj @ view @ np.array([*p, 1.0])
    x, y = q[0] / q[3], q[1] / q[3]
    cx, cy, w, h = REGION
    return np.array([cx + x * w / 2, cy + y * h / 2, 0.0])


def world_x(c):
    return c - W / 2


def world_z(r):
    return r - H / 2
