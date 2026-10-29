"""Shared data, timeline and camera for the space-time clip, v3 (used by render_volume_v3.py and
animation_lot_spacetime_v8.py).

v3 geometry: each latent frame stands upright like the video (columns → +x, rows → −y); frames go from back to
front, so time points out of the screen (latent frame k at z = k·TS). The octant origin is the back-bottom-left
corner of frame 1. Frame 1 starts face-on and flat (2D tokens); slabs get their depth only once the camera starts
to rotate, and every later frame steps out of the previous one toward the viewer.

Data: site showcase item 31 (VBench test prompt, depth-of-field layout, LoT-Wan2.1 14B, 81 frames → 21 latent
frames, 45 × 80 tokens per latent frame, 27,795 of 75,600 tokens = 2.72× fewer; the site's compression field).
Leaves are exact, decoded from ../web/showcase.json (token_grid: gzip-base64-uint8 (y0, y1, x0, x1) per latent
frame). Tokens have extent (1, e_h, e_w): every token is one latent frame thick; the stacked "tube" is a display of
per-frame layouts, not 3D tokens.
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
VOL = os.path.join(GEN, f"vol3_{ITEM}")                               # rendered volume frames, 60 fps

H, W, T = 45, 80, 21
DURATION, FPS = 10.0, 60

# ------------------------------------------------------------------ timeline (s)
ROTATE = (1.0, 2.8)               # camera turns from face-on to the oblique view
GROW = (1.0, 1.8)                 # slabs grow from flat (2D) to their depth as the rotation starts
DROP0, DROP_DT, DROP_LEN = 1.8, 0.12, 0.45     # latent frame k ≥ 1 flies in at DROP0 + (k − 1) DROP_DT
SWEEP = (4.6, 9.5)                # a plane sweeps latent frames 0 → 20 while the video plays (≈ real time)
ORBIT = (2.8, 10.0)
AXIS_GROW = (1.0, 2.2)            # time axis grows out of the origin as the camera turns
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
TS = 3.0                          # world units per latent frame along time (token cell = 1 unit)
THICK = 0.6                       # slab depth as a fraction of TS (gaps show separate frames)
VOL_D = T * TS                    # depth of the volume along time
REGION = (2.35, 0.45, 9.3, 6.0)   # manim (cx, cy, w, h) of the rendered volume
RES = (1256, 810)                 # output pixels of the region at 1080p
FOV = 28.0


def thick(t):
    """Slab depth (fraction of TS) at clip time t: ~0 (flat 2D tokens) until the rotation starts."""
    return max(THICK * ease(t, *GROW), 0.002)


def camera(t):
    """Eye, target at clip time t. Face-on at the start (camera on +z looking at frame 1), then oblique from the
    upper left: the y axis (back-left edge) and the time axis (bottom-left edge) stay in view."""
    a = ease(t, *ROTATE)
    o = ease(t, *ORBIT)
    th = np.radians(20.0 * a - 4.0 * o)
    ph = np.radians(-58.0 * a + 18.0 * o)
    d = 117.0 + (222.0 - 117.0) * a
    tgt = np.array([0.0, 0.0, (VOL_D / 2 - 1.0) * a])
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


# octant: origin just outside the back-bottom-left corner of frame 1; axes along the volume's edges, time outward
AX_OFF = 1.0
ORIGIN = (-W / 2 - AX_OFF, -H / 2 - AX_OFF, -AX_OFF)
AX_END = {"x": (W / 2 + 7.0, -H / 2 - AX_OFF, -AX_OFF),
          "y": (-W / 2 - AX_OFF, H / 2 + 7.0, -AX_OFF),
          "time": (-W / 2 - AX_OFF, -H / 2 - AX_OFF, VOL_D + 8.0)}
TIP = 2.6                         # arrowhead length (world units); the head is drawn by manim, the shaft in 3D


def axis_end(name, t):
    """Current end point of an axis (the time axis grows out of the origin as the camera turns)."""
    f = 1.0 if name != "time" else ease(t, *AXIS_GROW)
    return np.array(ORIGIN) + f * (np.array(AX_END[name]) - np.array(ORIGIN))


def world_x(c):
    return c - W / 2


def world_y(r):
    return H / 2 - r


def slab_z(k):
    """Back face of latent frame k (the slab spans slab_z(k) … slab_z(k) + THICK·TS)."""
    return k * TS
