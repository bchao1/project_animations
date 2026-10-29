"""Shared data, timeline, token motion and camera for the space-time clip, v8 (render_volume_v11.py and
animation_lot_spacetime_v16.py).

v8 (user feedback on v7: real fluid starts spreading the moment it lands, not after a pause): no "sand castle" —
friction is low from the start, and the hydrostatic push and jitter act from the first landing, but only on landed
cubes (resting on the pile, slow vertically), with the surface-height map built from landed cubes only, so cubes in
flight still fall straight down; damping rises earlier (SETTLE) so the sloshing dies out (as settle2d_clip v3).
Tokens also carry their mean video colour (img_col: mean of the patch over the latent frame's pixel frames) for
the colour variant ($LOT_COLOR=image).

v7 (user feedback on v6: the ease into the lattice slab looked unnatural): no snap — the cubes stay where the physics
puts them, and the simulation runs to the end of the clip with light damping, so the surface keeps moving around the
36.8% level.

v6 (user feedback on v5: the trickle did not move like real water / sand): the settle is a real particle simulation
(sim_physics.py, position-based dynamics on the GPU: gravity, contacts, floor + walls, friction high → ~0, then
viscosity); v6 then eased the rested cubes into an exact lattice slab, which v7 drops.

v5 (user feedback on v10): the settle is a particle-style simulation (settle_sim.py) instead of a choreographed
raster refill — cubes let go at random times and fall under gravity onto whatever is below them, then the piles
flatten by cubes trickling off the towers into lower neighbouring columns; a final squash brings the slab to the
exact level.

Story: frame 1 face-on as a flat 2D layout → the frames step out toward the viewer along time (octant x, y, time at
the back-bottom-left corner of frame 1) → every token shrinks to the size of one 1 × 1 token (keeping its colour) →
the cubes fall straight down into piles (one cube per token per (x, time) column) → the piles flatten into one slab
whose height on the y axis is the token fraction: 27,795 / 75,600 = 36.8% of the full-resolution height.

Data: site showcase item 31 (VBench test prompt, depth-of-field layout, LoT-Wan2.1 14B, 81 frames → 21 latent
frames, 45 × 80 tokens per latent frame). Leaves are exact, decoded from ../web/showcase.json. Every token is one
latent frame thick; stacking, piling and flattening are displays of the per-frame layouts and their token count.
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
COLOR = os.environ.get("LOT_COLOR", "extent")                         # extent | image
VOL = os.path.join(GEN, f"vol11{'c' if COLOR == 'image' else ''}_{ITEM}")                               # rendered volume frames, 60 fps
PHYS = os.path.join(GEN, f"physics3_{ITEM}.npz")                      # sim_physics_v3.py output

H, W, T = 45, 80, 21
DURATION, FPS = 13.8, 60

# ------------------------------------------------------------------ timeline (s)
ROTATE = (1.0, 2.6)               # camera turns from face-on to the oblique view
GROW = (1.0, 1.8)                 # slabs grow from flat (2D) to their depth as the rotation starts
AXIS_GROW = (1.3, 2.8)            # time axis grows out of the origin
STEP0, STEP_DT, STEP_LEN = 1.6, 0.1, 0.4       # latent frame k ≥ 1 steps out at STEP0 + (k − 1) STEP_DT
ORBIT = (2.6, 11.0)
SHRINK = (5.6, 6.5)               # every token → one 1 × 1-token cube at a cell of its footprint
FALL = (6.6, 13.8)                # physics: release, fall, pile, liquefy, flow out, keep moving to the end
SIM_END = FALL[1]
RELEASE, GRAV = 0.6, 140.0        # release window (s) after FALL[0]; gravity (cells / s²)
LIQUEFY = (6.6, 6.8)              # v8: fluid from the first landing (friction low, push + jitter on)
SETTLE = (8.3, 9.3)               # damping rises: the sloshing dies out, the surface keeps moving
VLAND = 6.0                       # |v_y| (cells / s) below which a touching cube counts as landed
D = 1.1                           # particle diameter (cell units): settled packing ≈ 1 cube per cell
FLAT = (99.0, 100.0)              # (renderer: voxel-edge softening — unused without the snap)
GHOST = (8.9, 9.5)                # wireframe of the full-resolution volume fades in
COLLAPSE = (11.6, 12.4)           # the token count flies onto the y axis at the slab's height


PRESSURE = (8.3, 99.0)            # hydrostatic (shallow-water) push down the height gradient, to the end
JITTER = 6.0                      # horizontal velocity noise (cells / s / √s) while fluid


def friction(t):
    """Contact friction rate (1 / s) on horizontal velocity: grippy sand → slippery."""
    return 30.0 - (30.0 - 0.3) * ease(t, *LIQUEFY)


def viscosity(t):
    """Bulk damping rate (1 / s): small while flowing, moderate later so the flow calms without freezing."""
    return 0.3 + 1.7 * ease(t, *SETTLE)


def pressure(t):
    """Gain (cells / s² per unit surface slope) of the hydrostatic push."""
    return 90.0 * ease(t, *LIQUEFY)


def grow(age):
    """Particle diameter vs time since release: just under the lattice spacing → D, in the air."""
    return 0.98 + (D - 0.98) * float(smooth((age - 0.15) / 0.6)) if np.isscalar(age) else \
        0.98 + (D - 0.98) * smooth((age - 0.15) / 0.6)
REPLAY = (3.9, 8.9)               # video replays at ≈ real time after the stacking
CAPS = [(0.0, "One LoT layout per latent frame."),
        (2.8, "Stacked in time, the fine tokens follow her."),
        (5.6, "Now shrink every token to a single cube…"),
        (8.6, "…and let them settle: 36.8% of the full-resolution tokens.")]


def decode():
    d = json.load(open(os.path.join(ROOT, "web", "showcase.json")))
    tg = d["items"][ITEM]["token_grid"]
    raw = np.frombuffer(gzip.decompress(base64.b64decode(tg["leaves"])), np.uint8).reshape(-1, 4).astype(int)
    off = tg["offsets"]
    assert (tg["height"], tg["width"], len(off) - 1) == (H, W, T)
    return [raw[off[t]:off[t + 1]] for t in range(T)]


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease(t, a, b):
    return float(smooth((t - a) / (b - a)))


def step_start(k):
    return -1.0 if k == 0 else STEP0 + (k - 1) * STEP_DT


def video_frame(t):
    """0-based index into the 81 generated frames shown next to the volume at clip time t."""
    if t < REPLAY[0]:
        landed = max([k for k in range(T) if t >= step_start(k) + STEP_LEN] + [0])
        return 4 * landed
    return int(min(80, round(80 * (t - REPLAY[0]) / (REPLAY[1] - REPLAY[0]))))


def latent_of(f):
    """Latent frame of pixel frame f (0-based): latent 0 ← frame 0, latent i ≥ 1 ← frames 4i − 3 … 4i."""
    return 0 if f == 0 else (f + 3) // 4


# ------------------------------------------------------------------ geometry
TS = 3.0                          # world units per latent frame along time (token cell = 1 unit)
THICK = 0.6                       # slab depth as a fraction of TS (gaps show separate frames)
VOL_D = T * TS
M = 0.1                           # token margin (fraction of a cell) on every side
FLOOR = -H / 2
REGION = (2.35, 0.45, 9.3, 6.0)   # manim (cx, cy, w, h) of the rendered volume
RES = (1256, 810)                 # output pixels of the region at 1080p
FOV = 28.0


def thick(t):
    """Slab depth (fraction of TS) at clip time t: ~0 (flat 2D tokens) until the rotation starts."""
    return max(THICK * ease(t, *GROW), 0.002)


def world_x(c):
    return c - W / 2


def world_y(r):
    return H / 2 - r


def slab_z(k):
    """Back face of latent frame k (the slab spans slab_z(k) … slab_z(k) + THICK·TS)."""
    return k * TS


def token_colors(k, y0, y1, x0, x1):
    """Mean 0–255 RGB of every token's patch in its latent frame (frames 4k − 3 … 4k; frame 0 for k = 0), cached."""
    p = os.path.join(GEN, f"tokcol_{ITEM}.npy")
    if os.path.exists(p):
        return np.load(p)
    from PIL import Image
    fs = sorted(os.listdir(CLIP))
    cell = []
    for kk in range(T):
        idx = [0] if kk == 0 else range(4 * kk - 3, 4 * kk + 1)
        a = np.mean([np.asarray(Image.open(os.path.join(CLIP, fs[i])).convert("RGB").resize((W * 16, H * 16)),
                                np.float64) for i in idx], 0)
        cell.append(a.reshape(H, 16, W, 16, 3).mean((1, 3)))
    col = np.array([cell[kk][a0:a1, b0:b1].mean((0, 1)) for kk, a0, a1, b0, b1 in zip(k, y0, y1, x0, x1)], np.float32)
    np.save(p, col)
    return col


class Tokens:
    """All 27,795 tokens with their four states (layout, unit cube, pile, flat) as box corners (lo, hi)."""

    def __init__(self, sim=True):
        frames = decode()
        L = np.concatenate([np.c_[np.full(len(f), k), f] for k, f in enumerate(frames)])   # k, y0, y1, x0, x1
        self.k, y0, y1, x0, x1 = L.T
        self.n = len(L)
        self.area = (y1 - y0) * (x1 - x0)
        self.fine = self.area == 1
        self.img_col = token_colors(self.k, y0, y1, x0, x1)
        # layout boxes (x, y; z added per frame)
        self.lay_lo = np.c_[world_x(x0) + M, world_y(y1) + M]
        self.lay_hi = np.c_[world_x(x1) - M, world_y(y0) - M]
        # unit cube at a cell of the footprint (top-left-of-centre cell)
        rs, cs = y0 + (y1 - y0 - 1) // 2, x0 + (x1 - x0 - 1) // 2
        self.cell = np.c_[rs, cs]
        self.unit_lo = np.c_[world_x(cs) + M, world_y(rs + 1) + M]
        self.unit_hi = np.c_[world_x(cs + 1) - M, world_y(rs) - M]
        self.LEVEL = self.n / (W * T)
        if sim:
            self.pos = np.load(PHYS)["pos"]                    # (frames, n, 3) particle centres, cell space

    @staticmethod
    def to_world(c):
        """Cell-space centres → world centres (z: latent frame k + 0.5 → middle of slab k)."""
        return np.c_[c[:, 0] - W / 2, FLOOR + c[:, 1], (c[:, 2] - 0.5) * TS + THICK * TS / 2]

    @staticmethod
    def col_xz(c):
        return world_x(c % W) + M, slab_z(c // W)

    def boxes(self, t):
        """(lo, hi, vis, glass) per token at clip time t; lo / hi are (n, 3) world corners."""
        n = self.n
        zt = thick(t) * TS
        # step-out along time
        st = np.array([step_start(k) for k in range(T)])[self.k]
        p = np.clip((t - st) / STEP_LEN, 0, 1)
        e = 1 - (1 - p) ** 3
        vis = smooth(p / 0.35)
        z0 = slab_z(self.k) - (1 - e) * 2 * TS
        lo = np.c_[self.lay_lo, z0]
        hi = np.c_[self.lay_hi, z0 + zt]
        glass = (~self.fine).astype(float)
        if t >= SHRINK[0]:
            s = ease(t, *SHRINK)
            lo[:, :2] = (1 - s) * self.lay_lo + s * self.unit_lo
            hi[:, :2] = (1 - s) * self.lay_hi + s * self.unit_hi
            glass *= 1 - s ** 3                       # glass turns solid only as the cubes reach unit size
        if t >= FALL[0]:                                  # particle physics (sim_physics.py)
            u = min((t - FALL[0]) * FPS, len(self.pos) - 1.0)
            a = int(u)
            b = min(a + 1, len(self.pos) - 1)
            c = self.pos[a] + (u - a) * (self.pos[b] - self.pos[a])
            hy = np.full(self.n, 0.5 - M)
            w = self.to_world(c)
            half = np.c_[np.full(self.n, 0.5 - M), hy, np.full(self.n, zt / 2)]
            lo, hi = w - half, w + half
        return lo, hi, vis, glass


# ------------------------------------------------------------------ camera + octant
def camera(t):
    """Face-on at the start (camera on +z looking at frame 1), then oblique from the upper left; the orbit comes to
    rest when the slab has flattened."""
    a = ease(t, *ROTATE)
    o = ease(t, *ORBIT)
    th = np.radians(20.0 * a - 2.0 * o)
    ph = np.radians(-58.0 * a + 18.0 * o)
    d = 150.0 + (212.0 - 150.0) * a
    tgt = np.array([6.0 * (1 - a), 6.0 - 15.0 * a, (VOL_D / 2 - 1.0) * a])
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


AX_OFF = 1.0
ORIGIN = (-W / 2 - AX_OFF, -H / 2 - AX_OFF, -AX_OFF)
AX_END = {"x": (W / 2 + 7.0, -H / 2 - AX_OFF, -AX_OFF),
          "y": (-W / 2 - AX_OFF, H / 2 + 7.0, -AX_OFF),
          "time": (-W / 2 - AX_OFF, -H / 2 - AX_OFF, VOL_D + 8.0)}
TIP = 2.6                         # arrowhead length (world units)
TICK = 2.4                        # y-axis tick length (toward −x)


def axis_end(name, t):
    f = 1.0 if name != "time" else ease(t, *AXIS_GROW)
    return np.array(ORIGIN) + f * (np.array(AX_END[name]) - np.array(ORIGIN))


def ytick(level):
    """Inner and outer end of a y-axis tick at height `level` above the floor."""
    y = FLOOR + level
    return np.array([ORIGIN[0], y, ORIGIN[2]]), np.array([ORIGIN[0] - TICK, y, ORIGIN[2]])
