"""LoT Diffusion as LEGO, 10-s clip, v1 (moderngl headless EGL + PIL text → ffmpeg). No manim.

Every token is a LEGO plate on a 64 × 64 baseplate, coloured by the mean of the corgi image under it.
  0.0–3.0 s  4,096 1 × 1 plates rain down (standard DiT: one token per patch); counter counts landed plates.
  3.0–7.0 s  plates fuse along the real LoT quadtree of the corgi demo: 1→2→4→8 (four plates lift, fuse, drop back
             as one bigger plate); counter ticks down to 1,546.
  7.0–10 s   camera to top-down, board on the left, generated image slides in on the right, title.
Usage: python render_lego_v1.py [l|h]   l = 960×540 30 fps preview, h = 1920×1080 60 fps → media/final/LoT_lego_v1_<res>.mp4
"""
import json
import os
import subprocess
import sys

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAYOUT = os.path.join(ROOT, "teaser_1min/data/generated/corgi_layout.json")
IMG = os.path.join(ROOT, "data/img/paint0_result.png")
TAG = "v1"

Q = sys.argv[1] if len(sys.argv) > 1 else "l"
W, H = (1920, 1080) if Q == "h" else (960, 540)
FPS = 60 if Q == "h" else 30
SS = 3 if Q == "h" else 2
DUR = 10.0
U = W / 1920                                     # text scale

N = 64
GAP = 0.035                                      # per-side inset between plates (stud pitch = 1)
PLATE_H = 0.4                                    # LEGO plate: 3.2 mm / 8 mm pitch
STUD_R, STUD_H = 0.30, 0.18
LIFT = 1.4                                       # fusing plates lift this high, the fused plate drops from here
BASE_COL = (0xA0, 0xA5, 0xA9)                    # light bluish grey baseplate
GROUND_COL = (0xEC, 0xE6, 0xDB)
INK = (0x2A, 0x2A, 0x2A)
RED, BLUE, YELLOW = (0xC9, 0x1A, 0x09), (0x00, 0x55, 0xBF), (0xF5, 0xCD, 0x2F)
FONT_BLACK = "/usr/share/fonts/truetype/lato/Lato-Black.ttf"
FONT_HEAVY = "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf"

INTRO0, INTRO_SPAN = 0.15, 1.7
STAGES = {2: (3.0, 0.9), 4: (4.45, 0.8), 8: (5.8, 0.6)}     # stage start, stagger window
FALL0 = (9.0, 0.42, 0.35, 0.16)                             # intro drop: height, fall time, bounce amp, bounce time
LIFT_T = 0.18
CAM_T0, CAM_T1 = 7.0, 8.4
INF = 1e4


# ------------------------------------------------------------------ bricks
def build():
    leaves = [tuple(l) for l in json.load(open(LAYOUT))]
    leaf_of = np.zeros((N, N), int)
    for i, j, b in leaves:
        leaf_of[i:i + b, j:j + b] = b
    img = np.asarray(Image.open(IMG).convert("RGB").resize((N * 16, N * 16), Image.BOX)).astype(float) / 255
    cell = img.reshape(N, 16, N, 16, 3).mean((1, 3))
    rng = np.random.default_rng(3)

    def colour(i, j, s):
        c = cell[i:i + s, j:j + s].mean((0, 1))
        g = c.mean()
        return np.clip(g + 1.12 * (c - g), 0, 1)                 # a touch more saturated: plastic

    event = {}                                                   # (s, i, j) → fuse time of that s-block
    for s, (t0, win) in STAGES.items():
        for i in range(0, N, s):
            for j in range(0, N, s):
                if leaf_of[i, j] >= s:
                    u = 0.75 * (j + 0.3 * i) / (1.3 * N) + 0.25 * rng.random()
                    event[(s, i, j)] = t0 + win * u
    bricks = []                                                  # x0 y0 sx sy z0 h r g b tin tout kind edge

    def add(i, j, s, tin, kind):
        par = 2 * s
        tout = event.get((par, i - i % par, j - j % par), INF) if leaf_of[i, j] >= par else INF
        x0, y0 = j - N / 2 + GAP, N / 2 - i - s + GAP
        bricks.append([x0, y0, s - 2 * GAP, s - 2 * GAP, 0, PLATE_H, *colour(i, j, s), tin, tout, kind, 1])

    for i in range(N):
        for j in range(N):
            add(i, j, 1, INTRO0 + INTRO_SPAN * (i + j) / (2 * N - 2) + 0.15 * rng.random(), 0)
    for (s, i, j), te in event.items():
        add(i, j, s, te, 1)
    bricks = np.array(bricks, "f4")
    studs = []
    for x0, y0, sx, sy, z0, h, r, g, b, tin, tout, kind, _ in bricks:
        s = int(round(sx + 2 * GAP))
        for di in range(s):
            for dj in range(s):
                studs.append([x0 - GAP + dj + 0.5, y0 - GAP + di + 0.5, STUD_R, STUD_R, h, STUD_H, r, g, b, tin, tout, kind, 0])
    base = [[-N / 2 - 0.5, -N / 2 - 0.5, N + 1, N + 1, -0.32, 0.32, *np.array(BASE_COL) / 255, -1, INF, 2, 1]]
    base_studs = [[j - N / 2 + 0.5, N / 2 - i - 0.5, STUD_R, STUD_R, 0, STUD_H, *np.array(BASE_COL) / 255, -1, INF, 2, 0]
                  for i in range(N) for j in range(N)]
    ground = [[-1500, -1500, 3000, 3000, -50.32, 50.0,              # thick: a 0.01 slab z-fought its own bottom face
                *np.array(GROUND_COL) / 255, -1, INF, 2, 0]]
    return bricks, np.array(studs, "f4"), np.array(base + ground, "f4"), np.array(base_studs, "f4")


def landed(br, t):
    """Plates resting on the board at time t (the live token counter)."""
    tf = np.where(br[:, 11] == 0, FALL0[1], 0.0)            # a fused plate counts as soon as it replaces its four
    return int(((br[:, 9] + tf <= t) & (t < br[:, 10])).sum())


# ------------------------------------------------------------------ geometry
def box():
    f = []
    faces = [((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
             ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
             ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
             ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
             ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
             ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])]
    for n, q in faces:
        for k in (0, 1, 2, 0, 2, 3):
            f.append((*q[k], *n))
    return np.array(f, "f4")


def cylinder(seg=28):
    f = []
    for k in range(seg):
        a0, a1 = 2 * np.pi * k / seg, 2 * np.pi * (k + 1) / seg
        c0, s0, c1, s1 = np.cos(a0), np.sin(a0), np.cos(a1), np.sin(a1)
        for p in [(c0, s0, 0), (c1, s1, 0), (c1, s1, 1), (c0, s0, 0), (c1, s1, 1), (c0, s0, 1)]:
            f.append((*p, p[0], p[1], 0))
        for p in [(0, 0, 1), (c0, s0, 1), (c1, s1, 1)]:
            f.append((*p, 0, 0, 1))
    return np.array(f, "f4")


VS = """
#version 330
uniform mat4 VP; uniform float t;
in vec3 pos; in vec3 nrm;
in vec4 rect; in vec2 zr; in vec3 col; in vec3 tim; in float edge;
out vec3 v_n; out vec3 v_col; out vec3 v_w; out vec3 v_l; out vec3 v_sz; out float v_flash; out float v_edge;
float fall(float s, float H, float tf, float a, float tb) {
    if (s < tf) { float u = s / tf; return H * (1.0 - u * u); }
    s -= tf;
    if (s < tb) return a * sin(3.14159 * s / tb);
    s -= tb;
    if (s < 0.6 * tb) return 0.3 * a * sin(3.14159 * s / (0.6 * tb));
    return 0.0;
}
void main() {
    float tin = tim.x, tout = tim.y, kind = tim.z;
    float dz = 0.0; bool hide = t < tin || t >= tout;
    if (kind < 0.5) dz = fall(t - tin, 9.0, 0.42, 0.35, 0.16);
    else if (kind < 1.5) dz = fall(t - tin, %(LIFT)f, 0.20, 0.18, 0.12);
    float lu = clamp((t - (tout - %(LIFT_T)f)) / %(LIFT_T)f, 0.0, 1.0);
    dz += %(LIFT)f * (1.0 - (1.0 - lu) * (1.0 - lu));
    vec3 w = vec3(rect.x + pos.x * rect.z, rect.y + pos.y * rect.w, zr.x + dz + pos.z * zr.y);
    if (hide) w = vec3(0.0, 0.0, -50.0);
    v_n = nrm; v_col = col; v_w = w; v_l = pos; v_sz = vec3(rect.zw, zr.y); v_edge = edge;
    v_flash = kind > 0.5 && kind < 1.5 ? exp(-max(t - tin, 0.0) / 0.18) : 0.0;
    gl_Position = VP * vec4(w, 1.0);
}
""" % dict(LIFT=LIFT, LIFT_T=LIFT_T)

FS = """
#version 330
uniform vec3 L; uniform vec3 eye; uniform mat4 LVP; uniform sampler2DShadow shadow; uniform vec3 ground;
in vec3 v_n; in vec3 v_col; in vec3 v_w; in vec3 v_l; in vec3 v_sz; in float v_flash; in float v_edge;
out vec4 frag;
void main() {
    vec4 lp = LVP * vec4(v_w, 1.0);
    vec3 sc = lp.xyz / lp.w * 0.5 + 0.5;
    float sh = 0.0;
    vec2 px = 1.0 / vec2(textureSize(shadow, 0));
    for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++)
        sh += texture(shadow, vec3(sc.xy + vec2(dx, dy) * px * 1.5, sc.z - 0.0015));
    sh /= 9.0;
    if (any(lessThan(sc.xy, vec2(0.0))) || any(greaterThan(sc.xy, vec2(1.0)))) sh = 1.0;
    vec3 N = normalize(v_n);
    vec3 V = normalize(eye - v_w);
    vec3 Hh = normalize(L + V);
    vec3 base = pow(v_col, vec3(2.2));
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, Hh), 0.0), 140.0) * 0.30;
    float fres = pow(1.0 - max(dot(N, V), 0.0), 5.0) * 0.10;
    float sky = 0.5 + 0.5 * N.z;
    vec3 c = base * (0.16 + 0.12 * sky) + base * diff * 0.80 * sh + vec3(spec + fres) * sh;
    if (v_edge > 0.5) {                                   // dark seam on plate edges (bevel)
        vec3 p = v_l * v_sz;
        vec2 uv = abs(N.z) > 0.5 ? p.xy : (abs(N.x) > 0.5 ? p.yz : p.xz);
        vec2 sz = abs(N.z) > 0.5 ? v_sz.xy : (abs(N.x) > 0.5 ? v_sz.yz : v_sz.xz);
        vec2 d = min(uv, sz - uv);
        float e = 1.0 - smoothstep(0.0, 0.045, min(d.x, d.y));
        c *= mix(1.0, 0.72, e);
    }
    c += vec3(1.0, 0.95, 0.8) * v_flash * 0.25;
    c = pow(c, vec3(1.0 / 2.2));
    frag = vec4(c, 1.0);
}
"""

SHADOW_FS = """
#version 330
void main() {}
"""


# ------------------------------------------------------------------ camera
def look_at(eye, tgt, up):
    f = tgt - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def persp(fov, aspect, n, f):
    t = 1 / np.tan(np.radians(fov) / 2)
    return np.array([[t / aspect, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]])


def ortho(l, r, b, t, n, f):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)], [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]])


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


FIN_FOV, FIN_FRAC, FIN_NDC_X = 20.0, 0.66, -0.45
FIN_D = (N / 2 + 0.5) / (FIN_FRAC * np.tan(np.radians(FIN_FOV / 2)))


def camera(t):
    a = smooth(t / CAM_T0) if t < CAM_T0 else 1.0
    az0, el0, d0, fov0, tg0 = -26 + 12 * a, 52 + 5 * a, 136 - 6 * a, 34.0, np.array([1.0, -5.0, 0.0])
    b = smooth((t - CAM_T0) / (CAM_T1 - CAM_T0))
    half = FIN_D * np.tan(np.radians(FIN_FOV / 2))
    tg1 = np.array([-FIN_NDC_X * (W / H) * half, 0.0, 0.0])
    az, el = (1 - b) * az0, (1 - b) * el0 + b * 90
    d, fov, tg = (1 - b) * d0 + b * FIN_D, (1 - b) * fov0 + b * FIN_FOV, (1 - b) * tg0 + b * tg1
    # keep the board's apparent size continuous through the fov change: blend in tan space
    fov = np.degrees(2 * np.arctan((1 - b) * np.tan(np.radians(fov0 / 2)) + b * np.tan(np.radians(FIN_FOV / 2))))
    ar, er = np.radians(az), np.radians(el)
    u = np.array([np.sin(ar), -np.cos(ar), 0.0])
    eye = tg + d * np.array([*(np.cos(er) * u[:2]), np.sin(er)])
    up = np.array([*(-np.sin(er) * u[:2]), np.cos(er)])
    return persp(fov, W / H, 10.0, 2000.0) @ look_at(eye, tg, up), eye


# ------------------------------------------------------------------ text
FONTS = {}


def font(path, size):
    k = (path, int(size * U))
    if k not in FONTS:
        FONTS[k] = ImageFont.truetype(path, int(size * U))
    return FONTS[k]


def ramp(t, a, b=None, fade=0.25):
    v = smooth((t - a) / fade)
    if b is not None:
        v *= 1 - smooth((t - b) / fade)
    return v


def text(im, xy, s, size, col, alpha, path=FONT_HEAVY, anchor="la"):
    if alpha <= 0.01:
        return
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((xy[0] * U, xy[1] * U), s, font=font(path, size), fill=(*col, int(255 * alpha)),
                               anchor=anchor)
    im.alpha_composite(layer)


def brick_tag(im, xy, s, alpha):
    """Text on a little yellow 1 × n LEGO brick (with studs)."""
    if alpha <= 0.01:
        return
    f = font(FONT_BLACK, 40)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tw = d.textlength(s, font=f)
    x, y = xy[0] * U, xy[1] * U
    w, h, r = tw + 44 * U, 62 * U, 8 * U
    a = int(255 * alpha)
    shade = tuple(int(c * 0.78) for c in YELLOW)
    ns = max(2, int(w / (36 * U)))
    for k in range(ns):
        cx = x + (k + 0.5) * w / ns
        d.rounded_rectangle([cx - 11 * U, y - 9 * U, cx + 11 * U, y + 4 * U], 3 * U, fill=(*shade, a))
    d.rounded_rectangle([x, y, x + w, y + h], r, fill=(*YELLOW, a), outline=(*shade, a), width=max(1, int(3 * U)))
    d.text((x + w / 2, y + h / 2 + 1 * U), s, font=f, fill=(*INK, a), anchor="mm")
    im.alpha_composite(layer)


def arrow(im, x0, x1, y, alpha):
    if alpha <= 0.01:
        return
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    x0, x1, y = x0 * U, x1 * U, y * U
    d.line([x0, y, x1 - 22 * U, y], fill=(*INK, a), width=max(2, int(8 * U)))
    d.polygon([(x1, y), (x1 - 30 * U, y - 20 * U), (x1 - 30 * U, y + 20 * U)], fill=(*INK, a))
    im.alpha_composite(layer)


def overlay(im, t, count, gen_img, board_px):
    a1 = ramp(t, 0.25, 2.9)
    a2 = ramp(t, 2.9, CAM_T0)
    text(im, (72, 58), "Standard diffusion transformer: one token per patch", 38, INK, a1)
    text(im, (72, 58), "Level-of-Token: big tokens where the image is simple", 38, INK, a2)
    ac = ramp(t, 0.25, CAM_T0)
    if ac > 0.01:
        col = RED if t < 2.9 else BLUE
        text(im, (68, 104), f"{count:,}", 120, col, ac, FONT_BLACK)
        nx = 68 + font(FONT_BLACK, 120).getlength(f"{count:,}") / U + 18
        text(im, (nx, 165), "tokens", 46, INK, ac)
    brick_tag(im, (76, 262), "2.6× fewer", ramp(t, 6.55, CAM_T0))
    # final layout
    bx0, by0, bx1, by1 = board_px
    side = bx1 - bx0
    ai = ramp(t, 8.0, fade=0.5)
    if ai > 0.01:
        cx = W * (1 - FIN_NDC_X) / 2
        dx = int(cx - side / 2 + (1 - ai) * 60 * U)
        g = gen_img.resize((int(side), int(side)), Image.LANCZOS).convert("RGBA")
        g.putalpha(int(255 * ai))
        sh = Image.new("RGBA", g.size, (0, 0, 0, int(60 * ai)))
        im.alpha_composite(sh, (dx + int(10 * U), int(by0 + 10 * U)))
        im.alpha_composite(g, (dx, int(by0)))
    af = ramp(t, 8.3, fade=0.4)
    text(im, (960, 82), "Level-of-Token Diffusion", 66, INK, af, FONT_BLACK, "mm")
    text(im, ((bx0 + bx1) / 2 / U, by1 / U + 46), "1,546 LoT tokens", 40, BLUE, af, FONT_BLACK, "mm")
    text(im, (W * (1 - FIN_NDC_X) / 2 / U, by1 / U + 46), "generated image", 40, INK, af, FONT_BLACK, "mm")
    arrow(im, bx1 / U + 40, W * (1 - FIN_NDC_X) / 2 / U - side / 2 / U - 40, (by0 + by1) / 2 / U, ramp(t, 8.15, fade=0.4))


# ------------------------------------------------------------------ main
def main():
    bricks, studs, statics, base_studs = build()
    print(f"plates {len(bricks)}, studs {len(studs) + len(base_studs)}; final count {landed(bricks, 9.9)}")
    ctx = moderngl.create_standalone_context(backend="egl", device_index=int(os.environ.get("EGL_DEV", "5")))
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    sprog = ctx.program(vertex_shader=VS, fragment_shader=SHADOW_FS)
    gbox, gcyl = ctx.buffer(box().tobytes()), ctx.buffer(cylinder().tobytes())
    fmt = "4f 2f 3f 3f 1f /i"
    names = ("rect", "zr", "col", "tim", "edge")
    sets = []
    for arr, geo, gid in [(statics, gbox, 0), (base_studs, gcyl, 1), (bricks, gbox, 0), (studs, gcyl, 1)]:
        ib = ctx.buffer(np.ascontiguousarray(arr, "f4").tobytes())
        va = ctx.vertex_array(prog, [(geo, "3f 3f", "pos", "nrm"), (ib, fmt, *names)])
        vs = ctx.vertex_array(sprog, [(geo, "3f 12x", "pos"), (ib, "4f 2f 12x 3f 4x /i", "rect", "zr", "tim")])
        sets.append((va, vs, gid, len(arr)))
    SM = 4096
    sdepth = ctx.depth_texture((SM, SM))
    sdepth.compare_func = "<="
    sdepth.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sfbo = ctx.framebuffer(depth_attachment=sdepth)
    dummy = ctx.depth_texture((1, 1))
    Ld = np.array([-0.42, 0.34, 1.0])
    Ld /= np.linalg.norm(Ld)
    LVP = ortho(-48, 48, -48, 48, 1, 300) @ look_at(Ld * 120, np.zeros(3), np.array([0, 1.0, 0]))
    w, h = W * SS, H * SS
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=4)],
                          depth_attachment=ctx.depth_renderbuffer((w, h), samples=4))
    out = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
    prog["L"].value = tuple(Ld)
    prog["LVP"].write(LVP.T.astype("f4").tobytes())
    prog["shadow"].value = 0
    gen_img = Image.open(IMG).convert("RGB")
    os.makedirs(os.path.join(HERE, "media/final"), exist_ok=True)
    res = "1080p60" if Q == "h" else "540p30"
    dst = os.path.join(HERE, f"media/final/LoT_lego_{TAG}_{res}.mp4")
    ff = None if os.environ.get("STILL") else subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    sheet = []
    nfr = int(DUR * FPS)
    still = [float(x) for x in os.environ.get("STILL", "").split(",") if x]
    for k in (range(len(still)) if still else range(nfr)):
        t = still[k] if still else k / FPS
        VP, eye = camera(t)
        # shadow pass
        dummy.use(0)                                     # unbind the shadow map while it is redrawn
        sfbo.use()
        ctx.viewport = (0, 0, SM, SM)
        sfbo.clear(depth=1.0)
        sprog["VP"].write(LVP.T.astype("f4").tobytes())
        sprog["t"].value = t
        for va, vs, gid, n in sets[1:]:
            vs.render(instances=n)
        # main pass
        fbo.use()
        ctx.viewport = (0, 0, w, h)
        fbo.clear(*(np.array(GROUND_COL) / 255), 1.0)
        sdepth.use(0)
        prog["VP"].write(VP.T.astype("f4").tobytes())
        prog["eye"].value = tuple(eye)
        prog["t"].value = t
        for va, vs, gid, n in sets:
            va.render(instances=n)
        ctx.copy_framebuffer(out, fbo)
        a = np.frombuffer(out.read(components=3), np.uint8).reshape(h, w, 3)[::-1]
        im = Image.fromarray(a).resize((W, H), Image.BOX).convert("RGBA")
        # board footprint on screen (exact once the camera is top-down)
        c = np.array([[-N / 2 - 0.5, N / 2 + 0.5, 0, 1], [N / 2 + 0.5, -N / 2 - 0.5, 0, 1]]) @ VP.T
        c = c[:, :2] / c[:, 3:]
        board = ((c[0, 0] + 1) / 2 * W, (1 - c[0, 1]) / 2 * H, (c[1, 0] + 1) / 2 * W, (1 - c[1, 1]) / 2 * H)
        overlay(im, t, landed(bricks, t), gen_img, board)
        rgb = im.convert("RGB")
        if still:
            rgb.save(os.path.join(HERE, f"scratch/still_{TAG}_{t:.2f}.png"))
            continue
        ff.stdin.write(rgb.tobytes())
        if k % (FPS // 2) == 0:
            sheet.append(rgb.resize((480, 270)))
        if k % FPS == 0:
            print(f"t = {t:.1f}s", flush=True)
    if still:
        return
    ff.stdin.close()
    ff.wait()
    cols = 4
    S = Image.new("RGB", (cols * 480, ((len(sheet) + cols - 1) // cols) * 270), "white")
    for k, s in enumerate(sheet):
        S.paste(s, ((k % cols) * 480, (k // cols) * 270))
    S.save(os.path.join(HERE, f"scratch/sheet_{TAG}_{res}.png"))
    print("wrote", dst)


if __name__ == "__main__":
    main()
