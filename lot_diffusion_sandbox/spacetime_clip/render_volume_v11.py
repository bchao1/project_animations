"""Render the space-time clip, v11 (moderngl, headless EGL) → data/generated/vol11[c]_<item>/NNNN.png, 60 fps.

v11: physics v3 (fluid from the first landing, spacetime_common_v8); $LOT_COLOR=image renders every token opaque in
its mean video colour (the settle2d_clip scheme) instead of the extent colours with 50% green faces → vol11c_<item>.

v10 = v9 (same pictures, same output dir) + EGL device from $LOT_EGL_DEVICE, for batch renders over several GPUs.

v9 (user feedback on v8): green (coarse) tokens have 50%-opaque faces before the collapse, so the stack reads as a
solid block rather than glass.

v8: token motion of the settle comes from the GPU particle simulation (sim_physics_v2.py via
spacetime_common_v7.Tokens.boxes) to the end of the clip — no snap into a lattice slab.

v6 (user feedback on v5): the settle is driven by settle_sim.py (random release, gravity fall, trickle-down flow,
via spacetime_common_v5.Tokens.boxes); green (coarse) token edges are more opaque before the collapse.

v5 (user feedback on v9): after the stacked space-time view, every token shrinks to the size of one 1 × 1 token
(keeping its colour, glass becomes solid), the cubes fall straight down into piles (one cube per token per (x, time)
column), the piles flatten into one slab of equal height, and the y axis gets ticks at the full-resolution height
(dim) and at the slab's height (amber, 36.8%), with the slab's top outlined in amber. A wireframe of the
full-resolution volume shows the empty space. All token motion lives in spacetime_common_v4.Tokens.boxes(t).
Octant (x, y, time) at the back-bottom-left corner of frame 1, drawn here with depth (shafts, cones, label sprites).
Usage: LOT_ITEM=<k> LOT_COLOR=extent|image LOT_EGL_DEVICE=<gpu> python render_volume_v11.py [first_frame last_frame]   (default: all DURATION × FPS frames)
"""
import os
import sys

import moderngl
import numpy as np
from PIL import Image

from spacetime_common_v8 import (COLOR, AX_END, COLLAPSE, DURATION, FLAT, FLOOR, FPS, GHOST, H, ORIGIN, RES, THICK, TIP, TS,
                                 VOL, VOL_D, W, Tokens, axis_end, ease, matrices, thick, ytick)

FONT = "/home/brianchc/.local/share/fonts/cmunrm.ttf"
SS = 2                                            # supersampling factor
HR, LR, LR3 = (0x44, 0x8A, 0xFF), (0x00, 0xC8, 0x53), (0x1B, 0x7F, 0x3B)
AMBER = (0xFF, 0xCA, 0x28)


def mix(a, b, t):
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))


AREA_COL = {1: HR, 2: mix(HR, LR, 0.5), 4: LR, 8: mix(LR, LR3, 0.5), 16: LR3}

VS = """
#version 330
uniform mat4 P; uniform mat4 V;
in vec3 pos; in vec3 nrm;
in vec3 lo; in vec3 hi; in vec3 col; in float glass; in float vis;
out vec3 v_l; out vec3 v_n; out vec3 v_col; out float v_glass; out float v_vis;
void main() {
    v_l = pos; v_n = nrm; v_col = col; v_glass = glass; v_vis = vis;
    gl_Position = P * V * vec4(lo + pos * (hi - lo), 1.0);
}
"""

FS = """
#version 330
uniform float flatness; uniform float soften;
in vec3 v_l; in vec3 v_n; in vec3 v_col; in float v_glass; in float v_vis;
out vec4 frag;
void main() {
    if (v_vis <= 0.001) discard;
    vec2 uv = abs(v_n.x) > 0.5 ? v_l.yz : (abs(v_n.y) > 0.5 ? v_l.xz : v_l.xy);
    vec2 d = min(uv, 1.0 - uv) / max(fwidth(uv), vec2(1e-5));
    float edge = 1.0 - smoothstep(0.6, 1.9, min(d.x, d.y));
    vec3 L = normalize(vec3(-0.45, 0.6, 0.65));
    float sh = 0.72 + 0.28 * max(dot(v_n, L), 0.0);
    vec3 c = v_col / 255.0;
    vec3 solid = mix(c * sh, c * 0.28, edge * 0.85 * (1.0 - 0.6 * soften));   // opaque token (voxel edges)
    float fa = 0.5 * sh;                                                      // 50%-opaque faces: reads as a solid
    float ea = 0.62 + 0.3 * flatness;                                         // opaque edges (brighter while flat)
    vec3 glassc = mix(c * sh, min(c * 1.25 + 0.08, 1.0), edge);
    float ga = mix(fa, ea, edge);
    frag = vec4(mix(solid, glassc, v_glass), mix(1.0, ga, v_glass) * v_vis);
}
"""

AXIS_VS = """
#version 330
uniform mat4 P; uniform mat4 V; uniform vec3 a; uniform vec3 b; uniform float r;
in vec3 pos; in vec3 nrm;
out vec3 v_n;
void main() {
    vec3 lo = min(a, b) - vec3(r), hi = max(a, b) + vec3(r);
    v_n = nrm;
    gl_Position = P * V * vec4(lo + pos * (hi - lo), 1.0);
}
"""
AXIS_FS = """
#version 330
uniform vec3 col; uniform float alpha;
in vec3 v_n;
out vec4 frag;
void main() {
    vec3 L = normalize(vec3(-0.45, 0.6, 0.65));
    frag = vec4(col * (0.75 + 0.25 * max(dot(v_n, L), 0.0)), alpha);
}
"""
AXIS_COL = {"x": (0.54, 0.54, 0.54), "y": (0.54, 0.54, 0.54), "time": (1.0, 1.0, 1.0)}
MESH_VS = """
#version 330
uniform mat4 P; uniform mat4 V;
in vec3 pos; in vec3 nrm;
out vec3 v_n;
void main() { v_n = nrm; gl_Position = P * V * vec4(pos, 1.0); }
"""
SPRITE_VS = """
#version 330
uniform mat4 P; uniform mat4 V; uniform vec3 c; uniform vec3 sx; uniform vec3 sy;
in vec2 q;
out vec2 v_uv;
void main() { v_uv = vec2(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5); gl_Position = P * V * vec4(c + q.x * sx + q.y * sy, 1.0); }
"""
SPRITE_FS = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 frag;
void main() { vec4 t = texture(tex, v_uv); if (t.a < 0.02) discard; frag = t; }
"""
LABEL_H = 6.5                     # label glyph-box height (world units)
LABEL_GAP = {"x": 3.2, "y": 3.0, "time": 3.6}


def cone(apex, base, r, n=32):
    """Triangles (pos, nrm) of a closed cone from `base` (disc centre) to `apex`."""
    d = apex - base
    L = np.linalg.norm(d)
    d = d / L
    a = np.cross(d, [0.0, 1.0, 0.0] if abs(d[1]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    out = []
    for k in range(n):
        t0, t1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
        p0, p1 = base + r * (np.cos(t0) * a + np.sin(t0) * b), base + r * (np.cos(t1) * a + np.sin(t1) * b)
        nm = np.cos((t0 + t1) / 2) * a + np.sin((t0 + t1) / 2) * b
        nm = (nm * L + d * r) / np.linalg.norm(nm * L + d * r)
        out += [(*p0, *nm), (*p1, *nm), (*apex, *nm), (*p1, *-d), (*p0, *-d), (*base, *-d)]
    return np.array(out, "f4")


def label_texture(ctx, text, rgb):
    from PIL import ImageDraw, ImageFont
    f = ImageFont.truetype(FONT, 200)
    x0, y0, x1, y1 = f.getbbox(text)
    im = Image.new("RGBA", (x1 - x0 + 40, 280), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((20 - x0, 40), text, font=f, fill=tuple(int(255 * v) for v in rgb) + (255,))
    tex = ctx.texture(im.size, 4, im.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    return tex, im.size[0] / im.size[1]


def cube():
    """36 vertices of the unit cube [0, 1]^3 with face normals."""
    faces = [((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
             ((0, 0, -1), [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)]),
             ((1, 0, 0), [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]),
             ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
             ((0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]),
             ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])]
    out = []
    for n, q in faces:
        for i in (0, 1, 2, 0, 2, 3):
            out.append((*q[i], *n))
    return np.array(out, "f4")



def box_edges(lo, hi):
    """12 axis-aligned edges of the box [lo, hi] as (a, b) pairs."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    P = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(a != b for a, b in zip(P[i], P[j])) == 1:
                out.append((np.array(P[i]), np.array(P[j])))
    return out


def main():
    tk = Tokens()
    cols = tk.img_col.astype("f4") if COLOR == "image" else np.array([AREA_COL[a] for a in tk.area], "f4")
    print(f"tokens: {tk.n}, slab height {tk.LEVEL:.3f} of {H} ({100 * tk.LEVEL / H:.1f}%)")
    ctx = moderngl.create_standalone_context(backend="egl", device_index=int(os.environ.get("LOT_EGL_DEVICE", 0)))
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    vbo = ctx.buffer(cube().tobytes())
    fmt_i = "3f 3f 3f 1f 1f /i"
    ibo_o = ctx.buffer(reserve=tk.n * 11 * 4, dynamic=True)
    ibo_g = ctx.buffer(reserve=tk.n * 11 * 4, dynamic=True)
    va_o = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm"), (ibo_o, fmt_i, "lo", "hi", "col", "glass", "vis")])
    va_g = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm"), (ibo_g, fmt_i, "lo", "hi", "col", "glass", "vis")])
    aprog = ctx.program(vertex_shader=AXIS_VS, fragment_shader=AXIS_FS)
    va_a = ctx.vertex_array(aprog, [(vbo, "3f 3f", "pos", "nrm")])
    mprog = ctx.program(vertex_shader=MESH_VS, fragment_shader=AXIS_FS)
    cone_buf = ctx.buffer(reserve=32 * 6 * 6 * 4)
    va_cone = ctx.vertex_array(mprog, [(cone_buf, "3f 3f", "pos", "nrm")])
    sprog = ctx.program(vertex_shader=SPRITE_VS, fragment_shader=SPRITE_FS)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    va_s = ctx.vertex_array(sprog, [(quad, "2f", "q")])
    labels = {k: label_texture(ctx, k, AXIS_COL[k]) for k in AX_END}
    w, h = RES[0] * SS, RES[1] * SS
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=4)],
                          depth_attachment=ctx.depth_renderbuffer((w, h), samples=4))
    out = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
    ghost = box_edges((-W / 2, FLOOR, 0.0), (W / 2, H / 2, VOL_D - (1 - THICK) * TS))
    top = FLOOR + tk.LEVEL
    level = box_edges((-W / 2, top, 0.0), (W / 2, top, VOL_D - (1 - THICK) * TS))
    level = [(a, b) for a, b in level if a[1] == top and b[1] == top and not np.allclose(a, b)]
    os.makedirs(VOL, exist_ok=True)
    n = int(DURATION * FPS)
    i0, i1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, n - 1)
    for i in range(i0, i1 + 1):
        t = i / FPS
        P, V, eye = matrices(t)
        for p in (prog, aprog, mprog, sprog):
            p["P"].write(P.T.astype("f4").tobytes())
            p["V"].write(V.T.astype("f4").tobytes())
        prog["flatness"].value = 1.0 - thick(t) / THICK
        prog["soften"].value = ease(t, FLAT[1] - 0.4, FLAT[1] + 0.4)
        lo, hi, vis, glass = tk.boxes(t)
        if COLOR == "image":                           # mean-colour tokens are solid, like the 2D settle
            glass = np.zeros_like(glass)
        inst = np.c_[lo, hi, cols, glass, vis].astype("f4")
        live = vis > 0.001
        op = inst[live & (glass <= 0.001)]
        gl = inst[live & (glass > 0.001)]
        if len(gl):
            c = (gl[:, 0:3] + gl[:, 3:6]) / 2
            gl = gl[np.argsort(-np.linalg.norm(c - eye, axis=1))]          # glass back to front
        fbo.use()
        fbo.clear(0, 0, 0, 1)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = True
        if len(op):
            ibo_o.orphan(tk.n * 11 * 4)
            ibo_o.write(op.tobytes())
            va_o.render(instances=len(op))

        def shaft(a, b, col, r=0.22, alpha=1.0):
            aprog["a"].value, aprog["b"].value = tuple(a), tuple(b)
            aprog["col"].value, aprog["r"].value, aprog["alpha"].value = col, r, alpha
            va_a.render()

        mprog["alpha"].value = 1.0

        for name in AX_END:
            e = axis_end(name, t)
            d = e - np.array(ORIGIN)
            L = np.linalg.norm(d)
            if L < TIP + 0.5:
                continue
            shaft(ORIGIN, e - d / L * TIP, AXIS_COL[name])
            cone_buf.write(cone(e, e - d / L * TIP, 0.85).tobytes())
            mprog["col"].value = AXIS_COL[name]
            va_cone.render()
        gfade = ease(t, *GHOST)
        lfade = ease(t, (COLLAPSE[0] + COLLAPSE[1]) / 2, COLLAPSE[1])
        ctx.depth_mask = False                         # fading overlays: alpha, never darker than the background
        if gfade > 0:                                  # full-resolution volume (wireframe) + 100% tick
            for a, b in ghost:
                shaft(a, b, (0.30, 0.30, 0.30), r=0.09, alpha=gfade)
            shaft(*ytick(H), (0.54, 0.54, 0.54), alpha=gfade)
        if lfade > 0:                                  # the slab's height: amber tick + top outline
            amb = tuple(v / 255 for v in AMBER)
            shaft(*ytick(tk.LEVEL), amb, alpha=lfade)
            for a, b in level:
                shaft(a + [0, 0.05, 0], b + [0, 0.05, 0], amb, r=0.16, alpha=lfade)
        ctx.depth_mask = True
        if len(gl):
            ctx.depth_mask = False
            ibo_g.orphan(tk.n * 11 * 4)
            ibo_g.write(gl.tobytes())
            va_g.render(instances=len(gl))
        ctx.depth_mask = False
        sx_, sy_ = V[0, :3], V[1, :3]                  # camera right / up in world space
        for name in AX_END:
            e = axis_end(name, t)
            d = e - np.array(ORIGIN)
            L = np.linalg.norm(d)
            if L < TIP + 0.5:
                continue
            tex, asp = labels[name]
            hh = LABEL_H / 2
            sprog["c"].value = tuple(e + d / L * (LABEL_GAP[name] + hh * asp * abs(np.dot(d / L, sx_))))
            sprog["sx"].value = tuple(sx_ * hh * asp)
            sprog["sy"].value = tuple(sy_ * hh)
            tex.use(0)
            sprog["tex"].value = 0
            va_s.render()
        ctx.depth_mask = True
        ctx.copy_framebuffer(out, fbo)
        img = Image.frombytes("RGBA", (w, h), out.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
        img.convert("RGB").resize(RES, Image.LANCZOS).save(os.path.join(VOL, f"{i:04d}.png"))
    print("done", i0, i1)


if __name__ == "__main__":
    main()
