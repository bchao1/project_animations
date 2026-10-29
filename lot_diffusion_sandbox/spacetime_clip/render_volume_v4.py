"""Render the space-time volume of per-frame LoT layouts, v4 (moderngl, headless EGL) → data/generated/vol4_<item>/.

v4: the whole octant (shafts, cone arrowheads, CMU Serif labels as depth-tested sprites) is rendered here, so the
opaque tube hides whatever lies behind it; manim draws no axes.

v3: frames stand upright (x = column, y = −row), go from back to front, and time points out of the screen (+z);
slab depth grows from ~0 (flat 2D tokens, face-on) as the camera starts to rotate (spacetime_common_v3.thick);
each new frame steps out of the previous one.
Each token of latent frame k is a box (rows y0..y1, cols x0..x1, one latent frame thick) with a 10% cell margin.
1 × 1 tokens are opaque (the subject's tube); larger tokens are glass boxes (faint faces, extent-coloured edges).
Usage: python render_volume_v4.py [first_frame last_frame]   (default: all DURATION × FPS frames)
"""
import os
import sys

import moderngl
import numpy as np
from PIL import Image

from spacetime_common_v3 import (AX_END, DROP0, DROP_DT, DROP_LEN, DURATION, FPS, H, ORIGIN, RES, T, THICK, TIP, TS,
                                 VOL, W, axis_end, decode, matrices, slab_z, sweep_tau, thick, world_x, world_y)

VOL = VOL.replace("vol3_", "vol4_")
FONT = "/home/brianchc/.local/share/fonts/cmunrm.ttf"

SS = 2                                            # supersampling factor
HR, LR, LR3 = (0x44, 0x8A, 0xFF), (0x00, 0xC8, 0x53), (0x1B, 0x7F, 0x3B)
AMBER = (0xFF, 0xCA, 0x28)


def mix(a, b, t):
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))


AREA_COL = {1: HR, 2: mix(HR, LR, 0.5), 4: LR, 8: mix(LR, LR3, 0.5), 16: LR3}

VS = """
#version 330
uniform mat4 P; uniform mat4 V; uniform float clock; uniform float drop0; uniform float dropdt;
uniform float droplen; uniform float ts; uniform float thick;
in vec3 pos; in vec3 nrm;
in vec4 rect; in float slab; in vec3 col; in float area;
out vec3 v_l; out vec3 v_n; out vec3 v_col; out float v_area; out float v_slab; out float v_vis; out vec3 v_w;
void main() {
    float st = slab < 0.5 ? -1.0 : drop0 + (slab - 1.0) * dropdt;
    float p = clamp((clock - st) / droplen, 0.0, 1.0);
    float e = 1.0 - pow(1.0 - p, 3.0);
    v_vis = smoothstep(0.0, 0.35, p);
    float m = 0.1;
    vec3 w = vec3(rect.x + m + pos.x * (rect.y - rect.x - 2.0 * m),
                  rect.z + m + pos.y * (rect.w - rect.z - 2.0 * m),
                  slab * ts + pos.z * thick * ts - (1.0 - e) * 2.0 * ts);     // steps out of the previous frame
    v_l = pos; v_n = nrm; v_col = col; v_area = area; v_slab = slab; v_w = w;
    gl_Position = P * V * vec4(w, 1.0);
}
"""

FS = """
#version 330
uniform float cur; uniform float glass; uniform float flatness;
in vec3 v_l; in vec3 v_n; in vec3 v_col; in float v_area; in float v_slab; in float v_vis; in vec3 v_w;
out vec4 frag;
void main() {
    if (v_vis <= 0.001) discard;
    vec2 uv = abs(v_n.x) > 0.5 ? v_l.yz : (abs(v_n.y) > 0.5 ? v_l.xz : v_l.xy);
    vec2 d = min(uv, 1.0 - uv) / max(fwidth(uv), vec2(1e-5));
    float edge = 1.0 - smoothstep(0.6, 1.9, min(d.x, d.y));
    vec3 L = normalize(vec3(-0.45, 0.6, 0.65));
    float sh = 0.72 + 0.28 * max(dot(v_n, L), 0.0);
    float hl = cur < -0.5 ? 0.0 : 1.0 - smoothstep(0.35, 0.65, abs(v_slab - cur));
    float dim = cur < -0.5 ? 1.0 : mix(0.62, 1.0, hl);
    vec3 c = v_col / 255.0;
    if (glass < 0.5) {                       // opaque fine token
        vec3 face = c * sh * dim * (1.0 + 0.35 * hl);
        vec3 col = mix(face, c * 0.28, edge * 0.85);
        frag = vec4(col, v_vis);
    } else {                                 // glass coarse token
        float fa = (0.015 + 0.12 * hl) * sh;               // thin glass: the tube shows through the stack
        float ea = (0.22 + 0.5 * hl + 0.5 * flatness) * dim;   // brighter edges while frame 1 is still flat (2D)
        vec3 col = mix(c * sh, min(c * 1.25 + 0.08, 1.0), edge);
        frag = vec4(col, mix(fa, ea, edge) * v_vis);
    }
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
uniform vec3 col;
in vec3 v_n;
out vec4 frag;
void main() {
    vec3 L = normalize(vec3(-0.45, 0.6, 0.65));
    frag = vec4(col * (0.75 + 0.25 * max(dot(v_n, L), 0.0)), 1.0);
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


def instances(frames):
    fine, coarse = [], []
    for k, L in enumerate(frames):
        for y0, y1, x0, x1 in L:
            a = (y1 - y0) * (x1 - x0)
            row = (world_x(x0), world_x(x1), world_y(y1), world_y(y0), k, *AREA_COL[a], a)
            (fine if a == 1 else coarse).append(row)
    return np.array(fine, "f4"), np.array(coarse, "f4")


def main():
    frames = decode()
    fine, coarse = instances(frames)
    print(f"tokens: {len(fine) + len(coarse)} (fine {len(fine)}, coarse {len(coarse)})")
    ctx = moderngl.create_standalone_context(backend="egl")
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    vbo = ctx.buffer(cube().tobytes())
    ibo_f = ctx.buffer(fine.tobytes())
    ibo_c = ctx.buffer(coarse.tobytes(), dynamic=True)
    fmt_i = "4f 1f 3f 4x /i"
    va_f = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm"), (ibo_f, fmt_i, "rect", "slab", "col")])
    va_c = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm"), (ibo_c, fmt_i, "rect", "slab", "col")])
    # sweep plane: one flat glass box per frame, drawn last
    plane = np.array([[world_x(-1), world_x(W + 1), world_y(H + 1), world_y(-1), 0, *AMBER, 99]], "f4")
    ibo_p = ctx.buffer(plane.tobytes(), dynamic=True)
    va_p = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm"), (ibo_p, fmt_i, "rect", "slab", "col")])
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
    prog["drop0"].value, prog["dropdt"].value, prog["droplen"].value = DROP0, DROP_DT, DROP_LEN
    prog["ts"].value = TS
    centers = np.stack([(coarse[:, 0] + coarse[:, 1]) / 2, (coarse[:, 2] + coarse[:, 3]) / 2,
                        np.array([slab_z(k) for k in coarse[:, 4]]) + THICK * TS / 2], 1)
    os.makedirs(VOL, exist_ok=True)
    n = int(DURATION * FPS)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, n - 1)
    for i in range(a, b + 1):
        t = i / FPS
        P, V, eye = matrices(t)
        prog["P"].write(P.T.astype("f4").tobytes())
        prog["V"].write(V.T.astype("f4").tobytes())
        prog["clock"].value = t
        prog["thick"].value = thick(t)
        prog["flatness"].value = 1.0 - thick(t) / THICK
        tau = sweep_tau(t)
        prog["cur"].value = -1.0 if tau is None else float(round(tau))
        fbo.use()
        fbo.clear(0, 0, 0, 1)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = True
        prog["glass"].value = 0.0
        va_f.render(instances=len(fine))
        aprog["P"].write(P.T.astype("f4").tobytes())
        aprog["V"].write(V.T.astype("f4").tobytes())
        aprog["r"].value = 0.22
        for name in AX_END:                              # shafts stop where manim's arrowhead starts
            e = axis_end(name, t)
            d = e - np.array(ORIGIN)
            n = np.linalg.norm(d)
            if n < TIP + 0.5:
                continue
            aprog["a"].value = tuple(ORIGIN)
            aprog["b"].value = tuple(e - d / n * TIP)
            aprog["col"].value = AXIS_COL[name]
            va_a.render()
            cone_buf.write(cone(e, e - d / n * TIP, 0.85).tobytes())
            mprog["P"].write(P.T.astype("f4").tobytes())
            mprog["V"].write(V.T.astype("f4").tobytes())
            mprog["col"].value = AXIS_COL[name]
            va_cone.render()
        order = np.argsort(-np.linalg.norm(centers - eye, axis=1))       # glass boxes back to front
        ibo_c.write(coarse[order].tobytes())
        ctx.depth_mask = False
        prog["glass"].value = 1.0
        va_c.render(instances=len(coarse))
        if tau is not None:
            pl = plane.copy()
            pl[0, 4] = round(tau)
            ibo_p.write(pl.tobytes())
            prog["thick"].value = 0.06 / TS + THICK                 # plane hugs the current slab
            prog["cur"].value = float(round(tau))
            va_p.render(instances=1)
            prog["thick"].value = thick(t)
        sprog["P"].write(P.T.astype("f4").tobytes())
        sprog["V"].write(V.T.astype("f4").tobytes())
        sx_, sy_ = V[0, :3], V[1, :3]                      # camera right / up in world space
        for name in AX_END:
            e = axis_end(name, t)
            d = e - np.array(ORIGIN)
            n = np.linalg.norm(d)
            if n < TIP + 0.5:
                continue
            tex, asp = labels[name]
            hh = LABEL_H / 2
            sprog["c"].value = tuple(e + d / n * (LABEL_GAP[name] + hh * asp * abs(np.dot(d / n, sx_))))
            sprog["sx"].value = tuple(sx_ * hh * asp)
            sprog["sy"].value = tuple(sy_ * hh)
            tex.use(0)
            sprog["tex"].value = 0
            va_s.render()
        ctx.depth_mask = True
        ctx.copy_framebuffer(out, fbo)
        img = Image.frombytes("RGBA", (w, h), out.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
        img.convert("RGB").resize(RES, Image.LANCZOS).save(os.path.join(VOL, f"{i:04d}.png"))
    print("done", a, b)


if __name__ == "__main__":
    main()
