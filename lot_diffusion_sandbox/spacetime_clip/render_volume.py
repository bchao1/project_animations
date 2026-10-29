"""Render the space-time volume of per-frame LoT layouts (moderngl, headless EGL) → data/generated/vol/NNNN.png.

Each token of latent frame k is a box (rows y0..y1, cols x0..x1, one latent frame thick) with a 10% cell margin.
1 × 1 tokens are opaque (the subject's tube); larger tokens are glass boxes (faint faces, extent-coloured edges).
Usage: python render_volume.py [first_frame last_frame]   (default: all DURATION × FPS frames)
"""
import os
import sys

import moderngl
import numpy as np
from PIL import Image

from spacetime_common import (DROP_LEN, DURATION, FPS, H, RES, T, THICK, TS, VOL, W, decode, drop_start,
                              matrices, sweep_tau, world_x, world_z)

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
                  slab * ts + pos.y * thick * ts + (1.0 - e) * 28.0,
                  rect.z + m + pos.z * (rect.w - rect.z - 2.0 * m));
    v_l = pos; v_n = nrm; v_col = col; v_area = area; v_slab = slab; v_w = w;
    gl_Position = P * V * vec4(w, 1.0);
}
"""

FS = """
#version 330
uniform float cur; uniform float glass; uniform vec3 eye;
in vec3 v_l; in vec3 v_n; in vec3 v_col; in float v_area; in float v_slab; in float v_vis; in vec3 v_w;
out vec4 frag;
void main() {
    if (v_vis <= 0.001) discard;
    vec2 uv = abs(v_n.x) > 0.5 ? v_l.yz : (abs(v_n.y) > 0.5 ? v_l.xz : v_l.xy);
    vec2 d = min(uv, 1.0 - uv) / max(fwidth(uv), vec2(1e-5));
    float edge = 1.0 - smoothstep(0.6, 1.9, min(d.x, d.y));
    vec3 L = normalize(vec3(-0.45, 0.85, 0.35));
    float sh = 0.72 + 0.28 * max(dot(v_n, L), 0.0);
    float hl = cur < -0.5 ? 0.0 : 1.0 - smoothstep(0.35, 0.65, abs(v_slab - cur));
    float dim = cur < -0.5 ? 1.0 : mix(0.62, 1.0, hl);
    vec3 c = v_col / 255.0;
    if (glass < 0.5) {                       // opaque fine token
        vec3 face = c * sh * dim * (1.0 + 0.35 * hl);
        vec3 col = mix(face, c * 0.28, edge * 0.85);
        frag = vec4(col, v_vis);
    } else {                                 // glass coarse token
        float fa = (0.07 + 0.16 * hl) * sh;
        float ea = (0.55 + 0.4 * hl) * dim;
        vec3 col = mix(c * sh, min(c * 1.25 + 0.08, 1.0), edge);
        frag = vec4(col, mix(fa, ea, edge) * v_vis);
    }
}
"""


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
            row = (world_x(x0), world_x(x1), world_z(y0), world_z(y1), k, *AREA_COL[a], a)
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
    plane = np.array([[world_x(-1), world_x(W + 1), world_z(-1), world_z(H + 1), 0, *AMBER, 99]], "f4")
    ibo_p = ctx.buffer(plane.tobytes(), dynamic=True)
    va_p = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm"), (ibo_p, fmt_i, "rect", "slab", "col")])
    w, h = RES[0] * SS, RES[1] * SS
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=4)],
                          depth_attachment=ctx.depth_renderbuffer((w, h), samples=4))
    out = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
    prog["drop0"].value, prog["dropdt"].value, prog["droplen"].value = 1.3, 0.13, DROP_LEN
    prog["ts"].value, prog["thick"].value = TS, THICK
    assert abs(drop_start(1) - 1.3) < 1e-9 and abs(drop_start(2) - drop_start(1) - 0.13) < 1e-9
    centers = np.stack([(coarse[:, 0] + coarse[:, 1]) / 2, coarse[:, 4] * TS + THICK * TS / 2,
                        (coarse[:, 2] + coarse[:, 3]) / 2], 1)
    os.makedirs(VOL, exist_ok=True)
    n = int(DURATION * FPS)
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, n - 1)
    for i in range(a, b + 1):
        t = i / FPS
        P, V, eye = matrices(t)
        prog["P"].write(P.T.astype("f4").tobytes())
        prog["V"].write(V.T.astype("f4").tobytes())
        prog["clock"].value = t
        tau = sweep_tau(t)
        prog["cur"].value = -1.0 if tau is None else float(round(tau))
        fbo.use()
        fbo.clear(0, 0, 0, 1)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = True
        prog["glass"].value = 0.0
        va_f.render(instances=len(fine))
        order = np.argsort(-np.linalg.norm(centers - eye, axis=1))       # glass boxes back to front
        ibo_c.write(coarse[order].tobytes())
        ctx.depth_mask = False
        prog["glass"].value = 1.0
        va_c.render(instances=len(coarse))
        if tau is not None:
            pl = plane.copy()
            pl[0, 4] = round(tau)
            ibo_p.write(pl.tobytes())
            prog["thick"].value = 1.0 / TS * 0.06 + THICK            # plane hugs the current slab
            prog["cur"].value = float(round(tau))
            va_p.render(instances=1)
            prog["thick"].value = THICK
        ctx.depth_mask = True
        ctx.copy_framebuffer(out, fbo)
        img = Image.frombytes("RGBA", (w, h), out.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
        img.convert("RGB").resize(RES, Image.LANCZOS).save(os.path.join(VOL, f"{i:04d}.png"))
    print("done", a, b)


if __name__ == "__main__":
    main()
