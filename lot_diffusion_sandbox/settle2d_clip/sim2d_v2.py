"""2D physics settle of the unit tokens of all six images, v2 → data/generated/physics2d_v2.npz.

v2: per-disk size spread (C.POLY), tunable surface smoothing (C.POOL) and jitter schedule (C.jitter); optional
surface-weighted push (C.SURF, off: it made the surface rougher). SIM_OVR / SIM_OUT env vars for parameter sweeps.

Same recipe as ../spacetime_clip/sim_physics_v2.py, in 2D: every token (already shrunk to a 1 × 1 token at a cell
of its footprint) is a disk of diameter D in cell space (x: 0 … 64 columns, y: 0 … 64 rows above the floor). Disks
let go at random times in RELEASE, then obey gravity, disk–disk contacts (Jacobi, uniform grid), the floor and the
two side walls. Contact friction is high at first (a "sand castle" of columns) and drops to ~0 (the pile liquefies
and flows out flat, helped by a hydrostatic push down the smoothed surface-height gradient); viscosity rises a
little at the end so the surface calms but keeps moving. Each disk grows from 0.98 to D only after it lets go.
The six images run in one batch, side by side with their own walls (tile i spans x ∈ [i·GAP, i·GAP + 64]).
Positions are saved at the render rate (FPS) from FALL[0] to the end; pos[f, j] = (x, y) of token j, x tile-local.
Usage: python sim2d_v2.py        (uses CUDA_VISIBLE_DEVICES; ~1 min on one A6000)
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

import json
import os

import settle2d_common_v2 as C

SUB, ITERS, K = 4, 8, 8           # substeps per render frame, contact iterations, disks per grid cell
SEED = 7
OVR = json.loads(os.environ.get("SIM_OVR", "{}"))     # parameter sweeps: {"POOL": 3, ...}
for k_, v_ in OVR.items():
    setattr(C, k_, v_)
OUT = os.environ.get("SIM_OUT", C.PHYS)
GAP = C.N + 4                     # tile spacing in cells (no contacts across tiles)


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    rng = np.random.default_rng(SEED)
    toks = [C.Tokens(i) for i in range(C.NI)]
    tile = np.concatenate([np.full(tk.n, i) for i, tk in enumerate(toks)])
    cell = np.concatenate([tk.cell for tk in toks])
    n = len(tile)
    rs, cs = cell[:, 0], cell[:, 1]
    off = torch.tensor(tile * GAP, dtype=torch.float32, device=dev)
    p = torch.tensor(np.c_[tile * GAP + cs + 0.5, (C.N - rs - 1) + 0.5], dtype=torch.float32, device=dev)
    p += torch.randn_like(p) * 0.01
    v = torch.zeros_like(p)
    release = torch.tensor(C.FALL[0] + rng.uniform(0, C.RELEASE, n), dtype=torch.float32, device=dev)
    ti = torch.tensor(tile, device=dev)
    size = torch.tensor(1 + C.POLY * rng.uniform(-1, 1, n), dtype=torch.float32, device=dev)
    D = C.D
    g = torch.tensor([0.0, -C.GRAV], device=dev)
    gx, gy = int(np.ceil(C.NI * GAP / D)) + 2, int(np.ceil((C.N + 4) / D)) + 2
    offs = torch.tensor([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], device=dev)
    nf = int(round((C.FALL[1] - C.FALL[0]) * C.FPS)) + 1
    out = np.empty((nf, n, 2), np.float32)
    dt = 1.0 / (C.FPS * SUB)
    idx = torch.arange(n, device=dev)
    hor = torch.tensor([1.0, 0.0], device=dev)

    def grid_cell(q):
        c = torch.clamp((q / D).long() + 1, min=0)
        c[:, 0].clamp_(max=gx - 1)
        c[:, 1].clamp_(max=gy - 1)
        return c

    for f in range(nf):
        out[f] = (p - torch.stack([off, torch.zeros_like(off)], 1)).cpu().numpy()
        for s in range(SUB):
            t = C.FALL[0] + (f * SUB + s) * dt
            dyn = (t >= release).float()[:, None]
            mu = 1 - np.exp(-C.friction(t) * dt)
            visc = 1 - np.exp(-C.viscosity(t) * dt)
            u = torch.clamp((t - release - 0.15) / 0.6, 0, 1)
            di = torch.where(t >= release, 0.98 + (D * size - 0.98) * u * u * (3 - 2 * u), torch.full_like(u, 0.98))
            v = v + g * dt * dyn
            kp = C.pressure(t)
            if kp > 0:                                                         # hydrostatic push
                col = torch.clamp((p[:, 0] - off).long(), 0, C.N - 1)
                ci = ti * C.N + col
                hm = torch.zeros(C.NI * C.N, device=dev).scatter_reduce(0, ci, p[:, 1] + 0.5, "amax")
                hm_raw = hm
                hm = F.avg_pool1d(F.pad(hm.view(C.NI, 1, C.N), (C.POOL // 2,) * 2, mode="replicate"), C.POOL, stride=1)[:, 0]
                gh = torch.zeros_like(hm)
                gh[:, 1:-1] = (hm[:, 2:] - hm[:, :-2]) / 2
                gh[:, 0], gh[:, -1] = hm[:, 1] - hm[:, 0], hm[:, -1] - hm[:, -2]   # one-sided at the walls
                w = torch.exp(-(hm_raw[ci] - p[:, 1] - 0.5).clamp_min(0) / C.SURF) if C.SURF > 0 else 1.0
                acc = torch.stack([gh.view(-1)[ci] * w, torch.zeros_like(p[:, 0])], 1)
                v = v - kp * acc * dt * dyn
                v = v + torch.randn_like(v) * hor * C.jitter(t) * np.sqrt(dt) * dyn
            v = v * (1 - visc)
            q = p + v * dt * dyn
            cl = grid_cell(q)
            for it in range(ITERS):
                lin = cl[:, 0] * gy + cl[:, 1]
                order = torch.argsort(lin)
                sl = lin[order]
                first = torch.searchsorted(sl, sl, right=False)
                slot = torch.arange(n, device=dev) - first
                table = torch.full((gx * gy, K), -1, dtype=torch.long, device=dev)
                keep = slot < K
                table[sl[keep], slot[keep]] = order[keep]
                nb = cl[:, None, :] + offs[None]                               # (n, 9, 2)
                ok = ((nb >= 0) & (nb < torch.tensor([gx, gy], device=dev))).all(-1)
                nb_lin = torch.where(ok, nb[..., 0] * gy + nb[..., 1], torch.zeros_like(nb[..., 0]))
                cand = table[nb_lin].reshape(n, -1)
                cand = torch.where(ok.repeat_interleave(K, 1), cand, torch.full_like(cand, -1))
                valid = (cand >= 0) & (cand != idx[:, None])
                cj = torch.where(valid, cand, idx[:, None])
                diff = q[:, None, :] - q[cj]
                dist = diff.norm(dim=-1).clamp_min(1e-6)
                dij = (di[:, None] + di[cj]) / 2
                pen = (dij - dist).clamp_min(0) * valid
                wi, wj = dyn[:, 0:1], dyn[cj, 0]
                wsum = (wi + wj).clamp_min(1e-6)
                corr = (pen * wi / wsum)[..., None] * diff / dist[..., None]
                cnt = (pen > 0).sum(1, keepdim=True).clamp_min(1)
                q = q + corr.sum(1) / cnt * 1.2 * dyn
                # floor + the tile's side walls
                q = torch.stack([torch.clamp(q[:, 0], off + di / 2, off + C.N - di / 2),
                                 torch.clamp_min(q[:, 1], di / 2)], 1)
                cl = grid_cell(q)
            v_new = (q - p) / dt * dyn
            touching = ((pen > 0).any(1) | (q[:, 1] <= di / 2 + 1e-3))[:, None].float()
            v = v_new - v_new * hor * mu * touching                          # contact friction (horizontal)
            p = torch.where(dyn.bool(), q, p)
        if f % 60 == 0:
            y = p[:, 1].cpu().numpy()
            lv = [f"{np.percentile(y[tile == i], 99) + 0.5:5.1f}/{toks[i].level:4.1f}" for i in range(C.NI)]
            print(f"t={C.FALL[0] + f / C.FPS:5.2f}  p99 top / level: {' '.join(lv)}  ({time.time() - t0:.0f} s)")
    np.savez_compressed(OUT, pos=out, tile=tile)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
