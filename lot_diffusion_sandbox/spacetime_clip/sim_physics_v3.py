"""Physics settle of the unit token cubes, v3: position-based dynamics on the GPU (torch) → data/generated/physics3_<item>.npz.

v3: fluid from the first landing — push, jitter and the height map act on landed cubes only (touching the pile or
floor in the previous substep and |v_y| < C.VLAND); friction low from the start (C.LIQUEFY).

v2: runs to the end of the clip (spacetime_common_v7.SIM_END) with light damping, so the surface keeps moving.

Every token (already shrunk to a 1 × 1-token cube at a cell of its footprint) is a particle of diameter D in cell
space (x: 0 … 80 columns, y: 0 … 45 rows above the floor, z: 0 … 21 latent frames). Particles let go at random
times in RELEASE and then obey gravity, particle–particle contacts, the floor and the volume's four side walls.
Contact friction is high at first (the cubes pile into a "sand castle") and drops to ~0 (the pile liquefies, slumps
under its own weight and flows out flat, helped by a hydrostatic push down the surface-height gradient — the
shallow-water pressure term); viscosity rises at the end so the surface comes to rest. Each particle's diameter
grows from 0.98 to D only after it lets go (no squeeze against still-suspended neighbours). Positions are
saved at the render rate (FPS) from FALL0 to SIM_END. Solver: substeps × Jacobi contact iterations on a uniform
grid (cell = D, up to K particles per cell).
Usage: python sim_physics_v3.py      (uses CUDA_VISIBLE_DEVICES; ~1 min on one A6000)
"""
import os
import time

import numpy as np
import torch

import spacetime_common_v8 as C

SUB, ITERS, K = 4, 8, 8           # substeps per render frame, contact iterations, particles per grid cell
SEED = 7


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    rng = np.random.default_rng(SEED)
    tk = C.Tokens(sim=False)
    n = tk.n
    # start: centre of the unit cube's cell, in cell space
    rs, cs = tk.cell[:, 0], tk.cell[:, 1]
    p = torch.tensor(np.c_[cs + 0.5, (C.H - rs - 1) + 0.5, tk.k + 0.5], dtype=torch.float32, device=dev)
    p += torch.randn_like(p) * 0.01                           # break the perfect lattice
    v = torch.zeros_like(p)
    release = torch.tensor(C.FALL[0] + rng.uniform(0, C.RELEASE, n), dtype=torch.float32, device=dev)
    lo_b = torch.tensor([0.0, 0.0, 0.0], device=dev)
    hi_b = torch.tensor([float(C.W), 1e9, float(C.T)], device=dev)
    D = C.D
    g = torch.tensor([0.0, -C.GRAV, 0.0], device=dev)
    gx, gy, gz = int(np.ceil(C.W / D)) + 2, int(np.ceil((C.H + 4) / D)) + 2, int(np.ceil(C.T / D)) + 2
    offs = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], device=dev)
    nf = int(round((C.SIM_END - C.FALL[0]) * C.FPS)) + 1
    out = np.empty((nf, n, 3), np.float32)
    dt = 1.0 / (C.FPS * SUB)
    idx = torch.arange(n, device=dev)
    touching = torch.zeros((n, 1), device=dev)
    for f in range(nf):
        out[f] = p.cpu().numpy()
        for s in range(SUB):
            t = C.FALL[0] + (f * SUB + s) * dt
            dyn = (t >= release).float()[:, None]
            mu = 1 - np.exp(-C.friction(t) * dt)
            visc = 1 - np.exp(-C.viscosity(t) * dt)
            di = 0.98 + (D - 0.98) * torch.clamp((t - release - 0.15) / 0.6, 0, 1) ** 2 * \
                (3 - 2 * torch.clamp((t - release - 0.15) / 0.6, 0, 1))
            di = torch.where(t >= release, di, torch.full_like(di, 0.98))
            v = v + g * dt * dyn
            kp = C.pressure(t)
            landed = dyn * touching * (v[:, 1:2].abs() < C.VLAND).float()
            if kp > 0:                                                         # hydrostatic push
                ci = torch.clamp(p[:, 0].long(), 0, C.W - 1) * C.T + torch.clamp(p[:, 2].long(), 0, C.T - 1)
                hy = torch.where(landed[:, 0] > 0, p[:, 1] + 0.5, torch.zeros_like(p[:, 1]))
                hm = torch.zeros(C.W * C.T, device=dev).scatter_reduce(0, ci, hy, "amax")
                hm = hm.view(1, 1, C.W, C.T)
                hm = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(hm, (2, 2, 2, 2), mode="replicate"),
                                                    5, stride=1)[0, 0]
                gxh = torch.zeros_like(hm)
                gzh = torch.zeros_like(hm)
                gxh[1:-1] = (hm[2:] - hm[:-2]) / 2
                gzh[:, 1:-1] = (hm[:, 2:] - hm[:, :-2]) / 2
                gxh[0], gxh[-1] = hm[1] - hm[0], hm[-1] - hm[-2]              # one-sided at the walls: piles
                gzh[:, 0], gzh[:, -1] = hm[:, 1] - hm[:, 0], hm[:, -1] - hm[:, -2]  # never get stuck in corners
                acc = torch.stack([gxh.view(-1)[ci], torch.zeros_like(p[:, 0]), gzh.view(-1)[ci]], 1)
                v = v - kp * acc * dt * landed
                jit = torch.randn_like(v) * torch.tensor([1.0, 0.0, 1.0], device=dev)   # topple perfect stacks
                v = v + jit * C.JITTER * (kp / 90.0) * np.sqrt(dt) * landed
            v = v * (1 - visc)
            q = p + v * dt * dyn
            # uniform grid: cell → up to K particles
            cell = torch.clamp((q / D).long() + 1, min=0)
            cell[:, 0].clamp_(max=gx - 1)
            cell[:, 1].clamp_(max=gy - 1)
            cell[:, 2].clamp_(max=gz - 1)
            for it in range(ITERS):
                lin = (cell[:, 0] * gy + cell[:, 1]) * gz + cell[:, 2]
                order = torch.argsort(lin)
                sl = lin[order]
                first = torch.searchsorted(sl, sl, right=False)
                slot = torch.arange(n, device=dev) - first
                table = torch.full((gx * gy * gz, K), -1, dtype=torch.long, device=dev)
                keep = slot < K
                table[sl[keep], slot[keep]] = order[keep]
                nb_cell = cell[:, None, :] + offs[None]                       # (n, 27, 3)
                ok = ((nb_cell >= 0) & (nb_cell < torch.tensor([gx, gy, gz], device=dev))).all(-1)
                nb_lin = (nb_cell[..., 0] * gy + nb_cell[..., 1]) * gz + nb_cell[..., 2]
                nb_lin = torch.where(ok, nb_lin, torch.zeros_like(nb_lin))
                cand = table[nb_lin].reshape(n, -1)                            # (n, 27·K)
                cand = torch.where(ok.repeat_interleave(K, 1), cand, torch.full_like(cand, -1))
                valid = (cand >= 0) & (cand != idx[:, None])
                cj = torch.where(valid, cand, idx[:, None])
                diff = q[:, None, :] - q[cj]                                   # (n, M, 3)
                dist = diff.norm(dim=-1).clamp_min(1e-6)
                dij = (di[:, None] + di[cj]) / 2
                pen = (dij - dist).clamp_min(0) * valid
                wi, wj = dyn[:, 0:1], dyn[cj, 0]
                wsum = (wi + wj).clamp_min(1e-6)
                corr = (pen * wi / wsum)[..., None] * diff / dist[..., None]
                cnt = (pen > 0).sum(1, keepdim=True).clamp_min(1)
                q = q + corr.sum(1) / cnt * 1.2 * dyn
                # floor + side walls
                q = torch.maximum(q, lo_b + di[:, None] / 2)
                q = torch.minimum(q, hi_b - di[:, None] / 2)
                cell = torch.clamp((q / D).long() + 1, min=0)
                cell[:, 0].clamp_(max=gx - 1)
                cell[:, 1].clamp_(max=gy - 1)
                cell[:, 2].clamp_(max=gz - 1)
            v_new = (q - p) / dt * dyn
            touching = ((pen > 0).any(1) | (q[:, 1] <= di / 2 + 1e-3))[:, None].float()
            tang = v_new * torch.tensor([1.0, 0.0, 1.0], device=dev)
            v = v_new - tang * mu * touching                                  # contact friction (horizontal)
            p = torch.where(dyn.bool(), q, p)
        if f % 30 == 0:
            ke = (v.norm(dim=1) ** 2).mean().item()
            print(f"t={C.FALL[0] + f / C.FPS:5.2f}  top={p[:, 1].max().item():5.1f}  "
                  f"p99={torch.quantile(p[:, 1], 0.99).item():5.1f}  "
                  f"mean_h={p[:, 1].mean().item():5.2f}  ke={ke:8.3f}  ({time.time() - t0:.0f} s)")
    np.savez_compressed(C.PHYS, pos=out)
    print(f"wrote {C.PHYS} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
