"""Particle-style settle of the unit token cubes (used by spacetime_common_v5.Tokens) → data/generated/settle_<item>.npz.

1. Fall: every cube (already shrunk to a 1 × 1-token cube at a cell of its footprint) lets go at a random time and
   falls straight down under gravity; it lands on whatever is below it in its (x, time) column at that moment — the
   floor, a settled cube, or a cube that has not let go yet (then it rides down with it). Result: one pile per column,
   the "sand castle".
2. Flow: the piles flatten like a granular / liquid surface. Each step, every column (random order) may pass its
   top cube to a 4-neighbour column (x ± 1 or time ± 1): always if that drops it by ≥ 2 cells (avalanche); by one
   cell only while the source is above, or the target below, the final level band {16, 17}; and toward lower ground
   of a blurred height field, so excess rolls across plateaus toward the holes. Stops when every column holds 16 or
   17 cubes (27,795 cubes over 80 × 21 columns).
Positions are recorded per fall tick (FALL_HZ). The flow is cut at the first step where every column holds 15–18
cubes (the squash does the rest), and each moving cube's lattice path is smoothed in time (Gaussian, SIGMA steps)
so the cubes glide and trickle instead of hopping one cell per step; the renderer interpolates the smoothed paths.
Usage: python settle_sim.py
"""
import os
import time

import numpy as np

import spacetime_common_v5 as C

FALL_HZ = 120
SEED = 7
SIGMA = 10                       # flow-path smoothing (steps)
BAND = 3                         # cut the flow once max − min column height ≤ BAND


def fall(tk, rng):
    """Bottom y (world) of every cube per tick from FALL[0] to FALL[1]."""
    n = tk.n
    y = tk.unit_lo[:, 1] - C.M                        # cell bottom
    col = tk.k * C.W + tk.cell[:, 1]
    order = np.lexsort((y, col))                      # per column, bottom → top
    rank = np.empty(n, int)
    counts = np.bincount(col, minlength=C.T * C.W)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank[order] = np.arange(n) - first[col[order]]
    below = np.full(n, -1)
    has = rank > 0
    below[order[1:]] = np.where(col[order[1:]] == col[order[:-1]], order[:-1], -1)
    release = C.FALL[0] + rng.uniform(0, C.RELEASE, n)
    v = np.zeros(n)
    g = C.GRAV
    dt = 1.0 / FALL_HZ
    ticks = int(round((C.FALL[1] - C.FALL[0]) * FALL_HZ)) + 1
    out = np.empty((ticks, n), np.float32)
    by_rank = [np.where(rank == r)[0] for r in range(rank.max() + 1)]
    for i in range(ticks):
        t = C.FALL[0] + i * dt
        free = t >= release
        for idx in by_rank:                           # bottom-up: supports move first
            f = idx[free[idx]]
            if len(f):
                v[f] += g * dt
                y_new = y[f] - v[f] * dt
                b = below[f]
                support = np.where(b >= 0, y[np.maximum(b, 0)] + 1.0, C.FLOOR)
                hit = y_new <= support
                y_new = np.where(hit, support, y_new)
                v[f] = np.where(hit, np.where(b >= 0, v[np.maximum(b, 0)], 0.0), v[f])
                y[f] = y_new
        out[i] = y
    lev = np.round(y - C.FLOOR).astype(int)
    assert np.allclose(y, C.FLOOR + lev, atol=1e-3), "cubes still falling at the end of FALL"
    assert all(np.array_equal(np.sort(lev[col == c]), np.arange(counts[c])) for c in np.unique(col)), "gaps in piles"
    return out, col, lev


def blur(hg, s=4):
    """Separable box blur (3 passes ≈ Gaussian) of the (T, W) height grid, edge-padded."""
    a = hg.astype(float)
    for _ in range(3):
        for ax in (0, 1):
            p = np.pad(a, [(s, s) if k == ax else (0, 0) for k in range(2)], mode="edge")
            c = np.cumsum(p, axis=ax)
            c = np.concatenate([np.zeros_like(np.take(c, [0], axis=ax)), c], axis=ax)
            a = (np.take(c, range(2 * s + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * s - 1),
                                                                                axis=ax)) / (2 * s + 1)
    return a


def flow(col0, lev0, n, rng, max_steps=4000):
    """Column states (col, level) of every cube per flow step until all columns hold lo or lo + 1 cubes."""
    ncol = C.T * C.W
    lo = n // ncol
    stacks = [[] for _ in range(ncol)]
    for i in np.lexsort((lev0, col0)):
        stacks[col0[i]].append(i)
    col, lev = col0.copy(), lev0.copy()
    hist_c, hist_l = [col.copy()], [lev.copy()]
    nbr = []
    for c in range(ncol):
        t, x = divmod(c, C.W)
        nbr.append([t2 * C.W + x2 for t2, x2 in ((t, x - 1), (t, x + 1), (t - 1, x), (t + 1, x))
                    if 0 <= x2 < C.W and 0 <= t2 < C.T])
    for step in range(max_steps):
        h = np.array([len(s) for s in stacks])
        if h.min() >= lo and h.max() <= lo + 1:
            break
        phi = blur(h.reshape(C.T, C.W)).ravel() + rng.uniform(0, 1e-3, ncol)
        moved = False
        for c in rng.permutation(ncol):
            hc = len(stacks[c])
            if hc == 0:
                continue
            best, bscore = None, None
            for m in rng.permutation(nbr[c]):
                hm = len(stacks[m])
                d = hc - hm
                if d >= 2 or (d == 1 and (hc > lo + 1 or hm < lo)):
                    score = (-d, phi[m])
                    if bscore is None or score < bscore:
                        best, bscore = m, score
            if best is None and hc > lo + 1:              # excess on a plateau: roll toward lower blurred ground
                cand = [m for m in nbr[c] if len(stacks[m]) == hc - 1 or (len(stacks[m]) == hc and phi[m] < phi[c] - 0.02)]
                if cand:
                    best = min(cand, key=lambda m: (len(stacks[m]), phi[m]))
                    if len(stacks[best]) == hc:
                        best = best if rng.random() < 0.5 else None
            if best is not None:
                i = stacks[c].pop()
                stacks[best].append(i)
                col[i], lev[i] = best, len(stacks[best]) - 1
                moved = True
        hist_c.append(col.copy())
        hist_l.append(lev.copy())
        if not moved:
            break
    h = np.array([len(s) for s in stacks])
    print(f"flow: {len(hist_c) - 1} steps, heights {h.min()}–{h.max()}")
    return np.array(hist_c, np.int16), np.array(hist_l, np.int8)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tk = C.Tokens(sim=False)
    fy, col, lev = fall(tk, rng)
    print(f"fall: {len(fy)} ticks, tallest pile {lev.max() + 1} ({time.time() - t0:.1f} s)")
    fc, fl = flow(col, lev, tk.n, rng)
    from scipy.ndimage import gaussian_filter1d
    ncol = C.T * C.W
    spread = np.array([np.ptp(np.bincount(c, minlength=ncol)) for c in fc])
    cut = int(np.argmax(spread <= BAND))
    fc, fl = fc[:cut + 1].astype(int), fl[:cut + 1].astype(int)
    moving = np.where((fc != fc[0]).any(0))[0]
    xyz = np.stack([fc[:, moving] % C.W, fl[:, moving], fc[:, moving] // C.W], -1).astype(np.float32)   # x, level, t
    xyz = gaussian_filter1d(xyz, SIGMA, axis=0, mode="nearest")
    xyz[0], xyz[-1] = np.stack([fc[0, moving] % C.W, fl[0, moving], fc[0, moving] // C.W], -1), \
        np.stack([fc[-1, moving] % C.W, fl[-1, moving], fc[-1, moving] // C.W], -1)
    print(f"flow cut at step {cut} (heights {np.bincount(fc[-1], minlength=ncol).min()}–"
          f"{np.bincount(fc[-1], minlength=ncol).max()}), {len(moving)} moving cubes")
    np.savez_compressed(C.SETTLE, fall_y=fy, fall_col=col, fall_lev=lev, flow_moving=moving, flow_xyz=xyz,
                        final_col=fc[-1], final_lev=fl[-1])
    print(f"wrote {C.SETTLE} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
