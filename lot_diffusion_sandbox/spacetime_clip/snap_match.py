"""Lattice snap for the rested physics cubes → data/generated/snap_<item>.npz (used by spacetime_common_v6.Tokens).

Slots: 16 / 17 cubes per (x, time) column (LEVEL tall on average, the extra cubes spread evenly). Every rested cube
starts in the column under its centre; an integral min-cost flow between 4-neighbouring columns (scipy linprog,
dual simplex → integral vertex of the network LP) balances the counts, each unit of flow moving the cube nearest the
shared border; finally every column is stacked in height order. So each cube moves only to a nearby column and
keeps its vertical order.
Usage: python snap_match.py
"""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import spacetime_common_v6 as C


def main():
    t0 = time.time()
    end = np.load(C.PHYS)["pos"][-1].astype(float)              # cell-space centres at rest
    n = len(end)
    W, T = C.W, C.T
    ncol = W * T
    base, extra = divmod(n, ncol)
    per = np.full(ncol, base)
    per[(np.arange(extra) * ncol) // extra] += 1
    cx = np.clip(np.floor(end[:, 0]).astype(int), 0, W - 1)
    cz = np.clip(np.floor(end[:, 2]).astype(int), 0, T - 1)
    col = cz * W + cx
    cnt = np.bincount(col, minlength=ncol)
    # directed edges between 4-neighbours; flow ≥ 0; out − in = cnt − per; minimise total flow
    E = []
    for c in range(ncol):
        z, x = divmod(c, W)
        for z2, x2 in ((z, x + 1), (z + 1, x)):
            if x2 < W and z2 < T:
                E += [(c, z2 * W + x2), (z2 * W + x2, c)]
    E = np.array(E)
    m = len(E)
    A = coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[E[:, 0], E[:, 1]], np.r_[np.arange(m), np.arange(m)])),
                   shape=(ncol, m)).tocsr()
    res = linprog(np.ones(m), A_eq=A, b_eq=(cnt - per).astype(float), bounds=(0, None), method="highs-ds")
    flow = np.round(res.x).astype(int)
    assert np.allclose(res.x, flow, atol=1e-6), "fractional flow"
    print(f"transport: {flow.sum()} cube moves over {np.count_nonzero(flow)} edges ({time.time() - t0:.1f} s)",
          flush=True)
    members = [set(np.where(col == c)[0].tolist()) for c in range(ncol)]
    todo = {(int(a), int(b)): int(f) for (a, b), f in zip(E, flow) if f > 0}
    while todo:
        progress = False
        for (a, b), f in list(todo.items()):
            k = min(f, len(members[a]))
            if k == 0:
                continue
            bz, bx = divmod(b, W)
            mem = np.array(sorted(members[a]))
            dd = np.abs(end[mem, 0] - (bx + 0.5)) + np.abs(end[mem, 2] - (bz + 0.5))      # nearest the border
            for i in mem[np.argsort(dd)[:k]]:
                members[a].discard(int(i))
                members[b].add(int(i))
            todo[(a, b)] = f - k
            if todo[(a, b)] == 0:
                del todo[(a, b)]
            progress = True
        assert progress, f"flow stuck with {len(todo)} edges left"
    assert all(len(members[c]) == per[c] for c in range(ncol))
    out_col = np.empty(n, int)
    out_lev = np.empty(n, int)
    for c in range(ncol):
        mem = np.array(sorted(members[c]))
        mem = mem[np.argsort(end[mem, 1])]
        out_col[mem] = c
        out_lev[mem] = np.arange(len(mem))
    h = (n / ncol) / per[out_col]
    centre = np.c_[out_col % W + 0.5, (out_lev + 0.5) * h, out_col // W + 0.5]
    d = np.linalg.norm(centre - end, axis=1)
    print(f"snap move (cells) median {np.median(d):.2f}  p90 {np.percentile(d, 90):.2f}  p99 {np.percentile(d, 99):.2f}"
          f"  max {d.max():.2f}  ({time.time() - t0:.0f} s)")
    np.savez_compressed(C.SNAPF, col=out_col, lev=out_lev, h=h, centre=centre)


if __name__ == "__main__":
    main()
