import numpy as np, importlib, sys
import spacetime_common_v3 as c
def ndc_ext(t):
    proj, view, _ = c.matrices(t)
    pts = [(x, y, z) for x in (-c.W/2, c.W/2) for y in (-c.H/2, c.H/2) for z in (0, c.VOL_D * min(1, max(0, (t-1.8)/2.7)))]
    for n in c.AX_END:
        e = c.axis_end(n, t); d = e - np.array(c.ORIGIN); L = np.linalg.norm(d)
        if L > 1: pts.append(e + d / L * 11.0)       # label beyond the tip
    q = np.array([proj @ view @ np.array([*p, 1.0]) for p in pts]); q = q[:, :2] / q[:, 3:4]
    return q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()
for t in np.arange(0, 10.01, 0.5):
    print(f"{t:4.1f}", " ".join(f"{v:+.2f}" for v in ndc_ext(t)))
