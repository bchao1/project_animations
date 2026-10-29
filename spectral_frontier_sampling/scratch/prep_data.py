"""Export everything the manim scenes need into data/sfs_data.npz (CPU, seconds).

Sources (research repo /home/brianchc/prompt-adaptive-speed-cvpr-2027):
- per-step x0hat radial power of the full-res run: spectrogram_<idx>.npz (Austin's, or scratch/spec/ for 577)
- schedules: /miele/data/austinchi/fractional_expansion/online/{speed3,cont2x,innov3}/<idx>.json
- SPEED 2-stage at 2K: 128 -> 256 at K0 = 36 (experiments/powerlaw_transition/params/klein_2k.json, speed_K0)
- FLOPs per step: envelope.flop_model() (fit to flops_per_step.json)
- rule params (prior m/s/ell, VAE lines, g, ladder): params/klein_2k_cont.json
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/brianchc/prompt-adaptive-speed-cvpr-2027")
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parents[1]
ONLINE = Path("/miele/data/austinchi/fractional_expansion/online")
SPECDIRS = [Path("/miele/data/austinchi/fractional_expansion"), HERE / "scratch" / "spec"]
NF, N = 256, 50
PROMPTS = [2580, 2348, 1026, 577, 548]


def ceiling(K, sizes, start):
    d, h, row = dict(zip(K, sizes)), start, []
    for i in range(N):
        h = d.get(i, h)
        row.append(h // 2)
    return np.array(row)


def main():
    import experiments.fractional_expansion.envelope as E
    fl, _ = E.flop_model()
    out = {}
    f = np.arange(0, NF // 2 + 1, dtype=float)
    out["compute_of_bin"] = fl(2 * f) / fl(NF)         # y of bin f on the compute axis (grid 2f, full step = 1)
    for idx in PROMPTS:
        src = next((d / f"spectrogram_{idx:06d}.npz" for d in SPECDIRS if (d / f"spectrogram_{idx:06d}.npz").exists()),
                   None)
        if src is None:
            print("missing spectrogram", idx)
            continue
        z = np.load(src)
        k = f"p{idx}_"
        out[k + "P_x0"] = z["P_x0"]
        out[k + "P_final"] = z["P_final"]
        out[k + "sigmas"] = z["sigmas"]
        s3 = json.load(open(ONLINE / "speed3" / f"{idx:06d}.json"))
        c2 = json.load(open(ONLINE / "cont2x" / f"{idx:06d}.json"))
        out[k + "speed2"] = ceiling([36], [256], 128)
        out[k + "speed3"] = ceiling(s3["K"], s3["sizes"][1:], 64)
        out[k + "cont2x"] = ceiling(c2["K"], c2["sizes"], 64)
        for arm in ("speed2", "speed3", "cont2x"):
            c = out[k + arm]
            out[k + arm + "_x"] = N * fl(NF) / sum(fl(2 * b) for b in c)   # FLOPs speedup vs full res
        out[k + "cont2x_K"] = np.array(c2["K"])
        out[k + "cont2x_sizes"] = np.array(c2["sizes"])
        print(idx, {a: round(float(out[k + a + "_x"]), 2) for a in ("speed2", "speed3", "cont2x")},
              "K", c2["K"][:3], "...")
    np.savez(HERE / "data" / "sfs_data.npz", **out)
    print("wrote", HERE / "data" / "sfs_data.npz")


if __name__ == "__main__":
    main()
