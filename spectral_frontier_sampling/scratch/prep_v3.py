#!/usr/bin/env python
"""v3 data (CPU): SNR maps, frontiers, staircases, compute, speedups and image copies for animation_sfs_v3.py.

Inputs (all on the Rhoda pod):
  data/capture_v3/<idx>/            capture_v3.py: x0_spectra.npz (final spectrum = step 49 x0hat power), rule.json
  $WD_BULK/figure_prompts/<set>/4klsdb/2k/<arm>/<idx>.{png,json}   release runs (times: total_s, B200, plain runs)
  $WD_BULK/figure_prompts/fig3_baselines/fig3/2k/<arm>/<idx>.*     Fig. 3 runs (one GPU, all arms)
  flops_per_step.json (research repo): the paper's per-step FLOP model F(n) = a T + b T^2, T = (n/2)^2 + 512
Outputs: data/sfs_v3.npz, data/sfs_v3.json, data/img/*.jpg

  /home/bchao/miniforge3/envs/anim/bin/python scratch/prep_v3.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CAP = DATA / "capture_v3"
IMG = DATA / "img"
WD = Path("/home/bchao/wavelet-diffusion")
FP = Path("/home/bchao/bulk/wd/figure_prompts")
FIG3 = FP / "fig3_baselines/fig3/2k"
NF, N, DELTA = 256, 50, 0.01
SPEED3 = {"K": [27, 36], "sizes": [64, 128, 256]}
SPEED2 = {"K": [36], "sizes": [128, 256]}
PROMPTS = [14, 50, 12, 35, 58, 99, 54]


def flop_fit():
    F = json.loads((WD / "experiments/klein_innovation_aesthetic4k/results/flops_per_step.json").read_text())["per_step_flops"]
    n = np.array([int(k) for k in F], float)
    T = (n / 2) ** 2 + 512
    coef, *_ = np.linalg.lstsq(np.stack([T, T ** 2], 1), np.array([F[k] for k in F], float), rcond=None)
    return lambda m: coef[0] * ((np.asarray(m, float) / 2) ** 2 + 512) + coef[1] * ((np.asarray(m, float) / 2) ** 2 + 512) ** 2


def sides(lg, start=64, cont=True):
    """grid side per step from K / sizes (cont: sizes[j] starts at K[j]; fixed: sizes[0] is the start)."""
    sz = lg["sizes"] if cont else lg["sizes"][1:]
    d, h, out = dict(zip(lg["K"], sz)), (start if cont else lg["sizes"][0]), []
    for i in range(N):
        h = d.get(i, h)
        out.append(h)
    out[-1] = NF
    return np.array(out)


def save_jpg(src, dst, side):
    im = Image.open(src).convert("RGB")
    if side and im.size[0] > side:
        im = im.resize((side, round(side * im.size[1] / im.size[0])), Image.LANCZOS)
    im.save(dst, quality=92)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    fl = flop_fit()
    out, meta = {}, {"delta": DELTA, "prompts": {}, "sets": {}}
    # compute axis: y(f) = F(2 f) / F(NF), f = radial bin; area under a ceiling = compute
    fgrid = np.linspace(0, NF // 2 + 1, 2001)
    out["axis_f"], out["axis_y"] = fgrid, fl(2 * fgrid) / fl(NF)
    names = {int(r["prompt_index"]): r["name"] for r in csv.DictReader(open(WD / "experiments/figure_prompts/cinematic.csv"))}
    for idx in PROMPTS:
        rj = json.loads((CAP / str(idx) / "rule.json").read_text())
        z = np.load(CAP / str(idx) / "x0_spectra.npz")
        sig = np.asarray(rj["sigmas"], float)[:N]
        P = z[f"P_{N - 1:02d}_n{NF}"]
        with np.errstate(divide="ignore"):
            lsnr = np.log10(np.maximum((1 - sig[:, None]) ** 2 * P[None] / sig[:, None] ** 2, 1e-12))   # [steps, bins 0..128]
        front = np.array([max([w for w in range(1, NF // 2 + 1) if lsnr[i, w] >= np.log10(DELTA)] or [0]) for i in range(N)])
        st = json.loads((FP / f"cinematic/4klsdb/2k/cont_mode/{idx}.json").read_text())
        fr = json.loads((FP / f"cinematic/4klsdb/2k/fullres/{idx}.json").read_text())
        ours = sides(st)
        s3, s2 = sides(SPEED3, cont=False), sides(SPEED2, cont=False)
        comp = {k: float(fl(v).sum() / (N * fl(NF))) for k, v in (("ours", ours), ("speed3", s3), ("speed2", s2))}
        out[f"lsnr_{idx}"], out[f"P_{idx}"], out[f"front_{idx}"] = lsnr, P, front
        out[f"ours_{idx}"], out["speed3"], out["speed2"], out["sig"] = ours, s3, s2, sig
        out[f"lps_{idx}"] = np.asarray(rj["lps"], float)
        tr = rj["trace"]
        out[f"tr_{idx}"] = np.array([[t["i"], t["n"], t["est"], t["sd"], t["u"], float(t["fire"])] for t in tr], float)
        meta["prompts"][idx] = dict(name=names[idx], K=st["K"], sizes=st["sizes"], t_ours=st["total_s"], t_full=fr["total_s"],
                                    speedup=fr["total_s"] / st["total_s"], compute=comp, match=rj["match"],
                                    first_grow=st["K"][0])
        save_jpg(CAP / str(idx) / "final.png", IMG / f"ours_{idx}.jpg", 768)
        save_jpg(FP / f"cinematic/4klsdb/2k/fullres/{idx}.png", IMG / f"full_{idx}.jpg", 768)
    # per-prompt speedups: cinematic (cont_mode) and the three style sets (cont_mode_hybk40, the paper's Fig. 4 arm)
    for s, arm in (("cinematic", "cont_mode"), ("lineart", "cont_mode_hybk40"), ("anime", "cont_mode_hybk40"),
                   ("pexels", "cont_mode_hybk40")):
        rows = []
        for f in sorted((FP / f"{s}/4klsdb/2k/{arm}").glob("*.json"), key=lambda p: int(p.stem)):
            fr = FP / f"{s}/4klsdb/2k/fullres/{f.stem}.json"
            if fr.exists():
                a, b = json.loads(f.read_text()), json.loads(fr.read_text())
                rows.append(dict(idx=int(f.stem), t=a["total_s"], t_full=b["total_s"], speedup=b["total_s"] / a["total_s"],
                                 first_grow=a["K"][0] if a["K"] else None))
        meta["sets"][s] = dict(arm=arm, rows=rows)
        if s == "cinematic":
            for r in rows:
                save_jpg(FP / f"{s}/4klsdb/2k/{arm}/{r['idx']}.png", IMG / f"thumb_{r['idx']}.jpg", 256)
    for r in csv.DictReader(open(WD / "experiments/figure_prompts/fig4_picks.csv")):
        s, i = r["set"], r["prompt_index"]
        save_jpg(FP / f"{s}/4klsdb/2k/cont_mode_hybk40/{i}.png", IMG / f"{s}_ours_{i}.jpg", 768)
        save_jpg(FP / f"{s}/4klsdb/2k/fullres/{i}.png", IMG / f"{s}_full_{i}.jpg", 768)
    meta["fig3"] = {}
    for i in (40, 66, 8, 79):
        meta["fig3"][i] = {}
        for arm in ("fullres", "speed2", "speed3", "cont_mode"):
            j = json.loads((FIG3 / arm / f"{i}.json").read_text())
            meta["fig3"][i][arm] = j["total_s"]
            save_jpg(FIG3 / arm / f"{i}.png", IMG / f"fig3_{arm}_{i}.jpg", 768)
    # 20,000 4KLSDB prompts (2K, B200): time per decile of ring-32 power (the first grid's edge), finding 2026-10-25
    pi = json.loads(Path("/home/bchao/bulk/wd/cont_mode_eval/runs/final/analysis/per_image_4klsdb_2k_cont_mode.json").read_text())
    lnP = np.array(pi["lnP"])[:, list(pi["rings"]).index(32)]
    t = np.array(pi["total_s"])
    q = np.quantile(lnP, np.linspace(0, 1, 11))
    out["dec_t"] = np.array([t[(lnP >= q[k]) & (lnP <= q[k + 1])].mean() for k in range(10)])
    out["dec_lnP"] = np.array([lnP[(lnP >= q[k]) & (lnP <= q[k + 1])].mean() for k in range(10)])
    meta["deciles"] = dict(n=len(t), full_s=70.8, note="finding 2026-10-25_cont-mode-4klsdb-20k: 6.45x lowest 10%, 3.95x highest")
    np.savez(DATA / "sfs_v3.npz", **out)
    (DATA / "sfs_v3.json").write_text(json.dumps(meta, indent=1))
    for idx, m in meta["prompts"].items():
        print(idx, m["name"], f"{m['speedup']:.2f}x", "first", m["first_grow"], "compute", {k: round(v, 3) for k, v in m["compute"].items()},
              "match", m["match"])
    for s, d in meta["sets"].items():
        sp = sorted(r["speedup"] for r in d["rows"])
        print(s, d["arm"], len(sp), f"{sp[0]:.2f} .. {np.median(sp):.2f} .. {sp[-1]:.2f}")


if __name__ == "__main__":
    main()
