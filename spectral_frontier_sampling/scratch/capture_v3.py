#!/usr/bin/env python
"""v3 data: rerun the release sampler (cont_mode, 2K) on picked cinematic prompts with previews, and keep the rule's
per-step record plus the inputs of its test, for the S05/S06 online-estimation scenes.

Imports experiments/cont_mode_benchmark/run.py as a module; nothing in the research repo changes.
Per prompt writes data/capture_v3/<idx>/:
  x0_<i>_n<grid>.png, x0_spectra.npz      (run.py --previews: x0hat and its radial power at every step)
  rule.json   trace (i, n, est, sd, u, fire, next), lps (ln P*(sigma_i; delta) over steps), sigmas, K, sizes,
              stored_K / stored_sizes (from the figure-prompts run) and match (schedule identical)
  rings.npz   per coarse step: stats [3, n/2+1] (Re<x, x0hat>, |x|^2, |x0hat|^2 per ring), sig, cnt
Timing is not valid here (the preview decodes); use the stored figure-prompts JSONs for times.

  source ~/wavelet-diffusion/configs/rhoda/env.sh
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY scratch/capture_v3.py 14 50 12 35 58 99 54
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

WD = Path("/home/bchao/wavelet-diffusion")
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "data" / "capture_v3"
STORED = Path("/home/bchao/bulk/wd/figure_prompts/cinematic/4klsdb/2k/cont_mode")
sys.path.insert(0, str(WD))

spec = importlib.util.spec_from_file_location("cmb_run", WD / "experiments/cont_mode_benchmark/run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from experiments.fractional_expansion.online_cont import ContRule  # noqa: E402

REC = {}
_observe = ContRule.observe


def observe(self, n, i, stats, sig, cnt):
    REC.setdefault("steps", []).append({"i": i, "n": n, "stats": np.asarray(stats, float).copy(), "sig": float(sig),
                                        "cnt": np.asarray(cnt, float).copy()})
    REC["lps"] = np.asarray(self.lps, float)
    REC["rule"] = self
    return _observe(self, n, i, stats, sig, cnt)


ContRule.observe = observe


def main(ids):
    rows = {int(r["prompt_index"]): r for r in csv.DictReader(open(WD / "experiments/figure_prompts/cinematic.csv"))}
    tr, vae, sched = R.load_models("cuda")
    bn = R.bn_stats(vae, "cuda")
    pipe = R.load_text_encoder()
    for idx in ids:
        r = rows[idx]
        d = OUT / str(idx)
        if (d / "rule.json").exists():
            print(idx, "done, skip", flush=True)
            continue
        REC.clear()
        img, log = R.run_arm("cont_mode", "2k", r["prompt"], int(r["seed"]), tr, vae, sched, bn, pipe, previews=d)
        R.save_png(img, d / "final.png")
        st = json.loads((STORED / f"{idx}.json").read_text())
        steps = REC["steps"]
        np.savez(d / "rings.npz", i=np.array([s["i"] for s in steps]), n=np.array([s["n"] for s in steps]),
                 sig=np.array([s["sig"] for s in steps]),
                 **{f"stats_{s['i']:02d}": s["stats"] for s in steps}, **{f"cnt_{s['i']:02d}": s["cnt"] for s in steps})
        trace = list(REC["rule"].trace)
        out = dict(idx=idx, name=r["name"], seed=int(r["seed"]), K=log["K"], sizes=log["sizes"], stored_K=st["K"],
                   stored_sizes=st["sizes"], match=(log["K"] == st["K"] and log["sizes"] == st["sizes"]),
                   lps=REC["lps"].tolist(), sigmas=sched.sigmas.detach().cpu().numpy().tolist(), trace=trace)
        (d / "rule.json").write_text(json.dumps(out, default=float))
        print(f"{idx} {r['name']}: K={log['K']} match={out['match']} trace={len(trace)}", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
