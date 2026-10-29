"""Recompute every estimator intermediate of the cont2x rule (ContRule, max_ratio 2) on CPU from a capture.

Input: data/capture/cont_<idx>/{steps.npz, trace.json}. Output: data/method_<idx>.npz with, per coarse step i
(all logs natural; the scene converts to log10):
  n[i], sig[i], lps[i]  (= log P*(sigma_{i+1}; delta), the threshold the test compares against)
  C[i, w], rel[i, w], ok[i, w]       ring measurements on bins 2..w_trust(n) (NaN elsewhere)
  mapped[i, w] = m + (log C - m - off) / lam   (the measurement mapped to the full-res scale)
  est[i, t], sd[i, t]  ML estimate of log P at every target bin t >= n/2 (NaN below)
  u[i] (band measure), ug[i, t] (g at target t), fire[i], next[i]
plus m[w] (prior mean of log P over generated images, w = 0..128), s[w], P_final[w] (this image).
The recomputed est/sd/u/fire are checked against the rule's own trace.

    PYTHONPATH=<repo> python scratch/prep_method.py 577
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/brianchc/prompt-adaptive-speed-cvpr-2027")
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parents[1]
NF, N, NB = 256, 50, 129


def main(idx):
    from experiments.fractional_expansion.online_cont import N_CH, PFILE, REL_SD_MAX, ContRule, band, w_trust
    cap = HERE / "data" / "capture" / f"cont_{idx:06d}"
    S = np.load(cap / "steps.npz")
    trace = json.load(open(cap / "trace.json"))
    p = json.load(open(PFILE))
    rule = ContRule(p, S["sigmas"], max_ratio=2)
    nan = lambda *sh: np.full(sh, np.nan)  # noqa: E731
    out = {k: nan(N, NB) for k in ("C", "rel", "mapped", "est", "sd", "ug")}
    out["ok"] = np.zeros((N, NB), bool)
    out.update(n=S["grid"], sig=S["sig"], lps=rule.lps[1:N + 1], u=nan(N), fire=np.zeros(N, bool), next=np.zeros(N, int))
    m = np.r_[np.nan, np.nan, rule.m]             # rule.m indexes bins 2..128
    sd_prior = np.r_[np.nan, np.nan, rule.s]
    bad = 0
    for i in range(N):
        n, sig = int(S["grid"][i]), float(S["sig"][i])
        if n >= NF:
            continue
        st = S["stats"][i][:, :n // 2 + 1]
        cnt = S[f"cnt_{n}"]
        obs = np.arange(2, w_trust(n) + 1)
        v = rule.vae[n]
        a = 1 - sig
        if a > 0:
            C = st[0, obs] / a
            c_prior = np.exp(m[obs] + v["offset_vs_corpus_m"])
            rel = sig * np.sqrt(st[2, obs] / (2 * cnt[obs] * N_CH / 2)) / (a * c_prior)
            out["C"][i, obs], out["rel"][i, obs] = C, rel
            out["ok"][i, obs] = (rel < REL_SD_MAX) & (C > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["mapped"][i, obs] = m[obs] + (np.log(C) - m[obs] - v["offset_vs_corpus_m"]) / v["lambda"]
        for t in [c // 2 for c in rule.ladder + [NF] if c >= n and c // 2 in rule.ell]:
            e, sd = rule.estimate(n, t, st, sig, cnt)
            out["est"][i, t], out["sd"][i, t] = e, sd
            if np.isfinite(e) and sig < 1:
                out["ug"][i, t] = float(rule.g(np.log((1 - sig) ** 2 * np.exp(e) * (2 * t / NF) ** 2 / sig ** 2)))
        lo, hi = band(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cb = st[0, lo:hi] / (1 - sig)
            out["u"][i] = float(1 - np.clip(st[2, lo:hi] / Cb, 0, 1).mean())
        tr = next(r for r in trace if r["i"] == i)
        out["fire"][i] = tr["fire"]
        out["next"][i] = tr.get("next", 0)
        e0 = out["est"][i, n // 2]
        if not (np.isclose(e0, tr["est"], atol=1e-6, equal_nan=True) and np.isclose(out["u"][i], tr["u"], atol=1e-6)):
            bad += 1
    out.update(m=m, s=sd_prior, z=rule.z, ladder=np.array(rule.ladder), K=S["K"], sizes=S["sizes"])
    spec = HERE / "scratch" / "spec" / f"spectrogram_{idx:06d}.npz"
    if not spec.exists():
        spec = Path("/miele/data/austinchi/fractional_expansion") / f"spectrogram_{idx:06d}.npz"
    out["P_final"] = np.load(spec)["P_final"]
    np.savez(HERE / "data" / f"method_{idx:06d}.npz", **out)
    fires = [i for i in range(N) if out["fire"][i]]
    print(f"steps not matching the trace: {bad}; fires at {fires}; next {[int(out['next'][i]) for i in fires]}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
