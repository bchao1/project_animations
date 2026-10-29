"""Animation data capture for one klein 2K prompt (full-res or cont2x run).

Same sampling code path as experiments/fractional_expansion/spectrogram.py (run / run_cont). Adds, per step:
decoded x_t and x0hat images (512 px PNGs), and for cont2x the raw ring stats the rule observes, so every
estimator intermediate can be recomputed on CPU. Output: data/capture/<src>_<idx>/{steps.npz, xt_ii.png, x0_ii.png}.

    CUDA_VISIBLE_DEVICES=<free> HF_HUB_OFFLINE=1 PYTHONPATH=<repo> python capture.py --prompt 577 --src cont
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/brianchc/prompt-adaptive-speed-cvpr-2027")
sys.path.insert(0, str(REPO))
OUTROOT = Path(__file__).resolve().parents[1] / "data" / "capture"
PROMPTS = REPO / "experiments/mlvae_aesthetic4k_2k/results/prompts_eval_all.csv"
ONLINE = Path("/miele/data/austinchi/fractional_expansion/online")
NF = 256
PX = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--src", choices=("full", "cont"), required=True)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from experiments.fractional_expansion.online_cont import PFILE, ContRule
    from experiments.fractional_expansion.run_oracle import dct2, idct2, upsample
    from experiments.fractional_expansion.spectrogram import radial_cross
    from experiments.klein_innovation_params.common_klein import (N_STEPS, bn_stats, decode, initial_noise, load_emb,
                                                                   load_models, make_forward, require_free,
                                                                   setup_scheduler)
    from experiments.mlvae_aesthetic4k_2k.common2k import ring_index, ring_stats
    from experiments.speed_table1_oracle_deciles.trajectory import radial_power_exact
    from inference_progressive import flux2klein_pack, flux2klein_unpack, reset_scheduler_state

    idx = a.prompt
    out = OUTROOT / f"{a.src}_{idx:06d}"
    out.mkdir(parents=True, exist_ok=True)
    r = next(r for r in csv.DictReader(open(PROMPTS)) if int(r["prompt_index"]) == idx)
    require_free(30)
    dev = "cuda"
    tr, vae, sched = load_models(dev)
    bn = bn_stats(vae, dev)
    fwd = make_forward(tr, sched, load_emb(f"ev_{idx:06d}", dev), load_emb("_neg", dev))

    def save(lat, path):
        img = F.interpolate(decode(vae, lat, bn), size=(PX, PX), mode="area")
        Image.fromarray(((img[0].permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()).save(path)

    rec = {k: [] for k in ("grid", "sig", "P_x0", "P_xt", "X", "stats")}
    K, sizes = [], []
    with torch.inference_mode():
        setup_scheduler(sched, N_STEPS, NF, dev)
        sig0 = sched.sigmas.detach().cpu().numpy().astype(np.float64)
        rule = ContRule(json.load(open(PFILE)), sig0, max_ratio=2) if a.src == "cont" else None
        if a.src == "cont":
            z_dct = dct2(initial_noise(int(r["seed"]), NF).to(dev).float())
            h = rule.ladder[0]
            x = flux2klein_pack(idct2(z_dct[..., :h, :h]).to(torch.bfloat16), h, h)
        else:
            h = NF
            x = flux2klein_pack(initial_noise(int(r["seed"]), NF).to(dev, torch.bfloat16), NF, NF)
        rings = {}
        for i in range(N_STEPS):
            if rule is not None:
                h2 = rule.next_grid(i, h)
                if h2 != h:
                    xf, t_eff = upsample(flux2klein_unpack(x, h, h).float(), float(sched.sigmas[i]), h2, z_dct)
                    sched.sigmas[i] = t_eff
                    sched.timesteps[i] = t_eff * 1000
                    reset_scheduler_state(sched, i)
                    h = h2
                    K.append(i), sizes.append(h)
                    x = flux2klein_pack(xf.to(torch.bfloat16), h, h)
            v = fwd(x, i, h)
            sig = float(sched.sigmas[i])
            xt = flux2klein_unpack(x, h, h).float()
            x0 = flux2klein_unpack(x.float() - sig * v.float(), h, h).float()
            pad = lambda q: np.r_[q, np.full(NF // 2 + 1 - len(q), np.nan)]  # noqa: E731
            rec["grid"].append(h), rec["sig"].append(sig)
            rec["P_x0"].append(pad(radial_power_exact(x0, dev)))
            rec["P_xt"].append(pad(radial_power_exact(xt, dev)))
            rec["X"].append(pad(radial_cross(xt, x0, dev)))
            st = np.full((3, NF // 2 + 1), np.nan)
            if h < NF:
                if h not in rings:
                    rings[h] = ring_index(h, dev)
                    rings[h] = (rings[h], rings[h][1].cpu().numpy())
                s = ring_stats(xt[0], x0[0], rings[h][0])
                st[:, :s.shape[1]] = s
                if rule is not None:
                    rule.observe(h, i, s, sig, rings[h][1])
            rec["stats"].append(st)
            save(xt, out / f"xt_{i:02d}.png")
            save(x0, out / f"x0_{i:02d}.png")
            x = sched.step(v, sched.timesteps[i], x, return_dict=False)[0]
        lat = flux2klein_unpack(x, NF, NF).float()
        save(lat, out / "final.png")
        F2 = torch.fft.fftshift(torch.fft.fft2(lat[0], norm="ortho"), dim=(-2, -1))
        P2D = (F2.abs() ** 2).mean(0).cpu().numpy().astype(np.float32)   # channel-mean 2D power, centered
    cnts = {n: rings[n][1] for n in rings}
    np.savez(out / "steps.npz", sigmas=sig0, P2D_final=P2D, K=np.array(K), sizes=np.array(sizes),
             **{k: np.array(v) for k, v in rec.items()}, **{f"cnt_{n}": c for n, c in cnts.items()})
    if rule is not None:
        json.dump(rule.trace, open(out / "trace.json", "w"), default=float)
        ref = json.load(open(ONLINE / "cont2x" / f"{idx:06d}.json"))
        print("schedule reproduced:", K == ref["K"] and sizes == ref["sizes"], K, sizes)
    print("done", out)


if __name__ == "__main__":
    main()
