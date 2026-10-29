"""Pack the 100-turn bottle session (LoT k = 4 arm) into data/generated/session.npz + turns.json.

Source (read-only): ~/LoT-george-branch/results/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn
  outputs/k4/samples/<sid>/canonical_k4_reflot/turn_NN/{output.png, gen.png, grid.pt, meta.json}
Per turn t ≥ 1: leaves (r0, r1, c0, c1) on the 64 × 64 grid, edit region, mean colour of every leaf over the
INPUT image (turn t − 1 output = the in-context reference) and over the decode gen.png, DiT seconds.
Also the base-model timings (50 steps, CFG 4) from results/2026-10-07_152323_ref-edit-multiturn-9b-bottles.

    ~/miniconda3/envs/imp_tome/bin/python prep_data.py
"""
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

RES = "/home/brianchc/LoT-george-branch/results"
RUN = f"{RES}/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn/outputs"
SID = "e0afb251d16658998c72d6cb200a4412"
K4 = f"{RUN}/k4/samples/{SID}/canonical_k4_reflot"
FULL = f"{RUN}/full/samples/{SID}/canonical_k1_refdense"
BASE = f"{RES}/2026-10-07_152323_ref-edit-multiturn-9b-bottles/outputs/recipe9b/samples/{SID}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "generated")
IMG_PX = 1080            # stored image size (image panel is 540 px at 1080p, rendered at 2×)


def mean_cols(im, leaves):
    a = np.asarray(im, dtype=np.float32)
    return np.array([a[r0 * 16:r1 * 16, c0 * 16:c1 * 16].reshape(-1, 3).mean(0) for r0, r1, c0, c1 in leaves])


def secs(d):
    ms = [json.load(open(f)) for f in glob.glob(d + "/turn_*/meta.json")]
    ms = [m for m in ms if m["turn"] > 0]
    return float(np.mean([m["sample_seconds"] for m in ms])), ms[0]["n_tokens"], ms[0].get("n_ref_tokens")


def main():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    turns, leaves_all, ctx_cols, gen_cols, regions = [], [], [], [], []
    prev = Image.open(f"{K4}/turn_00/output.png").convert("RGB")
    prev.resize((IMG_PX, IMG_PX), Image.LANCZOS).save(f"{OUT}/img/out_000.png")
    for t in range(1, 101):
        d = f"{K4}/turn_{t:02d}"
        m = json.load(open(f"{d}/meta.json"))
        g = torch.load(f"{d}/grid.pt", map_location="cpu", weights_only=False)
        lv = np.array(g["leaves"], dtype=np.int16)
        assert len(lv) == m["n_tokens"]
        gen_im = Image.open(f"{d}/gen.png").convert("RGB")
        out_im = Image.open(f"{d}/output.png").convert("RGB")
        leaves_all.append(lv)
        ctx_cols.append(mean_cols(prev, lv))
        gen_cols.append(mean_cols(gen_im, lv))
        regions.append(g["edit_region"].numpy())
        out_im.resize((IMG_PX, IMG_PX), Image.LANCZOS).save(f"{OUT}/img/out_{t:03d}.png")
        gen_im.resize((IMG_PX, IMG_PX), Image.LANCZOS).save(f"{OUT}/img/gen_{t:03d}.png")
        turns.append(dict(turn=t, instruction=m["prompt"], edit=m["edit"], n_tokens=m["n_tokens"],
                          n_ref=m["n_ref_tokens"], extents=m["extents"], seconds=m["sample_seconds"]))
        prev = out_im
    n = np.array([len(l) for l in leaves_all])
    off = np.concatenate([[0], np.cumsum(n)])
    np.savez_compressed(f"{OUT}/session.npz", leaves=np.concatenate(leaves_all), ctx=np.concatenate(ctx_cols),
                        gen=np.concatenate(gen_cols), off=off, region=np.stack(regions))
    base = {a: secs(f"{BASE}/{a}") for a in ("reencode_k1_refdense_noblend", "canonical_k4_reflot")}
    timing = dict(k4=secs(K4), full=secs(FULL), base_full=base["reencode_k1_refdense_noblend"],
                  base_k4=base["canonical_k4_reflot"])
    json.dump(dict(turn0_prompt=json.load(open(f"{K4}/turn_00/meta.json"))["prompt"], turns=turns, timing=timing),
              open(f"{OUT}/turns.json", "w"), indent=1, ensure_ascii=False)
    print(timing, "total k4 s", sum(t["seconds"] for t in turns))


if __name__ == "__main__":
    main()
