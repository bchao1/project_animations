"""Pack the context-size sweep (k = 1, 2, 4, 8) and drift run into data/results.json + data/leaves_turn1.npz.

Source (read-only): ~/LoT-george-branch/results/2026-10-08_173831_ref-edit-9b-distilled-bottles-100turn-ctx-sweep
  outputs/compare/per_turn.csv (metrics per arm and turn), outputs/samples/<sid>/<arm>/turn_NN/{output.png, grid.pt}
  re-encode loop: results/2026-10-08_084925_.../outputs/full_reenc/samples/<sid>/reencode_k1_refdense_noblend

    ~/miniconda3/envs/imp_tome/bin/python prep_data.py
"""
import csv
import json
import os

import numpy as np
import torch

RES = "/home/brianchc/LoT-george-branch/results"
SWEEP = f"{RES}/2026-10-08_173831_ref-edit-9b-distilled-bottles-100turn-ctx-sweep/outputs"
SID = "e0afb251d16658998c72d6cb200a4412"
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"k=1": f"{SWEEP}/samples/{SID}/canonical_k1_refdense",
        "k=2": f"{SWEEP}/samples/{SID}/canonical_k2_reflot",
        "k=4": f"{SWEEP}/samples/{SID}/canonical_k4_reflot",
        "k=8": f"{SWEEP}/samples/{SID}/canonical_k8_reflot",
        "re-encode loop": f"{RES}/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn/outputs/full_reenc/"
                          f"samples/{SID}/reencode_k1_refdense_noblend"}


def main():
    rows = list(csv.DictReader(open(f"{SWEEP}/compare/per_turn.csv")))
    out = {"arms": {}, "session": json.load(open(
        f"{RES}/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn/outputs/session.json"))["sessions"][0]["turns"]}
    for arm, d in ARMS.items():
        rs = sorted([r for r in rows if r["arm"] == arm], key=lambda r: int(r["turn"]))
        assert len(rs) == 100, arm
        out["arms"][arm] = dict(dir=d,
                                tokens=[int(r["tokens"]) for r in rs], ref=[int(r["ref_tokens"]) for r in rs],
                                dit_s=[float(r["dit_s"]) for r in rs], edit_ok=[float(r["edit_ok"]) for r in rs],
                                keep_psnr=[float(r["keep_psnr"]) for r in rs],
                                clip_margin=[float(r["clip_margin"]) for r in rs],
                                crop_musiq=[float(r["crop_musiq"]) for r in rs],
                                crop_clipiqa=[float(r["crop_clipiqa"]) for r in rs])
        a = out["arms"][arm]
        print(arm, "tok", np.mean(a["tokens"]), "s", np.mean(a["dit_s"]), "ok", np.mean(a["edit_ok"]),
              "psnr t1/10/25/100", [a["keep_psnr"][i] for i in (0, 9, 24, 99)])
    lv = {}
    for k in (1, 2, 4, 8):
        g = torch.load(f"{ARMS[f'k={k}']}/turn_01/grid.pt", map_location="cpu", weights_only=False)
        lv[f"k{k}"] = np.array(g["leaves"], dtype=np.int16)
        print(k, len(lv[f"k{k}"]))
    np.savez(f"{HERE}/data/leaves_turn1.npz", **lv)
    json.dump(out, open(f"{HERE}/data/results.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
