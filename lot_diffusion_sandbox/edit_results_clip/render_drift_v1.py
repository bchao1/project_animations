"""LoT edit results 2/2 — drift over 100 chained edits, 20 s, v1. See storyboard.md.

Left: LoT editor (4 × 4 context; carries the latent and pastes only the edit region).
Right: the usual in-context loop (re-encode the previous output, regenerate the whole image, full res).

    ~/miniconda3/envs/imp_tome/bin/python render_drift_v1.py h|l        → media/LoT_drift_v1_<res>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_drift_v1.py still 3 12 → scratch/LoT_drift_v1_<t>.png
"""
import json

from common_v1 import *

D = json.load(open(f"{HERE}/data/results.json"))
A = D["arms"]
SESSION = D["session"]
DUR = 20.0

P, GAP = 620, 80
X0 = (W - (2 * P + GAP)) / 2
PL, PR_ = (X0, 225, P), (X0 + P + GAP, 225, P)
INS_Y, LAB_Y, NUM_Y = 130, 195, 892
ARMS = [("k=4", "LoT editor (4 × 4 context)", PL), ("re-encode loop", "re-encode loop (full res)", PR_)]

# timeline: turn 0 hold, turns 1–10 slow, 11–100 fast, hold turn 100
T0, T_SLOW, DT_SLOW, DT_FAST = 1.2, 0.42, 0.42, 0.11     # turn 1 at 1.2 s; turn 10 ends 5.4 s; 100 at ~15.3 s
T_END = 15.6
CAPS = [(0.0, "The same 100 chained edits, two editors"),
        (T0 + 10 * DT_SLOW, "LoT edits only the edit region: the rest stays bit-exact"),
        (10.0, "Re-encoding the whole image every turn drifts"),
        (T_END, "After 100 edits: the scene holds, and edits still land")]


def turn_at(t):
    if t < T0:
        return 0
    if t < T0 + 10 * DT_SLOW:
        return 1 + int((t - T0) / DT_SLOW)
    return min(100, 11 + int((t - T0 - 10 * DT_SLOW) / DT_FAST))


def short(turn):
    e = SESSION[turn - 1]["edit"]
    return e.split("→")[-1].strip() if "→" in e else e


def render(t):
    c = Canvas()
    k = turn_at(t)
    if k == 0:
        c.text(W / 2, INS_Y, "turn 0 · source image", 34, AMBER)
    else:
        s, ins = f"turn {k}", f"“{SESSION[k - 1]['instruction']}”"
        w1 = font("rm", 34 * K).getlength(s) / K
        w2 = font("ti", 34 * K).getlength(ins) / K
        tot = w1 + 24 + w2
        sc = min(1.0, 1800 / tot)
        x0 = W / 2 - tot * sc / 2
        c.text(x0, INS_Y, s, 34 * sc, AMBER, anchor="l")
        c.text(x0 + (w1 + 24) * sc, INS_Y, ins, 34 * sc, WHITE, style="ti", anchor="l")
    for arm, lab, p in ARMS:
        cx = p[0] + p[2] / 2
        c.text(cx, LAB_Y, lab, 32)
        c.paste(load(f"{A[arm]['dir']}/turn_{k:02d}/output.png"), *p)
        frame_box(c, p)
        if k >= 1:
            psnr = A[arm]["keep_psnr"][k - 1]
            a = "untouched pixels vs. turn 0:  "
            b = "bit-exact" if not np.isfinite(psnr) else f"{psnr:.1f} dB"
            size = 32 if t < T_END else 38
            wa = font("rm", size * K).getlength(a) / K
            wb = font("bx", size * K).getlength(b) / K
            x0 = cx - (wa + wb) / 2
            c.text(x0, NUM_Y, a, size, anchor="l")
            c.text(x0 + wa, NUM_Y, b, size, AMBER if not np.isfinite(psnr) else RED, style="bx", anchor="l")
    caption(c, t, CAPS)
    return finish(c)


if __name__ == "__main__":
    run(render, DUR, "LoT_drift_v1")
