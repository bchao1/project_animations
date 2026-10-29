"""LoT edit results 1/2 — context token size (1 × 1 … 8 × 8), 20 s, v2 (user: no edit-success line). See storyboard.md.

    ~/miniconda3/envs/imp_tome/bin/python render_ctx_v2.py h|l      → media/LoT_ctx_v2_<res>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_ctx_v2.py still 2 9 → scratch/LoT_ctx_v2_<t>.png
"""
import json

from common_v1 import *

D = json.load(open(f"{HERE}/data/results.json"))
A = D["arms"]
SESSION = D["session"]
LV = dict(np.load(f"{HERE}/data/leaves_turn1.npz"))
KS = [1, 2, 4, 8]
DUR = 20.0

# geometry: four equal panels, centred row
P, GAP = 430, 40
X0 = (W - (4 * P + 3 * GAP)) / 2
PANELS = [(X0 + i * (P + GAP), 245, P) for i in range(4)]
INS_Y, LAB_Y, CNT_Y, SPD_Y = 150, 215, 715, 810
LABELS = ["1 × 1 context (full res)", "2 × 2 context", "4 × 4 context", "8 × 8 context"]

# timeline
T_SWEEP = (0.7, 1.9)
T_TURNS = 4.6
TURNS = [1, 2, 3, 10, 25, 50, 75, 100]
T_TURN = 1.35                                    # s per shown turn → 4.6 + 8 × 1.35 = 15.4
T_END = 15.6
CAPS = [(0.0, "Context tokens from 1 × 1 (full res) to 8 × 8"),
        (T_TURNS, "The same 100-turn session at each context size"),
        (10.0, "Coarser context, same edits"),
        (T_END, "Same edits with 16–34% of the tokens, up to 4.8× faster")]


def path(k, t):
    return f"{A[f'k={k}']['dir']}/turn_{t:02d}/output.png"


SRC = f"{A['k=4']['dir']}/turn_00/output.png"


@lru_cache(4)
def cell_cols():
    return np.asarray(load(SRC).resize((N, N), Image.BOX), dtype=np.float32)


@lru_cache(4)
def leaf_cols(k):
    a = np.asarray(load(SRC), dtype=np.float32)
    return np.array([a[r0 * 16:r1 * 16, c0 * 16:c1 * 16].reshape(-1, 3).mean(0) for r0, r1, c0, c1 in LV[f"k{k}"]])


def sweep(k, u):
    """Turn-1 layout of arm k over the source: 1 × 1 tokens merge along a diagonal sweep (u ∈ [0, 1])."""
    lv = LV[f"k{k}"].astype(int)
    e = lv[:, 1] - lv[:, 0]
    trig = (lv[:, 0] + lv[:, 2]) / 126.0 < u
    cc = cell_cols()
    ls, cs = [lv[trig | (e == 1)]], [leaf_cols(k)[trig | (e == 1)]]
    for (r0, r1, c0, c1) in lv[~trig & (e > 1)]:
        ii, jj = np.mgrid[r0:r1, c0:c1]
        ii, jj = ii.ravel(), jj.ravel()
        ls.append(np.stack([ii, ii + 1, jj, jj + 1], 1))
        cs.append(cc[ii, jj])
    lv2 = np.concatenate(ls)
    return lv2, np.concatenate(cs)


def instruction(c, turn):
    s = f"turn {turn}"
    ins = f"“{SESSION[turn - 1]['instruction']}”"
    w1 = font("rm", 34 * K).getlength(s) / K
    w2 = font("ti", 34 * K).getlength(ins) / K
    tot = w1 + 24 + w2
    sc = min(1.0, 1800 / tot)
    x0 = W / 2 - tot * sc / 2
    c.text(x0, INS_Y, s, 34 * sc, AMBER, anchor="l")
    c.text(x0 + (w1 + 24) * sc, INS_Y, ins, 34 * sc, WHITE, style="ti", anchor="l")


def count(c, panel, n, full=4096, unit="tokens"):
    a, b = fmt(n), f" of {fmt(full)} {unit}  ({pct(n, full)}%)"
    wa = font("bx", 30 * K).getlength(a) / K
    wb = font("rm", 30 * K).getlength(b) / K
    x0 = panel[0] + panel[2] / 2 - (wa + wb) / 2
    c.text(x0, CNT_Y, a, 30, AMBER, style="bx", anchor="l")
    c.text(x0 + wa, CNT_Y, b, 30, anchor="l")


def render(t):
    c = Canvas()
    if t < T_TURNS:
        instruction(c, 1)
    else:
        idx = min(len(TURNS) - 1, int((t - T_TURNS) / T_TURN))
        instruction(c, TURNS[idx])
    full_s = np.mean(A["k=1"]["dit_s"])
    for k, p, lab in zip(KS, PANELS, LABELS):
        cx = p[0] + p[2] / 2
        c.text(cx, LAB_Y, lab, 30)
        if t < T_TURNS:
            u = seg(t, *T_SWEEP)
            if u <= 0:
                c.paste(load(SRC), *p)
                n = 4096
            else:
                lv, cols = sweep(k, u)
                draw_tokens(c, p, lv, cols)
                n = len(lv)
            count(c, p, n)
        else:
            turn = TURNS[min(len(TURNS) - 1, int((t - T_TURNS) / T_TURN))]
            c.paste(load(path(k, turn)), *p)
            if t < T_END:
                count(c, p, A[f"k={k}"]["tokens"][turn - 1])
            else:
                n = int(round(np.mean(A[f"k={k}"]["tokens"]) * 2))
                count(c, p, n, full=8192)
        frame_box(c, p)
        if t >= T_END + 0.4:
            s = np.mean(A[f"k={k}"]["dit_s"])
            if k == 1:
                c.text(cx, SPD_Y, f"{s:.1f} s / edit", 56, style="bx")
            else:
                c.text(cx, SPD_Y, f"{full_s / s:.1f}× faster", 56, PURPLE, style="bx")
    caption(c, t, CAPS)
    return finish(c)


if __name__ == "__main__":
    run(render, DUR, "LoT_ctx_v2")
