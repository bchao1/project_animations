"""LoT edit race — LoT 4 × 4 context vs. full res on a shared wall clock, 22 s, v1.

Remake of ~/LoT-george-branch/results/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn/race_video.py in the
sandbox style (user, 2026-10-08): no input-image / token-layout column; each panel has a top text box with the prompt
of the turn it shows and a large "turn N" box at its bottom-left. A panel shows turn n once the cumulative measured
DiT time of turns 1..n (meta.json sample_seconds, idle GPUs) has elapsed; wall time is linear so that LoT finishes
turn 100 at 20 s (179 s of DiT time), then a 2-s hold. Full res reaches turn 23.

    ~/miniconda3/envs/imp_tome/bin/python render_race_v1.py h|l        → media/LoT_race_v1_<res>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_race_v1.py still 5 21 → scratch/LoT_race_v1_<t>.png
"""
import json

from common_v1 import *

D = json.load(open(f"{HERE}/data/results.json"))
SESSION = D["session"]
ARMS = [(D["arms"]["k=4"], "LoT 4 × 4 context", AMBER), (D["arms"]["k=1"], "full res", (215, 215, 215))]
for a, _, _ in ARMS:
    a["done_at"] = np.concatenate([[0.0], np.cumsum(a["dit_s"])])        # done_at[n] = DiT s when turn n is done
RACE, HOLD = 20.0, 2.0
DUR = RACE + HOLD
SCALE = ARMS[0][0]["done_at"][-1] / RACE                                 # DiT s per video s

P, GAP = 780, 90
X0 = (W - (2 * P + GAP)) / 2
PANELS = [(X0, 190, P), (X0 + P + GAP, 190, P)]
LAB_Y, BOX_Y, BAR_Y, CNT_Y = 36, 118, 1000, 1046


def textbox(c, cx, cy, s, max_w, size=30, style="ti"):
    """Prompt in a rounded box; wraps onto two lines (split at the space nearest the middle) if too wide."""
    f = font(style, size * K)
    lines = [s]
    if f.getlength(s) / K > max_w:
        sp = [i for i, ch in enumerate(s) if ch == " "]
        i = min(sp, key=lambda i: abs(i - len(s) / 2))
        lines = [s[:i], s[i + 1:]]
    w = max(f.getlength(l) / K for l in lines)
    if w > max_w:
        size *= max_w / w
        w = max_w
    lh = size * 1.3
    h = lh * len(lines) + 22
    c.rect(cx - w / 2 - 20, cy - h / 2, cx + w / 2 + 20, cy + h / 2, fill=(26, 26, 26), outline=(85, 85, 85),
           width=1.5, r=10)
    for j, l in enumerate(lines):
        c.text(cx, cy + (j - (len(lines) - 1) / 2) * lh, l, size, style=style)


def turn_box(c, panel, k, size=60):
    """Large "turn N" box at the panel's bottom-left."""
    x0, y0, S = panel
    s = f"turn {k}"
    w = font("bx", size * K).getlength(s) / K
    bx0, by1 = x0 + 16, y0 + S - 16
    c.rect(bx0, by1 - size * 1.35, bx0 + w + 32, by1, fill=(12, 12, 12), r=12)
    c.text(bx0 + 16, by1 - size * 0.675, s, size, style="bx", anchor="l")


def render(t):
    c = Canvas()
    clock = min(t, RACE) * SCALE
    for (arm, name, col), p in zip(ARMS, PANELS):
        cx = p[0] + p[2] / 2
        s = np.mean(arm["dit_s"])
        a, b = f"{name}  ·  ", f"{s:.1f} s / edit"
        wa = font("rm", 34 * K).getlength(a) / K
        wb = font("bx", 34 * K).getlength(b) / K
        c.text(cx - (wa + wb) / 2, LAB_Y, a, 34, col, anchor="l")
        c.text(cx - (wa + wb) / 2 + wa, LAB_Y, b, 34, col, style="bx", anchor="l")
        k = int(np.searchsorted(arm["done_at"], clock + 1e-9, side="right") - 1)
        k = min(k, 100)
        textbox(c, cx, BOX_Y, "source image" if k == 0 else f"“{SESSION[k - 1]['instruction']}”", P - 40,
                style="rm" if k == 0 else "ti")
        c.paste(load(f"{arm['dir']}/turn_{k:02d}/output.png"), *p)
        frame_box(c, p)
        turn_box(c, p, k)
        # progress: completed turns + the running one
        frac = k
        if k < 100:
            t0, t1 = arm["done_at"][k], arm["done_at"][k + 1]
            frac += np.clip((clock - t0) / (t1 - t0), 0, 1)
        c.rect(p[0], BAR_Y - 9, p[0] + P, BAR_Y + 9, fill=(40, 40, 40), r=6)
        c.rect(p[0], BAR_Y - 9, p[0] + P * frac / 100, BAR_Y + 9, fill=col, r=6)
        a, b = f"{k}", f" of 100 edits  ·  {clock:.0f} s"
        wa = font("bx", 32 * K).getlength(a) / K
        wb = font("rm", 32 * K).getlength(b) / K
        c.text(cx - (wa + wb) / 2, CNT_Y, a, 32, col, style="bx", anchor="l")
        c.text(cx - (wa + wb) / 2 + wa, CNT_Y, b, 32, anchor="l")
    if t >= RACE:
        full, lot = (np.mean(a["dit_s"]) for a, _, _ in ARMS[::-1])
        tag = f"{full / lot:.1f}× faster"
        w = font("bx", 84 * K).getlength(tag) / K
        cy = PANELS[0][1] + P / 2
        c.rect(W / 2 - w / 2 - 34, cy - 70, W / 2 + w / 2 + 34, cy + 70, fill=(12, 12, 12), outline=PURPLE,
               width=3, r=16)
        c.text(W / 2, cy, tag, 84, PURPLE, style="bx")
    return finish(c)


if __name__ == "__main__":
    run(render, DUR, "LoT_race_v1")
