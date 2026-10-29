"""LoT edit results 2/2 — 100 chained edits: LoT 4 × 4 context vs. full res, same editor, 20 s, v4 = v3 without captions (slide deck, 2026-10-14). See CLAUDE.md.

v3 (user): drop the CLIP edit margin (noisy) and keep the two image-quality curves; no re-encode loop (drift-freeness
comes from the editor — carried latent, blend, paste — not from LoT, and k = 1 has it too). The LoT-specific question:
does the compressed context cost quality over 100 turns? Lines: k = 4 (amber) vs. k = 1 full res (white), 10-turn
rolling means of CLIP-IQA and MUSIQ of the edited crop (compare/per_turn.csv). Axes 0.55–0.85 and 65–80.

    ~/miniconda3/envs/imp_tome/bin/python render_drift_v4.py h|l        → media/LoT_drift_v4_<res>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_drift_v4.py still 3 12 → scratch/LoT_drift_v4_<t>.png
"""
import json

from common_v1 import *

CAPTIONS = False                                 # v4: False = no bottom caption (one slide section)
_caption_v1 = caption


def caption(c, t, caps, y=1012):
    if CAPTIONS:
        _caption_v1(c, t, caps, y)

D = json.load(open(f"{HERE}/data/results.json"))
LOT, FULL = D["arms"]["k=4"], D["arms"]["k=1"]
CHANGE = {int(k): v for k, v in json.load(open(f"{HERE}/data/change_text.json")).items()}
DUR = 20.0
LINE_FULL = (215, 215, 215)

IMG = (80, 200, 640)
PX0, PX1 = 830, 1840
ROWS = [(200, "CLIP-IQA", "crop_clipiqa", (0.55, 0.85), "{:.2f}"),
        (530, "MUSIQ", "crop_musiq", (65.0, 80.0), "{:.1f}")]
ROW_H, PLOT_TOP = 310, 64
WIN = 10

T0, DT = 1.0, 0.14
T_END = 15.3
CAPS = [(0.0, "100 chained edits: LoT 4 × 4 context vs. full res"),
        (2.5, "Image quality of every edit, turn by turn"),
        (9.0, "LoT holds quality over 100 turns, 4.3× faster"),
        (T_END, "Mean over 100 turns: LoT at or above full res")]


def turn_at(t):
    return 0 if t < T0 else min(100, 1 + int((t - T0) / DT))


def textbox(c, cx, cy, s, size=34, style="ti"):
    w = font(style, size * K).getlength(s) / K
    c.rect(cx - w / 2 - 22, cy - size * 0.85, cx + w / 2 + 22, cy + size * 0.85, fill=(26, 26, 26),
           outline=(85, 85, 85), width=1.5, r=10)
    c.text(cx, cy, s, size, style=style)


def turn_box(c, panel, k, size=44):
    x0, y0, S = panel
    s = f"turn {k}"
    w = font("bx", size * K).getlength(s) / K
    bx1, by1 = x0 + S - 14, y0 + S - 14
    c.rect(bx1 - w - 28, by1 - size * 1.35, bx1, by1, fill=(12, 12, 12), r=10)
    c.text(bx1 - 14, by1 - size * 0.675, s, size, style="bx", anchor="r")


def rolling(v, k):
    v = np.asarray(v[:k], float)
    return np.array([v[max(0, i - WIN + 1):i + 1].mean() for i in range(len(v))])


def values(c, x, y, a, b):
    """Right-aligned legend + values: amber "LoT 4 × 4  a", white "full res  b"."""
    parts = [("LoT 4 × 4  ", "rm", AMBER), (a, "bx", AMBER), ("     full res  ", "rm", LINE_FULL), (b, "bx", LINE_FULL)]
    ws = [font("bx" if s == "bx" else "rm", 32 * K).getlength(p) / K for p, s, _ in parts]
    xc = x - sum(ws)
    for (p, s, col), w in zip(parts, ws):
        c.text(xc, y, p, 32, col, style=s, anchor="l")
        xc += w


def plot_row(c, t, k, top, label, key, rng, f):
    lo, hi = rng
    y0, y1 = top + PLOT_TOP, top + ROW_H
    X = lambda i: PX0 + (PX1 - PX0) * i / 99.0
    Y = lambda x: y1 - (y1 - y0) * (np.clip(x, lo, hi) - lo) / (hi - lo)
    c.text(PX0, top + 22, label, 34, style="bx", anchor="l")
    c.rect(PX0, y0, PX1, y1, outline=GHOST, width=1.5)
    c.text(PX0 + 10, y0 + 18, f.format(hi), 22, DIM, anchor="l")
    c.text(PX0 + 10, y1 - 18, f.format(lo), 22, DIM, anchor="l")
    if k >= 2:
        for arm, col, w in ((FULL, LINE_FULL, 3.0), (LOT, AMBER, 4.0)):
            r = rolling(arm[key], k)
            c.line([(X(i), Y(r[i])) for i in range(k)], col, w)
    if t >= T_END:
        values(c, PX1, top + 22, f.format(np.mean(LOT[key])), f.format(np.mean(FULL[key])))
    elif k >= 1:
        values(c, PX1, top + 22, f.format(rolling(LOT[key], k)[-1]), f.format(rolling(FULL[key], k)[-1]))


def render(t):
    c = Canvas()
    k = turn_at(t)
    textbox(c, W / 2, 112, "source image" if k == 0 else f"“{CHANGE[k]}”", 34, "rm" if k == 0 else "ti")
    c.paste(load(f"{LOT['dir']}/turn_{k:02d}/output.png"), *IMG)
    frame_box(c, IMG)
    turn_box(c, IMG, k)
    c.text(IMG[0] + IMG[2] / 2, IMG[1] + IMG[2] + 34, "LoT 4 × 4 context · 1.8 s / edit", 30, AMBER)
    for row in ROWS:
        plot_row(c, t, k, *row)
    caption(c, t, CAPS)
    return finish(c)


if __name__ == "__main__":
    run(render, DUR, "LoT_drift_v4")
