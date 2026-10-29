"""LoT edit results 2/2 — 100 chained edits keep their quality, 20 s, v2. See storyboard.md.

v2 (user): no re-encode comparison / untouched-pixel numbers; show that edit and image quality stay the same over
the 100 turns (k = 4 arm, compare/per_turn.csv: clip_margin, crop_clipiqa, crop_musiq). Instruction in a top text box,
a large "turn N" box on the image.

    ~/miniconda3/envs/imp_tome/bin/python render_drift_v2.py h|l        → media/LoT_drift_v2_<res>.mp4
    ~/miniconda3/envs/imp_tome/bin/python render_drift_v2.py still 3 12 → scratch/LoT_drift_v2_<t>.png
"""
import json

from common_v1 import *

D = json.load(open(f"{HERE}/data/results.json"))
ARM = D["arms"]["k=4"]
CHANGE = {int(k): v for k, v in json.load(open(f"{HERE}/data/change_text.json")).items()}
DUR = 20.0

IMG = (80, 200, 640)
PX0, PX1 = 830, 1840
ROWS = [(200, "edit accuracy · CLIP edit margin", "clip_margin", (0.0, 0.12), "{:.3f}"),
        (420, "image quality · CLIP-IQA of the edit", "crop_clipiqa", (0.5, 1.0), "{:.2f}"),
        (640, "image quality · MUSIQ of the edit", "crop_musiq", (50.0, 90.0), "{:.1f}")]
ROW_H, PLOT_TOP = 200, 56
WIN = 10                                         # causal rolling mean (turns)

T0, DT = 1.0, 0.14                               # turn 1 at 1.0 s, turn 100 at 14.9 s
T_END = 15.3
CAPS = [(0.0, "100 chained edits on one image, 1.8 s each"),
        (2.5, "Edit accuracy stays flat across all 100 turns"),
        (8.0, "Image quality of the edited region holds too"),
        (T_END, "First 20 vs. last 20 turns: edits hold, quality within 3%")]


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


def plot_row(c, t, k, top, label, key, rng, f):
    v = ARM[key]
    lo, hi = rng
    y0, y1 = top + PLOT_TOP, top + ROW_H
    X = lambda i: PX0 + (PX1 - PX0) * i / 99.0                    # turn i + 1
    Y = lambda x: y1 - (y1 - y0) * (np.clip(x, lo, hi) - lo) / (hi - lo)
    c.text(PX0, top + 20, label, 30, anchor="l")
    c.rect(PX0, y0, PX1, y1, outline=GHOST, width=1.5)
    ref = float(np.mean(v[:20]))
    for xa in np.arange(PX0, PX1, 22):                           # dashed: level of turns 1–20
        c.line([(xa, Y(ref)), (min(xa + 11, PX1), Y(ref))], (110, 110, 110), 1.5)
    if k >= 1:
        r = rolling(v, k)
        if k >= 2:
            c.line([(X(i), Y(r[i])) for i in range(k)], AMBER, 3.5)
        if t < T_END:
            c.text(PX1, top + 20, f.format(r[-1]), 34, AMBER, style="bx", anchor="r")
    if t >= T_END:
        a, b = f.format(np.mean(v[:20])), f.format(np.mean(v[-20:]))
        c.text(PX1, top + 20, f"{a} → {b}", 34, AMBER, style="bx", anchor="r")


def render(t):
    c = Canvas()
    k = turn_at(t)
    textbox(c, W / 2, 112, "source image" if k == 0 else f"“{CHANGE[k]}”", 34, "rm" if k == 0 else "ti")
    c.paste(load(f"{ARM['dir']}/turn_{k:02d}/output.png"), *IMG)
    frame_box(c, IMG)
    turn_box(c, IMG, k)
    for row in ROWS:
        plot_row(c, t, k, *row)
    caption(c, t, CAPS)
    return finish(c)


if __name__ == "__main__":
    run(render, DUR, "LoT_drift_v2")
