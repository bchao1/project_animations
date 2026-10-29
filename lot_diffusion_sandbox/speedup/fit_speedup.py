"""Writes speedup_table.csv (fraction 1 … 100% → speedup, both modalities) and speedup_fit.png (fit vs paper points)."""
import os

import numpy as np

import lot_speedup as L

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "speedup_table.csv"), "w") as fh:
    fh.write("percent_tokens,image_tokens_of_4096,image_speedup,video_tokens_of_75600,video_speedup\n")
    for p in range(1, 101):
        fh.write(f"{p},{round(p / 100 * 4096)},{L.speedup(p / 100 * 4096):.3f},"
                 f"{round(p / 100 * 75600)},{L.speedup(p / 100 * 75600, modality='video'):.3f}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
f = np.linspace(0.1, 1, 200)
for m, col in (("image", "#E6A800"), ("video", "#B000C8")):
    ax.plot(100 * f, L.speedup(f * L.FULL[m], modality=m), color=col, label=f"{m} fit")
    for p in L.POINTS[m]["points"]:
        ax.scatter(100 * p["tokens"] / L.FULL[m], p["speedup"], color=col, s=14,
                   marker="o" if p["fit"] else "x", zorder=3)
ax.set_xlabel("% of full-resolution tokens")
ax.set_ylabel("speedup (×)")
ax.set_title("LoT speedup vs tokens (o = fit points, x = not fit)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "speedup_fit.png"))
a, b, c = L.COEF["image"]
print(f"image 1/s = {a:.3f} + {b:.3f} f + {c:.3f} f²")
a, b, c = L.COEF["video"]
print(f"video 1/s = {a:.3f} + {b:.3f} f + {c:.3f} f²")
for p in L.POINTS["image"]["points"] + L.POINTS["video"]["points"]:
    m = "video" if p["tokens"] > 4096 else "image"
    print(f'{m:5s} {p["tokens"]:8.0f}  paper {p["speedup"]:.2f}  fit {L.speedup(p["tokens"], modality=m):.2f}  {"" if p["fit"] else "(not fit)"}')
