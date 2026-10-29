# Storyboard — Foveated Diffusion 1-minute teaser (3b1b style)

**Current version: v4** — `animation_foveated_diffusion_1min_v4.py` (480p15 total 59.1 s: S1 24.1 + S2 14.6 + S3 20.4)
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60). Env: `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`.
Final cut: `media/final/FoveatedDiffusion_1min_v4_1080p60.mp4` (3 scenes concatenated, 58.98 s, 1920×1080, 60 fps, h264; 1080p scenes 24.0 / 14.6 / 20.4 s). Target 55–70 s (hard max 75 s).

Compact cut of `../teaser_5min/` (v2, read-only). Paper: Chao*, Yariv*, Xiao, Wetzstein, *Foveated Diffusion:
Efficient Spatially Adaptive Image and Video Generation*, arXiv 2603.23491 (2026), bchao1.github.io/foveated-diffusion.
Assets are read by absolute path from `../teaser_5min/data/` (paper figure crops + `generated/`); the only new asset
(`data/generated/gaze4_{A,B,C}.png`, gaze-contingent blurs of the farmer whose sharp radius matches the token fovea)
is built on import into this folder (render.sh imports the script once before the parallel renders).

## Version history
- v1 — first cut, 3 scenes, 56.0 s. Blur = 5-min cache (sharp radius 40 px) → dashed circle much smaller than the token fovea.
- v2 — own blurs (sharp radius 100 px) + dashed circle = token fovea radius (2 blocks); S2 caption no longer faded out with
  the left column (it was missing 9–13 s); S3 user-study block 0.15 lower. 56.7 s.
- v3 — stronger periphery blur (k 0.05 → 0.08); S3 big numbers hold 3.0 s, user-study hold +3.5 s, small labels 18/20 → 20/22;
  S2 +1.2 s on the DiT frame. 58.2 s.
- v4 — gaze B/C moved inward (0.72, 0.26)/(0.26, 0.72) so the dashed circle stays inside the image; S2 DiT frame hold 2.5 s. 59.1 s.

**Takeaway.** Viewers only see full detail where they look. Foveated Diffusion keeps high-res tokens only in the fovea
and merges every 2×2 block elsewhere into one low-res token; a pretrained DiT is post-trained (LoRA) on
mixed-resolution-consistent targets → ≈ 2× faster images, ≈ 4× faster video, perceptually indistinguishable.

## Numbers shown (all from the paper)
| Number | Source |
|---|---|
| 2.2× image / 4.2× video speedup at 25% tokens | Fig. 6 curve (text: "over 2× and over 4×", §5) |
| user study: ours vs full high-res 47.3% (p = 0.48) | §5 user study, Fig. 7 (2AFC, gaze-contingent) |
| "≈ 2× faster images · ≈ 4× faster video · perceptually indistinguishable" | abstract + contributions ("up to 2× and 4×") |
| token counts in S1 (256 → e.g. 100) | computed live from the drawn 16×16 grid and fovea (illustrative, not a paper number) |
| FLUX.2 Klein 4B / Wan2.1 1.3B, LoRA | §4 setup |

## Style (unchanged from teaser_5min)
CMU Serif (+ Latin Modern Math for × ⊙ − → ≈ …), Pango markup only, black bg, HR `#448AFF`, LR `#00C853`, dim `#8A8A8A`,
red `#FF5252`, amber `#FFCA28` (image), purple `#E040FB` (video), LoRA `#FF7043`. Text appears instantly.
No section headers (teaser). Caption y −3.5, size 28, ≤ ~10 words, hold ≥ len/16 + 0.8 s.

## Beats (v4; durations from the 480p15 render)

### S1_Insight (24.1 s) — hook + key insight
1. **Title (0–4 s).** "Foveated Diffusion" (72 bold, y 2.75) + subtitle (32, y 1.95); 16×16 `FoveaField` (3.8 at (0, −0.75))
   fades in 1.2 s, fovea wanders 3 steps × 0.5 s, 0.6 s hold, fade out 0.5 s.
2. **Hook (4–10 s).** Farmer (4.6 at (−3.4, 0.0)), red gaze dot, label "what you see" (26 dim, above).
   "Keep your eyes on the red dot." → gaze blur A fades in 1.5 s + dashed circle (radius = token fovea, 1.15).
   "You only see full detail where you look."
3. **Problem (10–15 s).** Right: 16×16 all-HR token grid (4.6 at (3.4, 0.0)), label "what the model computes",
   count "256 high-res tokens" below. "Yet a diffusion transformer pays for every token equally."
4. **Insight (15–24 s).** Grid → `FoveaField` whose fovea sits under the gaze (2×2 merges, 1.5 s); count →
   "m high-res + k low-res = L tokens" (computed). "Foveated Diffusion: high-res tokens only where you look."
   Gaze A (0.50, 0.55) → B (0.72, 0.26) → C (0.26, 0.72) (1.2 s each + 1.0 s hold; blur cross-fades; fovea retargets in
   sync; count updates: 56 + 50 = 106 → 52 + 51 = 103 → 52 + 51 = 103).
   "Elsewhere, every 2 × 2 block becomes one token."

### S2_Method (14.6 s) — foveated training in one picture
Layout of the 5-min S05: x (1.9 at (−5.7, 1.3)), Down(x) (0.95 at (−5.7, −1.6)), encoders E (x −3.95),
16×16 HR token grid (1.9 at (−1.9, 1.3)), 8×8 LR grid (1.9 at (−1.9, −1.6)), mask M (1.1 at (0.55, −0.15)),
foveated target grid (2.6 at (3.5, −0.15)).
1. "Training targets: encode each image at two resolutions." (x, Down(x), E, both grids.)
   Grid labels "16 × 16 high-res" (HR) / "8 × 8 low-res" (LR). (0–5 s)
2. "Merge by the mask: a mixed-resolution-consistent target." Mask panel "mask M", "foveated target" (28 bold) above
   ghost grid; unused tokens dim to 0.2; tokens fly in (1.8 s); "both from the <i>same</i> image" (22 dim) below.
   No equation (zero math; the words carry it). (5–9 s)
3. Left side fades; target slides 5.6 left; arrow → DiT + LoRA box (1.9×2.4 at (1.6, −0.15)); right: "pretrained base
   models: FLUX.2 Klein 4B (images) / Wan2.1 1.3B (video)". "Post-train a pretrained DiT with LoRA on these targets."
   LoRA tag Indicate, 2.5 s hold. (9–14.6 s)

### S3_Payoff (20.4 s) — results + end card
1. **Speed (0–5 s).** Fig. 6 speedup plot (Axes 5.4×4.2 at (−3.4, 0.2)); right column: big "2.2×" (amber) faster images,
   "4.2×" (purple) faster video (84 bold), "with 25% of the tokens" (26 dim); 3 s hold. "Fewer tokens per step, much faster generation."
2. **Quality (5–15 s).** Horse row Full high-res / Naïve (no training) / Ours (fig5, 2.9 each, y 1.0; white circles = fovea,
   from the paper), red note "duplicated horse". "Without training it breaks; ours matches full resolution."
   User study (y −1.9): header "gaze-contingent user study: which image looks better?" (22 dim, y −1.3), label
   "ours vs. full high-res", bar x0 −2.2 W 4.4 filled 47.3%, dashed 50% line + "50% = chance", note "p = 0.48:
   indistinguishable". "Viewers could not tell ours from full resolution." 3.5 s hold.
3. **End card (15–20.4 s, 5 s hold).** Title, HR tagline "≈ 2× faster images · ≈ 4× faster video · perceptually indistinguishable",
   authors · Stanford, arXiv 2603.23491 · bchao1.github.io/foveated-diffusion.
