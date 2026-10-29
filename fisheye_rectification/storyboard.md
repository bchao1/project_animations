# Storyboard — Self-Supervised Deep Learning for Fisheye Image Rectification (FE-GAN) teaser (3b1b style)

**Current version: v3** — `animation_fisheye_v3.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
`render.sh` first imports the script once (builds the cached toy assets), then renders the 10 scenes in parallel
(logs in `media/log_<script>_<res>_<Scene>.txt`) and concatenates them.
Individual scene MP4s: `media/videos/animation_fisheye_v3/<res>/S0k_*.mp4`.
Final cut: `media/final/FisheyeRectification_animation_fisheye_v3_1080p60.mp4` (10 scenes, 1920×1080 @ 60 fps; duration in
the section "Final render" below: 5 min 24.7 s). 480p15 test cut: 5 min 27.2 s = 327.2 s.

Source: Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang, *Self-Supervised Deep Learning for Fisheye Image
Rectification*, IEEE ICASSP 2020, pp. 2248–2252 (National Taiwan University; Wang also ASUS Intelligent Cloud Services).
Not on arXiv and not open access; the PDF was fetched from IEEE Xplore over the Stanford network.
PDF, text, figures: `fisheye-rectification/pdfs/`; ingest notes with every number used:
`fisheye-rectification/summaries/chao2020selfsupervised.md`. Official code (reference only): https://github.com/ByronHsu/FEGAN
(cloned to `scratch/FEGAN`).
`data/fig1_*.png`, `data/fig2_<row>_<mc|wf|ch><k>.png` are crops of the paper's Fig. 1 / Fig. 2 (made by `scratch/crop_figs.py`
from `pdfs/figs/*_hi.png` = the images embedded in the PDF at native resolution). `data/icassp2020-result.gif` is the website
teaser (142 × 142, 2 frames × 800 ms), frames in `data/gif_frames/`. `data/generated/` is built on import by `build_assets()`
(flag `done_v1`, ≈ 45 s) and `build_assets_v2()` (flag `done_v2`, the finer strength family); file-locked.

## Version history
- v1 — first full cut, 10 scenes, 6 min 5 s (480p15). QA: too long; S02 ray crossed the r(θ) plot, curve labels planned
  off-frame (moved to an in-plot legend before render); S03 colour swatches rendered black, sampled pixels had tiny flows,
  toy note crowded Eq.; S04 "paired" label on the images; S05 y label on Eq. 1, self-attention tag and "warp (STN)" collided
  with blocks, heads at the frame edge; S06 all-black radial diff map invisible, flow arrows too faint; S07 toy note over the
  column labels, error-vs-strength curve clipped at 40 with a one-sample spike; S08 gain label off-frame.
- v2 — fixes for all of the above; captions merged/shortened (S01, S02, S05, S06, S07); strength family re-done with a
  3-px-blurred RMSE on 81 steps (graded dip, min at s = 0.99); thin space after 𝔼 subscripts; nbsp around every ×.
  5 min 32 s (480p15).
- v3 — S07 note corrected (amber ring arrows are ×1.6, not ×0.8); S01/S08 captions shortened; Fig. 2 captions reworded to
  what the figure shows ("garbles content", "leaves most of the bend"). 480p15: 5 min 27.2 s. Rendered 1080p60.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif (text); Latin Modern Math for glyphs CMU lacks (× − → ≈ ∑ 𝔼 ‖ ∇ ∫ ℒ π ↑ ↓ ↦ ✓ …, `MATHCH`); DejaVu Sans for ✗ |
| Math | Pango markup, no LaTeX. `sub()` rise −3000 + `<small><small>`; `sup()` rise 4500; θ is CMU italic (`TH`); x̃ = italic x + U+0303 |
| Background | black |
| Colours | dim `#8A8A8A`, orange `#FF9E40` (fisheye / distorted, `FISH`), cyan `#4DD0E1` (pinhole / distortion-free, `PIN`), green `#69F0AE` (FE-GAN / correct), red `#FF5252` (baselines / failure), amber `#FFCA28` (flow, highlights, headline), blue `#448AFF`, purple `#E040FB` (other lens models) |
| Flow fields | arrows (amber / white) on a grid, lengths scaled and labelled (×0.5 / ×0.7 / ×1.6); colour image: hue = direction, brightness = length |
| Error maps | `inferno` |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 centred lines), max width 13.2, bottom ≥ −3.85; held ≥ len/17 + 0.8 s; `hold()` waits for it before the next visual change |
| Part card | “Part k” (30, dim) over title (50, bold, ≤ 12.5 wide), 1.5 s, instant |
| Toy notes | 16 pt dim, “toy: …” under or above every computed panel |

## Toy computations (all numpy, cached in data/generated/)
Normalised image coordinates: frame = [−1, 1]², focal length f = 1 (half the image width). The world is the pinhole image
plane (a procedural street facade, `draw_scene`, canvas ±6.4 at 256 px/unit, 2× supersampled). A ray at angle θ lands at
tan θ on it. Lens family r = tan(κθ)/κ: κ = 1 pinhole (r = f tan θ), κ → 0 equidistant fisheye (r = f θ); 48 steps
(`morph.npy`). Ray angle to the image edge / corner: 45° / 55° (pinhole) → 57° / 81° (fisheye).
- Rectifying flow (exact for the toy lens, `flow_true_uv`): x̃(p) = x(p + f_p), p + f_p = p · atan|p| / |p| (inward);
  bilinear sampling (`map_coordinates`, order 1), 2× supersampled. Rectified vs pinhole RMSE 4.5/255 (interpolation only).
- Cross-rotation check (S06): R = 90° (`np.rot90`); radial rectifier mean |R(G(x)) − G(R(x))| = 2e−6 (shown 0.00);
  "x-only" rectifier (f = (atan u − u, 0)) = 27.0 on a 0–255 scale.
- Candidate flows (S07): noisy = true + 𝒩(0, 0.02) per pixel; swirl = true rotated 90° (tangential); radial = true.
  Toy loss terms in pixels of a 512² image — smoothness mean(|∂x f| + |∂y f|): 23.15 / 0.66 / 0.66; direction mean |cos|:
  0.92 / 0.00 / 1.00; circle variance (64 radius bins, r < 1): 19.37 / 0.10 / 0.10.
- Strength family (S07): s · f_true, s ∈ [0, 1.5], 81 steps; error = RMSE (0–255) after a 3-px Gaussian blur against the
  pinhole view: 38.6 at s = 0, minimum 1.9 at s = 0.99, 42.2 at s = 1.5.
- Unpaired set Y (S04): three recoloured toy facades (pinhole views) + the paper's Fig. 1 y image.

## Scenes (durations from the 480p15 render of v3)

### S00_Title (6.7 s)
Title “Self-Supervised Deep Learning” (54 bold, y 3.0) / “for Fisheye Image Rectification” (40, y 2.28). Website GIF frames
side by side, 3.0 tall at (∓2.0, −0.3): “fisheye input” (orange frame) → arrow → “rectified by FE-GAN” (green).
Authors (22 dim, y −2.5), “National Taiwan University · ASUS Intelligent Cloud Services · IEEE ICASSP 2020” (y −2.95),
“* equal contribution” (18, y −3.35).

### S01_Hook (27.4 s) — “1 · Why fisheye images look bent”
Live toy image 4.9 tall at (−3.0, 0.2) with the world grid overlaid (grid lines recomputed every frame through the lens
map, `grid_segments`), morphing pinhole → fisheye (κ 1 → 0, 4 s). Right: “lens projection” slider pinhole ↔ fisheye and live
readouts “ray angle at the image edge / corner”. Amber world lines (cornice, pavement edge, one pilaster) bend live
(κ 0 → 1 → 0). Captions: pinhole keeps straight lines straight · fisheye with the same focal length squeezes a wider view ·
lines bend more toward the edges · breaks the pinhole assumption behind recognition and segmentation (paper Sec. 1) ·
goal: undo it from a single fisheye image.

### S02_Projection (42.7 s) — “2 · Pinhole vs fisheye projection”
A. Left: camera centre O (−5.6, −2.15), quarter circle of radius f (1.55), image plane at x = O + f, ray at angle θ (live).
Pinhole height f tan θ (cyan segment + dot); fisheye: arc of length f θ (orange) unrolled onto the plane. Right: Axes
θ 0–90° × r/f 0–4 (6.4 × 4.4 at (3.35, 0.35)), in-plot legend; pinhole curve then equidistant, then equisolid, stereographic,
orthographic. θ sweeps 25° → 62° → 84° (“tan θ → ∞ at 90°”) → 40° → 80°.
B. Manim nonlinear transform: 150 cyan grid lines (spacing f/4) over the pinhole toy image; `ApplyPointwiseFunction`
r ↦ f atan(r/f) squeezes the plane into the circle r = f π/2 (amber dashed) while the image cross-fades to the fisheye;
then the inverse r ↦ f tan(r/f) pulls it back out (clamped at r = 1.52 f).

### S03_Flow (32.5 s) — “3 · Rectification as a pixel flow”
Panels 3.6 at x −4.4 / 0 / 4.4 (y 0.45): fisheye x (toy), flow f, rectified x̃ (empty frame). Three output pixels
((0.5, 0.42), (0.86, 0.72), (−0.84, −0.66)): ghost ring in x, amber arrow to p + f_p, colour swatch placed at p.
Eq. x̃(p) = x(p + f_p) (32, y −1.95) + “f_p : the flow at pixel p”. Flow colour image + 11 × 11 arrows (×0.5). Row-by-row
reveal of the rectified image (2.6 s). Closing: FE-GAN predicts the flow, not the parameters of one lens model.

### S04_Data (22.0 s) — “4 · Why not just supervise?”
Left (red): toy fisheye ↔ toy ground truth, “paired”, what prior deep methods need (paper Sec. 1 list; refs [8, 9, 10]).
Right (green): set X (four fisheye inputs from paper Fig. 2) and set Y (paper Fig. 1 y + three toy facades) in rounded boxes;
“no correspondences, no camera parameters”.

### S05_GAN (38.3 s) — “5 · Fisheye GAN (FE-GAN)”
Generator row (y 1.35): x (Fig. 1) → U (encoder–decoder with skip arcs, “self-attention” tag) → f (Fig. 1 schematic) →
W box (“warp (STN)”) → x̃ = G(x) (Fig. 1); curved arrow x → W; dashed box “generator G = (U, W)”; an amber pulse runs
through. Discriminator: trapezoid D (4.2, 1.35), y (Fig. 1) from below, heads D_adv (“real / fake”) and D_cls
(“distorted / distortion-free”). Eq. 1 (y −1.75) and Eq. 2 (y −2.45), 26 pt. Closing: adversarial losses judge appearance;
what pins down geometry?

### S06_Rotation (36.9 s) — “6 · Cross-rotation consistency”
Commutative square (images 1.9): x → G(x) (top), R_θ (animated 90° turn of the copy), G(R_θ(x)) = R_θ(G(x)) ?; Eq. 3 (two
lines, top right), θ ∈ {90°, 180°, 270°}. Toy check: diff maps (framed) radial (mean 0.00, green) vs x-only (27.0, red).
Then Eq. 4; radial and x-only flow fields (9 × 9, ×0.7) rotate by 90° over faint copies: “lands on itself ✓” /
“becomes a y-only field ✗”.

### S07_Radial (55.3 s) — “7 · Intra-image warping consistency”
A. Toy flow (3.5) with circles; amber ring of 16 equal arrows (×1.6). Right column: “Radial-symmetric” (direction
|f_p · p| / (‖f_p‖ ‖p‖) → 1, circles Var({‖f_p‖ : ‖p‖ = r}) → 0, “(6)”), “Smooth” (Eq. 5), assumption line.
B. Three candidate flows (noisy / swirl / radial): flow (2.0, ×0.5), warped fisheye (2.0), three toy loss values coloured
good/bad. C. Strength family: live warped output (3.4) + error-vs-strength curve with moving dot, s 0 → 1.5 → 1.0.
Caption: the constraints fix the shape, the adversarial losses pick the strength. D. The six losses (1)–(6) with one-line
meanings and group tags (adversarial / cross-rotation / intra-image warping) + G = arg min_G max_D ℒ(G, D).

### S08_Results (51.3 s) — “8 · Results”
1. Datasets: one fisheye input each (Fig. 2) for MCindoor20000 / Wireframe / LSUN Churches; “fisheye transform of Basu &
   Licardie (1995), random parameters: 1000 fisheye images per dataset”; ground truth only for scoring (PSNR, SSIM, RMSE).
2. Fig. 2 grid, 5 columns (fisheye, CycleGAN, UNIT, Aleman-Flores, FE-GAN) × 3 rows (mc0 / wf1 / ch0), then (mc2 / wf2 / ch2).
3. Website GIF toggling every 0.8 s (4.4 tall), label switches “fisheye input” / “rectified by FE-GAN”.
4. Table 1 (22 pt, FE-GAN row green); amber gains +2.3 / +1.3 / +1.8 dB under each dataset with the reference baseline.
5. Table 2 ablation: horizontal PSNR bars (0.40 units/dB), full FE-GAN green; SSIM · RMSE in small text.

### S09_Outlook (14.1 s) — “9 · Takeaways”
Three contributions; amber “best PSNR, SSIM and RMSE on all three datasets vs CycleGAN and UNIT”; dim “keep in mind:
assumes the optical axis through the image centre; evaluated on synthetic fisheye images”. End card (no header): title,
“IEEE ICASSP 2020 · pp. 2248–2252”, code URL, authors, affiliations, * equal contribution; held 4 s.

## Accuracy notes
- Numbers on screen from the paper: Table 1 (all 27 values), Table 2 (all 12), 1000 fisheye images per dataset,
  θ ∈ {90°, 180°, 270°}, H × W × 2, pp. 2248–2252. Derived: +2.3 / +1.3 / +1.8 dB = 15.215 − 12.959 (UNIT),
  12.846 − 11.505 (CycleGAN), 12.781 − 10.982 (CycleGAN); "4.2 dB" = 15.215 − 11.021 (w/o cross-rotation).
- Everything else (angles 45°/55°/57°/81°, flows, 0.00 / 27.0, loss values, error curve) is our toy computation, labelled.
- Standard projection formulas (pinhole, equidistant, equisolid, stereographic, orthographic) are textbook optics, not paper
  claims. The paper's synthetic data used the Basu & Licardie fisheye transform, not our equidistant toy lens.
- Eq. 6 as typeset sums |cos| (which would be minimised); the text and the official code maximise radial alignment
  (code returns −mean|cos|). The video shows the intent ("→ 1", "→ 0") rather than the sign-ambiguous form.
- The flow panel in S05 is the paper's own schematic (Fig. 1), labelled as such. Loss weights (code only) are not shown.
- "The adversarial losses pick the strength" (S07) is our reading: the flow constraints are satisfied by radial flows of any
  strength (shown in the toy), so only the adversarial terms depend on how distortion-free the output looks.
- Fig. 2 captions describe what the figure shows; the paper itself says CycleGAN/UNIT translate style (colour, texture)
  rather than geometry and that single-model classical methods regress a fixed number of parameters.

## Final render
`media/final/FisheyeRectification_animation_fisheye_v3_1080p60.mp4` — 1920×1080 @ 60 fps, 5 min 24.7 s = 324.7 s
(ffprobe); scenes 6.7 · 27.1 · 42.4 · 32.3 · 21.9 · 38.0 · 36.7 · 54.8 · 50.9 · 14.0 s.
Known cosmetic issue: in S05 the small “self-attention” tag touches the bottom edge of the dashed generator box.
