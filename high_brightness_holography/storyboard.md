# Storyboard — High-Brightness Holographic Projection teaser (3b1b style)

**Current version: v4** — `animation_hbh_v4.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`). `render.sh` runs `precompute_toy.py` if a toy cache
is missing, imports the script once (builds the display assets), renders the 10 scenes in parallel (logs in
`media/log_<script>_<res>_<Scene>.txt`) and concatenates them.
Individual scene MP4s: `media/videos/animation_hbh_v4/<res>/S0k_*.mp4`.
Final cut: `media/final/HighBrightnessHolography_animation_hbh_v4_1080p60.mp4` (10 scenes, 1920×1080 @ 60 fps, 5 min 12.1 s = 312.1 s;
scenes 7.4 · 35.9 · 33.1 · 24.0 · 36.8 · 31.9 · 26.8 · 39.8 · 53.8 · 22.7 s; 480p15 test cut 5 min 14.3 s).

Source: Chao, Gopakumar, Choi, Wetzstein, *High-brightness holographic projection*, Optics Letters 48(15), 4041–4044 (2023),
doi 10.1364/OL.489617, + Supplement 1 (figshare 23596260, CC-BY) + Visualization 1 (figshare 22290445).
Main text from the Wayback snapshot (Optica is captcha-walled); **main-paper Figs. 1–5 could not be obtained**, so the video
uses the supplement's figures (S1, S4, S5, S6, S7, S8, S9, S10, S12), Visualization 1 and the website clip. Records, PDFs,
figure rasters, and the numbers ledger: `high-brightness-holography/` (`summaries/chao2023high.md`).
`data/*.png` = supplement crops (`scratch/crop_figs.py`), `data/video/{OL2023,visualization1}/` = frames at 30 fps.
`data/generated/toy_*.npz` = toy simulations (`precompute_toy.py`, torch), display arrays built by `build_assets()` (flag `done_v1`).

## Version history
- v1 — first full cut, 10 scenes, 5 min 41 s (480p15). QA: title off-frame; S01 plot labels on the curves and axis label on the
  toy note; S03 meter labels overlapping; S05 "L = d tan θ" hidden on the bright cone, S4-plot labels colliding with the
  y label / source note; S06 four panels overlapping each other (3.7 wide at 3.1 spacing) and their labels; S07 loop note on
  the return arrow; S08 schematic and spec lines off-frame and under the caption; "(b)" label left in the S12 crops.
- v2 — all layout fixes above (title scaled to 12.6, curve labels at the line ends, 2-line meter labels, smaller S06 panels and
  shorter labels, S08 re-laid out in three columns, S12 labels painted out with the mirrored corner, S05 short/long views get
  target vs image profiles); 5 min 34 s.
- v3 — trimmed ~20 s of captions (merged S01 Fig. 3 caption, S04 γ caption, S06 "why" caption, S07 CITL caption); S08 panels
  3.3 tall; S05 PSNR labels moved clear of the profiles. 480p15: 5 min 14.3 s; 1080p60: 5 min 12.1 s.
- v4 — 1080p QA: S08 spec column ended ~20 px from the right frame edge → figures 3.2 tall, spec column ≤ 4.5 wide centred
  at x 4.5. Timing unchanged (480p15 5 min 14.3 s). Rendered 1080p60.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif; Latin Modern Math for glyphs CMU lacks (× − → ≈ ∑ ℒ ∂ ⇒ ≥ ∞ …, `MATHCH`); DejaVu Sans for ✗ |
| Math | Pango markup, no LaTeX. `sub()` / `sup()` / `supsub()`; no Unicode sub/superscript digits |
| Background | black |
| Colours | dim `#8A8A8A`, red `#FF5252` (without the loss / conventional / failure), green `#69F0AE` (with the loss / hologram / ours), amber `#FFCA28` (highlights, headline numbers), cyan `#4DD0E1` (image region, geometry, diagram boxes), blue, orange |
| Intensity maps | `inferno` (x–z side views, log image-plane views); captured / RGB toy images in colour (gamma 1/2.2) |
| Phase | `twilight` strip for 1D SLM phases |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 lines), max width 13.2; held ≥ len/17 + 0.8 s; `hold()` before visual changes |
| Part card | “Part k” (30, dim) over title (50, bold), 1.5 s |
| Toy notes | 16 pt dim “toy simulation: …” under every simulated panel; paper figures carry an 18 pt dim source line |

## Toy simulations (`precompute_toy.py`, torch autograd; all labelled "toy" on screen)
Phase-only SLM, 10.8 µm pixels, λ = 636.4 / 517.7 / 440.8 nm (paper), angular spectrum method on a padded grid, image plane at
α d* with d* = R / tan(sin⁻¹(λ/2p)) (supplement Eq. S4), α = 0.64. SGD = Adam, lr 0.02, phase init U(−0.5, 0.5), closed-form
scale s every step (Alg. S1 / Eq. S5); L = L_recon (ROI only) + γ L_reg (whole plane, target zero outside the ROI; Eq. 3), γ = 0.1.
- 2D: SLM 192 × 320, grid 576 × 704, ROI 180 × 300, 2000 iterations. Targets: skimage `chelsea` (cat), `coffee`.
  Cat, green: light in the ROI **64% → 94%**; RGB 70/64/51% → 93/94/94%; brightness 1/Σs² **+46%** (coffee +68%);
  mean PSNR 53 → 35 dB. Snapshots at 27 iterations (0 … 2000) for the live views.
- 1D (side views): 400 SLM px, grid 1280, ROI 372 px, three-bar target, 3000 iterations. In-ROI light 79% (no loss) / 96%
  (loss); PSNR 29.5 dB at 0.64 d*, 18.5 dB at 0.12 d*. Amplitude-panel comparison: light passed = Σ target / 400 px = 34%.
  Single-pixel cone out to 1.1 d* (normalised per distance).
- Sparsity sweep (green, 1200 it.): cat masked by smooth random blobs, 0–96% black. Brightness relative to a conventional
  display with the same power (s² = max I / max L): conventional 1.00 → 1.14, hologram (no loss) 1.87 → 21.0.
- CITL toy (green): "real" projector = off-centre Gaussian illumination, phase response φ + 0.35 sin φ + 0.15 sin 2φ, smooth
  aberration, +0.5% distance. SGD-ASM (ideal model) **16.6 dB**, SGD-CITL (captured amplitudes, ideal gradients; Alg. S2)
  **21.4 dB**, SGD with a toy calibrated model (learned illumination, SLM phase offset, 32-knot phase LUT, distance; fitted on
  80 captured SGD-ASM patterns + noise) **37.8 dB**, all "captured" on the toy real projector.
- Distance sweep (`toy_distance.npz`) is computed but not shown (the paper's own Fig. S4 data are shown instead).

## Scenes (durations from the v4 480p15 render; 1080p60 ≈ 0.2 s shorter per scene)

### S00_Title (7.4 s)
Title (56 bold, ≤ 12.6 wide, y 2.95), subtitle from the website summary (26 dim, y 2.3). Website clip `OL2023` (captured
swallowtail wipe, loops) 3.5 tall at (0, −0.1); side notes “captured on the prototype” and “the brighter half: with the
light-efficiency loss” (green). Authors (y −2.45), “Stanford University · Optics Letters 48(15), 4041 (2023)” (y −2.9).

### S01_Waste (36.1 s) — “1 · Projectors throw light away”
A. 1D x–z maps 3.3 × 4.6 at x ∓3.35: conventional amplitude panel (mask = target; light passes straight; “light that gets
through: 34%”) vs phase-only SLM (twilight strip; light flows into the three bars; “light that lands in the image: 79%”),
same colour scale. B. Toy sparsity sweep: live target thumbnail (x −4.3) + Axes 6.0 × 4.0 at (1.7, 0.45): brightness at equal
laser power vs sparsity, conventional (red, flat) and hologram (green, hyperbolic); labels at the line ends. Caption quotes the
paper's Fig. 3: over 70% vs nearly 0% efficiency. C. Paper Fig. S6 (simulated, same power budget): conventional | hologram at
0.17 / 0.33 / 0.67 d*, monarch + macaw (width 3.1); “+1042% to +1225%”.

### S02_Holography (33.3 s) — “2 · Steering light with phase”
Eq. 1 (y 2.75). 1D SGD live: x–z map 3.6 × 7.6 at (−0.6, 0.35) interpolated over 15 snapshots (0 … 3000 it.) in 6.5 s, SLM
phase strip updating, target (dim) vs image-plane profile (amber) on the right, “SGD iteration k”. Then Eq. 2 (y 1.6), the
scale s line and “brightness = 1 / (s_R² + s_G² + s_B²)” (amber; supplement Sec. 4). Captions mention 2000 iterations, lr 0.02.

### S03_Stray (24.1 s) — “3 · Where does the rest of the light go?”
Toy 2D green SGD without the loss: log-scale image plane 4.3 tall at (−2.2, 0.35), cyan dashed image region, dim SLM footprint;
live meters “light inside the image” (green) / “light outside (wasted)” (red) → 64% / 36%, iteration counter.

### S04_Loss (37.1 s) — “4 · The light-efficiency loss”
Eq. 3 (y 2.6) + Eq. 4 (y 1.8); amber “laser power ∑|u_src|²” and green “power of the scaled target” bars of equal length.
Side-by-side live SGD (log scale, 3.5 tall at x ∓3.35): L_recon only vs L_recon + γ L_reg, “light in the image” 64% vs 94%.
RGB results at equal laser power (3.5 tall): “toy: +46% brightness (1/∑s²)”; caption: paper simulations +66% on average.

### S05_Distance (32.1 s) — “5 · Why the image must be far from the SLM”
A. Toy single-pixel cone (6.2 × 3.4 at (−3.2, 0.55)) with cyan cone lines, θ arc, amber L = d tan θ; right column:
θ = sin⁻¹(λ/2p), d* = R / tan θ, prototype p = 10.8 µm, λ = 517 nm, R = 7.45 mm ⇒ d* = 31.11 cm, image at z ≈ 20 cm ≈ 0.64 d*.
B. 1D toy short (0.12 d*, stretched 2.2× in z) vs long (0.64 d*) with target / image profiles, toy PSNR 18.5 vs 29.5 dB.
C. Paper Fig. S4 re-plotted from the vector data: 45/60/66/67/63% vs 66/80/89/93/91% at 0.03–0.97 d*; peak line at 0.64 d*.

### S06_Reality (27.0 s) — “6 · Long distances break ideal models”
Paper Fig. S7 (captured, CITL): long 0.64 d* (fringes) vs short 0.31 d*, width 6.0. Toy: target | toy laser beam (+ phase
errors, + 0.5% distance error) | ideal-model prediction | toy ‘camera’ capture (16.6 dB), 2.3 tall at x −4.8 / −1.6 / 1.6 / 4.8.

### S07_Camera (40.0 s) — “7 · Learning the real projector”
CITL loop diagram (SLM phase φ → projector + screen → camera → error vs target → update φ), amber dot travels the loop once.
Paper Fig. S5 model + training facts (4800 patterns per colour, 8 : 1 : 1, batch 4, lr 4e−4, ≈ 24 h on an A6000).
Toy three-way (2.9 tall): SGD ideal model 16.6 dB | camera in the loop 21.4 dB | calibrated model 37.8 dB.
Paper Fig. S9 castle (captured): ASM / CITL / learned model × without / with the loss; caption +0% / +14% / +24%.

### S08_Results (54.1 s) — “8 · Results on the prototype”
Fig. S1 schematic + photo (3.2 tall) + spec column (≤ 4.5 wide, x 4.5) (SLM, laser, z, lens/screen/image size, 8 frames). Visualization 1 playing
(8.8 wide, ~10 s: monarch + macaw wipes). Fig. S10 bunny / cat / swallowtail without vs with (width 3.9) + amber “around 30%
brighter, image quality retained”; captions +36 / +33 / +32%, test-set 33% captured / 66% simulated. Fig. S8 γ = 0 / 1 / 100 / ∞
(monarch, eagle). Sparse glasses (+9%) and Fig. S12 small-ROI holograms with amber “10-fold … for a 100 × 160 pixel region
(simulated sinusoid)”.

### S09_Outlook (22.9 s) — “9 · Limitations and outlook”
Three limitations → fixes (20 min per pattern on an RTX A6000 → inverse network; zeroth order → efficient SLMs / off-axis phase
ramp; video → per-frame laser power); HUD caption; 3-point summary + amber result line; end card (title, Optics Letters
48(15), 4041 (2023) · doi, authors, Stanford University, bchao1.github.io), 4 s.

## Accuracy notes
- Every paper number on screen is in `high-brightness-holography/summaries/chao2023high.md` ("Numbers used on screen").
- The Fig. S4 values were digitised from the supplement's vector plot (exact path coordinates), not eyeballed.
- "+1042% to +1225%" = Fig. S6 monarch labels (simulation, conventional vs holographic at a fixed power budget).
- "around 30%" is the paper's own wording; the per-image gains are Fig. S10 labels; 33% / 66% are supplement Sec. 10.
- The sparsity toy uses no light-efficiency loss; its hologram-without-loss efficiency falls below the paper's "over 70%"
  claim, so the toy shows only brightness and the efficiency statement is attributed to the paper (Fig. 3 caption).
- Toy PSNRs and efficiencies are toy outputs only; the toy is continuous-phase, single-frame, 1 colour at a time.
