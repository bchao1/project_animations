# Storyboard — Random-phase Wave Splatting teaser (3b1b style)

**Current version: v3** — `animation_rpws_v3.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
`render.sh` first imports the script once (builds the cached assets), then renders the 10 scenes in parallel
(logs in `media/log_<script>_<res>_<Scene>.txt`) and concatenates them.
Individual scene MP4s: `media/videos/animation_rpws_v3/<res>/S0k_*.mp4`.
Final cut: `media/final/RandomPhaseWaveSplatting_animation_rpws_v3_1080p60.mp4` (10 scenes, 1920×1080 @ 60 fps, 5 min 17.3 s = 317.3 s; scenes 8.3 · 26.3 · 41.1 · 26.3 · 32.4 · 18.7 · 47.8 · 41.9 · 54.8 · 19.8 s;
480p15 test cut 5 min 19.7 s).

Source: Chao*, Yang*, Choi, Gopakumar, Koiso, Wetzstein, *Random-phase Wave Splatting of Translucent Primitives for
Computer-generated Holography*, arXiv 2508.17480 (v3, Dec 2025; v1 Aug 2025 titled "Random-phase Gaussian Wave Splatting");
Stanford University (Koiso also KDDI Research). No journal reference on arXiv; the video says "arXiv 2508.17480 (2025)".
PDF, text, LaTeX source, v1 HTML text: `random-phase-wave-splatting/pdfs/`; ingest notes with every number used:
`random-phase-wave-splatting/summaries/chao2025random.md`. Direct follow-up of Gaussian Wave Splatting (`../gaussian_wave_splatting/`).
`data/*.png` are crops of the paper's figures (made by `scratch/crop_figs.py` from `pdfs/figs/*_hi.png`, rendered from the
LaTeX-source figure PDFs). `data/generated/` is built on import by `build_assets()` (flag `done_v1`) and `build_assets_v2()`
(flag `done_v2`: domain-coloured SLM fields, 7-decade spectra); file-locked; ≈ 30 s.

## Version history
- v1 — first full cut, 10 scenes, 5 min 40 s (480p15). QA: visuals changed before the running caption was read (S02, S06,
  S07, S08 teaser pair shown < 1 s); overlapping labels (S01 arrow label over the Gaussians, S06 panel titles);
  broken superscript subscripts (e^{iφ₁}, e^{ikz_i}) and a cluttered exponent in Eq. 4; flat twilight phase maps; S04
  captions out of sync with the focus sweep; smooth spectrum only a dot (5 decades).
- v2 — `hold()` before every visual change that follows a caption; `supsub()` for sub-inside-superscript; Eq. 4 exponent
  written exp(…); SLM fields in domain colouring (smooth = one hue, random = all hues); spectra over 7 decades with the flat
  one capped; S04 captions re-synced; captions shortened; 5 min 20 s.
- v3 — caption splitter never breaks √(1 − α); S08 headline 26 pt (was near the frame edge); captured Fig. 6 grid 3.6 wide.
  480p15: 5 min 19.7 s. Rendered 1080p60.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif (text); Latin Modern Math for glyphs CMU lacks (× − → ≈ ∑ ∏ ℱ 𝒫 𝔼 𝒪 ∠ ✓ ⇒ …, `MATHCH`); DejaVu Sans for ✗ (`SYMCH`) |
| Math | Pango markup, no LaTeX. `sub()` rise −3000 + `<small><small>`; `sup()` rise 4500; `supsub()` keeps a subscript inside a superscript; long exponents written `exp(…)` |
| Background | black |
| Colours | dim `#8A8A8A`, red `#FF5252` (GWS / failure), green `#69F0AE` (RPWS / correct), amber `#FFCA28` (highlights, cross term, headline), cyan `#4DD0E1` (pupil), blue `#448AFF`, orange `#FF9E40` |
| Intensity images | greyscale (toy reconstructions), `inferno` (x–z side views, error maps) |
| Complex fields | domain colouring: hue = phase, value = amplitude^0.7 (`domain_rgb`) |
| Spectra | log10 |FFT|², 7 decades, `magma`; the flat random spectrum capped at 0.62 of full scale |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 centred lines), max width 13.2, bottom ≥ −3.85; held ≥ len/17 + 0.8 s; `hold()` waits for it before the next visual change |
| Part card | “Part k” (30, dim) over title (50, bold, ≤ 12.5 wide), 1.5 s, instant |
| Row labels | `label()`: 20–26 pt, common baseline 0.15 above (or below) the image |
| Toy notes | 16 pt dim, “toy simulation: …” under every simulated panel |

## Toy wave-optics simulations (all numpy, cached)
λ = 520 nm, 8 µm pixels (prototype SLM pitch), N = 512 grid, angular spectrum method (`asm`). Scene (`toy_scene`): translucent-edged
5-point star (α ≤ 0.97, intensity 0.95) at z_f = 6 mm in front of an opaque checkerboard plane (32-px squares, 0.18/0.90) at z_b = 18 mm.
- **GWS (smooth phase)** `smooth_slm`: amplitude weights α√c, transmittance (1 − α_front) on the rear plane, phases e^{ikz} matched.
- **RPWS** `rpws_slm` (paper Eqs. 4–5): rear √(α_b c_b)·e^{iφ} → 𝒫(·; −12 mm) → × √(1 − α_f) + √(α_f c_f)·e^{iφ} → 𝒫(·; −6 mm). φ iid 𝒰(−π, π) per pixel, fresh per frame.
- **'GWS with √'** `gwssqrt_slm` (paper Fig. 3 ablation): √ weights, but the (1 − α_f) mask applied on the rear plane.
- Eye pupil for focal stacks: circle of radius 0.4 × (half SLM bandwidth) centred; light-field views: radius 0.22, centre moved
  over ±0.72 of the half-bandwidth (41 positions); focus on the star. Focal sweep 2–24 mm, 45 planes. T = 24 frames (paper maximum).
- Speckle contrast = std / mean inside the star (α > 0.95) of the running average (0.97 → 0.20 at T = 24).
- 1D side views (`_xz_maps`): 0.6 mm slit, 1024 samples at 8 µm, z 0–20 mm, x ±1.6 mm shown (stretched vertically).
- Alpha toy (`_alpha_toy`): orange Gaussian (σ 40 px, o = 1) in front of blue (σ 52 px, o = 0.9) at one depth, per-channel
  per-pixel random phase, band-limited through a radius-0.5 pupil. Target = ray alpha blending on intensity (Eq. 7 on c²);
  GWS-NRP weights α₁√c₁, (1 − α₁)α₂√c₂; RPWS weights √(α₁c₁), √((1 − α₁)α₂c₂) (Eq. 16). Error = Σ_channels |blur₆(result) − blur₆(target)|, scale 0.9.
  The GWS-NRP error comes out as a ring around the front Gaussian (where α₁ ≈ ½), as in the paper's Fig. 2.

## Scenes (durations from the 480p15 render of v3)

### S00_Title (8.3 s)
Title “Random-phase Wave Splatting” (60 bold, y 2.95), subtitle (32, y 2.22). Two live panels 3.0 tall at (∓2.0, −0.35):
“one random-phase frame” (flicks through the 24 toy frames at star focus) and “average of k frames” (running mean), k 1 → 24
in 6 s (ease x^1.6). Authors (22 dim, y −2.5), “Stanford University · KDDI Research · arXiv 2508.17480 (2025)” (22, y −2.95),
“* equal contribution” (18, y −3.35).

### S01_Recap (26.5 s) — “1 · Recap: Gaussian Wave Splatting”
Three RGBA Gaussians (2.7 at (−4.6, 0.95)) → arrow “closed-form / wavefronts” → GWS SLM field of the toy scene (domain colour,
one hue) 2.7 at (0.55, 0.95) → arrow 𝒫(·; z) → target scene 2.7 at (4.6, 0.95). GWS Eqs. 1–2 (30/28, y −1.25/−1.95); toy note y −2.55.
Captions: GWS turns Gaussians into wavefronts · alpha wave blending front to back with T · each wave carries e^{ikz_i}, all in
step → one smooth phase (single hue) · “what does that smooth phase cost?”

### S02_Phase (41.3 s) — “2 · Smooth phase vs random phase”
A. x–z side views (width 8.4 at x 0.9; y 1.35 / −1.1): smooth slit beam stays collimated; random phase (1 frame) fans out into the
full ±λ/(2·8 µm) cone; crossfade to the 24-frame average (smooth cone). Row labels at x −4.9.
B. Four panels 2.75 tall at x −5.15 / −1.95 / 1.95 / 5.15, y 0.45: SLM field + angular spectrum for smooth (GWS) and random.
Notes “energy piled up at the centre” (red) / “fills the SLM's whole bandwidth” (green), y −1.25; spectrum note y −1.9.
Captions end with “Each frequency is a direction. In a near-eye display, the spectrum becomes the eyebox.”

### S03_Eyebox (26.5 s) — “3 · A full eyebox, with parallax”
Smooth | random (RPWS, 24 frames): spectrum (2.6) + live view (2.6) each, x −5.25 / −2.35 / 2.35 / 5.25, y 0.5; cyan pupil circle
(radius 0.22 half-band) slides 0 → −0.72 → +0.72 → −0.72 → 0 (views interpolated from 41 precomputed positions).
Readout “pupil position: ±k% of the way to the edge” (cyan, y −1.55), toy note y −2.1. Smooth view goes black off-centre;
RPWS stays bright, the checkerboard shifts behind the star (parallax).

### S04_Defocus (32.6 s) — “4 · Natural defocus blur”
Three live focal stacks 3.2 tall at x −4.5 / 0 / 4.5, y 0.7: smooth (GWS, red), random 1 frame, random 24 frames (green).
z ruler 2–24 mm (y −1.75) with star (6, amber) and checkerboard (18, cyan) markers, dot + “z = … mm”. Sweep 6 → 18 → 6 → 18 → 12.
Smooth: the far plane rings instead of blurring; 1 frame: speckle; 24 frames: natural blur.

### S05_Speckle (18.9 s) — “5 · Speckle and time multiplexing”
Frame k (3.1 at (−4.85, 0.7)), running average (3.1 at (−1.45, 0.7)), Axes 4.6×2.9 at (4.2, 0.55): measured speckle contrast
dots appear with k; dashed C₁/√T (amber). “Speckle reduction is the classic reason … RPWS finds a deeper one.”

### S06_Alpha (48.1 s) — “6 · Alpha blending random-phase waves”
Eq. 9–11 as |w₁e^{iφ₁} + w₂e^{iφ₂}|² = w₁² + w₂² + 2w₁w₂cos(φ₁ − φ₂) (34, y 2.5, cross term amber). Branches at (∓3.4, 0.6):
smooth cos = 1 ⇒ |u| = w₁ + w₂ (amplitude, GWS) · random 𝔼[cos] = 0 ⇒ 𝔼|u|² = w₁² + w₂² (intensity, on average).
GWS weights (red, y −1.25) vs RPWS √ weights (green, y −1.9; paper Eq. 16).
Toy: target / GWS-NRP (naïve) / RPWS (√ weights), 2.6 tall at x −4.4 / 0 / 4.4, y 1.05; error maps 1.35 under the last two;
“frames averaged: k” (1 → 24 in 5 s). GWS-NRP keeps a ring of error; RPWS error → 0. Closing caption: time multiplexing is what
makes the blending correct.

### S07_Compositing (42.3 s) — “7 · Back-to-front wave compositing”
Eq. 4 (30, y 2.6, exponent as exp(i∠u_i + iφ_i^(t))) and Eq. 5 (28, y 1.85). Side-view diagram (y −0.35): far / i / near primitive
planes at x −4.6 / −1.6 / 1.4, SLM at 5.9; amber arrows 𝒫(·; Δz) appear one by one; “× √(1 − α) / + own wave” at each plane.
Toy light-field views 2.15 tall (x −1.3 / 1.25 / 3.8; rows y 1.2 “GWS with √ weights / mask on the far plane” (red) and −1.15
“RPWS / mask at the occluder” (green)); dark halos in the top row's off-axis views.
Paper Fig. 3 ablation (width 3.15 at x −4.95 / −1.65 / 1.65 / 4.95; lego y 1.2, garden y −0.95) with ✓/✗ rows
“back-to-front” ✗✗✓✓ and “√ blending” ✗✓✗✓ (y −2.3 / −2.72).

### S08_Results (55.3 s) — “8 · Results”
1. Paper Fig. 1 panels (width 6.3 at x ∓3.3): GWS front focus vs RPWS back focus, then GWS-NRP front vs RPWS front.
2. Fig. 4b bicycle light field: GWS / RPWS (24 frames) × centre / right view (width 3.5 at x ∓2.0, y ±1.2).
3. Table 1 (24 pt; rows y 2.35 − 0.55 r; RPWS 24 f green, GWS red) + amber headline “+10.5 dB (Blender) and +10.7 dB
   (Mip-NeRF 360) light-field PSNR over GWS” (26, y −2.3).
4. Fig. 6 captured kitchen lego: GWS vs RPWS-GS (24 frames) at front / back focus (width 3.6 at x ∓2.0, y ±1.2).
5. Fig. 6 captured zoom crops RPWS-GS 1 / 8 / 24 frames (width 3.2 at x −3.9 / 0 / 3.9).
6. Fig. 5 RPWS-Δ (triangle splats): 4 scenes, front / rear focus (width 3.0).

### S09_Outlook (19.9 s) — “9 · Limitations and outlook”
Limitations (paper Sec. 5): many frames ↔ SLM refresh (→ 3600 fps FLCoS, 5760 fps MEMS phase modulators); reduced contrast
(→ learned calibration for complex fields). Summary of 3 contributions + amber “full eyebox · parallax · natural defocus ·
+10.5 dB light-field PSNR over GWS (Blender, 24 frames)”. End card (no header): title, subtitle, arXiv 2508.17480, authors,
Stanford University · KDDI Research, * equal contribution; held 4 s.

## Accuracy notes
- Numbers on screen: Table 1 of the paper (all 6 rows, both datasets); +10.5 / +10.7 dB = 24.87 − 14.35 and 21.21 − 10.50
  (derived, labelled as over GWS with 24 frames); 24 frames = the paper's largest setting; 3600 fps / 5760 fps from Sec. 5.
  Everything else is a parameter or output of our toy simulations, labelled as such on screen.
- GWS's LPIPS (0.63 / 0.60) beats RPWS at 1 and 8 frames; the video only claims RPWS 24 f is best on every metric (true).
- The toy simulations model complex SLM fields (no phase-only encoding) and an idealised eye pupil in the SLM's Fourier plane.
  The paper's phase-only holograms were made by SGD with a complex-field loss (v1 hardware text); not shown in the video.
- No runtimes are reported in the paper, so none are shown.
- "GWS with √" in S07's toy matches the paper's ablation name; its mechanism (mask on the far plane / on-axis-only
  transmittance) follows the v1 text's explanation of the halo artifacts.
