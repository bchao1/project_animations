# Storyboard — Gaussian Wave Splatting teaser (3b1b style)

**Current version: v4** — `animation_gws_v4.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
`render.sh` first imports the script once (builds the cached assets), then renders the 10 scenes in parallel
(logs in `media/log_<script>_<res>_<Scene>.txt`) and concatenates them.
Final cut: `media/final/GaussianWaveSplatting_animation_gws_v4_1080p60.mp4` (10 scenes, 1920×1080 @ 60 fps, 5 min 19.4 s; 480p15 test cut 5 min 22 s).

Source: Choi*, Chao*, Yang, Gopakumar, Wetzstein, *Gaussian Wave Splatting for Computer-Generated
Holography*, ACM TOG 44(4) / SIGGRAPH 2025, arXiv 2505.06582. PDF, extracted text and LaTeX source:
`gaussian-wave-splatting/pdfs/`; ingest notes with every number used: `gaussian-wave-splatting/summaries/choi2025gaussian.md`.
`data/*.png` are crops of the paper's figures (made by `scratch/crop_figs.py` from `pdfs/figs/`).
`data/generated/` is built by `build_assets()` on import (file-locked; delete `data/generated/done_v3` to rebuild).
v4 reuses the v3 asset set (the v1/v2 scripts expect older caches and would rebuild if re-run).

## Version history
- v1 — first full cut, 10 scenes, ≈ 6.7 min.
- v2 — scene clock (waves keep moving during caption holds); S02 ASM formula no longer overlaps images, note on screen;
  S03 arcs end at the SLM; S04 Gaussian sprites on 400² canvas (no square clipping); S05 2D Gaussian/spectrum
  shown as central crops of a 256² FFT (spectrum visible), equation labels under their terms, SLM/rec labels on screen;
  S06/S07/S08/S09 off-screen or colliding labels fixed; end title split in two lines; ≈ 6.3 min.
- v3 — captions 17 chars/s + 0.8 s; merged captions (S01, S03, S05, S06, S07); “excessive number of points” (paper's wording);
  reference wave back to 60 cycles/mm (90 aliased); higher-gamma LUT for the tilted-splat side view; ≈ 5.4 min.
- v4 — row labels share a common baseline (`label()` corrects for CMU descenders; “naïve sum” no longer sits low).

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif (text); Latin Modern Math for glyphs CMU lacks: × − → ≈ ∼ ∈ ℝ ≤ ↔ ∑ ∏ ∫ ℱ 𝒫 𝒢 ℋ ∗ ≠ ∞ ⇒ … (`MATHCH`) |
| Math | Pango markup, no LaTeX. Subscript `rise=-3000` + `<small><small>`; superscript `rise=4500`; complex exponents written `exp(…)` |
| Background | black |
| Colours | dim `#8A8A8A`, red `#FF5252` (failure labels), green `#69F0AE` (ours / good), blue `#448AFF`, amber `#FFCA28` (highlights, fast GWS), cyan `#4DD0E1`, purple `#E040FB` |
| Real wave fields | diverging LUT blue (−) · black (0) · amber (+), gamma 0.8 (`WLUT`); side view gamma 1.5 (`WLUT2`) |
| Complex fields | domain colouring: hue = phase, value = amplitude^0.6–0.7 (`domain_rgb`) |
| Phase-only maps | matplotlib `twilight` (cyclic), −π…π |
| Intensity | `inferno` (S01) / greyscale (reconstructions) |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 centred lines), max width 13.2, bottom ≥ −3.85; held ≥ len/17 + 0.8 s |
| Part card | “Part k” (30, dim) over title (50, bold, ≤ 12.5 wide), 1.5 s, instant |
| Row labels | `label()`: 22–26 pt, common baseline 0.15 above (or below) the image |
| Scene clock | `self.clock` ValueTracker advanced by dt every frame; drives all travelling-wave animations |

Optics constants of the toy simulations: λ = 520 nm (the prototype's green laser), pixel pitch 8 µm (the prototype's SLM),
angular spectrum method with evanescent waves removed (`asm`). Closed-form splat field `gws_field` = paper Eq. 10 with
on-axis plane-wave illumination (spectrum evaluated at S R⁻¹(f − f_r)), det J = k_oz / k_z, lateral shift + back-propagation phase.

## Scenes (durations from the 480p15 render of v4)

### S00_Title (8.5 s)
Title “Gaussian Wave Splatting” (66, bold, y 2.95), “for Computer-Generated Holography” (36, y 2.2).
Live field (height 3.5 at (0, −0.1), dim frame): Re{U e^(−j2π·0.9t)} of three coherent point sources at
(−3.2, −3.2), (0, 2.6), (3.2, −3.2), λ = 0.8 units (`three_U.npy`, 640×280). Authors (24, dim, y −2.55),
“Stanford University · ACM Transactions on Graphics (SIGGRAPH 2025)” (22, y −3.0), “* equal contribution” (18, y −3.4).

### S01_Waves (22 s) — “1 · Light is a wave”
Axes 6.0×2.1 at (−3.5, 1.35) with travelling cos(x − 2π·0.5t) (amber), label u(x,t) = A cos(kx − ωt + φ).
Phasor: circle r 0.85 at (−5.3, −1.45), rotating arrow; “u = A e^{jφ}” (34) + amplitude / phase notes at x −2.6.
Right: two coherent sources (`two_U.npy`, 480×270, sources (±1.4, −4.9), λ 0.9), height 3.6 at (3.55, 0.55), live;
then crossfade to intensity |u|² (inferno). Captions: wave with amplitude & phase · complex number, arrow spins ·
two sources add · camera records only |u|², phase lost, holography keeps it.

### S02_Hologram (37 s) — “2 · What a hologram does”
A. Recording (2 µm samples, 360²): object wave O = point at 5 mm (Re part), tilted reference R (60 cycles/mm),
recorded |O + R|² (off-axis zone plate); panels 2.6 tall at x −4.3 / 0 / 4.3, y 0.75, “+” and “=”.
Equation |O + R|² = |O|² + |R|² + O R* + O* R (34, y −1.55, cross terms amber) + note “the cross terms carry the phase of O”.
B. Digital: hologram phase (`holo_phase.png`, twilight) 3.3 tall at (−4.35, 0.6); arrow with
𝒫(u; z) = ℱ⁻¹{ℱ{u} · exp(j k_z z)} (20) / “angular spectrum method”; live reconstruction |𝒫(H; z)|² (3.3 at (4.35, 0.6))
interpolated from 91 precomputed frames z = 0…45 mm (`focus_frames.npy`). Scene: star outline at 12 mm, “3D” (CMU bold) at 36 mm,
H = 𝒫(A₁; −12 mm) + 𝒫(A₂; −36 mm) (512², complex, smooth phase). z ruler (NumberLine 0–45, length 7, y −2.35) with
markers (star amber @12, “3D” cyan @36), moving dot, “z = … mm” readout. Sweep 0→45 (5 s) →12 (2 s) →36 (2.5 s) →24 →36.
Note “toy simulation: λ = 520 nm, 8 µm pixels” under the hologram.

### S03_WhyHard (35 s) — “3 · Why photoreal 3D holograms are hard”
SLM bar at x −6.1; three coloured points (−3.6,1.35) red, (−2.4,0.2) green, (−4.3,−0.8) blue; 3 arcs each (half-angle 0.75 rad),
last arc touches the SLM. Arrow → computed phase of the sum of 3 spherical waves (z = 3, 4, 5.5 mm; 3.4 tall at (3.4, 0.45)).
Then Fig. 5 target renderings (lego) 2.75 tall at x −4.5 / 0 / 4.5: point cloud (“holes; poor occlusion”, red),
mesh per-face colours (“overly smooth; many triangles”, red), Gaussian splats (“photoreal, from a few photos”, green).

### S04_Gaussians (25 s) — “4 · Scenes made of Gaussians”
One 2D Gaussian (`g_one`, 3.6 at (−3.9, 0.75)) with tangent arrows t_u (amber, 95 px) and t_v (red, 40 px), labels s_u, s_v, μ;
parameter lines (22) at y −1.55 / −2.0. Three RGBA Gaussians (blue far, amber, red near; 2.6 each) fan out, then slide
together to (3.65, 1.0) — Manim's alpha “over” compositing *is* front-to-back blending. Eq. 1 (30, y −1.2) and Eq. 2 (26, y −1.9).

### S05_Transform (57 s) — “5 · One Gaussian, one hologram, in closed form”
1. 1D pair: Axes 5.4×2.2 at (∓3.5, 1.05); cyan Gaussian, amber |FFT| computed live with numpy (σ: 1 → 0.45 → 1.7 → 1).
   Eq. 7: ℱ{exp(−(x² + y²)/2)} = 2π exp(−2π² kᵀk) (32, y −1.05).
2. 2D: live 256² Gaussian (σ 9 px; crops: space 96², spectrum 48² of the FFT), 3.0 tall at (∓3.4, 0.95), spectrum in domain colouring.
   Aspect 1 → 2.2, rotation 0 → 60°, shift μ 0 → 22 → −14 px (phase ramp appears). x = R S x_c + μ and Σ = R S Sᵀ Rᵀ (30, y −1.2).
3. Tilt: x–z side view (`tilted_U.npy`, 420×210, z 0–40 λ, x ±10 λ): plane wave up to a disk tilted 35° (Gaussian σ = 3 λ,
   drawn with brightness = amplitude), Huygens sum of 240 wavelets beyond it (obliquity factor), width 9.6 at (−0.4, 0.45), SLM bar right.
4. Equation (40, y 2.2) built term by term with labels under each term: Û_i(k) = 2π det(J) [tilt Jacobian, amber]
   det(S_i) [scale, green] exp(−2π² kᵀ Σ_i k) [Gaussian shape, cyan] exp(j k·μ_i) [position and depth, purple].
   Below: closed-form SLM field of one splat (40 × 8 µm, rotations 0/45°/30°, depth 40 mm; 256² crop, 2.7 tall at (−3, −0.95))
   → 𝒫(·; z_i) → refocused splat (60² crop, “4× zoom”, 2.7 at (3, −0.95)).

### S06_Occlusion (41 s) — “6 · Occlusion: alpha wave blending”
Rays: Eq. 1 (28, dim, y 2.75); Eq. 5 (30, y 1.95); Eq. 6 (30, y 1.25).
Toy (256², σ 34 / 42 px, red at 6 mm in front of blue at 10 mm, o = 1): naïve sum · binary silhouette (T = 1 − [α > 0.1]) ·
alpha wave blending (T = 1 − o|u|) · ray-traced target, each channel propagated to the SLM and back to its own depth;
2.3 tall at x −4.95 / −1.65 / 1.65 / 4.95, y −0.75; note (18, y −2.1).
Paper Fig. 3 (stump): naïve / silhouette (binary mask) / alpha wave blending / target, width 3.15 at y 0.7.

### S07_Fast (38 s) — “7 · Making it fast: splat in the Fourier domain”
Eq. 11 (30, y 2.6, “∗ ℱ(T_i)” red) + red note; “T_i ≈ 1 ⇒” Eq. 12 (30, y 1.35).
Two 10×10 frequency grids (2.5, centres x ∓3.6, y −1.1): exact fills sequentially with a “Gaussian n of many” counter;
fast lights every cell at once (amber) — “one CUDA thread per frequency”.
Eq. 13 (OIT, McGuire & Bavoil 2013) and view-dependent opacity α_i = 𝒢_i · SH(o_i, d) over paper Fig. 2 crops (width 5.4).
Table 2 (24 pt; rows y 2.3 − 0.58 r; fast rows amber): Chen & Wilkinson 2.47 s · Matsushima 13.5 s · exact 180 s / 470 s · fast 6.84 s / 18.06 s.

### S08_Results (39 s) — “8 · Results on a real holographic display”
Setup photo (supplement, 4.6 tall at (−3.6, 0.35)) + specs: lasers 638/520/488 nm (FISBA READYBeam) · Holoeye PLUTO phase-only LCoS ·
1920 × 1080, 8 µm · FLIR Grasshopper3 · double-phase coding · depth range ≈ 1 cm.
Fig. 5 lego captured front focal slice: point / polygon / fast GWS / exact GWS (width 2.95 at x −3.95, −0.85, 2.25, 5.35, y 1.45);
Table 1 PSNR rows Blender 18.34 / 20.79 / 26.55 / 27.30 and Mip-NeRF 360 9.45 / 15.77 / 20.70 / 22.37 (28, y −0.95 / −1.6).
Close-up Fig. 5 materials: point vs exact GWS (3.9 tall at (∓3.4, 0.35)).
Fig. 6 garden (exact GWS, captured): big 4.1-tall image at (−1.9, 0.3) crossfading focus slices; 3 viewpoint thumbnails
(1.45 tall at x 4.6) with amber selector; order v0 rear→mid→front, v1 front→rear, v2 rear→front.

### S09_Outlook (20 s) — “9 · Limitations and outlook”
Three limitations (26) with amber “→” directions (24): smooth phase → random-phase GWS; not real time → closed-form tricks,
neural networks, custom hardware; small étendue, 0th-order SH → étendue-expanded partially coherent displays.
Summary: 3 numbered contributions (28) + “5–11 dB PSNR over point / polygon CGH · fast variant nearly 30× faster than exact” (amber).
End card (no header): title (48 bold) / subtitle (32), arXiv 2505.06582 · bchao1.github.io/gaussian-wave-splatting,
code github.com/computational-imaging/hsplat, authors, “Stanford University · ACM TOG 44(4), SIGGRAPH 2025”, held 4 s.

## Accuracy notes
- Every number on screen is from the paper (Tables 1–2, Sec. 5, supplement hardware section) or is a parameter of our own toy simulation (labelled as such).
- Table 2 ratio 180 / 6.84 ≈ 26×; the paper describes it as “nearly 30×”, and the video quotes that wording next to the raw numbers.
- The toy holograms are complex (smooth-phase) fields; the paper encodes its complex SLM field to phase-only with double-phase coding (stated in S08).
