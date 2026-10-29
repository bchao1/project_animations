# Storyboard — Fast Wave-optics Rendering of Multiplane Images teaser (3b1b style)

**Current version: v4** — `animation_mpi_v4.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
`render.sh` first imports the script once (builds the cached assets), then renders the 10 scenes in parallel
(logs in `media/log_<script>_<res>_<Scene>.txt`) and concatenates them.
Individual scene MP4s: `media/videos/animation_mpi_v4/<res>/S0k_*.mp4`.
Final cut: `media/final/MPIHolography_animation_mpi_v4_1080p60.mp4` (10 scenes, 1920×1080 @ 60 fps, 5 min 20.5 s = 320.5 s;
scenes 7.8 · 28.2 · 37.7 · 34.5 · 40.5 · 39.9 · 25.0 · 40.1 · 47.8 · 19.0 s). 480p15 test cut (v3): 5 min 23.2 s.

Source: Chao, Seyb, Matsuda, Cossairt, Zhou, Lanman, Wetzstein, Kuo, Jang, *Fast Wave-optics Rendering of Multiplane Images
for 3D Holographic Displays*, arXiv 2607.19731 (v1, 22 Jul 2026); Stanford University · Meta. No project page; the author
homepage (https://bchao1.github.io/) has two teaser videos (`data/videos/mpi_light_field.mp4`, `mpi_focal_stack.mp4`).
PDF, text, LaTeX source, figure rasters: `mpi-holography/pdfs/`; ingest notes with every number used:
`mpi-holography/summaries/chao2026fast.md`. Direct follow-up of RPWS (`../random_phase_wave_splatting/`).
`data/*.png` are crops of the paper's figures (made by `scratch/crop_figs.py` from `pdfs/figs/*_hi.png`, rendered from the
LaTeX-source figure PDFs). `data/clips/{lf,fs}.npy` = all frames of the two homepage videos at 960×640 (ffmpeg), played as
live ImageMobjects (ping-pong loop at 25 fps, driven by the scene clock: `video()`). `data/generated/` is built on import by
`build_assets()` (flag `done_v1`; file-locked; ≈ 3 min).

## Version history
- v1 — first full cut, 10 scenes, 5 min 57 s (480p15). QA: title / author lines wider than the frame (S00, S09 end card);
  S02 arrow labels (“× exp(…)”, “inverse FFT”) overlapped the images; S03 sheet labels overlapped each other and the
  (c_i, α_i, d_i) line; S07 source note on top of the x-axis title, “≈ 30 layers” on top of the plot titles; S08 Fig. 1 panels
  overlapped each other and the caption; too long.
- v2 — `fit()` for long lines; ASM arrows labelled FFT / × H / FFT⁻¹ above the arrows, H defined in the equation; sheet labels
  = layer numbers + one legend line; `src_note()` puts source notes top right on the header line; Fig. 1 2×2 at width 4.3;
  shorter sweeps; 5 min 42 s.
- v3 — 24 captions shortened (hold time = len/17 + 0.8 s dominated the pacing); 480p15: 5 min 23.2 s. Rendered 1080p60;
  HD check: the S07 runtime plot drew its x-axis line at 10⁰ (manim puts the x-axis at y = 0 in data units).
- v4 — S07 runtime axis offset by +2 decades internally (y range 0…7 ↔ 10⁻² … 10⁵ s) so the axis sits at the bottom.
  Rendered 1080p60: 5 min 20.5 s.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif (text); Latin Modern Math for glyphs CMU lacks (× − → ≈ ∑ ℱ 𝒫 𝔼 △ ← ✓ …, `MATHCH`); DejaVu Sans for ✗ (`SYMCH`) |
| Math | Pango markup, no LaTeX. `sub()` rise −3000 + `<small><small>`; `sup()` rise 4500; `supsub()` keeps a subscript inside a superscript |
| Background | black |
| Colours | dim `#8A8A8A`, red `#FF5252` (RGBD), green `#69F0AE` (MPI, ours), orange `#FF9E40` (RPWS), amber `#FFCA28` (highlights, headline), cyan `#4DD0E1` (pupil / camera), blue `#448AFF` |
| Toy images | RGB intensity (3 wavelengths), displayed with gamma 1/2 |
| Complex fields | domain colouring: hue = phase, value = amplitude^γ (`domain_rgb`) |
| Spectra | log10 |FFT|², 7 decades, `magma`, flat random spectrum capped at 0.62 of full scale; transfer function phase in `twilight` |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title”; source notes top right on the same line (16, dim, `src_note`) |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 centred lines), max width 13.2, bottom ≥ −3.85; held ≥ len/17 + 0.8 s; `hold()` waits for it before the next visual change |
| Part card | “Part k” (30, dim) over title (50, bold, ≤ 12.5 wide), 1.5 s, instant |
| Row labels | `label()`: 20–26 pt, common baseline 0.15 above (or below) the image |
| Toy notes | 16 pt dim, “toy simulation: …” under every simulated panel |

## Toy wave-optics simulations (all numpy, cached in data/generated)
8 µm pixels (paper SLM), RGB lasers 638 / 520 / 450 nm, N = 512 grid (352 × 352 shown), angular spectrum method (`asm`,
scipy.fft). Toy MPI (`toy_mpi`), colour = target intensity: layer 1 soft-edged red star (α ≤ 0.97) at 4 mm, layer 2 green
triangle at 9 mm, layer 3 yellow disc at 14 mm, layer 4 blue checkerboard wall (28-px squares) at 20 mm from the SLM.
- **MPI hologram** `mpi_slm` = paper Eq. 2: from layer 4 to the SLM, u ← 𝒫(√(1 − α)u + √(cα)e^{iφ}; −Δz), φ iid 𝒰(−π, π) per
  pixel, per layer, per frame and per colour.
- **RGBD hologram** `rgbd_slm`: RGB = ray-traced ‘over’ composite of the MPI, depth = front-most layer with α > 0.5; 4 hard
  depth slices √(RGB)·mask·e^{iφ}, each propagated to the SLM and summed (conventional layer-based CGH).
- Eye: pupil in angle space (same physical pupil for all colours) in units of blue's max angle λ_b/(2·8 µm). Focal stacks:
  radius 0.5, 45 planes 2–24 mm. Light-field views: radius 0.25, centre over ±0.7 (29 positions), focus 4 mm.
  Intensities normalised by the flat random-phase spectrum through the pupil (`pupil_ref`) — no per-image auto-exposure.
- T = 24 frames (paper setting). Per-plane running-wave intensities (S05) averaged over the 24 frames, full bandwidth.
- Single-plane propagation sweep (S02): the RGB composite, smooth (constant) phase vs one random phase map, z 0–20 mm, full band.
- Ray-optics MPI novel views (S03): layers shifted by 30 / 19 / 9 / 0 px × camera shift, composited with ‘over’.

## Scenes (durations from the 480p15 render of v3; v4 differs only in the S07 axis)

### S00_Title (7.8 s)
Title (50 bold, fit 13, y 3.0) + “for 3D Holographic Displays” (36, y 2.3); homepage videos playing, 3.0 tall at (∓2.45, −0.2):
“light field” | “focal stack”, note “MPI holograms (paper teaser videos)” (y 1.55); 9 authors (20 dim, fit, y −2.6);
“Stanford University · Meta · arXiv 2607.19731 (2026)” (22, y −3.05).

### S01_Hook (28.5 s) — “1 · Holograms need fast 3D content”
Focal-stack and light-field videos 3.3 tall at (∓3.2, 0.7), labels “refocus” / “look around”. Captions: display rebuilds the
wave (refocus, look around) · a computer must first compute it: CGH. Then paper Table 1 re-typeset: rows Point cloud △✗✗,
RGBD ✓△△, Focal stack △✓△, Light field △✓✓, Mesh ✗✗✗, Gaussians ✗✓✓ appear one by one (0.3 s), then “MPIs (this paper)” ✓✓✓
(green, amber box). Columns fast / focal stack / light field.

### S02_Waves (38.0 s) — “2 · Propagating a plane of light”
A. Live propagation sweep 0 → 20 → 10 mm: smooth phase (ringing, stays put) vs random phase (spreads into speckle), 3.2 tall at
(∓2.4, 0.75), readout “propagation distance z = … mm”. B. ASM chain 2.3 tall at x −5.1 / −1.7 / 1.7 / 5.1: plane u (domain
colour) → FFT → flat spectrum → × H (transfer-function phase rings, z = 10 mm, green) → FFT⁻¹ → 𝒫(u; z); equation
𝒫(u; z) = ℱ⁻¹{ℱ{u}·H}, H = exp(i2πf_z z), f_z = √(1/λ² − f_x² − f_y²). C. Cost: 700 coloured dots “millions of primitives, one
wavefront each” (primitive-based CGH, orange) vs 4 sheared toy layers “a handful of layers (typically fewer than 100)” (green).

### S03_MPI (34.8 s) — “3 · Multiplane images”
A. Toy composite (3.0) → arrow → 4 pseudo-3D sheared RGBA layers (pre-sheared in PIL; Cairo cannot shear ImageMobjects) with
outlines, numbered 4 … 1; legend “layer i = 1 (front) … 4 (back): colour c_i, opacity α_i, depth d_i”.
B. Ray-optics novel view (3.4 at (3.2, 0.75)) live with a camera shift −1 → 1 → −1 → 0; layer tiles 2 × 2 on the left;
C ← α c + (1 − α) C, i = L … 1 (“over”). C. Paper Fig. 2a MPI (width 6.4) + bullets: networks from a stereo pair / a single
photo / binned from Gaussian splats (this paper) / amber “typically fewer than 100 layers”; small-eyebox caption.

### S04_RGBD (40.7 s) — “4 · Why not RGBD layers?”
A. RGB + depth (viridis) tiles; row of 4 RGBD slices (red, holes behind occluders) vs 4 MPI layers (green, complete).
B. Live light-field views, RGBD | MPI holograms (24 frames, 3.3 tall at ∓3.4), eyebox circle with cyan pupil sliding
0 → −0.7 → 0.7 → −0.7 → 0: RGBD opens dark gaps at occlusion edges, MPI shows the hidden content.
C. Paper Fig. 4 zoom-ins (width 2.5): ground truth / RPWS / RGBD 4 / RGBD 512 / MPI 4; top defocused garden edge, bottom
bicycle light-field view.

### S05_Compositing (40.2 s) — “5 · Wave compositing, back to front”
Eq. 2 at top (32). A. Five 2.05-tall panels (layer 4 → 3 → 2 → 1 → SLM, x −5.45 … 5.45) of the running-wave intensity
(24-frame average) appear one by one with amber 𝒫 arrows; op notes “√(c₄α₄) e^{iφ}”, “× √(1 − α) + own wave”, “hologram”.
B. ray optics C ← αc + (1 − α)C · wave optics u ← √(1 − α)u + √(αc)e^{iφ} · average over φ: 𝔼|u|² = (1 − α)|u|² + αc
(amber “cross term → 0”); note: random-phase alpha blending from RPWS (Chao et al. 2025).
C. One SLM frame (domain colour, green channel) + its flat angular spectrum: fills the bandwidth → wide eyebox.

### S06_TimeMux (25.3 s) — “6 · Time multiplexing and focus”
Eq. 3 I = (1/T) Σ_t |𝒪(u_SLM^(t))|². Frame k vs average of k frames (1 → 24; 3.0 tall at ∓2.2). Then live focal sweep
4 → 20 → 4 → 12 mm of RGBD (24 frames) / MPI (1 frame) / MPI (24 frames), 3.3 tall at x −4.5 / 0 / 4.5; ruler 2–24 mm with the
4 layer markers (star 4, triangle 9, disc 14, wall 20). RGBD shows the star's silhouette cut into in-focus back layers.

### S07_Speed (40.4 s) — “7 · Cost: layers, not primitives”
A. Runtime log plot (7.6 × 4.6), layers 4 … 512 (log2 axis), 10⁻² … 10⁵ s: MPI (green) and RGBD (red dashed, “same”) from
0.05 s to 7.2 s; RPWS dashed at ≈ 1.4 × 10⁴ s (≈ 4 h); amber double arrows “up to 250,000×” (4 layers, “4 layers: 0.05 s”) and
“≈ 2000×” (512 layers). B. Light-field PSNR (21.0–23.75 dB) and SSIM (0.45–0.53) vs layers: RGBD low, MPI above at every
count, RPWS dashed; amber “≈ 30 layers” line. All values read from paper Fig. 3 (vector data; see summary).

### S08_Results (48.4 s) — “8 · Results”
1. Fig. 1 2×2 (width 4.3): RGBD front (light leakage ✗), RPWS front (long generation time ✗), MPI back, MPI front.
2. Fig. 7 simulated lego (width 2.65): GT / RPWS / RGBD 4 / MPI 4 × front / back focus.
3. Fig. 8 captured garden (width 3.7): RPWS / RGBD 4 / MPI 4, front focus + zoom-ins.
4. Fig. 5 captured light field: front-focus and back-focus panels cycle left → centre → right → centre (2.2 Hz), “pupil: …”.
5. Fig. 6 Stereo Magnification MPIs → holograms, 3 scenes × back / front focus.
6. Homepage teaser videos playing (3.6 tall): light field | focal stack.

### S09_Outlook (19.1 s) — “9 · Limitations and outlook”
Up to 250,000× faster but not real time → real-time random-phase hologram networks; random phase lowers contrast →
learning-based calibration. Summary 1–3 + amber “quality close to RPWS · far better than RGBD · up to 250,000× faster than RPWS”.
End card: title, subtitle, arXiv 2607.19731, 9 authors, Stanford University · Meta; held 4 s.

## Accuracy notes
- On-screen numbers from the paper: 250,000× (abstract / text, vs 4-layer MPI), “more than 2000×” (Fig. 3 caption, shown as
  ≈ 2000× at 512 layers), “typically fewer than 100 layers”, “roughly 30 layers”, 4 / 64 / 512 layer labels, T = 24 frames,
  8 µm SLM pitch, Table 1 symbols. Runtime / PSNR / SSIM curves and 0.05 s, ≈ 7 s, ≈ 1.4 × 10⁴ s (≈ 4 h) are read from the
  vector data of Fig. 3 (Mip-NeRF 360 average; table in the summary) and labelled “values read from paper Fig. 3”.
- The ‘over’-on-intensity identity in S05 is our one-line restatement of RPWS's random-phase alpha blending, labelled as such.
- The RGBD toy baseline is plain summation of depth slices; the paper does not detail its RGBD compositing.
- The homepage videos are not labelled simulated or captured by the source, so the video only calls them teaser videos.
- The paper's supplement (pseudocode, contrast discussion, hardware details) is not public; not used.
