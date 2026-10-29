# [chao2026fast] Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays

Brian Chao (Stanford University and Meta; corresponding author), Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou,
Douglas Lanman (Meta), Gordon Wetzstein (Stanford University), Grace Kuo, Changwon Jang (Meta).
arXiv 2607.19731 (cs.GR), v1 22 Jul 2026, CC BY 4.0. Author homepage: "arXiv, 2026"; one-line blurb "An efficient rendering
algorithm for multiplane images (MPIs) that enables fast 3D hologram generation for next-generation holographic displays."
No project page, no supplement on arXiv ("Please refer to the supplementary materials" for pseudocode, hardware details,
contrast discussion; not available). Code: "Source code and example datasets will be made public."
PDF (11 pp.): `../pdfs/chao2026fast.pdf` · text: `../pdfs/chao2026fast.txt` · LaTeX: `../pdfs/src/` ·
figure rasters: `../pdfs/figs/*_hi.png` (PyMuPDF from the source figure PDFs, long side 5200 px).
Homepage teaser videos (downloaded to `../../data/videos/`): `mpi_light_field.mp4` (1536×1024, 60 frames @ 60 fps, 1.0 s;
garden-table scene, horizontal parallax sweep) and `mpi_focal_stack.mp4` (1536×1024, 59 frames @ 25 fps, 2.36 s; focus sweep).
The homepage labels them "Light Field" and "Focal Stack"; it does not say whether they are simulated or captured (they look
like the simulated results of Fig. 1), so the video labels them only as the paper's teaser videos.

## Figure numbering (compiled PDF)
Fig. 1 teaser · Table 1 representation comparison · Fig. 2 pipeline · Fig. 3 runtime / quality plots (Mip-NeRF 360) ·
Fig. 4 depth-transition zoom-ins (simulated) · Fig. 5 captured light-field parallax at three focus depths · Fig. 6 Stereo
Magnification MPIs → holograms · Fig. 7 simulated focal stacks + light fields · Fig. 8 experimentally captured.

## Problem
Holographic near-eye displays reconstruct physically accurate wavefronts (true focus cues, parallax), but CGH is expensive.
Each 3D representation has a trade-off (Table 1, ✓ / △ / ✗ for efficient inference · focal-stack quality · light-field quality):
Point cloud △ ✗ ✗ · RGBD ✓ △ △ · Focal stack △ ✓ △ · Light field △ ✓ ✓ · Mesh ✗ ✗ ✗ · Gaussians ✗ ✓ ✓ · **MPIs ✓ ✓ ✓**.
- RGBD images: fast but occlusion artifacts ("light leakage and halo artifacts in defocus regions, dark occlusion boundaries
  in light field views", "depth-tearing" near depth discontinuities); more RGBD layers do not fix it.
- Mesh / point-cloud / Gaussian pipelines "must process millions of primitives" (primitive wavefronts propagated and
  composited one by one) → RPWS is state-of-the-art quality but "extremely slow runtime".
- Dense light fields: content rarely produced in casual capture.
- Optimization-based CGH: "tens to hundreds of wave propagation steps and gradient updates per frame".

## Key idea
Use **multiplane images** (Zhou et al. 2018): a set of L RGBA images (c_i ∈ ℝ^{H×W×3}, α_i ∈ ℝ^{H×W}) at depths d_i,
fronto-parallel to a reference camera. Novel views: warp each layer by a planar homography, alpha-composite back to front
with Porter–Duff **over**. MPIs are ideal for near-eye holography: only *local, front-facing* light fields are needed
(small eyebox), and MPIs have only "a handful of RGBA images (<100)" ("typically fewer than 100") vs millions of primitives.
"Layered formats … require only a limited number of depth planes (typically fewer than 100) to be propagated, in contrast to
the thousands or millions of primitives processed in Gaussian- or polygon-based holography."

## Method (Sec. 3, Fig. 2: (a) MPI · (b) apply random phase, view space → hologram space · (c) sequential propagation and
compositing · (d) time multiplexing)
- Eq. 1: MPI = {(c_i, α_i)}, i = 1 … L.
- Eq. 2 (the algorithm), back to front from the farthest layer:
  u_{i−1}^{(t)}(x) = 𝒫( √(1 − α_i(x)) u_i(x) + √c_i √α_i(x) e^{iφ_i^{(t)}(x)} ; Δz ),  Δz = z_{i−1} − z_i,
  𝒫 = angular spectrum method (Goodman). Random-phase alpha blending from RPWS (chao2025rpws). Recursion ends at the
  display plane z_0 = 0: u_SLM^{(t)} = u_0^{(t)}. A fresh random phase φ_i^{(t)} per layer and per frame t.
- Eq. 3: I(x) = (1/T) Σ_t |𝒪(u_SLM^{(t)}(x); ·)|², 𝒪 = free-space propagation (2D image / focal stack), STFT (4D light field,
  Choi et al. 2022) or pupil masks (specific views, Schiffers et al. 2023).
- Why it is right (from RPWS, restated in our video): with independent uniform random phases the cross term averages out
  over frames, so the time-averaged intensity after the step obeys E|√(1−α)u + √(αc)e^{iφ}|² = (1 − α)|u|² + αc — the
  **over** operator applied to intensity. (Our derivation, a one-line consequence of RPWS Eqs. 9–14; labelled as such.)
- The composited hologram's phase is random (Fig. 1 colour-coded phase map): expands the eyebox, uses the SLM's
  space–bandwidth product, natural defocus, accurate parallax.
- Cost: one angular-spectrum propagation (FFT → multiply → inverse FFT) per layer per frame → runtime ∝ number of layers,
  "nearly identical for both MPI and RGBD".

## Data / baselines (Sec. 4.1)
- Scenes: NeRF Blender synthetic, Mip-NeRF 360. Fig. 3 averages "selected scenes of the Mip-NeRF 360 dataset".
- MPIs extracted from pre-optimized Gaussian splats by binding Gaussians to predetermined depth planes (Eq. 4:
  c_{d_k} = Σ α_i c_i Π(1 − α_j) / (1 − Π(1 − α_i)), α_{d_k} = 1 − Π(1 − α_i)); planes uniformly distributed in
  (z_min, z_max). Lets the layer count be varied freely. Method is "agnostic of the MPI generation process" (Fig. 6: MPIs from
  Stereo Magnification network).
- RGBD baseline: DepthAnything v2 depth, same uniform depth bins → depth masks and masked RGB images (layer-based CGH,
  Peng 2020 / Shi 2021 / 2022).
- RPWS baseline: 2DGS via gsplat, Gaussians optimized on the squared image, sqrt of colours for CGH.
- **T = 24 frames** time multiplexing for all baselines. Layer counts shown: **4, 64, 512** (Figs. 4, 7, 8); Fig. 3 sweeps
  **4 … 512** (2², …, 2⁹).
- PyTorch implementation.
- Hardware: HOLOEYE Pluto-2.1 phase-only SLM, 1080p, **8 µm** pixel pitch; FISBA READYBeam Ind2 fiber-coupled laser;
  FLIR GS3-U3-123S6C-C colour camera on a Thorlabs MTS25-Z8 (50 mm) z-stage; pupil on a Newport CONEX-AGP stage at the
  Fourier plane (horizontal parallax / light fields). Phase-only holograms via SGD with complex-field loss at a fixed
  **4 cm** from the SLM.

## Key numbers (text)
- "up to **250,000×** speedup compared to state-of-the-art primitive-based methods" (abstract, intro, Fig. 7 caption: "(for 4
  MPI layers)", conclusion). Fig. 3 caption: "rendering up to 250,000× faster while matching RPWS in SSIM and surpassing
  RGBD in overall visual quality".
- RPWS is "more than **2000×** slower than layer-based approaches (MPI and RGBD)" (Fig. 3 caption).
- Runtime of MPI / RGBD "scales approximately linearly with the number of depth layers".
- "perceptual quality saturates beyond roughly **30 layers**" (Sec. 4.2).
- Fig. 4 caption: MPI holograms "reconstruct natural defocus blur and accurate occlusion effects using as few as **four**
  depth layers"; RGBD: more layers "does not alleviate these issues".
- MPIs: "<100" / "typically fewer than 100" layers.
- Limitations: "still not real-time" → combine with real-time random-phase hologram networks (Chu et al. 2025, "Artifact-
  resilient real-time holography"); random phase lowers contrast → learning-based calibration (Peng 2020, Choi 2022, Jang
  2024, Choi 2025).

## Fig. 3 plot values (read from the vector PDF `mpi_rgbd_metrics_lf_mipnerf.pdf`, marker centres mapped through the
axis tick positions; precision ≈ ±0.01 dB / ±0.0005 SSIM / ±1 % runtime). Light-field metrics, Mip-NeRF 360 scenes.
| layers | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| MPI PSNR (dB) | 22.47 | 22.59 | 22.64 | 22.68 | 22.72 | 22.74 | 22.74 | 22.72 |
| RGBD PSNR (dB) | 21.24 | 21.45 | 21.50 | 21.50 | 21.49 | 21.48 | 21.46 | 21.46 |
| MPI SSIM | 0.508 | 0.516 | 0.519 | 0.520 | 0.522 | 0.523 | 0.524 | 0.524 |
| RGBD SSIM | 0.453 | 0.467 | 0.471 | 0.472 | 0.472 | 0.473 | 0.474 | 0.476 |
| MPI runtime (s) | 0.050 | 0.110 | 0.220 | 0.441 | 0.878 | 1.78 | 3.61 | 7.20 |
| RGBD runtime (s) | 0.050 | 0.110 | 0.220 | 0.440 | 0.880 | 1.78 | 3.61 | 7.22 |
RPWS (dashed, one value): PSNR 23.50 dB, SSIM 0.521, runtime ≈ 1.41 × 10⁴ s (≈ 3.9 h).
Derived: RPWS / MPI runtime = 2.8 × 10⁵ at 4 layers (paper rounds to "up to 250,000×"), 1958 at 512 layers (paper: "more
than 2000×" — i.e. ≈ 2000×); MPI − RGBD PSNR ≈ +1.2 dB at every layer count; MPI SSIM ≥ RPWS from 64 layers on
(0.522 vs 0.521), "matching RPWS in SSIM"; MPI PSNR ≈ 0.8–1.0 dB below RPWS.
**On screen we use**: the recreated plots with these values (labelled "values read from paper Fig. 3"), "0.05 s" (MPI,
4 layers), "≈ 7 s" (512 layers), "≈ 4 hours" / "≈ 1.4 × 10⁴ s" (RPWS), "up to 250,000×", "more than 2000×", "~30 layers".

## Figures used on screen
Fig. 1 (teaser panels c–f + zooms; MPI stack illustration), Fig. 2 (pipeline panels a–d), Fig. 4 (zoom-ins: garden focal
slice, bicycle light-field view), Fig. 7 (lego front / back focus: GT, RPWS, RGBD 4, MPI 4), Fig. 8 (captured garden
front / back focus: RPWS, RGBD 4, MPI 4; zooms), Fig. 5 (captured light field, left / centre / right at front and back
focus), Fig. 6 (Stereo Magnification MPIs, back / front focus). Table 1 re-typeset.

## Toy simulations in the video (ours, numpy; labelled "toy simulation")
Parameters are ours, chosen to match the paper's hardware where possible: 8 µm pixels (paper SLM), RGB lasers 638 / 520 /
450 nm (ours), N = 512 grid, 4 MPI layers at 4 / 9 / 14 / 20 mm from the SLM (ours), T = 24 frames (paper), angular spectrum
propagation, Eq. 2 compositing for MPI, depth-sliced summation for RGBD. Eye pupil in angle space.

## Accuracy notes
- The paper gives no per-scene runtimes in text; all runtime values come from Fig. 3 (log plot, read from vector data).
- "250,000×" is quoted exactly as the paper states it (vs 4-layer MPI). We never claim real time.
- The RGBD toy uses plain summation of depth-sliced random-phase layers (conventional layer-based CGH); the paper does not
  spell out its RGBD compositing, only that it uses depth masks and masked RGB images.
