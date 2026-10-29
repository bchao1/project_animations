# [chao2024etendue] Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation

Brian Chao, Manu Gopakumar, Suyeon Choi (Stanford), Jonghyun Kim (NVIDIA), Liang Shi (MIT), Gordon Wetzstein (Stanford)
SIGGRAPH Asia 2024 Conference Papers (Tokyo, Dec 3–6 2024), DOI 10.1145/3680528.3687600 · arXiv 2409.03143 (cs.GR) · 12 pp., 7 figures
Project page: https://bchao1.github.io/holo_dfm/
PDF: `../pdfs/chao2024etendue.pdf` · text: `../pdfs/chao2024etendue.txt` · LaTeX source: `../pdfs/src/` · figure PNGs: `../pdfs/figs/`

## Problem
- Étendue of a holographic display is fixed by the SLM pixel count → direct tradeoff between field of view (FoV) and eyebox.
  A 1080p SLM supports e.g. ~80° FoV with an eyebox < 1 mm, or vice versa (Related Work).
- Increasing SLM pixel count to holographic-film feature sizes (tens of nm) is infeasible.
- Existing étendue expansion:
  - post-SLM: static high-resolution (scattering/phase) masks → insufficient degrees of freedom → low contrast, speckle;
    pupil replication (waveguides/HOEs) → copies of the same wavefront, no 3D parallax; HDOs + pupil optimization → slight expansion; lens arrays.
  - pre-SLM: steered illumination (MEMS, sequential LD arrays, tilting cascades) → needs very high-speed SLMs, HDOs hurt quality;
    simultaneous multi-source + fixed random Fourier binary mask (Jo et al. 2022) → not content adaptive, no time multiplexing.
- With multiple sources, the HDOs of one source intermingle with the shifted spectrum of another in the Fourier domain → cannot be separated by a fixed filter; must be modeled.

## Method
### Optical system analysis (Sec. 4)
- Étendue: G = 4A sin²θ (A display area, 2θ emission cone). Conserved through reflection, refraction, free-space propagation.
- SLM diffraction angle for pixel pitch p: θ_SLM = ± sin⁻¹(λ / 2p).
- G_SLM = 4 L_x L_y sin²θ_SLM = λ² N_x N_y → étendue ∝ number of SLM pixels.
- Fresnel holography: FoV = 2 tan⁻¹(L / 2g) = 2 tan⁻¹(Np / 2g), eyebox w = gλ / p (g = eyepiece focal length).
  Paraxially FoV_x · FoV_y · w² ≈ λ² N_x N_y = G_SLM.
- α × α grid of directional illuminations matched to the HDO angles: 1D eyebox expands by α (w = αfλ/p), FoV unchanged → 2D étendue × α².
- Fig. 3: FoV vs eyebox iso-étendue curves for α = 1…5; assumes pitch 10.8 µm, 1000 × 1000 pixels, λ = 632.8 nm.

### Image formation (Sec. 3)
- Single source, ASM (Eq. 1): f(φ, z) = F⁻¹{ F{ e^{iφ(x,y)} u_src(x,y) } · H(f_x, f_y; z) },
  H = exp(i 2π/λ · z · sqrt(1 − (λf_x)² − (λf_y)²)) inside the band limit, 0 otherwise.
  Time-multiplexed: Σ_t |f(φ^(t), z)|² / T (Choi et al. 2022).
- Multi-source with Fourier modulation (Eq. 2): for source j with wave vector k^(j),
  f^(j)(φ, P, z) = F⁻¹{ U_slm^(j)(f_x, f_y; φ) · H(f_x, f_y; z) · P(f_x, f_y) },
  U_slm^(j) = F{ e^{iφ} u_src^(j) e^{i k^(j)·x} } — a tilted plane wave shifts the spectrum.
  Sources are mutually incoherent → intensities add over j. All sources on at all times (config V).
- Optimization (Eq. 3): minimize over {φ^(t), P^(t)}
  ‖ s · sqrt( (1/T) Σ_t Σ_j |H2LF( f^(j)(φ^(t), P^(t), z) )|² ) − lf_target ‖.
- Stochastic H2LF (Eq. 4): each iteration picks one view p; H2LF^(p)(f) = f^(j)(φ^(t), P^(t) · M^(p), z) with a binary circular
  pupil mask M^(p) in the Fourier plane (center (c_x,p, c_y,p), radius r_p). Avoids STFT memory blow-up.
- Implementation: 4-bit phase SLM quantization handled as in Choi et al. 2022; HDOs modeled as in Gopakumar et al. 2021; PyTorch.
  r_p = 2 mm (4 mm pupil); 81 pupils equally spaced in the Fourier (eyebox) plane → 9 × 9 light field.
  Illumination angles match the ±1st HDO angles of blue → blue spectrum copies tile the Fourier plane perfectly; red/green gaps removed.

## Baselines (Fig. 4, Table 1)
I single source + Fourier filter · II single source + phase mask · III multiple sources · IV multiple sources + fixed random Fourier mask (Jo 2022) ·
V ours: multiple sources + dynamic Fourier filter (amplitude SLM) · V* ours + laser amplitude control · VI steered illumination · VII steered + shifting Fourier filter (Lee 2020).
Simulation: 3 × 3 sources, 800 × 1280 phase SLM, 20 × 20 Fourier display (IV, V, V*), 9 × 9 target light field, 6 test scenes.

## Key numbers
Table 1 (simulation, PSNR / SSIM, averaged over 6 scenes, 9 × 9 light field unless noted):
- I: 46.23 / 0.97 (2D image) · 22.43 / 0.46 (3 × 3 LF) · 16.60 / 0.23 (9 × 9 LF)
- II: 14.61 / 0.15 · III: 13.35 / 0.24 · IV: 18.63 / 0.26
- V (ours): 21.05 / 0.39 (1 frame) · 24.65 / 0.66 (6 frames) · 25.17 / 0.71 (9 frames)
- V*: 21.43 / 0.42 · 25.02 / 0.69 · 25.66 / 0.74
- VI: 20.02 / 0.43 (9 frames) · VII: 23.95 / 0.51 (9 frames)
- Ours beats VII with fewer frames (6 vs 9); V* gain < 0.5 dB, not worth complexity.
Table 2 (experimental, PSNR / SSIM): III 12.72 / 0.19 · IV 12.46 / 0.17 · V 13.83 / 0.23 (1 frame) · 14.43 / 0.40 (6 frames).
Fig. 6 (one scene, per-row labels): I 16.13/0.25 · II 12.62/0.14 · III 11.28/0.22 · IV 18.27/0.28 · VI 18.28/0.43 · VII 25.07/0.56 · V 1 frame 21.32/0.42 · V 6 frames 26.01/0.75.

Prototype (Sec. 5): cascaded 1:4 fiber splitters (Thorlabs TWQ560HA) → 9 fiber tips in a 3 × 3 array (3D-printed mount), spacing 8.17 mm,
200 mm collimating lens → 2.34° incidence per source; TI DLP6750Q1EVM phase SLM; 1080p SiliconMicroDisplay LCoS amplitude SLM at the Fourier plane;
75 mm Fourier transform lens; diagonal FoV 7.78°; eyebox 8.53 mm × 8.53 mm; FLIR Grasshopper 2.3 MP + Canon EF 50 mm f/1.4.
Capture: Thorlabs SM1D12 iris on a translation stage at the Fourier plane mimics pupil movement.

## Limitations / future work
Benchtop; miniaturize sources with nanophotonic phased arrays, fold with waveguides. Frame rate limited by 240 Hz amplitude display
→ ~13.33 Hz in color-sequential 6-frame mode (LCoS > 720 Hz would help). No real-time synthesis (neural CGH for 4D LF is future work).
No calibrated neural wave propagation model.

## Figures
1. Teaser: 3 × 3 fiber source array photo (a); captured Fourier spectra single vs multi-source with red/blue pupils (b, e); center/right views (c, d, f, g).
2. System architecture schematic (multiple sources → phase SLM → amplitude SLM at Fourier/eyebox plane).
3. FoV vs eyebox tradeoff, iso-étendue lines α = 1…5.
4. Baseline configurations I–VII schematics.
5. Prototype photograph with labeled optical path.
6. Simulated comparison of configurations (center view, rear/front focus, left/right views) with PSNR/SSIM.
7. Experimental focal stack + parallax comparison (I, III, IV, ours 1 frame, ours 6 frames).
