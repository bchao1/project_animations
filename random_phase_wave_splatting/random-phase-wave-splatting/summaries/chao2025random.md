# [chao2025random] Random-phase Wave Splatting of Translucent Primitives for Computer-generated Holography

Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein
(Stanford University; Koiso also KDDI Research; * equal contribution)
arXiv 2508.17480 (cs.GR). v1 24 Aug 2025 as "Random-phase Gaussian Wave Splatting for Computer-generated Holography"
(method then called GWS-RP); v3 11 Dec 2025 (current title, method RPWS). No journal-ref on arXiv; author homepage: "arXiv, 2025".
PDF (v3, 11 pp., main paper only): `../pdfs/chao2025random.pdf` · text: `../pdfs/chao2025random.txt` ·
LaTeX source: `../pdfs/src/` · figure rasters: `../pdfs/figs/` · v1 HTML text (had hardware + analysis sections now in the
supplement): `../pdfs/chao2025random_v1_html.txt`. Supplement: not on arXiv. Code: "will be made public".

## Problem
Conventional CGH (incl. Gaussian Wave Splatting, GWS) produces **smooth-phase** holograms: sharp in focus, but energy sits in
the low frequencies of the angular spectrum → SLM space–bandwidth product under-used, **small eyebox**, little parallax,
large depth of field (little blur change with depth), unnatural coherent defocus with **ringing**. Natural reflectance needs
**random phase**. But (i) GWS only handles Gaussians, and (ii) naively putting random phase on each primitive and running
GWS (**GWS-NRP**, suggested in the GWS supplement) gives **wrong colours** and broken defocus/parallax.

Intro Table 1 (limitations): GWS — large eyebox ✗, correct colour ✓, defocus & parallax △, other primitives ✗;
GWS-NRP — ✓, ✗, △, ✗; RPWS — ✓ ✓ ✓ ✓.

## Background (Sec. 3.1)
- Primitives: 2D Gaussians (mean μ_i, Σ_i = R_i S_i S_iᵀ R_iᵀ) or soft-edged triangles (vertices V_i, smoothness σ_i);
  both with opacity o_i and colour c_i. Sorted by depth from the SLM; "holographics" pipeline of GWS.
- GWS alpha wave blending (Eqs. 1–2): u_SLM(x) = Σ_i 𝒫( c_i o_i |u_i(x)| T_i(x) e^{ikz_i} ; −z_i ),
  T_i(x) = Π_{j<i} (1 − o_j |u_j(x)|). Phases matched (∠u_i = kz_i) → smooth-phase hologram.
- GWS-NRP (Eq. 3): same as Eq. 1 with e^{ikz_i + φ_i}, φ_i iid 𝒰(−π, π) — "provably fails".

## Method
### Random-phase wave splatting (Sec. 3.2, Eqs. 4–6)
Each primitive wavefront modulated by a random phase map φ_i^(t)(x) (scatters light off-axis); composited **back to front**
(far → SLM), T time-multiplexed frames with independent random phases:
- Eq. 4: g_{i−1}^(t)(x) = 𝒫( M_i(x) g_i(x) + √c_i √(o_i |u_i(x)|) e^{i∠u_i(x) + φ_i^(t)(x)} ; Δz ),  Δz = z_{i−1} − z_i
- Eq. 5: M_i(x) = √(1 − o_i |u_i(x)|)   (sqrt transmittance mask at the primitive's own plane)
- u_SLM^(t) = g_0^(t) (SLM at z_0 = 0). Like the Porter–Duff *over* operator / painter's algorithm, plus propagation and
  square roots, because primitives are optimized to reproduce **intensity** c² instead of amplitude.
- Eq. 6: I(x) = (1/T) Σ_t |𝒪(u_SLM^(t)(x); ·)|², 𝒪 = one propagation (2D image), several (focal stack), or STFT (4D light
  field, as in Choi et al. 2022).
- u_i can be any primitive's wavefront (Gaussian or translucent triangle) → plug-and-play.

### Random-phase alpha wave blending (Sec. 3.3, Eqs. 7–17)
Two wavefronts at the same depth, u_1 = α_1 e^{iφ_1} in front of u_2 = α_2 e^{iφ_2}, α = o·ℳ (opacity × falloff).
- Ray splatting (Eq. 7): c = α_1 c_1 + (1 − α_1) α_2 c_2.
- Blended wave (Eq. 8): u = w_1 e^{iφ_1} + w_2 e^{iφ_2}. Then (Eqs. 9–11)
  c² = |u|² = w_1² + w_2² + 2 w_1 w_2 cos(φ_1 − φ_2)  → depends on the phase difference (cross term).
- Smooth phase (φ_1 − φ_2 = 2kπ): |u| = w_1 + w_2 → w_1 = α_1 c_1, w_2 = (1 − α_1) α_2 c_2 (Eq. 12) = GWS (amplitude-linear).
- Random phase (φ iid 𝒰(−π, π)): E[cos(φ_1 − φ_2)] = 0 → E[|u|²] = w_1² + w_2² = c² (Eqs. 13–14): blending is
  **linear in intensity, in expectation**. Optimize primitives on intensity: c² = α_1′c_1′ + (1 − α_1′) α_2′ c_2′ (Eq. 15),
  w_1 = √(α_1′c_1′), w_2 = √((1 − α_1′) α_2′ c_2′) (Eq. 16), u = √(α_1′c_1′) e^{iφ_1} + √((1 − α_1′)α_2′c_2′) e^{iφ_2} (Eq. 17).
- **Time multiplexing is required**: the blend holds only in expectation; one realization has cross-term interference.
  Speckle reduction is a by-product. Supplement: uniform 2π phase is in a class achieving optimal O(1/√N) decay of cross terms.
- Toy (Fig. 2): orange Gaussian in front of blue; GWS-NRP error concentrated as a ring near the occlusion border (MSE colour
  bar up to 1.3e−2); RPWS sqrt blending: error map ≈ 0.

### Applying random phase (Sec. 3.4)
Adopts the heuristic from the GWS supplement; proves it exact with statistical optics; enables programmable depth of field.
v1 detail: convolve each primitive's angular spectrum with a random complex kernel Q(k) e^{jφ^(t)(k)} ("structured random
phase"); Q = 1 fully diffuse, Q = circular pupil P(k, r) sets depth of field, Q = spherical harmonics for specular.
v1 analysis: smooth phase → PSD ∝ δ(k) (small eyebox); random phase → E[|û(k)|²] = const (flat, full eyebox);
spread after propagation σ²(z) ≈ ∬ x²|u|² dx + (z/k)² ∬ |k|² |û(k)|² dk (grows ∝ z² with angular spread).

## Setup (Sec. 4.1)
Blender (NeRF synthetic) + Mip-NeRF 360 scenes; 2DGS via gsplat; textured meshes via NeRF2Mesh; Triangle Splatting
(Held et al. 2025) for translucent triangles. PyTorch. Primitive counts matched between Gaussians and polygons via 3DGS-MCMC.
Baselines: Polygons-RP (random-phase polygon CGH with silhouette method; 1 and 8 frames), GWS (1 frame).
Variants: RPWS-GS (Gaussians; quantitative), RPWS-Δ (triangles; qualitative only, triangle count not controllable).
Hardware (v1 text; v3 defers to supplement): HOLOEYE Pluto-2.1 phase-only SLM, 1080p, 8 µm pitch; FISBA READYBeam
fiber-coupled laser; FLIR GS3-U3-123S6C-C colour camera on a Thorlabs MTS25-Z8 stage; phase-only holograms by SGD with a
complex-field loss at 4 cm from the SLM. Frames and colour channels captured separately and merged.

## Key numbers — Table 1 (v3; simulated 10 × 10 dense light fields, PSNR ↑ / SSIM ↑ / LPIPS ↓)
| CGH algorithm | Blender | Mip-NeRF 360 |
|---|---|---|
| Polygons-RP (1 frame) | 17.02 / 0.14 / 0.75 | 12.77 / 0.14 / 0.76 |
| Polygons-RP (8 frames) | 20.27 / 0.30 / 0.65 | 15.63 / 0.27 / 0.69 |
| RPWS-GS (1 frame) | 18.11 / 0.15 / 0.76 | 14.17 / 0.17 / 0.76 |
| RPWS-GS (8 frames) | 23.42 / 0.36 / 0.64 | 19.65 / 0.36 / 0.65 |
| RPWS-GS (24 frames) | **24.87 / 0.49 / 0.56** | **21.21 / 0.48 / 0.58** |
| GWS | 14.35 / 0.18 / 0.63 | 10.50 / 0.17 / 0.60 |
Derived: RPWS-GS (24 f) − GWS = +10.52 dB (Blender), +10.71 dB (Mip-NeRF 360); RPWS-GS (8 f) − Polygons-RP (8 f) =
+3.15 / +4.02 dB. Note GWS LPIPS (0.63 / 0.60) is better than RPWS-GS 1 f and 8 f; RPWS 24 f best on all metrics.
Other numbers: up to 24 multiplexed frames; experimental 24-frame RPWS "near speckle-free"; future SLMs: 3600 fps FLCoS
(Holographic Parallax), 5760 fps TI MEMS PLM. No runtimes are reported in the v3 main paper.

## Figures (v3)
1. Teaser: (a) primitives (Gaussians, soft triangles) → (b) time-multiplexed random-phase holograms; (c) GWS front focus
   "Unnatural defocus, small eyebox ✗"; (d) GWS-NRP "Incorrect appearance ✗"; (e) RPWS back focus "Natural defocus,
   large eyebox ✓"; (f) RPWS front focus "Correct appearance ✓". (Kitchen lego scene, simulated.) → `figs/teaser.png`
2. Toy sqrt-blending: ground truth / GWS-NRP (ring error map) / RPWS (zero error). → `figs/alpha_blending_toy_example.png`
3. Ablation (lego focal slice, garden light-field view): GWS-NRP (Eq. 3) · GWS with sqrt · RPWS without sqrt · RPWS full
   (Eqs. 4, 5); rows "back-to-front compositing" ✗✗✓✓ and "square root alpha blending" ✗✓✗✓. GWS+sqrt: halos / light
   leakage / black holes at occlusion borders; RPWS w/o sqrt: wrong colours. → `figs/alpha_blending_ablation.png`
4. Simulated focal stacks (materials; front / rear focus, zoom crops) and light fields (bicycle; centre / right view,
   epipolar image) for GT, Polygons-RP 1f/8f, GWS 1f, RPWS 1f/8f/24f. GWS right view is almost black (energy at eyebox
   centre). → `figs/sim_fs_lf.png`
5. RPWS-Δ (triangle splats) simulated focal stacks (bicycle, bonsai, garden, counter) and light fields (kitchen lego, room).
   → `figs/triangle_splats.png`
6. Experimentally captured focal stacks (bicycle, kitchen lego: front / back focus) and light fields (left / right view) for
   GT, Polygons-RP 1f/8f, GWS 1f, RPWS-GS 1f/8f/24f, RPWS-Δ 24f. → `figs/experimental_main.png`

## Limitations (Sec. 5)
Many frames needed for exact blending, limited by SLM refresh rate (fast FLCoS / PLM may help; quantization is an open issue);
reduced contrast from random phase in experiments (learned calibration for complex fields may help). v1 additionally: not real
time; neural networks that output random-phase holograms would help.

## Supplement (added 2026-10-03) — `../data/Random_phase_Wave_Splatting (2).pdf` (49 pp., SIGGRAPH submission 1220)
Text: `pdfs/rpws_supp_clean.txt` (line-number gutters stripped). The submission main paper `(1).pdf` (11 pp.) →
`pdfs/rpws_main_submission_clean.txt`; same content as arXiv v3.
- **S1.2–S1.3 pseudocode.** Alg. 1 GWS: sort front→back; per primitive `u_SLM += 𝒫(T·u_k, z_k)·c_k·o_k`, then
  `α_k = o_k|u_k|`, `T ← T·(1 − α_k)` — T is one global 2-D map in SLM-plane xy coordinates, applied to the primitive
  **at its own plane before propagation** (a projected on-axis silhouette). Alg. 2 Polygons-RP (silhouette method): sort
  back→front; `u′ = 𝒫(u, z_k)`; `u ← (1 − |u_k|o_k)·u′ + u_k c_k o_k e^{iφ}`; `u ← 𝒫(u, −z_k)` (binary-alpha style mask,
  amplitude). Alg. 3 RPWS: same back→front loop as Alg. 2 but `u ← √(1 − |u_k|)·u′ + √(|u_k||q_k|)·√c_k·e^{iφ}`,
  q_k = ℱ⁻¹{Q_k e^{iφ_k}} (structured random phase); model optimized on the **square** of the target.
  Ablations: Alg. 4 GWS-NRP (Alg. 1 + e^{iφ_k}); Alg. 5 GWS + √ (`𝒫(√(T·|u_k|) e^{iφ_k}, z_k)·√c_k·√o_k`, intensity-optimized
  model); Alg. 6 RPWS without √ (Alg. 3 with (1 − |u_k|) and |u_k||q_k| c_k, amplitude-optimized model).
- **S1.4** 2DGS via gsplat, 3DGS-MCMC, ≤ 100k Gaussians; GWS models fit the image (amplitude), RPWS models fit image²
  (intensity) and use √colour.
- **S3.1 bandwidth (Wiener–Khinchin):** E|û(k)|² = ℱ{R_u(ρ)}; smooth phase: R_u = R_a (broad) → narrow spectrum;
  random phase: E[e^{i(φ(x) − φ(x+ρ))}] = δ → R_u = (∫a²) δ(ρ) → flat spectrum, full SLM bandwidth.
- **S3.2 depth of field:** σ²(z) ≈ σ²(0) + (z/k)² σ_k², σ_k² = ∬|k|²|û(k)|²dk; cross term vanishes in expectation for
  fully random wavefronts. Spread grows ∝ z² × angular spread.
- **S4 cross-term convergence:** |a₁e^{iφ₁} + a₂e^{iφ₂}|² = a₁² + a₂² + 2a₁a₂cos(φ₁ − φ₂); E[cos(φ₁ − φ₂)] = |𝜑_φ(1)|²
  (characteristic function 𝜑_φ(t) = E[e^{itφ}]). N-frame mean C̄_N → 𝒩(|𝜑(1)|², Var/N),
  Var = ½(1 + |𝜑(2)|²) − |𝜑(1)|⁴. Fastest decay iff 𝜑(1) = 𝜑(2) = 0 → SE(C̄_N) = 1/√(2N). Uniform 𝒰(−π, π) satisfies it;
  so does the 4-point {0, π/2, π, 3π/2} (but uses less SLM bandwidth); the 2-point {0, π} has 𝜑(1) = 0 but 𝜑(2) = 1 →
  slower (SE = 1/√N). Fig. S3: empirical mean vs N = 10…10⁴, 1000 trials each.
- **S5 structured random phase:** ũ(x) = ℱ⁻¹{û(k) ∗ Q(k)e^{iφ(k)}}, φ iid 𝒰(−π, π); time-multiplexed √E|ũ|² keeps the
  primitive's spatial amplitude while the Fourier amplitude follows Q (Q = 1 diffuse, pupil P(k, r) → DoF control, SH → specular).
- **S6.1 ground truth:** orthographic LF from 2D Gaussians; pupil at Fourier position (f_x, f_y) ∈ [−λ/2p, λ/2p] ↔ view
  angle θ_x = sin⁻¹(f_x); shear of a primitive at depth z: Δx = −z·tan θ_x. Focal stacks by shift-and-add.
- **S6.2:** 10 × 10 LFs, 60-slice focal stacks; per-scene tables (GWS wins focal-stack metrics because it is sharp;
  RPWS wins all LF metrics).

## Nature-format manuscript (added 2026-10-03) — `../data/main_old_nature_format.pdf` (21 pp.)
Text: `pdfs/nature_main.txt`. Authors: Chao†, Yang†, Choi (Stanford; Seoul National University), Gopakumar, Koiso
(KDDI Research), Wetzstein*. **The explainer video now accompanies this version.**
- Framing: "RPWS, a new primitive-based CGH formulation for composing large collections of continuously translucent
  random-phase wavefronts … The central challenge is that independently phase-modulated primitives do not obey
  conventional alpha compositing: interference cross terms … require a different compositing law. We derive this law in
  the intensity domain … square-root wave compositing … time multiplexing as the statistically exact mechanism …
  Because the formulation depends on phase statistics and continuous alpha rather than a particular primitive geometry,
  it applies to arbitrary translucent primitives." Discussion: "establishes a new primitive-based CGH regime";
  "extends wave splatting from the smooth-phase regime to a distinct random-phase regime".
- Numbering is identical to arXiv v3: Eqs. 1–2 (smooth-phase alpha wave blending), 3 (naïve random phase), 4–5 (RPWS
  back-to-front compositing + mask), 6 (time-multiplexed intensity), 7–17 (alpha-blending derivation; 12 amplitude law,
  13–14 expectation, 15–17 √ law). Figs. 1 teaser, 2 toy, 3 ablation, 4 sim FS/LF, 5 triangles, 6 captured.
  Table 1 capability, Table 2 LF metrics (same numbers as above). Derivations deferred to "supplemental materials".
- PI guidance (2026-10-03): for Nature it must look entirely new, never incremental; earlier video/paper were too
  defensive about GWS. Video v6 removes "GWS" from screen text and frames RPWS as a new compositing law.
