# [chao2026foveated] Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation

Chao*, Yariv*, Xiao, Wetzstein (Stanford) · arXiv 2603.23491 (Mar 2026) · 46 pp. incl. supplement
PDF: `../pdfs/chao2026foveated.pdf` · full text: `../pdfs/chao2026foveated.txt` · figure pages/crops: `../pdfs/figs/`

## Problem
DiT cost grows quadratically with token count; resolution, frame rate and context length keep rising.
When the viewer's gaze is known (eye tracking) or can be estimated (saliency), full resolution is wasted
outside the fovea because acuity falls off rapidly with eccentricity.

## Method
- **Foveated tokenization.** Binary mask M ∈ {0,1}^(h×w). HR tokens where M = 1; each 2×2 block of the
  periphery becomes one LR token (d = 2, 4× saving). Sequence length L = m + (h·w − m)/4, L ≪ h·w.
- **Foveated generation.** Sample z_1^fov ~ N(0, I) of length L; denoise t = 1 → 0;
  (z_0^high, z_0^low) = Split(z_0^fov, M); x^high = D(z_0^high), x^low = D(z_0^low);
  x^fov = M′ ⊙ x^high + (1 − M′) ⊙ Up(x^low), M′ = Up(M).
  Training-free use of a pretrained DiT = “naïve mixed-resolution denoising”; fails (duplicated / fused objects, scale mismatch).
- **Mixed-resolution RoPE.** Phase-aligned RoPE (Wu et al. 2025, arXiv 2511.19778): HR queries see LR keys
  at their fine-grid positions; LR queries see subsampled HR keys re-indexed on the coarse grid (Fig. 3).
- **Foveated training.** z_0^high = E(x), z_0^low = E(Down(x)) (bicubic), z_0^fov = Merge(z_0^high, z_0^low, M):
  consistent across resolutions by construction. Standard flow matching on the L-token sequence; LoRA rank 32.
- **Masks.** Random circles (images), random cubic-spline gaze paths through 3 control points (video);
  saliency-guided (DeepGaze) and bounding-box (YOLO) variants; no architecture change.

## Setup
Images: FLUX.2 Klein 4B, Aesthetic-Train-V2 (90k train / 10k eval), 1024². Video: Wan2.1 1.3B, Vchitect-T2V-Dataverse, 480p (720p generalization).
10k steps, 8 × H100, effective batch 8, DiffSynth-Studio defaults.

## Results
- Speedup (Fig. 6): at 25% tokens ≈ 2.2× image, ≈ 4.2× video (3D attention costs more).
- Table 1 (HPSv2.1 naïve / ours; full 0.280): 43% 1.61× 0.268 / 0.279 · 30% 1.98× 0.270 / 0.279 · 26% 2.08× 0.275 / 0.280.
  FID judged unreliable for foveated images (naïve beats full on FID despite artifacts).
- User study (2AFC, fixation dot, 1 s exposures, 11 participants, 220 votes/pair, fovea = 1/3 image width ≈ 15°, 1.85× speedup):
  ours vs full 47.3% (p = 0.48, indistinguishable); ours vs naïve 87.4%; full vs naïve 90.8%.
- Video (VBench, 38% tokens, 3.5× speedup): subject consistency full 0.9407, naïve 0.9072, ours 0.9446.

## Limitations / future work
Color seams at the fovea boundary from separate VAE decodes + blending (fix: mixed-resolution VAE);
only two foveation levels (multi-level needs multi-level phase-aligned RoPE); integrate into a streaming,
eye-tracked video world model.
