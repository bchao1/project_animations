# Storyboard — RPWS explainer, 3 × 1 min (3b1b style)

**Current version: v7** — `animation_rpws_explainer_v7.py` (accompanies the Nature-format manuscript `../data/main_old_nature_format.pdf`)
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60). Outputs: one MP4 per part in `media/final/`
(`RPWS_Part1…3_<script>_<res>.mp4`) plus the 3-min cut `RPWS_3min_<script>_<res>.mp4` (= S00 title + P1 + P2 + P3).

Goal (user, 2026-10-03): replace the 5-min teaser's glossing with three standalone ≈ 1-min explainers that make RPWS look
**very novel** by following the paper's exact math and logic, each built on computed toy experiments that replicate the
paper's own toy figures (look/feel = the multisource explainer, `../../large_etendue_holography/`).

Sources (every claim on screen): main paper `../random-phase-wave-splatting/pdfs/src/` (Eqs. 1–17, Figs. 2–3) and the
supplement `../data/Random_phase_Wave_Splatting (2).pdf` (text: `../random-phase-wave-splatting/pdfs/rpws_supp_clean.txt`;
Algs. 1, 3–6; S3.1 Wiener–Khinchin bandwidth; S3.2 σ²(z) ≈ σ²(0) + (z/k)²σ_k²; S4 cross-term SE = 1/√(2N); S6.1 shear
ground truth). Notes: `../random-phase-wave-splatting/summaries/chao2025random.md` (Supplement section).

Toy experiments: `sims.py` (numpy + torch; ≈ 2.5 min) → `data/generated/`. All labelled "toy simulation".
- A — one primitive, smooth vs random phase, x–z side view (sim units λ = 6 px, SLM pitch p = 10 px → θ_max = 17.5°,
  24 frames). Predicted spread σ(z) from S3.2 vs measured: smooth 21.7 / 21.6 px, random 90.0 / 84.3 px at z = 480.
- C — paper Fig. 2: orange Gaussian in front of blue at one depth. Intensity model (RPWS, defines GT, Eq. 15); amplitude
  model fitted for GWS (S1.4). MSE: GWS 6e−5 (fit), GWS-NRP 6.0e−3 (does not converge), RPWS 1.1e−2 (T = 1) →
  4.6e−4 (24) → 4.3e−5 (256). NRP deficit ≈ cross term 2w₁w₂.
- D — Fig. S3: C̄_N for uniform 2π / 4-point / {0, π}, 1000 trials; std at N = 10⁴: 0.0067 (1/√2N = 0.0071),
  0.0073, 0.0103 (1/√N = 0.0100).
- E — Fig. 3: three translucent Gaussians at 4 mm over an opaque checker plane at 16 mm; λ = 520 nm, 8 µm, 512²,
  24 frames; Algs. 4 (GWS-NRP), 5 (GWS + √), 6 (RPWS w/o √), 3 (RPWS); GT by S6.1 shear + shift-and-add over the same
  pupil. LF MSE centre / edge: NRP .039/.049, GWS+√ .019/.033, w/o √ .039/.038, RPWS .019/.017.
- F — Part 3 side view (sim units): rear primitives Q (x = −28, behind the occluder), P (+92, beside it), R (−150,
  control) at z = 0; occluder |x| < 55, α = 0.985 at z = 200; SLM at z = 440. GWS + √ (Alg. 5) vs RPWS (Alg. 3).
  Eye at θ = −13° (pupil-normalised energy): Q — GT 1.0, RPWS 0.6, GWS+√ 0 (black hole); P — GT 0.05, RPWS 0.2,
  GWS+√ 0.8 (leak).

## Global style (= multisource explainer)
Black background · CMU Serif, Latin Modern Math for math glyphs (`mathify`) · Pango markup, no LaTeX · header top-left
(24, dim, "k · title") · caption y = −3.5 (28; held ≥ len/16 + 0.9 s) · text added instantly · part card "Part k of 3"
(30, dim) over the title (48, bold), 1.6 s · laser green `#39E27D` light, amber `#FFCA28` cross term / formula guides,
purple `#E040FB` √, cyan `#4DD0E1` back-to-front, red `#FF5252` failures · toy visuals tagged "toy simulation".

## S00_Title (≈ 5 s, only in the 3-min cut)
Title, subtitle, authors, "arXiv 2508.17480"; three part titles listed.

## P1_Difference — "1 · How RPWS differs from GWS" (≈ 65 s)
1. Card. Two live x–z wave panels (sim A, stacked, isotropic): smooth-phase primitive (GWS) → narrow beam; random
   phase (RPWS) → fills the ±θ_max cone (frames change every 0.4 s). Spectra |û(k)|² beside each: spike vs flat over
   the SLM band ±λ/2p (S3.1). Switch to the 24-frame average; amber dashed ±σ(z) from σ²(z) ≈ σ²(0) + (z/k)²σ_k² (S3.2)
   lies on the computed glow.
2. Equations: GWS Eqs. 1–2 vs RPWS Eqs. 4–6, then a 4-row table revealed with matching colours —
   phase (smooth ∠u_i = kz_i | random φ_i^(t) ~ 𝒰(−π, π)), blending (amplitude c o|u| | intensity √c √(o|u|)),
   order (front → back, one global T map | back → front, mask at each plane), frames (1 | T, averaged).
3. "Just add random phase to GWS?" Eq. 3 (GWS-NRP). Computed Fig. 3 replica (sim E): GT · GWS-NRP · GWS + √ ·
   RPWS w/o √ · RPWS, with the paper's ✗/✓ rows (back-to-front, √ blending); viewpoint sweeps left ↔ right.
   Captions: NRP wrong colours; GWS + √ breaks off-axis at occlusion borders; only both changes work → Parts 2, 3.

## P2_AlphaBlending — "2 · Why random-phase alpha blending works" (≈ 65 s)
1. Card. Eq. 7 (ray splatting) and Eq. 8 (wave) → Eq. 11 |u|² = w₁² + w₂² + 2w₁w₂cos(φ₁ − φ₂) (cross term amber).
2. Phasor diagram (example α₁ = 0.5, c₁ = c₂ = 1, α₂ = 1 → c = 1): smooth phase, phasors aligned → |u| = w₁ + w₂
   (Eq. 12, GWS). Random phase redrawn each frame with GWS weights 0.5, 0.5: running mean of |u|² → w₁² + w₂² = 0.5,
   not c² = 1 (GWS-NRP). √ weights √0.5, √0.5 (Eqs. 15–17): running mean → 1 = c². Bars: this frame / running mean,
   dashed targets.
3. Pixel toy (sim C, paper Fig. 2): GT · GWS · GWS-NRP · RPWS with frame counter 1 → 256; error maps; measured
   GT − NRP next to predicted 2w₁w₂ — the same ring.
4. Convergence (sim D, Fig. S3): C̄_N vs N (10 … 10⁴), ±1 std envelopes for uniform 2π and {0, π}, dashed ±1/√(2N);
   "optimal iff 𝔼[e^{iφ}] = 𝔼[e^{2iφ}] = 0". Close: time multiplexing *is* the alpha blend; speckle reduction is a by-product.

## P3_BackToFront — "3 · Why back-to-front compositing works" (≈ 65 s)
1. Card. Ray diagram in the exact geometry of sim F (rear plane with Q, P, R; occluder; SLM; eye at θ = −13°).
   GWS's single transmittance map T = ∏(1 − o_j|u_j|) is the occluder projected straight back along z (Alg. 1):
   Q inside it is deleted — but its ray at −13° clears the occluder (black hole); P outside keeps its whole cone —
   its −13° ray goes through the occluder (leak). Fine for smooth phase (on-axis light), wrong for random phase.
2. Waves (sim F, 24-frame average) fade in under the rays: GWS + √ — no Q cone, P's cone crosses the occluder.
   RPWS recursion Eq. 4 revealed left → right: propagate to the occluder plane, × M = √(1 − α) there, add its wave,
   propagate on. Eye strips at −13°: GT / GWS + √ / RPWS.
3. 2-D toy (sim E): GT · GWS + √ · RPWS with error maps, viewpoint sweep — identical on-axis, GWS + √ fails off-axis.
4. Close: RPWS = painter's algorithm (Porter–Duff *over*) for waves, with propagation and square roots; paper Fig. 3
   crops (GWS with sqrt vs RPWS full).

## Version history
- v1 — first cut, P1 88 s / P2 92 s / P3 75 s. QA: Latin Modern Math √ oversized; labels clipped; FrameSeq ignored the
  shared tracker (`tracker or …` with a falsy ValueTracker) so every sweep / frame counter image stayed on frame 0.
- v2 — √ via CMU Serif fallback; tracker bug fixed; captions condensed (≈ 60 s per part); P1 equation layout;
  P2 axes clipped to ±0.3, right column moved; P3 occluder label below, eye label below the eye, missing-Q circle and
  leak box on the GWS + √ wave field.
- v3 — holds before visual changes (P2 smooth-phase step and P3 GWS + √ wave step were skipped in v2); P2 right
  column fitted to ≤ 4.95 wide; P1 method labels flush left. Rendered 1080p60.
- v4 — user: "the checkerboard and weird blob of colours" unclear → both toys rebuilt as a see-through amber star
  (α = 0.72, soft edge) over a blue/white checkerboard. Sim E: star at 4 mm, checker at 12 mm, views focused at 8 mm,
  pupil offsets ±0.8; LF MSE centre/edge NRP .026/.055, GWS+√ .021/.041, w/o √ .026/.032, RPWS .021/.024 (GWS+√ opens a
  black gap beside the star off-axis). Sim C (Fig. 2 experiment): same star over checker at one depth; GWS-NRP MSE 1.1e−2,
  RPWS 3.0e−2 (T = 1) → 1.2e−3 (24) → 1.2e−4 (256). Part 3 error maps speckle-smoothed (σ 2.5 px, floor 0.08).
- v5 — user: use the talk deck `../data/Random-Phase-GWS_font_embedded.pptx` and show real result *videos* instead of
  static images. Clips copied to `data/deck/` (mapping from slide layouts): P1 + focal sweep GWS (media18) vs RPWS
  (media17) and RPWS light field (media27) with the moving-pupil inset (media34); P2 + GT / GWS-NRP / RPWS focal sweep
  (media10 / 9 / 36) and the 1 / 8 / 24 / 96-frame grid (media55–58); P3 static Fig. 3 crops replaced by simulated (media29
  + pupil) and captured (media69, 18 views ping-pong) bicycle light fields. Frames cached by `build_videos()`
  (flag `done_vid1`).

- v6 — accompanies the Nature-format manuscript; PI guidance: must look entirely new, never incremental, not defensive
  about GWS. Parts retitled: 1 · Holograms from translucent primitives (hook = RPWS light field; why random phase; real
  focal sweep; RPWS formulation Eqs. 4–6 as four ingredients — random phase, √ alpha blending, back to front, time
  multiplexing; "no assumption on primitive shape" + soft-edged-triangle focal sweep, media40; Fig. 3 ablation relabelled
  naïve random phase (Eq. 3) / √ only / back-to-front only / RPWS (Eqs. 4–5)). 2 · A compositing law for random-phase
  light (Eq. 12 = coherent amplitude blending; "naïve random phase"; "we prove … 1/√(2N)"). 3 · Occlusion through depth
  (hook "How does a wave know it is hidden?"; "front to back" vs RPWS back to front). No "GWS" in on-screen text; the
  GWS-vs-RPWS equation table is gone. S00: Stanford · Seoul National University · KDDI Research, no arXiv id.
  Equation/figure numbers match the Nature PDF (identical to arXiv v3 numbering).
- v7 — P3 closing: the captured prototype clip (media69) only has horizontal parallax, so it no longer sits next to the
  circular pupil path; replaced by two simulated RPWS light fields (bicycle media29, counter media31) sharing one pupil
  inset (media34).

## Talk deck review (2026-10-03)
109 slides, 80 videos, 19 narration MP3s (transcribed: `../data/pptx_narration.md`; slide text/media map:
`../data/pptx_slides.md`, `../data/pptx_videos.txt`). Borrowed: the result videos above. Framing that matches ours:
"time multiplexing is the exact mathematical mechanism … not merely a speckle heuristic"; "the current primitive masks
the wavefronts behind it, then its own wavefront is added". Not used: the deck's back-to-front lego build-up and Mask/ASM
diagrams (our x–z wave recursion shows the same thing with physics), the GWS small-eyebox animation (media5, white
background), captured focal stacks with insets (media1, 83–93) — candidates if more results are wanted.
