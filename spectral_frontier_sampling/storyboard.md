# Storyboard: Spectral Frontier Sampling (SFS), 3b1b style

**Current version: v6**: `animation_sfs_v6.py` (1080p60 final: `media/final/spectral_frontier_sampling_animation_sfs_v6_1080p60.mp4`, about 3 min; slide sections: `slides_v6/sections.json`, copied to `~/slides/assets/clips/sfs/`), a 5-min-format teaser with one slide section per beat,
like the SPEED animation (`../spectral_progressive_diffusion/animation_spectral_progressive_v8.py`). A 1-min cut
(`teaser_1min/`) follows after v3 is approved.

Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) · `CAPTIONS=0 ./render.sh s` (slide sections).
Env: conda env `anim` (manim 0.19.1, Pillow 11.1). No libffi preload on the Rhoda pod.

Source: the CVPR 2027 draft `~/wavelet-diffusion/submission/adaptive-speed-cvpr2027/` (commit 707e9d7). Context with
every number and its baseline: `~/slides/assets/papers/sfs/CONTEXT.md`. Assets: `~/slides/assets/papers/sfs/ASSET_INVENTORY.md`.
v1/v2 (below, under "Old plan") were built on the older draft and older data. They are reference only.

## Decisions (Brian, 2026-10-29)
- Name on screen: **SFS / Spectral Frontier Sampling** (the paper name is still open; one find-and-replace later).
  Subtitle: "Per-sample continuous expansion for fast diffusion sampling".
- Method: the release rule **frontier tracking** (`cont_mode`): grid grows in steps of 2 latents, 2× cap, online estimate.
- Baselines on screen: **full res and SPEED-2 / SPEED-3** only. SPEED-Conti (SPEED + tracking) only in a backup slide.
- The two points of the story: (1) SFS saves compute by **tracing the frontier**; (2) it finds the frontier **online, per sample**.
  Honest framing: tracking gives most of the speed; the online estimate times each sample (smooth images stay small longer).

## Takeaway
Grow the grid just above the spectral frontier, so no compute goes to frequencies that are still noise. The frontier
leads the story. Second, also important: every image has its own frontier, and SFS finds it while sampling, so smooth
images stay small longer (2.9× to 7.7× faster across 100 cinematic prompts, up to 10× on smooth styles).
(Brian 2026-10-29: the frontier leads; adaptivity, as in Fig. 2 and the 7–10× low-power prompts, is also important.)

## Running example
`climber_ridge` (cinematic prompt 14, the paper's Fig. 1), FLUX.2 [klein] 9B, 2048², 50 steps, B200.
- Per-step x̂₀ frames and spectra: `$WD_BULK/figure_prompts_previews/cinematic/4klsdb/2k/previews/cont_mode/14/`
  (`x0_<step>_n<grid>.png`, `x0_spectra.npz`: sigmas + per-step x̂₀ ring power).
- Schedule (`$WD_BULK/figure_prompts/cinematic/4klsdb/2k/cont_mode/14.json`): 64² for steps 0–19, then
  74, 78, 84, 92, 102, 110, 132, 156, 180, 214, 256 at steps 20, 35–38, 41–46. Time 13.2 s vs full res 70.5 s (5.3×).
- Per-sample pair (paper Fig. 2): `mountain lake` (cinematic 50, lower detail) and `Kyoto torii` (cinematic 12, higher detail),
  both with frames, spectra and schedules.
`$WD_BULK` = `/home/bchao/bulk/wd`.

## New data to make (before the render)
- D1 `scratch/capture_v3.py` (GPU, local B200, about 1 min per prompt): reruns the release sampler on cinematic prompts
  14, 50, 12, 35, 58, 99, 54 with the same seeds (previews: x̂₀ frames + per-step spectra) and saves, per coarse step, the rule record (`est`, `sd`, `u`, `fire`, `next`), the ring
  cross-power C on the current grid, and the threshold ln P*(σ_{i+1}). Check: the schedule equals the stored JSON.
  Writes `data/capture/<idx>/`. It imports the research code and changes nothing in `~/wavelet-diffusion`.
- D2 `scratch/prep_v3.py` (CPU): SNR maps (1 − σ)² P_ω / σ² from each prompt's final spectrum, the frontier (SNR = δ = 0.01),
  the compute-scaled frequency axis (paper's FLOP model F(n) = aT + bT², T = (n/2)² + 512), per-staircase compute,
  and 768-px copies of the result images. Writes `data/sfs_v3.npz` and `data/img/`.

## Style notes
Sibling: SPEED v8 (`../spectral_progressive_diffusion/animation_spectral_progressive_v8.py`): copy its helpers
(`tx`, `caption`, `header`, part cards, `sec`), sizes and colors.
- Signal `#58C4DD`, noise `#FC6255`, frontier and ours `#FFCA28`, SPEED stages `#B0BEC5` (solid), full res `#8A8A8A`,
  wasted compute red `#FF5252` hatching, speedups purple `#E040FB`, counts amber `#FFCA28` bold.
- Spectrogram: x = denoising step 0–49, y = frequency on the compute-scaled axis (area under a staircase = its compute,
  as in paper Figs. 1–2). Color = log10 SNR of x_t: dark red tint where SNR < δ (still noise), blue ramp above, a white
  line on the frontier (SNR = δ). Same meaning as the paper (white = frontier, area = compute), 3b1b colors.
- Token grids: one constant gap, inset outlines (house `TokenGrid`), blue 1× tokens.

## Scenes (target 4:45–5:15)
| Scene | Beat (section name) | Visual (what moves) | Caption (CAPTIONS=1 only) | Time | Numbers (source) |
|---|---|---|---|---|---|
| S00_Title | `title` | Title + subtitle. Emblem: the climber's spectrogram fills column by column, a yellow staircase hugs the white frontier. | none | 7 s | none |
| S01_Frontier "1 · The spectral frontier" | `xhat` | Part card "Part 1 · The problem". Left: climber x̂₀ sharpening, step 0 → 49 (real frames). | "Diffusion builds an image from coarse to fine." | 6 s | |
| | `column` | Right: the spectrogram fills one column per step, in sync with the left image. Each column = SNR at each frequency at that step. | "Each column shows which frequencies have left the noise." | 8 s | δ = 0.01 (SPEED) |
| | `frontier` | The white frontier line is drawn over the filled map. Label "spectral frontier". | "The edge between signal and noise moves up over time." | 6 s | |
| S02_Waste "2 · Where compute goes" | `area` | The y axis stretches to the compute scale (frequency rows re-space, labels move). A grey bar at the top = full res. Its area fills. | "Make the height match compute: area is cost." | 7 s | FLOP model (paper Sec. 2.10) |
| | `fullres` | The area above the frontier hatches red: "still noise". | "Full resolution pays for every frequency, even pure noise." | 6 s | 70.5 s (prompt 14, B200) |
| | `speed` | SPEED-3 staircase (64 → 128 → 256 at steps 27, 36) drawn in grey. Hatching shrinks to the octave above the frontier. | "SPEED grows in 2× stages. Each stage adds a whole octave of noise." | 8 s | SPEED-3 schedule (CONTEXT 4.2) |
| | `more_stages` | Ghost staircases with 4, 6, 8 stages: hatching shrinks each time. "more stages → less waste, more hand-set choices". | "More stages waste less, but each one is a hand-set choice." | 7 s | |
| S03_PerSample "3 · Every image has its own frontier" | `morph` | The climber's spectrogram morphs through four prompts, thumbnail beside it: astronaut in dunes (35) → mountain lake (50) → Kyoto torii (12) → koi pond (58). The white frontier slides right (smooth: late) or left (detailed: early). | "Each image has its own frontier. Smooth images reach detail late." | 10 s | D1 spectra of 35, 50, 12, 58 |
| | `pair` | It settles on two panels, lake (left) and torii (right), as in paper Fig. 2. | "A smooth lake and a detailed temple: very different edges." | 5 s | prompts 50, 12 |
| | `same_speed` | The same grey SPEED staircase on both, compute above each frontier hatched; label = share of SPEED's compute that is still noise (lake 57%, torii 46%). No "too late" (no frontier rises above SPEED-3's grid on these prompts). | "SPEED uses one schedule for every image. The smooth one wastes more of it." | 8 s | SPEED-3 K 27, 36; compute from the FLOP model |
| | `question` | Center text: "Grow with each image's own frontier?" (CAPTIONS=0: kept as the slide's bridge, not drawn) | | 3 s | |
| S04_Tracking "4 · Frontier tracking" | `ladder` | Part card "Part 2 · The method". Back to the climber. Candidate grid sides 64, 66, …, 256 as thin ticks on the y axis. | "Any grid size is allowed, not only 2× steps." | 6 s | Δ = 2 latents (release code) |
| | `grow` | Yellow staircase is drawn step by step just above the frontier, using the real schedule. Beside it, the climber's x̂₀ at each new grid, drawn at a size proportional to the grid (Fig. 1 triangle: 64, 92, 110, 156, 214, 256). | "When the grid's edge fills with signal, grow to the next size that is still noise." | 12 s | real schedule (14.json) |
| | `saved` | The area between full res and our staircase hatches red, labelled "saved"; SPEED's ghost stays. | "We skip almost all the noise-only compute." | 6 s | |
| | `cap` | A small inset: a jump 64 → 256 crossed out, "at most 2× per step". | "One growth is at most 2×, so one bad guess cannot jump to full size." | 5 s | 2× cap (CONTEXT 2.9) |
| S05_Measure "5 · Reading the frontier while sampling" | `unknown` | The final spectrum is hidden (frontier line fades): "but we do not know it while sampling". | "The true frontier needs the final image, which we do not have yet." | 6 s | |
| | `xhat_now` | Climber x̂₀ at step 12 on the 64² grid (64×64 token grid overlay fades in and out). | "At every step the model already guesses the clean image." | 6 s | |
| | `rings` | Ring sweep on the guess's 2D spectrum; dots of ring power C for rings 2–12 with error bars appear on log axes. Unreliable rings fade. | "Its low frequencies are already clear: measure their power." | 8 s | rings 2–12 = 0.375·n/2 (CONTEXT 2.6); C from D1 |
| S06_Estimate "6 · From low rings to the edge" | `extrapolate` | A band from the dots to the grid edge (ring 32): estimate ± error. One line of text: "power is correlated across frequencies" (no formula). | "Power at nearby frequencies moves together, so the low rings predict the edge." | 8 s | est, sd (D1) |
| | `test` | Plot: estimate at the edge vs step (yellow, with band); the threshold line falls step by step (it is what SNR = δ needs). They cross: "grow". | "When the edge's estimated power passes the noise threshold, grow." | 8 s | threshold ln P*(σ; δ = 0.01) |
| | `two_samples` | Same plot for lake and torii: they cross at different steps; SPEED's single dataset line crosses at step 27 for both. | "Each image gets its own moment. SPEED uses one average for all." | 8 s | first growth steps from D1; SPEED 27 |
| | `free` | Text: "no extra network calls · no training". | "It costs only a few FFTs per step." | 4 s | CONTEXT 3 |
| S07_Loop "7 · Close the loop" | `loop` | Measure → estimate → test → grow as four small boxes in a loop; each turn adds one step to the staircase on the climber's spectrogram. | "Repeat every step: the grid follows the frontier." | 7 s | |
| | `pair_loop` | Lake and torii side by side (Fig. 2), each with its own yellow staircase traced live; grey SPEED for reference; the saved area hatches; time under each. | "Smooth images stay small longer. Detailed ones grow early." | 8 s | 50, 12: JSON times |
| | `gallery` | Six thumbnails in a row, smooth to detailed (astronaut 35, beach 99, jellyfish 54, climber 14, torii 12, koi 58), each with its own mini staircase above and its speedup in purple below (7.7× … 2.9×). | "Same model, same settings: each image gets its own schedule and speed." | 9 s | cinematic JSONs (fullres / cont_mode wall time, B200) |
| S08_Results "8 · Results" | `spread` | Part card "Part 3 · Results". The 100 cinematic prompts as small thumbnails, sorted by speedup into a strip; a purple speedup axis below, 2.9× … 7.7×, median 5.0× marked. The four fastest pop out with their numbers. | "100 cinematic prompts: from 2.9× to 7.7× faster, median 5×." | 9 s | cinematic set, per-image wall time vs full res, B200 |
| | `styles` | Line art / anime / product photo: full res vs ours, side by side, speedups in purple (9.9×, 9.5×, 9.5×; range 7.9–10.2×). | "Smooth styles gain the most: up to 10× faster." | 8 s | Fig. 4 (`cont_mode_hybk40`, per-image wall time) |
| | `power` | Binned time vs band power for 20,000 prompts, rising line. "6.45× on the smoothest 10%", "3.95× on the most detailed 10%". | "On 20,000 prompts, time follows each image's detail." | 8 s | CONTEXT 5.2 item 10 (ρ = 0.82) |
| | `bars` | Average over 1,000 random prompts: time bars full res 70.5 s, SPEED-2 29.0 s, SPEED-3 24.9 s, ours 15.1 s. Purple "4.68× vs full res", "1.65× vs SPEED-3"; dim "HPSv3 −0.18 vs SPEED-3". | "On average: 4.7× faster than full resolution, 1.65× faster than SPEED-3." | 8 s | CONTEXT 4.2 (B200, FLUX.2 klein, 4KLSDB, not yet in the paper) |
| | `images` | One row of real results, same prompt and seed: full res (70 s), SPEED-2, ours (with time). | "Same prompt and noise, a fraction of the time." | 7 s | Fig. 3 source images + JSON |
| S09_Outlook "9 · Takeaways" | `three` | "In three ideas": 1 trace the frontier, 2 estimate it online, 3 no training. Small icons from earlier scenes. | per line | 12 s | |
| | `end` | End card: title, yellow tagline "Pay only for frequencies that have left the noise", "CVPR 2027 submission" dim. No authors (anonymous). | none | 5 s | |

Number rules: every speedup names its baseline; on-screen numbers come from CONTEXT.md and are marked "2K, 1,000 prompts,
B200" where they are not yet in the paper. HPSv3 cost is shown next to the 1.65×. No "same quality" claim against SPEED.

## Deck plan (research-talk-slides, `~/slides`)
Part file `parts/01b_sfs.html` (after SPEED, before Foveated), about 13 slides, roughly as long as SPEED (12):
| # | Slide id | Title | TLDR | Visual |
|---|---|---|---|---|
| 1 | div-sfs | 1b · Fewer tokens over time, per sample | Follow each image's own frontier | roadmap, frequency row lit |
| 2 | sfs-recap | The spectral frontier | Frequencies leave the noise one by one | clip S01 (3 beats) |
| 3 | sfs-waste | Where compute goes | Area is cost; most of it buys noise | clip S02 b1–b3 |
| 4 | sfs-persample | Every image has its own frontier | Too early for smooth, too late for detailed | clip S03, ask "Grow with each image's frontier?" |
| 5 | sfs-fig2 | Same model, different frontiers | Smooth lake grows late, detailed temple early | interactive: pick a prompt (1–6), its SNR map, frontier, SPEED's and our staircase (paper Fig. 2 style) |
| 6 | sfs-tracking | Frontier tracking | Grow to the next size that is still noise | clip S04 |
| 7 | sfs-measure | Read the frontier while sampling | Low frequencies are already clear: measure them | clip S05 |
| 8 | sfs-estimate | Estimate the edge, then test | Each image grows at its own step | clip S06 |
| 9 | sfs-loop | Close the loop | Smooth images stay small longer | clip S07 |
| 10 | sfs-spread | Every prompt gets its own speed | 100 cinematic prompts: 2.9× to 7.7×, median 5× | clip S08 `spread`, then a click-through gallery of the fastest cinematic picks (full res vs ours, times) |
| 11 | sfs-styles | Smooth styles gain most | Line art, anime, products: up to 10× faster | Fig. 4 source images, full res vs ours, speedup per pair |
| 12 | sfs-results | On average | 4.7× vs full res, 1.65× vs SPEED-3 | HTML bars + small quality table (HPSv3, IR, FID, each vs its baseline) + time-vs-power line |
| 13 | sfs-qual | Side by side | Same prompt and noise, a fraction of the time | Fig. 3 source images + crops, time per column |
| 14 | sfs-bridge | What is left | Last steps still full resolution everywhere | the old SPEED bridge, moved here (leads to Foveated) |
Backup (`05_backup.html`): SPEED-Conti decomposition (tracking vs estimate, honest numbers), estimator vs oracle per power
decile (Fig. S3), "SPEED is a special case" 2×2 table, few-step video limit (Self-Forcing 197/200).

SPEED part additions (`01_spd.html`, after spd-video): 2 RAE slides.
- `spd-rae-spectrum` "Semantic latents follow the power law too": RAE (DINOv2-B) β 1.65–1.72 vs klein VAE 1.51–1.57,
  same per-band SNR = 1 noise level; coarse-to-fine x̂₀ strip of the RAE ImageNet DiT. Source `~/slides/assets/papers/rae/CONTEXT.md`.
- `spd-rae-progressive` "Progressive sampling in an RAE text-to-image model": SVG-T2I (DINOv3, 1024², 50 steps).
  Out of the box 2.5× faster but HPSv3 −2.61 → cause: the DiT is weak on the half-size grid → 40-step low-res fine-tune:
  +0.15 HPSv3 vs full res at 2.4× (B200, denoising time, 32 prompts, one seed). Triplet: full res / SPEED / SPEED + FT (prompt 13).

Outlook additions (`04_outlook.html`, after takeaways): 2 discussion slides, iterated with Brian.
- `disc-learned` "What we learned": frequency order depends on the representation (pixels, VAE, RAE, video: yes; text: weak);
  the handoff, not the schedule, limits quality; short post-training on the sampler's states fixes it; per-sample adaptivity is real but small.
- `disc-next` "What this framework enables next": 5–8 projects with their evidence and status (DISCUSSION_SOURCES.md).

---

## Old plan (v1–v2, 2026-10-15, older draft and apollo data, reference only)
The full v2 storyboard is in git history (commit 0299ea5). Main ideas kept: spectrogram with steps on x and compute on y,
waste as hatched area, per-sample pair, the residual and 2× cap ideas. Dropped: magma colors, old prompts (002580,
002348, 001026), `cont2x` details (Δ = 8, z = 0.5, g curve), old headline numbers (1.7×, 4.7–5.0×, HPSv3 −0.19, 80 prompts, A6000).

## Version history
- v4 (2026-10-29): v3 480p sheet fixes: L-shaped axes, S02 label column, S03 first-growth dots, S04 thumbnail triangle
  clear of the panel, S05/S06 at step 19 (the climber's estimate exists from step 16), S06 test zoomed, S06 extremes koi
  pond (grows at 11) vs astronaut (38) vs SPEED (27), S06 extrapolate is a labelled sketch, S08 spacing, grow_bar.
- v5 (2026-10-29): S03 shows SPEED's wasted share per image instead of "too late"; label fixes S02, S04, S06.
- v6 (2026-10-29): slide-mode S00 without title text (bigger spectrogram); S06 label positions. Deck: slides_v6.
- v1 (2026-10-15): first cut, teaser + method video.
- v2 (2026-10-15): SNR spectrograms, real x_t, VAE scatter, residual cartoon.
- v3 (2026-10-29, planned): rebuilt on the 707e9d7 draft and the release rule, one 5-min video with slide sections, new
  running example (climber, Fig. 1) and pair (lake / torii, Fig. 2), results from the 1,000-prompt 2K runs.
