# Spectral Frontier Sampling animation

3b1b-style teaser with slide sections for the SFS paper (CVPR 2027 submission, paper name still open; on screen "SFS").
Source: `~/wavelet-diffusion/submission/adaptive-speed-cvpr2027/` (commit 707e9d7). Context with every number:
`~/slides/assets/papers/sfs/CONTEXT.md`; assets: `~/slides/assets/papers/sfs/ASSET_INVENTORY.md`. Ground truth for the
animation is `storyboard.md`; the current script is named at its top (v6 on 2026-10-29).

- Environment (Rhoda pod): conda env `anim` (manim 0.19.1, Pillow pinned 11.1, ffmpeg): `export PATH=/home/bchao/miniforge3/envs/anim/bin:$PATH`.
  `render.sh` is the kit's (l = 480p15, h = 1080p60, s = 1080p60 sections; `CAPTIONS=0` for slides).
  The v1/v2 render script (imp_tome + libffi preload, apollo) is kept as `scratch/render_v2.sh`.
- Data: `scratch/capture_v3.py` (GPU, about 1 min per prompt after a long model load from /home) reruns the release sampler on
  cinematic prompts 14, 50, 12, 35, 58, 99, 54 and writes `data/capture_v3/<idx>/` (x0hat frames, spectra, rule trace, ring
  stats; schedules match the stored runs). `scratch/prep_v3.py` (CPU) writes `data/sfs_v3.{npz,json}` and `data/img/`.
  Data files are not in git; rerun these two scripts on a new machine.
- The scripts set `config.disable_caching`, a per-process text dir and pre-made media dirs: parallel renders on the network
  disk raced without them.
- Slide export: `CAPTIONS=0 ./render.sh s animation_sfs_vN.py`, then
  `python sections_manifest.py animation_sfs_vN.py slides_vN scratch/section_desc_v5.json`, copy the section mp4s to
  `~/slides/assets/clips/sfs/` with `-movflags +faststart`, point `~/slides/build.py` MANIFESTS at `slides_vN`.
- Style and helpers follow SPEED v8 (`../spectral_progressive_diffusion`): CMU Serif, Pango markup via `tx()`, no LaTeX.
- Captions are plain, simple English. Every on-screen number comes from the data files or CONTEXT.md.
- Versioning: a new fix round makes a new `animation_sfs_vN.py` (never overwrite) and updates `storyboard.md`.
- Scratch files stay in `scratch/`.
