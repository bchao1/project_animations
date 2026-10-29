# Spectral Frontier Sampling animation

3b1b-style teaser + method video for the SFS paper draft (CVPR 2027). Source project:
`/home/brianchc/prompt-adaptive-speed-cvpr-2027` (paper in `submission/adaptive-speed-cvpr2027/`). Ground truth for
the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env with `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render.sh`).
- Style and helpers follow `../spectral_progressive_diffusion` (SPEED animation): CMU Serif, Pango markup via `tx()`,
  no LaTeX.
- Data: `scratch/prep_data.py` (CPU) writes `data/sfs_data.npz`. `scratch/capture.py` (GPU, about 8 min per run)
  writes `data/capture/<src>_<idx>/`. Every on-screen number must come from the paper or from these files.
- Captions are plain, simple English (see the simple-english skill).
- Versioning: a new fix round makes a new `animation_sfs_vN.py` (never overwrite) and updates `storyboard.md`.
- Scratch files stay in `scratch/`.
