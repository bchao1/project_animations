# Level-of-Token (LoT) Diffusion — teasers + thread clips

Anonymous ICLR 2027 submission 7014 (`7014_Level_of_Token_Diffusion.pdf`, text `paper/lot_text.txt`); supplementary
site mirrored in `web/`. **Style rules: `STYLE.md`** (distilled from the other 20 teasers in this repo; template =
foveated_diffusion/teaser_5min + teaser_1min). Every video follows it. v1 (off-style clip montage) is archived in
`superseded_v1/` — do not build on it.

- `animation_lot_vN.py` — 5-min teaser, 10 scenes; every beat is a `Beats.b_*` method. Ground truth `storyboard.md`.
- `teaser_1min/animation_lot_1min_vN.py` — 3 scenes, imports helpers from the 5-min script. `teaser_1min/storyboard.md`.
- `clips/animation_lot_clips_vN.py` — thread clips = the 5-min beats in clip mode (`clip=True`: header
  "Level-of-Token Diffusion", first animation lands on frame 1, beat ends on a hold, captions at 19 chars/s).
- `social/` — 4-person X thread plan: `tweets.py` (text + length check) → `build_threads_md.py` → `threads.md`.
- Numbers: `level-of-token-diffusion/summaries/lot_diffusion.md`. Toy layout = paper Eq. 23 on the site's painted
  corgi map (`data/img/paint0_canvas.png`, 16 × 16 scores in `paint0_detail16.npy`): 104 + 26 + 3 = 133 tokens.
- Credits: `TITLE`, `AUTHORS`, `LINKS` constants at the top of `animation_lot_vN.py` (authors unset: anonymous paper).
- Env `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`; `./render.sh l|h` in each folder (one media dir
  per scene: parallel manim processes race on the text-SVG cache otherwise). Never overwrite `*_vN.py`.
- QA: `python scratch/qa_sheets.py <video dir> <out dir> 2`.
- Data prep: `scratch/prep_data.py` (site media, exact layouts, paper-figure crops; PDF rasters are stored flipped).
