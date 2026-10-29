# World Models 10 s clip

10-second 3Blue1Brown-style clip on world models (four papers, ~2.5 s each) for a Marlow promo video; ten style
variants to pick from. Ground truth: `storyboard.md`.

- Environment: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render_styles.sh`). No LaTeX.
- Render: `./render_styles.sh l [NN ...]` (480p15) or `./render_styles.sh h` (1080p60, trimmed to exactly 600 frames). All ten run in parallel; finals
  land in `media/final/WorldModels_sNN_<style>_<res>.mp4`, logs in `media/logs/`. Parallel jobs can race on the shared
  text cache, so the script retries.
- Code: `wm_common.py` holds everything shared (segment data + exact overlay text, `VideoCard` playing clip mobject,
  control icons/chips, concept glyphs, captions, figure stacks, `TimedScene` absolute clock). One scene per file in `styles/`.
- `VideoCard`: position/size/opacity/darken are ValueTrackers applied by an updater, so clips keep playing during moves
  and fades (don't FadeIn/Transform the ImageMobject directly — that freezes it). Videos sit at z-index −2 under overlays.
- Assets: `python scratch/prep_assets.py` → `data/clips/<key>/%04d.jpg` (30 fps, 1920 wide) and `data/figcrops/*.png`.
  Raw demo videos (from the project pages) in `data/videos/`, raw figure crops in `data/figs/`.
- Paper-finder records (PDFs, extracted text, project-page HTML, summaries, memory bank, mind graph, BibTeX): `world-models/`.
- QA: `scratch/qa.sh <video> <out.jpg>` makes a 12-frame contact sheet.
- Full-bleed styles (02, 10): Generated Reality's headset inset sits bottom-left, so its caption is right-aligned on two lines.
- Style picker website: `site/` (index.html + symlinked finals + posters). Serve with `python3 site/serve.py 8420`
  (range-capable, needed for video seeking) → http://apollo.stanford.edu:8420/
