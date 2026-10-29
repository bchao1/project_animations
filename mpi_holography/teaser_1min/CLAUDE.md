# MPI holography — 1-minute teaser

Compact (~60 s) cut of the 5-min teaser in the parent folder (`../animation_mpi_v3.py`, `../storyboard.md`).
Ground truth: `storyboard.md` (current script named at its top). Scenes: T1_Hook, T2_Method, T3_Payoff.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set in `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]` → `media/final/MPIHolography_1min_vN_<res>.mp4` (all scenes only).
- Assets are read by absolute path from `../data` (figure crops, `clips/*.npy` homepage-video frames) and `../data/generated`
  (toy caches built by the 5-min script, flag `done_v1`; this script never writes them and asserts they exist).
  Helpers are copied from the 5-min script, not imported.
- No LaTeX (Pango markup via `tx()`/`mathify()`), CMU Serif, text added instantly, captions short, one equation (Eq. 2).
- Versioning: never overwrite `animation_mpi_1min_vN.py`; copy to vN+1 and update `storyboard.md`.
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets; keep scratch files in `scratch/`.
