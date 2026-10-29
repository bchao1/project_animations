# LoT edit results clips (edit_results_clip/)

Started 2026-10-08. Two 20-s 1080p60 clips from ~/LoT-george-branch/results/2026-10-08_173831_ref-edit-9b-distilled-bottles-100turn-ctx-sweep
(read-only), in the edit_clip v3 look (equal centred panels, label above, amber count below, one caption).
- `render_ctx_v1.py` → `media/LoT_ctx_v1_1080p60.mp4`: k = 1 / 2 / 4 / 8 context. Turn-1 layouts sweep in
  (4,096 / 1,423 / 826 / 694 tokens), then turns 1, 2, 3, 10, 25, 50, 75, 100 side by side, then avg target + context
  tokens of 8,192 (100 / 34 / 19 / 16%), 7.7 s · 2.6× · 4.3× · 4.8×, edit success 77 / 78 / 73 / 76% (seed floor 74–77%).
- `render_drift_v1.py` → `media/LoT_drift_v1_1080p60.mp4`: LoT editor (k = 4) vs re-encode loop, 100 turns; untouched-pixel
  PSNR vs turn 0 under each (bit-exact vs 28.5 → 3.5 dB, from compare/per_turn.csv keep_psnr).
- Data: `prep_data.py` (imp_tome) → `data/results.json`, `data/leaves_turn1.npz`. Shared drawing: `common_v1.py`.
- Render: `~/miniconda3/envs/imp_tome/bin/python render_<ctx|drift>_v1.py h|l` (~1 min) or `still t …`. Never overwrite a version.
- Don't show the re-encode loop's edit success / crop metrics (README: indicative only, objects move) or its 16.1 s timing (shared GPU).
- v2 (2026-10-08, user): `render_ctx_v2.py` drops the edit-success line (caption "Same edits with 16–34% of the tokens,
  up to 4.8× faster"). `render_drift_v2.py` drops the re-encode loop and untouched-pixel numbers: k = 4 image + "turn N"
  box + top "Change … to …" text box, three live rolling-mean (10-turn) curves — CLIP edit margin, CLIP-IQA, MUSIQ of the
  edited crop — with a dashed turns-1–20 level; end shows first-20 → last-20 means (0.051 → 0.058, 0.76 → 0.76, 76.2 → 74.1).
- v3 ctx (2026-10-08, user): `render_ctx_v3.py` shows "X s / edit" under every k (7.7 / 3.0 / 1.8 / 1.6) and a purple
  "N× faster" line under 2 × 2, 4 × 4, 8 × 8 (2.6 / 4.3 / 4.8×). Current: ctx v3, drift v2.
- v3 drift (2026-10-08, user): `render_drift_v3.py` — no re-encode (its drift is about the editor design, not LoT: k = 1
  with the same editor is also bit-exact), no CLIP margin (noisy). CLIP-IQA + MUSIQ of each edit, k = 4 (amber) vs k = 1
  full res (white), 10-turn rolling means, axes 0.55–0.85 / 65–80; end = 100-turn means 0.75 vs 0.69, 74.8 vs 71.3.
- `render_race_v1.py` → `media/LoT_race_v1_1080p60.mp4` (22 s): remake of George's race video in this style — LoT 4 × 4
  vs full res on a shared clock (measured per-turn dit_s; LoT's 100 turns = 179 s at 20 s, full res reaches 23), no side
  column; per panel: label "name · X s / edit", 2-line prompt text box, large "turn N" box bottom-left, progress bar
  "N of 100 edits · T s"; 2-s hold with a purple "4.3× faster" box. User kept drift v3 ("at or above full res").
