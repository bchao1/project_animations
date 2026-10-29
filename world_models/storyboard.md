# World Models — 10 s clip (ground truth)

For a promo video about Marlow: 10 seconds on world models, built from four lines of work. Ten style variants in
`styles/` (pick one); all share the same beats, text and assets via `wm_common.py`.

## Beats (all styles)
| t (s)     | Paper (control)                      | Demo clip (data/clips/)                                  | Paper figure (data/figcrops/)        |
|-----------|--------------------------------------|----------------------------------------------------------|--------------------------------------|
| 0.0–2.5   | CameraCtrl / CameraCtrl II (Camera)  | camctrl2: CC-II `citywalk1.mp4` generated panel, 1.3×    | cc1_row (CC Fig 1), cc2_rows (CC-II Fig 1) |
| 2.5–5.0   | Generated Reality (Hands)            | genreal: `teaser_final2.mp4` 22.25–24.6 s (dog + headset inset), 0.82× | gr_inputs (Fig 1 left: prompt + head/hand poses) |
| 5.0–7.5   | BulletTime (Time)                    | bullettime: `intro_public_1.mp4` 13.25–16.25 s (coupled → controlled camera & time) | bt_fig1, bt_bullet (Fig 5 row) |
| 7.5–10.0  | Spatial Memory (Memory)              | spmem: `demo_page.mp4` 5.6–8.8 s (memory diagram + generated rider) | sm_teaser (Fig 1, hi-res) |

Transitions ≈0.26–0.42 s centred on the boundaries. Styles end on a "all four controls lit" beat where it fits.

## Text overlays (exact, sans-serif Lato)
- Camera-controlled World Generation — ICLR 2025, ICCV 2025
- Interactive World Generation with Hand-object Interaction — CVPR 2026
- Space-time Controlled World Generation — CVPR 2026
- World Models with Spatial Memory — NeurIPS 2025

## Styles
01 chips · 02 full-bleed stories · 03 equation + brace · 04 grid zoom · 05 carousel · 06 paper → demo ·
07 paper white (light) · 08 concept glyph → video · 09 camera frustum · 10 kinetic type.
Each file's docstring describes its look.
