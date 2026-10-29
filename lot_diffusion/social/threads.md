# Level-of-Token Diffusion — X thread plan (4 people)

Every thread clip is used exactly once across the four threads, so the threads look different even when
they are reposted or quoted side by side. Post text lives in `tweets.py` (run it to check lengths; X counts a
URL as 23 characters). Clips: `../clips/media/final/1080p60/LoT_<clip>.mp4` (1920×1080, 60 fps, 8–20 s, no audio; each is one beat of the 5-min teaser).
Teasers: `../media/final/LevelOfTokenDiffusion_5min_v2_1080p60.mp4`, `../teaser_1min/media/final/LevelOfTokenDiffusion_1min_v2_1080p60.mp4`.

## Before posting

- The paper is an anonymous ICLR 2027 submission. Link an arXiv / project page, not the anonymous supplementary
  site, and check the venue's current policy on promoting submissions under review.
- Fill in the `<…URL>` placeholders, the co-author handles in A7, and who person D is.
- The clips have no sound; X autoplays muted, so nothing is lost.

## Who posts what

| | account | role |
|---|---|---|
| A | @GordonWetzstein | Main thread (7 posts). Big picture: the problem, the layout idea, the method, results, numbers. |
| B | @BrianCChao | Quote of A1 (4 posts). Origin story (Foveated Diffusion → any layout); layouts as control; the two DiT changes; the budget knob. |
| C | @YarivLior | Quote of A1 or B1 (4 posts). Plans already exist; attention cost; how much a layout saves; quality at the same budget. |
| D | <4th co-author> | Quote of A5, the video post (4 posts). Video comparison; why the naive approach fails; the projection; image comparison. |

## Schedule (T = launch time)

| when | who | action |
|---|---|---|
| T+0 | A | Post the 7-post main thread A1–A7. |
| T+10 min | B | Quote A1 with B1, reply-chain B2–B4 under B1. Repost A1. |
| T+45 min | C | Quote A1 (or B1) with C1, reply-chain C2–C4. Repost A1 and B1. |
| T+2 h | D | Quote A5 (the driving-video post) with D1, reply-chain D2–D4. Repost A1. |
| T+2–4 h | A | Repost B1, C1, D1. Reply to B2 or D5 with one sentence of your own (not a copy of the post). |
| T+4 h | B, C | Repost each other's quote and D1. |
| Day 1 | A | Standalone post with the 1-min cut ('In case you missed it…'), linking to A1. |
| Day 2 | any | Reply to A7 with the YouTube link of the 5-min explainer once it is up. |

## Thread A — @GordonWetzstein

Main thread (7 posts). Big picture: the problem, the layout idea, the method, results, numbers.

### A1 · media: `LoT_K01_TokenCost.mp4` · 263/280

```
Diffusion transformers spend the same compute on every token: empty sky costs as much as a face. But where detail matters is often known in advance.

Introducing Level-of-Token Diffusion: multiresolution token layouts for efficient image & video generation. 1/7🧵
```

### A2 · media: `LoT_K04_Merge.mp4` · 208/280

```
A Level-of-Token layout decides, per region, how big a token is. Where the plan asks for less detail, 2×2 or 4×4 blocks of tokens merge into one.

Same full-resolution image, far fewer tokens for the DiT. 2/7
```

### A3 · media: `LoT_K09_Flow.mp4` · 223/280

```
The trick that makes this work with pretrained DiTs: patch-wise asymmetric flow. Per-size heads predict the flow for each patch, and the rest is recovered in closed form, so every step still yields full-resolution flow. 3/7
```

### A4 · media: `LoT_K11_Sources.mp4` · 178/280

```
Layouts can come from anything that says where detail matters: bounding boxes, semantic masks, texture variance, depth.

A single fine-tuned FLUX.2 model follows all of them. 4/7
```

### A5 · media: `LoT_K14_Driving.mp4` · 162/280

```
It works for video too (Wan2.1 14B). Here fine tokens follow cars, people and signs frame by frame, while sky and asphalt get coarse ones: 1.97× fewer tokens. 5/7
```

### A6 · media: `LoT_K15_Speed.mp4` · 173/280

```
Up to 2.04× faster images and 3.53× faster video than the full-resolution models, and better quality than ToMe-SD, DDiT and Foveated Diffusion at the same token budgets. 6/7
```

### A7 · media: no media (links only) · 168/280

```
Paper: <arXiv URL>
Project page: <URL>
5-min explainer: <YouTube URL>

Joint work with @BrianCChao, @YarivLior, <co-authors> 7/7
```


## Thread B — @BrianCChao

Quote of A1 (4 posts). Origin story (Foveated Diffusion → any layout); layouts as control; the two DiT changes; the budget knob.

### B1 · media: `LoT_K05_Shapes.mp4` · 268/280

```
With Foveated Diffusion we used two token sizes around a gaze point. The next question: what if a DiT took ANY layout of detail, with tokens of any size and shape, from 1×1 to 8×8, squares or rectangles?

That's Level-of-Token Diffusion 👇 (thread by @GordonWetzstein)
```

### B2 · media: `LoT_K13_MoveBox.mp4` · 217/280

```
A layout is also a control signal: move the fine-token box and the subject follows.

Over 2,096 COCO prompts, the subject lands inside the fine region 66.8% of the time with LoT-FLUX.2 9B vs 44.9% for dense FLUX.2 9B.
```

### B3 · media: `LoT_K10_RoPE.mp4` · 224/280

```
Two small changes teach a pretrained DiT about token size: RoPE indexed at each patch centre (exactly the pretrained RoPE for 1×1 tokens), and a zero-initialised size embedding. No new architecture, no training from scratch.
```

### B4 · media: `LoT_K12_Budget.mp4` · 172/280

```
And one knob sets the trade-off: lower token budgets simplify textures while keeping the subject and the scene; higher budgets bring the detail back. Same model throughout.
```


## Thread C — @YarivLior

Quote of A1 or B1 (4 posts). Plans already exist; attention cost; how much a layout saves; quality at the same budget.

### C1 · media: `LoT_K03_Plan.mp4` · 241/280

```
Image models increasingly plan a layout before they draw: boxes, regions, even painted detail maps.

But that plan only decides WHAT goes where, never how much compute each region gets. Level-of-Token Diffusion turns the plan into compute 👇
```

### C2 · media: `LoT_K02_Attention.mp4` · 171/280

```
Why it matters: self-attention compares every token with every other token, at every denoising step. Double the resolution and you pay 4× the tokens and 16× the attention.
```

### C3 · media: `LoT_K06_Sequence.mp4` · 153/280

```
So a layout that halves the tokens cuts attention by far more: in this toy layout, 126 tokens instead of 256 means about a quarter of the attention work.
```

### C4 · media: `LoT_K16_Quality.mp4` · 183/280

```
And the savings don't cost quality: at the same budget LoT scores closest to full-resolution FLUX.2 on HPSv3 (9.55 vs 10.43), well ahead of token merging, DDiT and Foveated Diffusion.
```


## Thread D — <4th co-author>

Quote of A5, the video post (4 posts). Video comparison; why the naive approach fails; the projection; image comparison.

### D1 · media: `LoT_K18_VideoCompare.mp4` · 197/280

```
Same token budget, four video models (Wan2.1 14B): token merging turns to noise, Foveated Diffusion distorts at resolution seams, Level-of-Token Diffusion holds up, and is the fastest of the three.
```

### D2 · media: `LoT_K07_Naive.mp4` · 201/280

```
Why not just feed big tokens to a pretrained DiT and ask for full-resolution flow? It produces noise everywhere. Two things were missing: the right flow target, and telling the model each token's size.
```

### D3 · media: `LoT_K08_Patch.mp4` · 163/280

```
The fix starts here: a fixed basis per token size projects each patch of the latent onto one ordinary-size token, so the pretrained DiT sees inputs it understands.
```

### D4 · media: `LoT_K17_ImageCompare.mp4` · 166/280

```
Images at the same budget (1,269 tokens): baselines blur, get noisy or show seams at token borders. LoT keeps the detail and is the fastest, 2.29× vs full resolution.
```

## Clip index

| clip | what it shows | thread |
|---|---|---|
| `LoT_K01_TokenCost.mp4` | corgi image → 16 × 16 patch-colour tokens; a face token vs a sky token cost the same; 256 tokens | A1 |
| `LoT_K02_Attention.mp4` | attention matrix with a moving query row; N → 4N tokens, N² → 16N² pairs | C2 |
| `LoT_K03_Plan.mp4` | plan boxes → agent-painted detail map → generated image; the plan never sets compute | C1 |
| `LoT_K04_Merge.mp4` | 256 tokens merge by the detail map (2×2, then 4×4) → count column, L = 133 (toy) | A2 |
| `LoT_K05_Shapes.mp4` | square tokens become rectangles (2×4, 8×4); R_i equation; L = 126 (toy) | B1 |
| `LoT_K06_Sequence.mp4` | the layout flattens to a 126-token sequence; attention area ≈ 24% (toy) | C3 |
| `LoT_K07_Naive.mp4` | paper Fig. 21: direct full-res prediction gives noise; ours is clean | D2 |
| `LoT_K08_Patch.mp4` | noisy latent → one 2×2 patch → A_eᵀ → one ordinary-size token | D3 |
| `LoT_K09_Flow.mp4` | short sequence → DiT (LoRA) → size head → Eq. 9 (colour-coded) → full-res velocity → VAE | A3 |
| `LoT_K10_RoPE.mp4` | RoPE at the patch centre; size embedding s(e) bars → zero-init MLP | B3 |
| `LoT_K11_Sources.mp4` | paper Fig. 4: box / mask / texture / depth layouts → images, compression · speed-up | A4 |
| `LoT_K12_Budget.mp4` | paper Fig. 10: deer at B = 1,024 … 4,096 tokens | B4 |
| `LoT_K13_MoveBox.mp4` | paper Fig. 19: move the fine box, subject follows; Tab. 4 44.9% → 66.8% | B2 |
| `LoT_K14_Driving.mp4` | site driving video: semantic layout + LoT-Wan2.1 video, 1.97× fewer tokens | A5 |
| `LoT_K15_Speed.mp4` | speed-up vs compression curves (Tab. 1, 2); 2.04× images, 3.53× video | A6 |
| `LoT_K16_Quality.mp4` | HPSv3 bars at the same budget: full / ToMe-SD / DDiT / Foveated / ours | C4 |
| `LoT_K17_ImageCompare.mp4` | paper Fig. 7a row with speed-ups and failure notes | D4 |
| `LoT_K18_VideoCompare.mp4` | site video comparison (paper Fig. 7b example), 4 playing videos | D1 |
