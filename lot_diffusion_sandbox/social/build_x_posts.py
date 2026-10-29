# Builds x_posts.md with per-post character counts (URLs count as 23 on X).
import re
URL = re.compile(r"https?://\S+|<[^>]*(link|page)[^>]*>")
def xlen(s): return len(URL.sub("x" * 23, s))

master = [
("1/7 🧵", "`clips/post1_hook.mp4` (11.3 s): full-res corgi from the 1,546-token layout first (0–2 s), then face vs sky at 4,096 tokens",
"""Image and video diffusion models spend the same compute on every patch: a face and an empty sky each get one token, at every denoising step. But does every region need the same level of detail?

Introducing Level-of-Token (LoT) Diffusion: adaptive, efficient generation that puts fine tokens where detail matters and coarse tokens everywhere else.

1/7🧵"""),
("2/7", "`clips/post2_layout.mp4` (13.6 s): painted detail map → merge passes → 1,546 of 4,096, 2.65× fewer",
"""Our key idea comes from level-of-detail rendering in graphics: spend resources where the detail is. And where detail matters is often known before generation, from masks, boxes, depth of field, or a plan.

LoT turns that into a layout of tokens of different sizes: fine on the subject, coarse on the sky.

The result? This image needs 1,546 tokens instead of 4,096, 2.65× fewer."""),
("3/7", "`clips/post3_method.mp4` (12.4 s): blocky toy vs LoT output, then DiT + LoRA → size-specific head → 16 fine flows",
"""Naively feeding big tokens to a pretrained DiT fails: a coarse token is a single vector, so decoding it as-is gives a blocky image, and the model only predicts flow for the tokens it sees.

We develop a patch-wise asymmetric flow parametrization with size-specific heads and shape embeddings, so a fine-tuned DiT predicts full-resolution flow for every patch while reading only the short token sequence."""),
("4/7", "`clips/post4_results.mp4` (12.9 s): 2.04× / 3.53× with the driving clip, then Fig. 7a at 1,269 tokens",
"""LoT is a general framework: we fine-tune FLUX.2 [klein] 4B for images and Wan2.1 14B for video, reaching up to 2.04× and 3.53× speedups at ≈3× fewer tokens.

At the same token budget, token-reduction baselines turn noisy or break at token borders (best HPSv3: 7.84), while LoT stays close to full resolution (9.55 vs 10.43)."""),
("5/7", "`clips/post5_sources.mp4` (12.5 s): one deer from box, mask, texture-variance and depth layouts, compression · speedup",
"""A neat property: the layout can come from anywhere. We derive it from semantic masks, bounding boxes, texture variance, depth of field, or a detail map painted by a user or an agent.

The layout also doubles as a control signal: on 2,096 COCO prompts, LoT raises target-region coverage of FLUX.2 9B from 44.9% to 66.8%."""),
("6/7", "`clips/post6_applications.mp4` (6.0 s): driving, robotics and game videos with their layouts (the site clips are 81 frames)",
"""This makes LoT a natural fit wherever the region of interest is known ahead of time: driving and robotics simulation from semantic masks, games from bounding boxes, and agentic pipelines where a planner draws the detail map before the model renders a single pixel."""),
("7/7", "none",
"""Work led by @BrianCChao, together with @YarivLior and @<4th co-author>
📄 Paper: <arXiv link>
🌐 Project page: <project page>"""),
]

quotes = [
("@BrianCChao", "origin → deep dive", 15, [
("q1_1_plan.mp4 (13.7 s): prompt → an agent's detail map → “Can a diffusion model spend its tokens where the plan says?”",
"""After Foveated Diffusion, one question kept coming back: gaze is only one way to know where detail matters. Why not use everything else we know before generating, like masks, boxes, depth of field, or an agent's plan? In driving simulation, games, and design tools, that knowledge is usually already there.

Our new work, Level-of-Token Diffusion, turns it into a layout of variable-size tokens, so a pretrained DiT spends fine tokens only where the scene needs them and still predicts every pixel at full resolution.

See the full breakdown in @GordonWetzstein's post below:"""),
("q1_2_flow.mp4 (13.3 s): the DiT sees a 4×4 patch through one token → Eq. 9 fills the rest, exact with a perfect head",
"""The part I'm proudest of: how a coarse token still gives full-resolution output. A 4×4 token is one vector, so the DiT sees that patch only through it.

A size-specific head predicts the part of the flow that one vector can carry, and Eq. 9 completes the rest from the noisy input itself. With a perfect head, all 16 positions get their detail back exactly.

[2/4]"""),
("q1_3_dit.mp4 (14.0 s): per-extent projections and heads, shape embedding, RoPE at the patch centre",
"""The DiT also has to know what each token is. Every token size gets its own input projection and output head, initialised from the pretrained ones; a zero-initialised shape embedding tells each token its size and aspect; RoPE indexes it at its patch centre, so a 1×1 token keeps its exact pretrained position.

Only a rank-256 LoRA, the heads and the shape MLP are trained, on FLUX.2 klein 4B and Wan2.1 14B.

[3/4]"""),
("q1_4_ideas.mp4 (7.6 s): “LoT in three ideas” summary",
"""Foveated Diffusion spent tokens where you look. LoT spends them wherever the plan says: a mask, a box, a depth map, or an agent's sketch. Same pretrained model, one layout, up to 2.04× faster images and 3.53× faster video.

📄 Paper: <arXiv link>
🌐 Project page: <project page>

[4/4]"""),
]),
("@<4th co-author>", "skeptic → evidence", 45, [
("q2_1_cost.mp4 (14.3 s): 4,096² ≈ 16.8 M query–key pairs; 2× resolution → 4× tokens, 16× attention",
"""Level-of-detail is one of the oldest ideas in graphics: the mountain on the horizon gets fewer triangles than the character in front. Diffusion models still spend the same on both, and token-reduction methods guess on the fly, with no way to say where the detail should go.

Our new work, Level-of-Token Diffusion, takes that as an input layout: NO merging heuristics, NO training from scratch, NO fixed patch size. In generative models, tokens are the new triangles.

(Full thread below)"""),
("q2_2_baselines.mp4 (11.8 s): HPSv3-vs-speedup plot (Tab. 1) → Wan2.1 video at the same budget (Fig. 7b)",
"""Why not just merge or prune tokens at runtime? At the same budgets, those baselines blur or break at token borders.

On FLUX.2 images, LoT is the fastest method and scores closest to full resolution (HPSv3 9.55 vs 10.43 full; best baseline 7.84). On Wan2.1 video it runs 3.53× faster with the best imaging quality of the compressed methods (0.565 vs 0.495).

[2/4]"""),
("q2_3_ablation.mp4 (6.4 s): Tab. 3 ablation bars",
"""Every piece earns its place. HPSv3 at ≈1.52× compression: predicting the high-res output directly 7.06, a binary 1×1/2×2 grid 8.20, no size embedding 10.03, full LoT 10.35.

[3/4]"""),
("q2_4_limit.mp4 (6.5 s): Fig. 22 sheet music at 3,189 vs 1,466 tokens, broken staff lines",
"""Where it breaks: when detail is everywhere, like a page of sheet music, there is little to save (3,189 tokens, 1.28×), and forcing a low budget (1,466 tokens, 2.79×) breaks the staff lines. The layout has to match the content.

[4/4]"""),
]),
("@YarivLior", "question → vision", 105, [
("q3_1_emblem.mp4 (6.3 s): title with the layout emblem splitting 8×8 → 1×1 tokens",
"""Why should an empty sky cost as many tokens as a face?
In our new work, Level-of-Token Diffusion, each region gets only as many tokens as its detail needs, inspired by level-of-detail rendering in graphics.
(See full thread below)"""),
("q3_2_agent.mp4 (10.5 s): site demos, agent-built Blender scenes → detail maps → LoT-Wan2.1 video (1.49× and 2.36× fewer tokens)",
"""Where does the plan come from? More and more, from an agent.

Here an agent builds and animates a Blender scene and marks what needs detail; we rasterize that from the moving camera into a layout, and LoT-Wan2.1 renders the video with fine tokens on the train and the people, coarser tokens everywhere else.

[2/3]"""),
("q3_3_outlook.mp4 (8.3 s): plan → compute → image",
"""I think layouts can become an editable plan between user, agent and model: first decide where the compute goes, then generate.

🌐 <project page>

[3/3]"""),
]),
]

out = ["# Level-of-Token Diffusion – X launch set", "",
"Assumptions:",
"- Master account @GordonWetzstein; quotes from @BrianCChao (origin story, as first author), @YarivLior, and an unnamed 4th co-author (handle unknown). The sandbox paper is anonymous, so the author list and order need your check.",
"- Links are placeholders: the paper is under double-blind review (ICLR 2027 #7014) and the supplementary site is anonymous. Post once the arXiv version and a named project page exist.",
"- 7 posts (not 5): the paper has both an extra property (any layout source + controllability) and a clear application story. For a 5-post cut, drop 5/7 and 6/7.",
"- Clips: `social/clips/` (1080p60 h264, no audio), cut from the 1-min v4 and 5-min v3 scene renders and the site's video demos by `social/make_clips.sh` (cut points are in the script).",
"- Every number comes from `level-of-token-diffusion/summaries/lot_diffusion.md` (Tab. 1, 2, 4; site demo).", "",
"All 17 clips are in `social/clips/`; each content post has its own clip and no two clips share footage (quote-thread clips use teaser scenes and site demos the master thread does not).", "",
"## Master thread — @GordonWetzstein — Tuesday 8:30 am PT", ""]
for tag, media, text in master:
    out += [f"### {tag} ({xlen(text)} chars) — media: {media}", "", text, ""]
for i, (who, lane, mins, posts) in enumerate(quotes, 1):
    t = f"+{mins} min" if mins < 60 else f"+{mins//60} h {mins%60} min"
    out += [f"## Quote thread {i} — {who} — lane: {lane} — {t}", ""]
    for k, (media, text) in enumerate(posts, 1):
        q = " quotes master 1/7" if k == 1 else ""
        out += [f"### [{k}/{len(posts)}]{q} ({xlen(text)} chars) — clip: `clips/{media.split(' ', 1)[0]}` {media.split(' ', 1)[1]}", "", text, ""]
open("x_posts.md", "w").write("\n".join(out))
p1 = master[0][2]; i = p1.find("Level-of-Token")
print("post1 name ends at char", i + len("Level-of-Token (LoT) Diffusion"), "| post1 len", xlen(p1))
for tag, _, text in master: print(tag, xlen(text))
for who, a, _, posts in quotes: print(who, a, [xlen(t) for _, t in posts])
