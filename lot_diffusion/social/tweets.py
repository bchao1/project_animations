"""Thread drafts (v2 clips: ../clips/media/final/1080p60/LoT_<clip>.mp4). Run to check the 280-char limit (URL = 23)."""
import re

T = {
"A1": ("K01_TokenCost", """Diffusion transformers spend the same compute on every token: empty sky costs as much as a face. But where detail matters is often known in advance.

Introducing Level-of-Token Diffusion: multiresolution token layouts for efficient image & video generation. 1/7🧵"""),
"A2": ("K04_Merge", """A Level-of-Token layout decides, per region, how big a token is. Where the plan asks for less detail, 2×2 or 4×4 blocks of tokens merge into one.

Same full-resolution image, far fewer tokens for the DiT. 2/7"""),
"A3": ("K09_Flow", """The trick that makes this work with pretrained DiTs: patch-wise asymmetric flow. Per-size heads predict the flow for each patch, and the rest is recovered in closed form, so every step still yields full-resolution flow. 3/7"""),
"A4": ("K11_Sources", """Layouts can come from anything that says where detail matters: bounding boxes, semantic masks, texture variance, depth.

A single fine-tuned FLUX.2 model follows all of them. 4/7"""),
"A5": ("K14_Driving", """It works for video too (Wan2.1 14B). Here fine tokens follow cars, people and signs frame by frame, while sky and asphalt get coarse ones: 1.97× fewer tokens. 5/7"""),
"A6": ("K15_Speed", """Up to 2.04× faster images and 3.53× faster video than the full-resolution models, and better quality than ToMe-SD, DDiT and Foveated Diffusion at the same token budgets. 6/7"""),
"A7": ("no media (links only)", """Paper: <arXiv URL>
Project page: <URL>
5-min explainer: <YouTube URL>

Joint work with @BrianCChao, @YarivLior, <co-authors> 7/7"""),

"B1": ("K05_Shapes", """With Foveated Diffusion we used two token sizes around a gaze point. The next question: what if a DiT took ANY layout of detail, with tokens of any size and shape, from 1×1 to 8×8, squares or rectangles?

That's Level-of-Token Diffusion 👇 (thread by @GordonWetzstein)"""),
"B2": ("K13_MoveBox", """A layout is also a control signal: move the fine-token box and the subject follows.

Over 2,096 COCO prompts, the subject lands inside the fine region 66.8% of the time with LoT-FLUX.2 9B vs 44.9% for dense FLUX.2 9B."""),
"B3": ("K10_RoPE", """Two small changes teach a pretrained DiT about token size: RoPE indexed at each patch centre (exactly the pretrained RoPE for 1×1 tokens), and a zero-initialised size embedding. No new architecture, no training from scratch."""),
"B4": ("K12_Budget", """And one knob sets the trade-off: lower token budgets simplify textures while keeping the subject and the scene; higher budgets bring the detail back. Same model throughout."""),

"C1": ("K03_Plan", """Image models increasingly plan a layout before they draw: boxes, regions, even painted detail maps.

But that plan only decides WHAT goes where, never how much compute each region gets. Level-of-Token Diffusion turns the plan into compute 👇"""),
"C2": ("K02_Attention", """Why it matters: self-attention compares every token with every other token, at every denoising step. Double the resolution and you pay 4× the tokens and 16× the attention."""),
"C3": ("K06_Sequence", """So a layout that halves the tokens cuts attention by far more: in this toy layout, 126 tokens instead of 256 means about a quarter of the attention work."""),
"C4": ("K16_Quality", """And the savings don't cost quality: at the same budget LoT scores closest to full-resolution FLUX.2 on HPSv3 (9.55 vs 10.43), well ahead of token merging, DDiT and Foveated Diffusion."""),

"D1": ("K18_VideoCompare", """Same token budget, four video models (Wan2.1 14B): token merging turns to noise, Foveated Diffusion distorts at resolution seams, Level-of-Token Diffusion holds up, and is the fastest of the three."""),
"D2": ("K07_Naive", """Why not just feed big tokens to a pretrained DiT and ask for full-resolution flow? It produces noise everywhere. Two things were missing: the right flow target, and telling the model each token's size."""),
"D3": ("K08_Patch", """The fix starts here: a fixed basis per token size projects each patch of the latent onto one ordinary-size token, so the pretrained DiT sees inputs it understands."""),
"D4": ("K17_ImageCompare", """Images at the same budget (1,269 tokens): baselines blur, get noisy or show seams at token borders. LoT keeps the detail and is the fastest, 2.29× vs full resolution."""),
}


def weight(s):
    s2 = re.sub(r"https?://\S+|<[^>]*URL>", "x" * 23, s)
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s2)


if __name__ == "__main__":
    used = [c for c, _ in T.values() if c.startswith("K")]
    assert len(used) == len(set(used)), "a clip is used twice"
    for k, (clip, s) in T.items():
        w = weight(s)
        print(f"{k:3s} {w:4d} {'OK ' if w <= 280 else 'TOO LONG'} {clip}")
    print(len(set(used)), "clips used")
