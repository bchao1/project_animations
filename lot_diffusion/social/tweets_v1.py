"""Thread drafts; run to check the 280-character limit (X counts every URL as 23 chars)."""
import re

T = {
"A1": ("C01_EqualCompute", """Diffusion transformers spend the same compute on every patch: empty background costs as much as a face. But where detail matters is often known in advance.

Introducing Level-of-Token Diffusion: multiresolution token layouts for efficient image & video generation. 1/7🧵"""),
"A2": ("C03_AnySize", """A Level-of-Token layout splits the latent grid into rectangles of different sizes and shapes (1×1 up to 8×8, including 1×4, 2×4, 4×1 …).

Each rectangle becomes ONE token, so the DiT processes L tokens instead of H×W. 2/7"""),
"A3": ("C05_AsymFlow", """How do you run a pretrained DiT on such a sequence? Patch-wise asymmetric flow: each patch is projected to one token, and per-extent heads recover a dense, full-resolution velocity at every step.

The VAE and the pretrained prior stay as they are. 3/7"""),
"A4": ("C04_Sources", """Layouts can come from anything that says where detail matters: bounding boxes, semantic masks, texture variance, depth of field.

A single fine-tuned FLUX.2 model follows all of them. 4/7"""),
"A5": ("R05_Driving", """It works for video too (Wan2.1 14B). In driving scenes, vehicles, people, lanes and traffic signs get fine tokens, while sky and asphalt get coarse ones. 5/7"""),
"A6": ("C08_Numbers", """At matched token budgets, LoT is faster than ToMe-SD, DDiT and Foveated Diffusion and scores best among the compressed methods on most quality metrics: up to 2.04× faster images, 3.53× faster video. 6/7"""),
"A7": ("no media (links only)", """Paper: <arXiv URL>
Project page: <URL>
5-min explainer: <YouTube URL>

Joint work with @BrianCChao, @YarivLior, <co-authors> 7/7"""),

"B1": ("R09_Typography", """With Foveated Diffusion we used two token sizes around a gaze point. The obvious next question: what if the model took ANY layout of detail, with tokens of ANY size and shape?

That's Level-of-Token Diffusion. You can even draw with tokens 👇 (full thread by @GordonWetzstein)"""),
"B2": ("R08_MoveTheBox", """A layout is not only a budget, it is also a control signal: move the fine-token box and the subject follows.

On 2,096 COCO prompts, the subject lands inside the fine region 66.8% of the time with LoT-FLUX.2 9B vs 44.9% for dense FLUX.2 9B."""),
"B3": ("C06_ShapeRoPE", """Two small changes teach a pretrained DiT about token size: RoPE indexed at each patch centre (exactly the pretrained RoPE for 1×1 tokens), and a zero-initialised patch-shape embedding. No new architecture, no training from scratch."""),
"B4": ("C07_Budget", """And one knob sets the trade-off: lower token budgets simplify textures while keeping the subject and the scene; higher budgets bring the fine detail back. Same model throughout."""),

"C1": ("R10_Boxes", """Layouts are becoming the interface of image models: Ideogram 4.0 is trained on JSON boxes per element, Reve 2 plans a hierarchical layout first.

But the layout only decides WHAT goes where, never how much compute it gets. Level-of-Token Diffusion changes that 👇"""),
"C2": ("C02_Lineage", """Graphics has done this for decades: level-of-detail meshes, variable-rate shading, image compression. They all adapt to content that already exists.

Generation is different: the plan exists before the first pixel, so compute can follow the plan."""),
"C3": ("R02_Agent", """The plan can come from a sentence: an agent paints a detail map, which becomes a token layout, which LoT-FLUX.2 turns into the image. Here: 2.6× fewer tokens."""),
"C4": ("R01_PaintGUI", """…or paint it yourself. White = fine, black = coarse; the map becomes a quadtree of tokens and you regenerate in seconds. (Recorded session, sped up.)"""),

"D1": ("R03_Scene3D", """For world models and simulators, Level-of-Token layouts can be keyframed: an animated 3D scene decides, frame by frame, where the fine tokens go. Here: 50,832 of 75,600 tokens (1.49× fewer), 81 frames at 720p, Wan2.1 14B."""),
"D2": ("R04_Games", """Games are a natural fit: the player and the level geometry get fine tokens, open sand and sky get coarse ones. About 2× fewer tokens in each of these clips."""),
"D3": ("R06_Robotics", """Robot manipulation: fine tokens on the arm, the gripper and the object of interest; the workspace and background get coarser tokens."""),
"D4": ("R07_DepthOfField", """Depth of field comes almost for free: out-of-focus regions tolerate big tokens. Depth → blur radius → token size, with 2.1–2.7× fewer tokens in these clips."""),
"D5": ("R11_VideoBaselines", """Same token layout for all three compressed models: ToMe-SD gets noisy, Foveated Diffusion distorts near resolution seams, LoT keeps the detail, and is the fastest of the three here (386 s vs 1,108 s full-res)."""),
}


def weight(s):
    s2 = re.sub(r"https?://\S+|<[^>]*URL>", "x" * 23, s)
    # X counts emoji and most CJK as 2; count non-BMP chars double
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s2)


if __name__ == "__main__":
    for k, (clip, s) in T.items():
        w = weight(s)
        print(f"{k:3s} {w:4d} {'OK ' if w <= 280 else 'TOO LONG'} {clip}")
