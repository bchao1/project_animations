"""Level-of-Token Diffusion — ~5-minute 3b1b-style teaser, v1.

Built from the 10-second thread clips (clips/animation_lot_clips_v1.py, replayed as "bodies") plus explainer-only
beats: attention cost, flow-matching / AsymFlow background, layout-source rules (paper Sec. A.3), training setup,
ablation (Tab. 3), limitation (Fig. 22) and outlook. Numbers: 7014_Level_of_Token_Diffusion.pdf / supplementary site.
Storyboard: storyboard.md. Render: ./render.sh l|h
Scenes: S00_Title S01_Problem S02_Layouts S03_Sources S04_Method S05_LoTDiT S06_Video S07_Control S08_Compare
        S09_Outlook
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "clips"))
from animation_lot_clips_v1 import *  # noqa: E402,F403  (lot_common + clip bodies)
import animation_lot_clips_v1 as CL  # noqa: E402


class T(CL.Clip):
    """teaser scene: no corner tag; body() replays a 10 s clip; narrate() shows timed captions"""

    def setup(self):
        Base.setup(self)
        self.head = None

    def body(self, cls, extra=0.0):
        self.t0, self.extra = self.now, extra
        cls.construct(self)
        self.t0, self.extra = 0.0, 0.0

    def wipe(self, run_time=0.5):
        self._hold_caption()
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)
        for m in mobs:
            m.clear_updaters()
        self.clear()
        self.head, self.cap = None, None

    def part(self, top, title, hold=1.6):
        self.title_card(top, title, hold=hold)

    def say(self, s, size=28, y=CAPTION_Y):
        """caption that holds for its reading time before the next one replaces it"""
        return self.caption(s, size=size, y=y)


def card_box(lines, w=6.2, color=WHITE, title=None, size=24):
    items = VGroup(*[tx(s, size) for s in lines]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
    if title:
        items = VGroup(tx(title, size + 2, color, bold=True), items).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    box = RoundedRectangle(width=max(w, items.width + 0.6), height=items.height + 0.55, corner_radius=0.14)
    box.set_stroke(color, 2).set_fill(BOX, 0.9)
    return VGroup(box, items.move_to(box))


# ------------------------------------------------------------------ scenes
class S00_Title(T):
    def construct(self):
        S = 5.4
        g = img("fig1_grid.png", h=S).move_to([0, 0.35, 0])
        im = img("fig1_image.png", h=S).move_to([0, 0.35, 0])
        fr = framed(g)
        self.add(g, fr)
        self.wait(1.2)
        self.play(FadeIn(im), run_time=1.4)
        self.wait(0.8)
        self.play(Group(g, im, fr).animate.scale(0.62).move_to([-3.75, 0.3, 0]), run_time=0.9)
        t = tx("Level-of-Token Diffusion", 56, bold=True)
        s1 = tx("spend tokens where the detail is", 30, DIM)
        s2 = tx("efficient, layout-adaptive image and video generation", 26, DIM)
        grp = VGroup(t, s1, s2).arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to([2.95, 0.3, 0])
        if grp.width > 7.3:
            grp.scale_to_fit_width(7.3).move_to([2.95, 0.3, 0])
        self.add(grp)
        self.wait(3.0)


class S01_Problem(T):
    def construct(self):
        self.part("Part 1 · The problem", "Every token costs the same")
        self.body(CL.C01_EqualCompute, extra=0.4)
        self.wipe()
        self.header("1 · Why it matters")
        # attention cost: N tokens -> N^2 pairs
        sq1 = Square(0.9).set_fill(FINE, 0.8).set_stroke(width=0).move_to([-4.2, 0.6, 0])
        sq2 = Square(1.8).set_fill(FINE, 0.8).set_stroke(width=0).move_to([-1.6, 0.6, 0])
        att1 = Square(0.9).set_fill(AMBER, 0.8).set_stroke(width=0).move_to([1.6, 0.6, 0])
        att2 = Square(3.6).set_fill(AMBER, 0.8).set_stroke(width=0).move_to([4.6, 0.6, 0])
        l1 = VGroup(tx(f"1× resolution: {it('N')} tokens", 22, DIM).next_to(sq1, DOWN, buff=0.25),
                    tx(f"2×: 4{it('N')} tokens", 22, DIM).next_to(sq2, DOWN, buff=0.25),
                    tx(f"{it('N')}{sup('2')} pairs", 22, DIM).next_to(att1, DOWN, buff=0.25),
                    tx(f"16 {it('N')}{sup('2')} attention pairs", 22, DIM).next_to(att2, DOWN, buff=0.25))
        hdr = VGroup(tx("tokens", 26).move_to([-2.9, 3.0, 0]), tx("self-attention", 26).move_to([3.1, 3.0, 0]))
        self.add(hdr)
        self.play(FadeIn(sq1), FadeIn(l1[0]), run_time=0.4)
        self.play(TransformFromCopy(sq1, sq2), FadeIn(l1[1]), run_time=0.7)
        self.play(FadeIn(att1), FadeIn(l1[2]), run_time=0.4)
        self.play(TransformFromCopy(att1, att2), FadeIn(l1[3]), run_time=0.8)
        self.say("Double the resolution: 4× the tokens, 16× the attention. Video multiplies it again by frames")
        self.say("Yet empty sky, flat walls and blurred backgrounds need far less detail than faces or text")
        self.wipe()
        self.body(CL.C02_Lineage, extra=0.5)
        self.wipe()


class S02_Layouts(T):
    def construct(self):
        self.part("Part 2 · The idea", "Level-of-Token layouts")
        self.body(CL.C03_AnySize, extra=0.6)
        self.say("A layout partitions the token grid into rectangles; each one becomes a single DiT token", y=-3.62)
        self.say(f"The DiT now processes {it('L')} tokens instead of {it('HW')} ({it('THW')} for video)", y=-3.62)
        self.wipe()


class S03_Sources(T):
    def construct(self):
        self.body(CL.C04_Sources, extra=0.5)
        self.wipe()
        self.header("2 · Turning cues into layouts (paper Sec. A.3)")
        cards = [
            card_box([f"each region gets a level ℓ ∈ {{0, 1, 2, 3}}", "→ tokens 8×8, 4×4, 2×2, 1×1",
                      "finer level wins where regions overlap"], w=6.2, color=MID, title="Semantic masks · boxes", size=21),
            card_box(["coarsen where luminance error stays", f"below τ({it('R')}) = {it('s')} (𝔼{sub(it('R'))}[{it('I')}] + {it('a')})",
                      "variable-rate shading rule"], w=6.2, color=AMBER, title="Texture variance", size=21),
            card_box([f"blur radius {it('r')} = {it('K')} |1/{it('d')} − 1/{it('d')}{sub(it('f'))}|",
                      "small blur → fine tokens", "large blur → coarse tokens"], w=6.2, color=TEAL, title="Depth of field", size=21),
            card_box([f"paint a detail map {it('D')}({it('q')}), or let an agent", f"split a block when max {it('D')} ≥ {it('t')}{sub(it('b'))}",
                      "quadtree from 8×8 down to 1×1"], w=6.2, color=PURPLE, title="Painted / agent maps", size=21)]
        pos = [[-3.35, 1.45, 0], [3.35, 1.45, 0], [-3.35, -1.35, 0], [3.35, -1.35, 0]]
        for c, p in zip(cards, pos):
            c.move_to(p)
            self.play(FadeIn(c, shift=UP * 0.15), run_time=0.45)
            self.wait(1.6)
        self.say("One model is trained on a mix of all of these, so any of them can drive generation")
        self.wipe()
        self.body(CL.C07_Budget, extra=0.5)
        self.wipe()


class S04_Method(T):
    def construct(self):
        self.part("Part 3 · The method", "Adapting a pretrained DiT")
        self.header("3 · Two obstacles")
        b1 = card_box(["the VAE decoder needs the full-resolution", f"latent {it('H')}{sub('0')} × {it('W')}{sub('0')} × {it('D')}′"],
                      w=5.4, color=AMBER, title="1 · Decoder", size=22).move_to([-3.6, 0.6, 0])
        b2 = card_box(["a pretrained DiT predicts flow only for the", f"{it('L')} tokens it receives, not all {it('HW')}"],
                      w=5.4, color=AMBER, title="2 · Denoiser", size=22).move_to([3.6, 0.6, 0])
        self.play(FadeIn(b1), run_time=0.4)
        self.wait(1.4)
        self.play(FadeIn(b2), run_time=0.4)
        self.say("So every step must still produce a dense velocity field from a reduced token sequence")
        self.wipe()
        self.header("3 · Background: flow matching and asymmetric flow (AsymFlow)")
        e1 = tx(f"{it('x')}{sub(it('t'))} = (1 − {it('t')}) {it('x')}{sub('0')} + {it('t')} ε,     "
                f"{it('u')} = ε − {it('x')}{sub('0')}", 36).move_to([0, 1.9, 0])
        n1 = tx("flow matching: the DiT learns the velocity from noise to data", 22, DIM).next_to(e1, DOWN, buff=0.2)
        e2 = tx(f"{it('u')}{sub('A')} = {it('P')} ε − {it('x')}{sub('0')},     {it('P')} = {it('A')}{it('A')}{sup('⊤')}", 36).move_to([0, 0.05, 0])
        n2 = tx("AsymFlow: predict the velocity from low-rank noise to full-rank data", 22, DIM).next_to(e2, DOWN, buff=0.2)
        e3 = tx(f"{it('u')} = {it('P')} {it('u')}{sub('A')} + ({it('I')} − {it('P')}) ({it('x')}{sub(it('t'))} + {it('u')}{sub('A')}) / σ{sub(it('t'))}",
                36).move_to([0, -1.8, 0])
        n3 = tx("…and the full-rank velocity is recovered in closed form", 22, DIM).next_to(e3, DOWN, buff=0.2)
        for e, n in ((e1, n1), (e2, n2), (e3, n3)):
            self.add(e, n)
            self.wait(2.3)
        self.say("LoT applies this patch by patch: a big patch is a low-rank view of the dense latent", y=-3.62)
        self.wipe()
        self.body(CL.C05_AsymFlow, extra=0.6)
        self.wipe()
        self.header("3 · Details that keep the pretrained prior")
        rows = [(f"{it('A')}{sub(it('e'))}: one semi-orthonormal basis per extent", "fitted by orthogonal Procrustes on multi-scale VAE tokens"),
                (f"{it('s')}{sub(it('e'))}: one data scale per extent", "clean patches are divided by it, so every token keeps one shared timestep"),
                (f"extent heads start from the pretrained head", f"{it('W')}{sub(it('e'))} = {it('A')}{sub(it('e'))} {it('W')}{sub('out')}, input heads likewise from {it('A')}{sub(it('e'))}{sup('⊤')}"),
                ("everything else is LoRA on the pretrained DiT", "the generative prior is kept, not retrained")]
        g = VGroup(*[VGroup(tx(a, 28), tx(b, 22, DIM)).arrange(DOWN, buff=0.08, aligned_edge=LEFT) for a, b in rows])
        g.arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        g.scale_to_fit_width(min(g.width, 12.8)).move_to([0, 0.2, 0])
        for r in g:
            self.add(r)
            self.wait(1.9)
        self.wait(0.6)
        self.wipe()


class S05_LoTDiT(T):
    def construct(self):
        self.body(CL.C06_ShapeRoPE, extra=0.6)
        self.wipe()
        self.header("3 · Training")
        a = card_box(["FLUX.2-klein 4B and 9B, 1024 × 1024", f"extents {it('e')}{sub('h')}, {it('e')}{sub('w')} ∈ {{1, 2, 4, 8}}",
                      "LoRA rank 256 · 50k steps (4B)", "Aesthetic-Train-V2 (4B) · LAION-3M (9B)"],
                     w=6.1, color=FINE, title="LoT-FLUX.2 (images)", size=22).move_to([-3.4, 0.4, 0])
        b = card_box(["Wan2.1-T2V 14B, 81 frames at 720p", f"extents (1, {it('e')}{sub('h')}, {it('e')}{sub('w')}), {it('e')} ∈ {{1, 2, 4}}",
                      "LoRA rank 256 · 4k steps", "100k clips of Vchitect-T2V-Dataverse"],
                     w=6.1, color=MID, title="LoT-Wan2.1 (video)", size=22).move_to([3.4, 0.4, 0])
        self.play(FadeIn(a), run_time=0.4)
        self.wait(1.6)
        self.play(FadeIn(b), run_time=0.4)
        self.say("Layouts in training come from masks, boxes, texture variance and depth, mixed together")
        self.wipe()


class S06_Video(T):
    def construct(self):
        self.part("Part 4 · Results", "Layout-adaptive video")
        self.body(CL.R03_Scene3D, extra=0.3)
        self.wipe(0.35)
        self.body(CL.R05_Driving, extra=0.3)
        self.wipe(0.35)
        self.body(CL.R04_Games, extra=0.3)
        self.wipe(0.35)
        self.body(CL.R07_DepthOfField, extra=0.3)
        self.wipe()


class S07_Control(T):
    def construct(self):
        self.body(CL.R08_MoveTheBox, extra=0.4)
        self.wipe(0.35)
        self.body(CL.R10_Boxes, extra=0.3)
        self.wipe(0.35)
        self.body(CL.R09_Typography, extra=0.4)
        self.wipe()


class S08_Compare(T):
    def construct(self):
        self.body(CL.C08_Numbers, extra=0.6)
        self.wipe(0.35)
        self.body(CL.R11_VideoBaselines, extra=0.4)
        self.wipe()
        self.header("4 · Ablation at a matched budget (≈ 1.52× compression, paper Tab. 3)")
        rows = [("Direct HR prediction", "7.06", "26.49", "63.35", RED), ("Binary grid (1×1 / 2×2 only)", "8.20", "20.60", "66.44", AMBER),
                ("No size embedding", "10.03", "14.58", "69.43", DIM), ("Full LoT", "10.35", "13.80", "70.41", FINE)]
        head = VGroup(tx("variant", 24, DIM), tx("HPSv3 ↑", 24, DIM), tx("FID ↓", 24, DIM), tx("MUSIQ ↑", 24, DIM))
        xs = [-3.6, 1.2, 3.2, 5.2]
        for m, x in zip(head, xs):
            m.move_to([x, 2.0, 0])
        head[0].align_to([-6.2, 0, 0], LEFT)
        self.add(head, Line([-6.3, 1.65, 0], [6.3, 1.65, 0]).set_stroke(DIM, 1.5))
        for k, (n, a, b, c, colr) in enumerate(rows):
            y = 1.1 - 0.75 * k
            r = VGroup(tx(n, 26, colr, bold=(k == 3)).move_to([xs[0], y, 0]).align_to([-6.2, 0, 0], LEFT),
                       tx(a, 26, colr).move_to([xs[1], y, 0]), tx(b, 26, colr).move_to([xs[2], y, 0]),
                       tx(c, 26, colr).move_to([xs[3], y, 0]))
            self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.35)
            self.wait(0.9)
        self.say("Predicting dense flow directly, or allowing only two token sizes, costs the most quality")
        self.wipe()


class S09_Outlook(T):
    def construct(self):
        self.body(CL.R02_Agent, extra=0.4)
        self.wipe()
        self.header("5 · Limitation (paper Fig. 22)")
        S = 2.9
        ims = [img(f"fig22_{k}.png", h=S) for k in range(4)]
        Group(*ims).arrange(RIGHT, buff=0.25).move_to([0, 0.5, 0])
        labs = VGroup(tx("3,189 tokens (1.28×)", 22, DIM).next_to(Group(ims[0], ims[1]), UP, buff=0.15),
                      tx("1,466 tokens (2.79×)", 22, DIM).next_to(Group(ims[2], ims[3]), UP, buff=0.15))
        self.add(*ims, labs)
        self.say("Fine structure everywhere, like sheet music, leaves little to save: aggressive budgets break staff lines")
        self.say("LoT pays off when the need for detail varies across the image")
        self.wipe()
        self.header("5 · Outlook")
        pts = ["layouts as an editable plan that users and agents refine, together with its compute budget",
               "faster planning ↔ synthesis loops for robotics and interactive simulation",
               "compatible with other speed-ups: caching, distillation, fewer sampling steps"]
        g = VGroup(*[tx("· " + p, 26) for p in pts]).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        g.scale_to_fit_width(min(g.width, 12.6)).move_to([0, 0.6, 0])
        for p in g:
            self.add(p)
            self.wait(2.2)
        self.wait(0.8)
        self.wipe()
        t = tx("Level-of-Token Diffusion", 56, bold=True).move_to([0, 0.9, 0])
        s = tx("up to 2.04× faster images · up to 3.53× faster video · layout-adaptive", 28, DIM).next_to(t, DOWN, buff=0.35)
        g2 = img("fig1_image.png", h=2.2).next_to(s, DOWN, buff=0.4)
        self.add(t, s, g2)
        self.wait(4.0)
