"""Level-of-Token Diffusion — ~1-minute teaser, v2 (house style: ../STYLE.md; template foveated_diffusion/teaser_1min).

Compact cut of ../animation_lot_v2.py (helpers, toy layout and beat pieces imported from it; data read-only).
Scenes, in order: S1_Insight, S2_Method, S3_Payoff. Render: ./render.sh l|h
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animation_lot_v2 import *  # noqa: E402,F403


def count_text(field, center):
    n1, n2, n4 = field.count(1), field.count(4), field.count(16)
    other = field.count() - n1 - n2 - n4
    s = f"{col(str(n1), HR)} + {col(str(n2), LR)} + {col(str(n4), LR3)}"
    if other:
        s += f" + {other}"
    return fit(tx(f"{s} = <b>{field.count()}</b> tokens", 28), 6.2).move_to(center)


# ================================================================== S1 Hook + insight
class S1_Insight(Beats):
    def construct(self):
        # --- title over the token-field hero
        title = fit(tx(TITLE, 72, bold=True), 12.8).move_to([0, 2.75, 0])
        subt = tx(SUBTITLE, 32).move_to([0, 1.95, 0])
        g0 = Grid(16, 3.8, (0, -0.75))
        f0 = LoTField(g0, uniform(16))
        self.add(title, subt)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in f0.mobs()], lag_ratio=0.004), run_time=1.0)
        for rects in (TOY,):
            self.play(LaggedStart(*f0.retarget(rects, order_from=(8, 8)), lag_ratio=0.01), run_time=0.7)
            self.wait(0.25)
        self.wait(0.6)
        self.play(FadeOut(f0.group()), run_time=0.4)
        self.remove(title, subt)

        # --- hook: the image vs what the model computes
        S = 4.6
        pic = img(RESULT, S).move_to([-3.4, 0.0, 0])
        lab_l = tx("the image", 26, DIM).next_to(pic, UP, buff=0.25)
        self.play(FadeIn(pic), run_time=0.5)
        self.add(lab_l)
        g = Grid(16, S, (3.4, 0.0))
        field = LoTField(g, uniform(16))
        lab_r = tx("what the model computes", 26, DIM).move_to([3.4, lab_l.get_y(), 0])
        self.caption("A diffusion transformer pays for every token equally.")
        self.add(lab_r)
        self.play(LaggedStart(*[FadeIn(m) for m in field.mobs()], lag_ratio=0.004), run_time=1.0)
        cnt_c = [3.4, -2.72, 0]
        cnt = tx(f"{col('256', HR)} high-res tokens", 28).move_to(cnt_c)
        self.add(cnt)
        self.caption("Empty sky costs as much as her face.")
        # --- insight: the plan says where detail matters
        pm = img(CANVAS, S).move_to(pic)
        self.hold()
        self.play(FadeIn(pm), run_time=0.8)
        self.remove(lab_l)
        lab_l = tx("the plan: a painted detail map", 26, DIM).move_to(lab_l)
        self.add(lab_l)
        self.caption("But the plan already says where detail matters.")
        mid = [r for r in TOY if (r[1] - r[0]) <= 2]
        stage1 = mid + [(y, y + 2, x, x + 2) for r in TOY if (r[1] - r[0]) == 4
                        for y in (r[0], r[0] + 2) for x in (r[2], r[2] + 2)]
        self.caption("Level-of-Token Diffusion: tokens only as fine as the plan asks.")
        self.play(LaggedStart(*field.retarget(stage1, order_from=(8, 8)), lag_ratio=0.02), run_time=1.6)
        self.play(LaggedStart(*field.retarget(TOY, order_from=(8, 8)), lag_ratio=0.1), run_time=0.8)
        self.remove(cnt)
        cnt = count_text(field, cnt_c)
        self.add(cnt)
        self.caption("Tokens of any size and shape, from 1 × 1 up to 8 × 8.")
        self.play(LaggedStart(*field.retarget(rect_merge_toy()), lag_ratio=0.15), run_time=1.2)
        self.remove(cnt)
        cnt = count_text(field, cnt_c)
        self.add(cnt, toy_note("toy layout on a 16 × 16 grid").next_to(cnt, DOWN, buff=0.15))
        self.hold(0.3)
        self.clear_all(run_time=0.5)
        self.wait(0.1)


# ================================================================== S2 Method
class S2_Method(Beats):
    def construct(self):
        # one patch → one token → pretrained DiT → per-size head → dense flow
        g = Grid(8, 2.6, (-5.2, 0.3))
        noise = noise_array(8, 3)
        cells = VGroup(*[token(g.hs, rgb_to_color(noise[i, j])).move_to(g.cell(i, j)) for i in range(8) for j in range(8)])
        pr = (0, 2, 4, 6)
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.01), run_time=0.7)
        self.add(tx(f"noisy latent {XT}", 24).next_to(cells, UP, buff=0.2))
        sel = cell_box(g, pr, WHITE, 4)
        self.play(Create(sel), run_time=0.3)
        self.caption("Each patch is projected to one ordinary-size token.")
        pj = op_box(AET, [-2.9, 0.3, 0])
        one = token(0.5, LR, WHITE, 1.2).move_to([-1.6, 0.3, 0])
        self.play(GrowArrow(arrow(sel.get_right(), pj.get_left(), buff=0.12)), FadeIn(pj), run_time=0.4)
        self.play(GrowArrow(arrow(pj.get_right(), one.get_left(), buff=0.12)), FadeIn(one), run_time=0.4)
        dit = dit_box([0.3, 0.3, 0], w=1.8, h=2.2, lora=True)
        self.play(GrowArrow(arrow(one.get_right(), dit.get_left())), FadeIn(dit), run_time=0.5)
        self.caption("A pretrained DiT, post-trained with LoRA, runs on the short sequence.")
        head = op_box(f"{I('h')}{sub('(2,2)')}", [2.45, 0.3, 0], LR, w=1.1)
        self.play(GrowArrow(arrow(dit.get_right(), head.get_left())), FadeIn(head), run_time=0.4)
        vel = VGroup(*[token(0.42, "#5C9DFF", WHITE, 1.2) for _ in range(4)]).arrange_in_grid(2, 2, buff=0.06).move_to([4.2, 0.3, 0])
        self.play(GrowArrow(arrow(head.get_right(), vel.get_left())), FadeIn(vel), run_time=0.4)
        self.add(tx("full-resolution flow", 22).next_to(vel, UP, buff=0.2))
        self.caption("A head per token size predicts full-resolution flow for the whole patch.")
        base = VGroup(tx("FLUX.2 klein 4B / 9B (images)", 24), tx("Wan2.1 14B (video)", 24)).arrange(DOWN, buff=0.15)
        base.move_to([0.3, -1.9, 0])
        self.add(base)
        self.hold(0.3)
        self.clear_all(run_time=0.5)
        # any cue becomes a layout (paper Fig. 4)
        fig = os.path.join(DATA, "fig4_full.png")
        names = ["bounding box", "semantic mask", "texture variance", "depth"]
        xs = [-4.95, -1.65, 1.65, 4.95]
        S = 2.3
        tops = Group(*[img(panel(fig, 0, 2 * k, f"fig4_r0_c{2 * k}.png"), S).move_to([x, 1.2, 0]) for k, x in enumerate(xs)])
        bots = Group(*[img(panel(fig, 0, 2 * k + 1, f"fig4_r0_c{2 * k + 1}.png"), S).move_to([x, -1.25, 0]) for k, x in enumerate(xs)])
        self.play(LaggedStart(*[FadeIn(Group(a, b)) for a, b in zip(tops, bots)], lag_ratio=0.15), run_time=1.2)
        for k, x in enumerate(xs):
            self.add(tx(names[k], 24).move_to([x, 1.2 + S / 2 + 0.25, 0]))
        self.add(tx("paper Fig. 4 · one LoT-FLUX.2 model", 18, DIM).move_to([0, -2.72, 0]))
        self.caption("Boxes, masks, texture or depth: any cue becomes a layout.")
        self.hold(0.5)
        self.clear_all(run_time=0.5)
        self.wait(0.1)


# ================================================================== S3 Payoff + end card
class S3_Payoff(Beats):
    def construct(self):
        self.b_speed()
        self.b_imgcompare()
        t = tx(TITLE, 64, bold=True).move_to([0, 1.5, 0])
        tag = fit(tx("up to 2.04× faster images  ·  up to 3.53× faster video  ·  layout-adaptive", 30, HR), 12.8)
        tag.move_to([0, 0.55, 0])
        self.add(t, tag)
        if AUTHORS:
            self.add(fit(tx(AUTHORS, 24), 12.8).move_to([0, -0.6, 0]))
        if LINKS:
            self.add(tx(LINKS, 26, DIM).move_to([0, -1.3, 0]))
        self.wait(5.0)
