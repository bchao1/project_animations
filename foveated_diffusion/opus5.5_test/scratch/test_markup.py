from manim import *
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤‖⊕∑")
def mathify(s):
    out=[]
    for ch in s:
        out.append(f'<span font_family="Latin Modern Math">{ch}</span>' if ch in MATHCH else ch)
    return "".join(out)
def sb(x, r): return f'<span rise="{r}"><small><small>{x}</small></small></span>'
class T1(Scene):
    def construct(self):
        rows=[]
        for rs, rp in [(-3000,4500),(-3000,6000),(-2500,7500)]:
            s = f'<i>z</i>{sb("0",rs)}{sb("fov",rp)} = <i>M</i>′ ⊙ <i>x</i>{sb("high",rp)} + (1 − <i>M</i>′) ⊙ Up(<i>x</i>{sb("low",rp)})  ‖<i>v</i>{sb("θ",rs)} − (<i>z</i>{sb("1",rs)} − <i>z</i>{sb("0",rs)})‖{sb("2",rp)} 16 × 16 → 𝒩(0, <i>I</i>) ℒ 𝔼 ≈ 39%'
            rows.append(MarkupText(mathify(s), font="CMU Serif", font_size=40).scale_to_fit_width(13.5))
        VGroup(*rows).arrange(DOWN, buff=0.6)
        self.add(*rows)
