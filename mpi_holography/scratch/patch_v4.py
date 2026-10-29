"""v3 -> v4 patch: S07 runtime plot x-axis at the bottom (log axis offset by +2 so the axis line is not at 10^0)."""
import re
p = "/home/brianchc/project_animations/mpi_holography/animation_mpi_v4.py"
s = open(p).read()
s = s.replace("teaser animation, v3.", "teaser animation, v4.").replace("animation_mpi_v3.py <Scene>", "animation_mpi_v4.py <Scene>")
a = s.index("class S07_Speed(Base):")
b = s.index("        # --- quality plots")
blk = s[a:b]
blk = blk.replace("y_range=[-2, 5, 1]", "y_range=[0, 7, 1]")
blk = blk.replace("next_to(ax.c2p(k, -2), DOWN", "next_to(ax.c2p(k, 0), DOWN")
blk = blk.replace("next_to(ax.c2p(2, k), LEFT", "next_to(ax.c2p(2, k + 2), LEFT")
blk = blk.replace("DashedLine(ax.c2p(2, k), ax.c2p(9, k)", "DashedLine(ax.c2p(2, k + 2), ax.c2p(9, k + 2)")
blk = re.sub(r"np\.log10\(([^()]*(?:\([^()]*\))?[^()]*)\)", r"(np.log10(\1) + 2)", blk)
blk = blk.replace("yr = np.log10(RT_RPWS)", "yr = np.log10(RT_RPWS) + 2")
blk = blk.replace("yr = (np.log10(RT_RPWS) + 2)", "yr = np.log10(RT_RPWS) + 2")
blk = blk.replace("ax.c2p(3.25, 1.95)", "ax.c2p(3.25, 3.95)").replace("ax.c2p(3.45, 1.35)", "ax.c2p(3.45, 3.35)")
s = s[:a] + blk + s[b:]
open(p, "w").write(s)
print("patched")
