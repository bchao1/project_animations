"""v2 -> v3 patch (QA round 2): shorter captions (total was 5:42), minor timing."""
p = "/home/brianchc/project_animations/mpi_holography/animation_mpi_v3.py"
s = open(p).read()


def R(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (n, a[:70])
    s = s.replace(a, b)


R("teaser animation, v2.", "teaser animation, v3.")
R("animation_mpi_v2.py <Scene>", "animation_mpi_v3.py <Scene>")
C = [
    ("A holographic display rebuilds the light wave of a 3D scene: you can refocus and look around.",
     "A holographic display rebuilds the light wave of a 3D scene: refocus, look around."),
    ("But a computer must first turn 3D content into that wave: computer-generated holography.",
     "A computer must first turn 3D content into that wave: computer-generated holography."),
    ("RGBD images are fast but break at occlusions. Gaussians look great but come by the million.",
     "RGBD is fast but breaks at occlusions. Gaussians look great but come by the million."),
    ("With smooth phase it rings but stays put. With random phase it spreads into every direction.",
     "Smooth phase rings but stays put. Random phase spreads into every direction."),
    ("Two FFTs and one multiply: the cost depends on the pixels, not on what the plane contains.",
     "Two FFTs and a multiply: the cost depends on pixels, not on content."),
    ("Primitive-based CGH computes one wavefront per point, polygon or Gaussian: millions of them.",
     "Primitive-based CGH needs one wavefront per point, polygon or Gaussian: millions."),
    ("A multiplane image needs one propagation per layer, and it has only a handful of layers.",
     "A multiplane image needs one propagation per layer, and has only a handful."),
    ("A multiplane image (MPI) is a stack of semi-transparent RGBA images, each at its own depth.",
     "A multiplane image (MPI) is a stack of RGBA images, each at its own depth."),
    ("Where a layer is transparent, you see the layers behind it, including what the front hides.",
     "Through transparent regions you see the layers behind, even what the front hides."),
    ("To render a new view: shift each layer by its parallax, then composite back to front.",
     "New view: shift each layer by its parallax, composite back to front."),
    ("Both as holograms, 24 frames each. Now move the eye's pupil across the eyebox.",
     "As holograms (24 frames each), move the eye's pupil across the eyebox."),
    ("RGBD: dark gaps open along every occlusion edge. The MPI shows what was behind.",
     "RGBD: dark gaps open along occlusion edges. The MPI shows what was behind."),
    ("Paper Fig. 4: RGBD leaks light and tears at depth edges, even with 512 layers.",
     "Paper Fig. 4: RGBD leaks and tears at depth edges, even with 512 layers."),
    ("Start with the farthest layer: square-rooted colour and opacity, times a random phase.",
     "Start at the farthest layer: √(colour × opacity), times a random phase."),
    ("Propagate it to the next layer. There, dim it by √(1 − α) and add that layer's own wave.",
     "Propagate to the next layer, dim by √(1 − α), add that layer's own wave."),
    ("The occluder's mask sits at the occluder's own depth, so it blocks light from every angle.",
     "Each mask sits at its occluder's depth, blocking light from every angle."),
    ("Random phases make the cross term average away: intensities obey the over operator exactly.",
     "The cross term averages away: intensities obey the over operator exactly."),
    ("The result: a random-phase hologram that fills the SLM's bandwidth, for a wide eyebox.",
     "The result: a random-phase hologram filling the SLM's bandwidth, a wide eyebox."),
    ("One frame is speckled; 24 frames give clean defocus. RGBD shows seams around the star.",
     "24 frames give clean defocus. RGBD shows seams around the star."),
    ("RGBD costs exactly the same: the price is set by the layer count, not the content.",
     "RGBD costs the same: the layer count sets the price."),
    ("Quality, from the same plots: RGBD stays low however many layers it gets.",
     "Quality: RGBD stays low however many layers it gets."),
    ("The MPI hologram refocuses from the grass to the vase with clean, natural blur.",
     "The MPI hologram refocuses from grass to vase with natural blur."),
    ("It works with MPIs straight from an off-the-shelf network (Stereo Magnification).",
     "It also works with MPIs from an off-the-shelf network."),
    ("Captured parallax: things away from the focal plane shift as the pupil moves.",
     "Captured parallax: things off the focal plane shift as the pupil moves."),
]
for a, b in C:
    R(a, b)
open(p, "w").write(s)
print("patched")
