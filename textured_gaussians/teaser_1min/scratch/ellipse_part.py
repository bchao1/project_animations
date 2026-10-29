def ellipse_overlay(params, res=384, fit_res=128, k=2.0):
    """RGBA image: white k-sigma outlines of the fitted Gaussians, clipped to the image."""
    sc = res / fit_res
    g = (np.arange(res) + 0.5) / sc
    X, Y = np.meshgrid(g, g)
    out = np.zeros((res, res), bool)
    for mx, my, s1, s2, th in params:
        c, s_ = np.cos(th), np.sin(th)
        dx, dy = X - mx, Y - my
        u = (c * dx + s_ * dy) / s1
        v = (-s_ * dx + c * dy) / s2
        r = np.sqrt(u * u + v * v)
        out |= np.abs(r - k) * min(s1, s2) * sc < 0.55
    a = np.zeros((res, res, 4), np.uint8)
    a[out] = (255, 255, 255, 200)
    return a

