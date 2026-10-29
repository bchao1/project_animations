import numpy as np
import torch
import time
import sys
import os
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from optics_proto import *

OUT = "/home/brianchc/project_animations/coded_aperture_lf_display/scratch/proto_" + sys.argv[3]
os.makedirs(OUT, exist_ok=True)


def showcase():
    """Back: letter wall (d = -1.5). Middle: green leaf (d = 0, display plane). Front: orange star (d = +1.5)."""
    im = Image.new("RGB", (C, C), (40, 46, 70))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 22)
    cols = [(255, 202, 40), (77, 208, 225), (255, 255, 255), (255, 82, 82)]
    words = ["LIGHT FIELD", "DISPLAY", "CODED", "APERTURE", "TIME", "DIVISION"]
    for r, y in enumerate(range(-4, C, 24)):
        s = "  ".join(words[(r + j) % len(words)] for j in range(3))
        d.text((-8 - 11 * (r % 3), y), s, fill=cols[r % 4], font=f)
    back = np.asarray(im) / 255.
    # middle: green ellipse with vein stripes
    mid = np.zeros((C, C, 3))
    yy, xx = np.mgrid[0:C, 0:C]
    mm = ((((xx - 62) / 30.0) ** 2 + ((yy - 120) / 40.0) ** 2) <= 1).astype(float)
    mid[..., 0], mid[..., 1], mid[..., 2] = 0.2, 0.75, 0.35
    mid *= (0.75 + 0.25 * (np.sin((xx + yy) / 3.0) > 0))[..., None]
    # front: orange 5-point star with checker texture
    sm = Image.new("L", (C, C), 0)
    ds = ImageDraw.Draw(sm)
    cx, cy, r1, r0 = 122, 86, 44, 19
    pts = []
    for i in range(10):
        r = r1 if i % 2 == 0 else r0
        a = -np.pi / 2 + i * np.pi / 5
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    ds.polygon(pts, fill=255)
    fm = np.asarray(sm) / 255.
    chk = ((xx // 5 + yy // 5) % 2).astype(float)
    front = np.stack([1.0 - 0 * chk, 0.62 - 0.25 * chk, 0.16 + 0 * chk], -1)
    return [(back, np.ones((C, C)), -1.0), (mid, mm, 0.0), (front, fm, 1.0)]


def make(seed):
    torch.set_num_threads(1)
    lay = showcase() if seed < 0 else rand_scene(np.random.default_rng(seed + 100))
    return render_views(lay).astype(np.float32)


def prep(V, E):
    Vt = torch.tensor(V).permute(0, 1, 4, 2, 3)
    Vf = torch.fft.rfft2(Vt)
    Fg = refocus_views(Vf, E)
    corners = torch.stack([Vt[v, u] for v, u in CORN])
    Fb = refocus_views(Vf, E, CORN)
    return corners, Fg, focus_weights(Fg), Fb


def save_img(x, name):
    a = crop(x).permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(OUT, name))


if __name__ == "__main__":
    NS = int(sys.argv[1]); IT = int(sys.argv[2])
    t0 = time.time()
    with Pool(48) as p:
        Vs = p.map(make, [-1] + list(range(NS)))
    print("render", time.time() - t0, flush=True)
    E = bases()
    data = [prep(V, E) for V in Vs]
    show = data[0]
    train = data[1:]
    for z in (0, 4, 8):
        save_img(show[1][z], f"gt_{z}.png")
        save_img(show[3][z], f"base_{z}.png")
    print("baseline psnr", psnr(show[3], show[1]))
    rng = np.random.default_rng(1)
    res = {}
    for mode in sys.argv[3].split(","):
        torch.manual_seed(0)
        net = EncNet() if mode == "fP" else None
        Pp = torch.nn.Parameter(torch.randn(4, L, L) * 0.5)
        params = [Pp] + (list(net.parameters()) if net else [])
        opt = torch.optim.Adam([{"params": [Pp], "lr": 3e-2}] + ([{"params": list(net.parameters()), "lr": 1e-3}] if net else []))
        for it in range(IT):
            cn, Fg, W, _ = train[rng.integers(len(train))]
            I = net(cn.reshape(1, 12, C, C)) if net else cn
            Fc = ctdm(I, torch.sigmoid(Pp), E)
            loss = (crop(W) * (crop(Fc) - crop(Fg)).abs()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 250 == 0 or it == IT - 1:
                with torch.no_grad():
                    I = net(show[0].reshape(1, 12, C, C)) if net else show[0]
                    Fs = ctdm(I, torch.sigmoid(Pp), E)
                print(mode, it, round(float(loss), 4), "show psnr", round(psnr(Fs, show[1]), 2), round(time.time() - t0), flush=True)
        with torch.no_grad():
            for z in (0, 4, 8):
                save_img(Fs[z], f"{mode}_{z}.png")
            if net:
                for i in range(4):
                    save_img(I[i], f"enc_{i}.png")
            P = torch.sigmoid(Pp).numpy()
            np.save(os.path.join(OUT, f"P_{mode}.npy"), P)
