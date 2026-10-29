"""Prototype of the toy light-field display simulation (ray-optics limit of the paper's Eqs. 8-10).
Layered scenes -> dense 9x9 light field -> GT focal stack (Eq. 8, all 81 views), baseline TDM-LFD (4 corner views through
4 corner windows), CTDM-LFD (k = 4 encoded images through 4 learned 9x9 coded apertures, Eq. 10), trained end to end with the
focus-weighted L1 loss (Eqs. 12-15, beta = 2)."""
import numpy as np
import torch
import time
import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

torch.set_num_threads(int(os.environ.get('NT', '24')))
C = 192           # canvas
B = 16            # border cropped away (FFT wrap)
L = 9             # aperture / angular resolution (l = 9)
US = np.arange(L) - L // 2
AS = np.linspace(-1.2, 1.2, 9)   # focal planes (disparity in px per aperture cell)
CORN = [(0, 0), (0, L - 1), (L - 1, 0), (L - 1, L - 1)]
FONT = "/home/brianchc/.local/share/fonts/cmunbx.ttf"


def blur(a, s=0.7):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s))) / 255.


def rand_texture(rng):
    kind = rng.integers(4)
    im = Image.new("RGB", (C, C), tuple(int(v) for v in rng.integers(20, 200, 3)))
    d = ImageDraw.Draw(im)
    if kind == 0:     # rectangles
        for _ in range(60):
            x, y = rng.integers(0, C, 2)
            w, h = rng.integers(4, 30, 2)
            d.rectangle([x, y, x + w, y + h], fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
    elif kind == 1:   # stripes
        p = int(rng.integers(5, 14))
        c1, c2 = [tuple(int(v) for v in rng.integers(0, 256, 3)) for _ in range(2)]
        for x in range(-C, 2 * C, p):
            d.polygon([(x, 0), (x + p // 2, 0), (x + p // 2 + C // 2, C), (x + C // 2, C)], fill=c1 if rng.random() < 0.8 else c2)
    elif kind == 2:   # letters
        f = ImageFont.truetype(FONT, int(rng.integers(16, 34)))
        for y in range(-10, C, int(f.size * 1.1)):
            s = "".join(rng.choice(list("ABCDEFGHKLMNOPRSTUVWXYZ"), 12))
            d.text((int(rng.integers(-20, 0)), y), s, fill=tuple(int(v) for v in rng.integers(0, 256, 3)), font=f)
    else:             # checker
        p = int(rng.integers(6, 16))
        c1, c2 = [tuple(int(v) for v in rng.integers(0, 256, 3)) for _ in range(2)]
        for i in range(0, C, p):
            for j in range(0, C, p):
                if (i // p + j // p) % 2:
                    d.rectangle([i, j, i + p - 1, j + p - 1], fill=c1)
                else:
                    d.rectangle([i, j, i + p - 1, j + p - 1], fill=c2)
    return np.asarray(im) / 255.


def rand_mask(rng):
    im = Image.new("L", (C, C), 0)
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.integers(40, C - 40, 2)
        r = rng.integers(18, 45)
        n = int(rng.integers(3, 9))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rr = r * rng.uniform(0.5, 1.0, n)
        d.polygon([(cx + a * np.cos(t), cy + a * np.sin(t)) for a, t in zip(rr, ang)], fill=255)
    return np.asarray(im) / 255.


def rand_scene(rng):
    back = rand_texture(rng)
    mid, mm = rand_texture(rng), rand_mask(rng) * (rng.random() < 0.6)
    front, fm = rand_texture(rng), rand_mask(rng)
    db = rng.uniform(-1.2, -0.5)
    dm = rng.uniform(-0.3, 0.3)
    df = rng.uniform(0.5, 1.2)
    return [(back, np.ones((C, C)), db), (mid, mm, dm), (front, fm, df)]


def fshift(a, dx, dy):
    """Sub-pixel shift (periodic) of HxW[xC] array: out(x) = a(x - d)."""
    F = np.fft.fft2(a, axes=(0, 1))
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    if a.ndim == 3:
        ph = ph[..., None]
    return np.real(np.fft.ifft2(F * ph, axes=(0, 1)))


def render_views(layers):
    """Dense L x L light field, back-to-front 'over' compositing of shifted layers."""
    V = np.zeros((L, L, C, C, 3))
    pre = [(blur(t, 0.6) if t.ndim == 2 else np.stack([blur(t[..., c], 0.6) for c in range(3)], -1),
            blur(m, 0.6), d) for t, m, d in layers]
    for iv, v in enumerate(US):
        for iu, u in enumerate(US):
            img = np.zeros((C, C, 3))
            for t, m, d in pre:
                ms = np.clip(fshift(m, u * d, v * d), 0, 1)[..., None]
                ts = fshift(t * m[..., None], u * d, v * d)
                img = img * (1 - ms) + np.clip(ts, 0, 1)
            V[iv, iu] = np.clip(img, 0, 1)
    return V


# ---------------------------------------------------------------- torch image formation
def freq(dev="cpu"):
    fy = torch.fft.fftfreq(C, device=dev)[:, None]
    fx = torch.fft.rfftfreq(C, device=dev)[None, :]
    return fx, fy


def sinc(x):
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(np.pi * x) / (np.pi * x + 1e-12))


def bases(dev="cpu"):
    """E[z, v, u, f] = exp(+j2π(fx u + fy v) a_z) · sinc(fx a) sinc(fy a)   (cell = box of width |a|)."""
    fx, fy = freq(dev)
    E = torch.zeros(len(AS), L, L, C, C // 2 + 1, dtype=torch.complex64)
    for z, a in enumerate(AS):
        Bz = sinc(fx * a) * sinc(fy * a)
        for iv, v in enumerate(US):
            for iu, u in enumerate(US):
                E[z, iv, iu] = Bz * torch.exp(2j * np.pi * (fx * u + fy * v) * a)
    return E


def refocus_views(Vf, E, cells=None):
    """Vf: rfft2 of views [L, L, 3, C, C/2+1]. Returns focal stack [Z, 3, C, C] averaged over the given cells."""
    if cells is None:
        S = torch.einsum("vucxy,zvuxy->zcxy", Vf, E) / (L * L)
    else:
        S = sum(Vf[v, u][None] * E[:, v, u][:, None] for v, u in cells) / len(cells)
    return torch.fft.irfft2(S, s=(C, C))


def ctdm(I, P, E):
    """I: [k, 3, C, C] encoded images; P: [k, L, L] apertures in [0, 1]. Eq. 10 in the ray-optics limit."""
    If = torch.fft.rfft2(I)
    A = torch.einsum("kvu,zvuxy->kzxy", P.to(torch.complex64), E)
    S = torch.einsum("kcxy,kzxy->zcxy", If, A) / P.sum()
    return torch.fft.irfft2(S, s=(C, C))


def crop(x):
    return x[..., B:C - B, B:C - B]


def focus_weights(Fg, beta=2.0):
    """Eqs. 13-15: U = |∂x R| + |∂y R|, normalised per plane, W = exp(beta Ū)."""
    g = Fg.mean(1)
    U = torch.zeros_like(g)
    U[:, :, 1:] += (g[:, :, 1:] - g[:, :, :-1]).abs()
    U[:, 1:, :] += (g[:, 1:, :] - g[:, :-1, :]).abs()
    mn = U.amin((1, 2), keepdim=True)
    mx = U.amax((1, 2), keepdim=True)
    Ub = (U - mn) / (mx - mn + 1e-8)
    return torch.exp(beta * Ub)[:, None]


class Res(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = torch.nn.Conv2d(c, c, 3, padding=1, padding_mode="circular")
        self.b = torch.nn.Conv2d(c, c, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        return x + self.b(torch.relu(self.a(x)))


class EncNet(torch.nn.Module):
    """Scaled-down light field encoding network f: conv -> residual blocks -> conv -> sigmoid (paper: 10 blocks, 128 ch)."""

    def __init__(self, n=4, k=4, c=48, nb=5):
        super().__init__()
        self.i = torch.nn.Conv2d(3 * n, c, 3, padding=1, padding_mode="circular")
        self.r = torch.nn.Sequential(*[Res(c) for _ in range(nb)])
        self.o = torch.nn.Conv2d(c, 3 * k, 3, padding=1, padding_mode="circular")
        self.k = k

    def forward(self, x):
        y = torch.sigmoid(self.o(torch.relu(self.r(torch.relu(self.i(x))))))
        return y.view(self.k, 3, C, C)


def psnr(a, b):
    return float(10 * torch.log10(1 / ((crop(a) - crop(b)) ** 2).mean()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    E = bases()
    t0 = time.time()
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    data = []
    for s in range(NS):
        V = render_views(rand_scene(rng))
        Vt = torch.tensor(V, dtype=torch.float32).permute(0, 1, 4, 2, 3)
        Vf = torch.fft.rfft2(Vt)
        Fg = refocus_views(Vf, E)
        corners = torch.stack([Vt[v, u] for v, u in CORN])
        data.append((corners, Fg, focus_weights(Fg)))
    print("data", time.time() - t0)
    base = [psnr(refocus_views(torch.fft.rfft2(torch.stack([torch.stack([d[0][0]] * L)] * L)), E, None), d[1]) for d in data[:1]]
    net = EncNet()
    Pp = torch.nn.Parameter(torch.randn(4, L, L) * 0.5)
    opt = torch.optim.Adam(list(net.parameters()) + [Pp], lr=1e-3)
    for it in range(int(sys.argv[2]) if len(sys.argv) > 2 else 300):
        idx = rng.integers(len(data))
        cn, Fg, W = data[idx]
        I = net(cn.reshape(1, 12, C, C))
        Fc = ctdm(I, torch.sigmoid(Pp), E)
        loss = (crop(W) * (crop(Fc) - crop(Fg)).abs()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(it, float(loss), psnr(Fc.detach(), Fg), time.time() - t0, flush=True)
