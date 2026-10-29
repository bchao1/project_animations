import torch, numpy as np, time, skimage.data as sd
from skimage.transform import resize
dev = 'cuda:4'
P = 10.8e-6; LAMS = {'r': 636.4e-9, 'g': 517.7e-9, 'b': 440.8e-9}
H, W = 192, 320
GH, GW = 512, 640
RH, RW = 180, 300
def dstar(lam): return np.sqrt(H**2 + W**2) * P / 2 / np.tan(np.arcsin(lam / (2 * P)))
def Hfun(lam, z):
    fy = torch.fft.fftfreq(GH, P, device=dev); fx = torch.fft.fftfreq(GW, P, device=dev)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    arg = 1 / lam**2 - FX**2 - FY**2
    fz = torch.sqrt(torch.clamp(arg, min=0))
    return torch.exp(1j * 2 * np.pi * fz * z) * (arg > 0)
def pad(u):
    out = torch.zeros(GH, GW, dtype=torch.complex64, device=dev)
    y0, x0 = (GH - H) // 2, (GW - W) // 2
    out[y0:y0 + H, x0:x0 + W] = u; return out
roi = torch.zeros(GH, GW, device=dev)
ry, rx = (GH - RH) // 2, (GW - RW) // 2
roi[ry:ry + RH, rx:rx + RW] = 1
def target(img):
    a = torch.zeros(GH, GW, device=dev)
    a[ry:ry + RH, rx:rx + RW] = torch.tensor(np.sqrt(img), device=dev, dtype=torch.float32)
    return a
def optimize(a_t, lam, alpha, gamma, iters=1000, lr=0.02, seed=0):
    torch.manual_seed(seed)
    z = alpha * dstar(lam); Hz = Hfun(lam, z)
    phi = (torch.rand(H, W, device=dev) - 0.5).requires_grad_()
    opt = torch.optim.Adam([phi], lr=lr)
    Psrc = H * W
    k = np.sqrt(Psrc / (a_t**2).sum().item())
    for it in range(iters):
        u = torch.fft.ifft2(torch.fft.fft2(pad(torch.exp(1j * phi))) * Hz)
        amp = u.abs()
        with torch.no_grad():
            s = ((amp * roi * a_t).sum() / (a_t**2).sum())
            s = 1 / s    # s * amp ~ a_t  -> s = <amp,a>/<amp,amp>? use closed form on s*amp
        # closed form (S5): s* = argmin ||s*arecon - a||  => s = <arecon,a>/<arecon,arecon>
        with torch.no_grad():
            ar = amp * roi
            s = (ar * a_t).sum() / (ar**2).sum()
        lrec = ((s * amp - a_t)**2 * roi).sum() / roi.sum()
        lreg = ((amp - k * a_t)**2).mean() if gamma > 0 else 0
        loss = lrec + gamma * lreg
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u = torch.fft.ifft2(torch.fft.fft2(pad(torch.exp(1j * phi))) * Hz)
        I = u.abs()**2
        eff = (I * roi).sum().item() / Psrc
        ar = u.abs() * roi; s = ((ar * a_t).sum() / (ar**2).sum()).item()
        rec = (s * u.abs() * roi)
        mse = ((rec - a_t)**2 * roi).sum() / roi.sum()
        psnr = 10 * np.log10(1 / (((rec**2 - a_t**2)**2 * roi).sum() / roi.sum()).item())
    return I.cpu().numpy(), eff, s, psnr
img = resize(sd.chelsea().mean(-1) / 255, (RH, RW), anti_aliasing=True)
a = target(img)
for alpha in (0.16, 0.32, 0.64):
    for g in (0, 1, 10, 100):
        t = time.time(); I, eff, s, ps = optimize(a, LAMS['g'], alpha, g)
        print(f'alpha {alpha} gamma {g}: eff {eff*100:.1f}% s {s:.4f} bright 1/s2 {1/s**2:.1f} psnr {ps:.2f} ({time.time()-t:.1f}s)')
