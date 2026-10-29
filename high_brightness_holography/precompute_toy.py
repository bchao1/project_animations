"""Toy wave-optics simulations for the High-Brightness Holographic Projection teaser (torch autograd, GPU if free).

Everything here is a *toy* re-implementation of the paper's image formation and losses, at a small size:
  - phase-only SLM, 10.8 um pixels (the prototype's TI DLP6750Q1EVM pitch), lambda = 636.4 / 517.7 / 440.8 nm (paper)
  - Fresnel holography: u_z = ASM(e^{i phi} u_src, z), image plane at z = alpha d*  (supplement Eq. S4)
  - SGD (Adam, lr 0.02, phase init U(-0.5, 0.5), closed-form scale s, supplement Alg. S1 / Eq. S5)
  - L = L_recon + gamma L_reg, L_reg = L2(|u_z|, sqrt(sum|u_src|^2 / sum a^2) a_target)  (paper Eqs. 3-4)
  - a toy "real display" (illumination falloff, SLM phase nonlinearity, aberration, distance error) for
    SGD-ASM vs SGD-CITL (supplement Alg. S2) vs SGD with a toy camera-calibrated model.
Writes data/generated/toy_*.npz.  Run:  python precompute_toy.py   (≈ 5-10 min)
"""
import os
import sys
import time
import numpy as np
import torch
import skimage.data as sd
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "data", "generated")
os.makedirs(GEN, exist_ok=True)


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        f, t = torch.cuda.mem_get_info(i)
        free.append((f, i))
    return f"cuda:{max(free)[1]}"


dev = pick_device()
print("device", dev)
P = 10.8e-6
LAMS = [636.4e-9, 517.7e-9, 440.8e-9]
LG = LAMS[1]
H, W = 192, 320            # toy SLM (aspect ≈ the prototype's 700 × 1190 ROI)
GH, GW = 576, 704          # padded simulation grid (room for light that lands outside the ROI)
RH, RW = 180, 300          # region of interest, slightly smaller than the SLM (as in the paper)
Y0, X0 = (GH - H) // 2, (GW - W) // 2
RY, RX = (GH - RH) // 2, (GW - RW) // 2
ALPHA = 0.64               # paper: z ≈ 20 cm ≈ 0.64 d*
GAMMA = 0.1                # toy light-efficiency weight
ITERS = 2000               # paper: 2000 iterations, lr 0.02
SNAPS = [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64, 85, 110, 145, 190, 250, 330, 430, 560, 730, 950,
         1250, 1600, 2000]


def dstar(lam, h=H, w=W):
    return np.sqrt(h ** 2 + w ** 2) * P / 2 / np.tan(np.arcsin(lam / (2 * P)))


def transfer(lam, z, gh=GH, gw=GW):
    fy = torch.fft.fftfreq(gh, P, device=dev)
    fx = torch.fft.fftfreq(gw, P, device=dev)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    arg = 1 / lam ** 2 - FX ** 2 - FY ** 2
    return torch.exp(1j * 2 * np.pi * torch.sqrt(torch.clamp(arg, min=0)) * z) * (arg > 0)


def pad(u):
    out = torch.zeros(GH, GW, dtype=torch.complex64, device=dev)
    out[Y0:Y0 + H, X0:X0 + W] = u
    return out


ROI = torch.zeros(GH, GW, device=dev)
ROI[RY:RY + RH, RX:RX + RW] = 1


def to_target(img):
    a = torch.zeros(GH, GW, device=dev)
    a[RY:RY + RH, RX:RX + RW] = torch.tensor(np.sqrt(np.clip(img, 0, 1)), device=dev, dtype=torch.float32)
    return a


def scale(amp, a_t):
    ar = amp * ROI
    return (ar * a_t).sum() / (ar ** 2).sum().clamp(min=1e-12)


def ideal_fwd(phi, Hz):
    return torch.fft.ifft2(torch.fft.fft2(pad(torch.exp(1j * phi))) * Hz)


def optimize(a_t, Hz, gamma, iters=ITERS, lr=0.02, seed=0, fwd=None, snaps=None, citl=None):
    """SGD phase optimization (supplement Alg. S1; Alg. S2 if citl = real-display forward is given)."""
    torch.manual_seed(seed)
    fwd = fwd or (lambda p: ideal_fwd(p, Hz))
    phi = (torch.rand(H, W, device=dev) - 0.5).requires_grad_()
    opt = torch.optim.Adam([phi], lr=lr)
    k = np.sqrt(H * W / (a_t ** 2).sum().item())       # sqrt(sum|u_src|^2 / sum a_target^2), unit illumination
    snap_I, snap_eff, snap_s = [], [], []
    for it in range(iters + 1):
        u = fwd(phi)
        amp = u.abs()
        if citl is not None:                            # replace(|u|, sqrt(I_captured)), keep model gradients
            with torch.no_grad():
                cap = citl(phi.detach()).abs()
            amp = amp + (cap - amp).detach()
        if snaps is not None and it in snaps:
            with torch.no_grad():
                snap_I.append((amp ** 2).cpu().numpy().astype(np.float16))
                snap_eff.append(((amp ** 2) * ROI).sum().item() / (H * W))
                snap_s.append(scale(amp, a_t).item())
        if it == iters:
            break
        with torch.no_grad():
            s = scale(amp, a_t)
        loss = ((s * amp - a_t) ** 2 * ROI).sum() / ROI.sum()
        if gamma > 0:
            loss = loss + gamma * ((amp - k * a_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return phi.detach(), dict(I=snap_I, eff=snap_eff, s=snap_s)


def metrics(I, a_t):
    """Efficiency (power in ROI / source power), scale s, PSNR of the s-scaled intensity inside the ROI."""
    amp = I.sqrt()
    s = scale(amp, a_t)
    eff = (I * ROI).sum().item() / (H * W)
    rec = (s * amp) ** 2
    mse = (((rec - a_t ** 2) ** 2) * ROI).sum() / ROI.sum()
    return eff, s.item(), 10 * np.log10(1 / mse.item())


# ------------------------------------------------------------------ targets
def rgb_target(name):
    im = {"cat": sd.chelsea, "coffee": sd.coffee, "astronaut": sd.astronaut, "rocket": sd.rocket}[name]()
    im = im.astype(float) / 255
    h0, w0 = im.shape[:2]
    ar = RW / RH
    if w0 / h0 > ar:
        nw = int(h0 * ar)
        im = im[:, (w0 - nw) // 2:(w0 - nw) // 2 + nw]
    else:
        nh = int(w0 / ar)
        im = im[(h0 - nh) // 2:(h0 - nh) // 2 + nh]
    im = resize(im, (RH, RW), anti_aliasing=True)
    return np.clip(im, 0, 1) ** 2.2                       # linear intensity


def hud_target():
    """Sparse head-up-display style target: arrow, speed, lane marks (white on black)."""
    im = Image.new("L", (RW * 4, RH * 4), 0)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 190)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 70)
    except OSError:
        f = f2 = ImageFont.load_default()
    d.text((110, 180), "60", font=f, fill=255)
    d.text((140, 390), "km/h", font=f2, fill=255)
    cx, cy = 860, 330
    d.polygon([(cx, cy - 210), (cx + 150, cy - 40), (cx + 55, cy - 40), (cx + 55, cy + 200), (cx - 55, cy + 200),
               (cx - 55, cy - 40), (cx - 150, cy - 40)], fill=255)
    for k in range(3):
        d.rectangle([80 + 380 * k, 640, 260 + 380 * k, 668], fill=255)
    im = im.resize((RW, RH), Image.LANCZOS)
    return np.asarray(im) / 255.0


def sparse_versions(gray, levels, seed=3):
    """Zero out a growing fraction of pixels (smooth random blobs), paper's sparsity = fraction of zero pixels."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.standard_normal((RH, RW)), 9)
    out = []
    for sp in levels:
        if sp <= 0:
            out.append(gray.copy())
            continue
        thr = np.quantile(field, sp)
        out.append(np.where(field > thr, gray, 0.0))
    return out


# ------------------------------------------------------------------ 1D toys (x–z side views)
def run_1d():
    n, g = 400, 1280                                    # SLM pixels, grid
    x0 = (g - n) // 2
    lam = LG
    R = n * P / 2
    ds = R / np.tan(np.arcsin(lam / (2 * P)))
    z_img = ALPHA * ds
    fx = torch.fft.fftfreq(g, P, device=dev)
    arg = 1 / lam ** 2 - fx ** 2
    fz = torch.sqrt(torch.clamp(arg, min=0))
    band = (arg > 0)
    Hz = lambda z: torch.exp(1j * 2 * np.pi * fz * z) * band
    roi = torch.zeros(g, device=dev)
    rn = 372
    r0 = (g - rn) // 2
    roi[r0:r0 + rn] = 1
    xx = np.arange(rn) / rn
    prof = np.zeros(rn)
    for c, w, v in ((0.18, 0.07, 1.0), (0.50, 0.12, 0.55), (0.80, 0.05, 1.0)):
        prof += v * (np.abs(xx - c) < w)
    from scipy.ndimage import gaussian_filter1d
    prof = np.clip(gaussian_filter1d(prof, 1.2), 0, 1)
    a_t = torch.zeros(g, device=dev)
    a_t[r0:r0 + rn] = torch.tensor(np.sqrt(prof), device=dev, dtype=torch.float32)
    k = np.sqrt(n / (a_t ** 2).sum().item())
    HZ = Hz(z_img)
    res = {}
    snaps1 = [0, 2, 5, 10, 20, 35, 60, 100, 160, 250, 400, 650, 1000, 1600, 3000]
    keep = slice(g // 2 - 420, g // 2 + 420)
    for name, gamma, al in (("noloss", 0.0, ALPHA), ("loss", 0.1, ALPHA), ("short", 0.0, 0.12)):
        zi = al * ds
        HZ = Hz(zi)
        torch.manual_seed(1)
        phi = (torch.rand(n, device=dev) - 0.5).requires_grad_()
        opt = torch.optim.Adam([phi], lr=0.02)
        zsn = torch.linspace(0, zi, 180, device=dev)
        snap_xz, snap_phi = [], []
        for it in range(3001):
            u0 = torch.zeros(g, dtype=torch.complex64, device=dev)
            u0[x0:x0 + n] = torch.exp(1j * phi)
            U0 = torch.fft.fft(u0)
            if name in ("loss", "noloss") and it in snaps1:
                with torch.no_grad():
                    snap_xz.append(torch.stack([torch.fft.ifft(U0 * Hz(z)).abs() ** 2 for z in zsn], 1)[keep]
                                   .cpu().numpy().astype(np.float16))
                    snap_phi.append((phi.detach() % (2 * np.pi)).cpu().numpy().astype(np.float32))
            if it == 3000:
                break
            amp = torch.fft.ifft(U0 * HZ).abs()
            with torch.no_grad():
                s = (amp * roi * a_t).sum() / ((amp * roi) ** 2).sum()
            loss = ((s * amp - a_t) ** 2 * roi).sum() / roi.sum()
            if gamma:
                loss = loss + gamma * ((amp - k * a_t) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            u0 = torch.zeros(g, dtype=torch.complex64, device=dev)
            u0[x0:x0 + n] = torch.exp(1j * phi)
            U = torch.fft.fft(u0)
            zs = torch.linspace(0, zi, 360, device=dev)
            xz = torch.stack([torch.fft.ifft(U * Hz(z)).abs() ** 2 for z in zs], 1).cpu().numpy()
            I = xz[:, -1]
            eff = (I * roi.cpu().numpy()).sum() / n
            amp = np.sqrt(I) * roi.cpu().numpy()
            at = a_t.cpu().numpy()
            sc = (amp * at).sum() / (amp ** 2).sum()
            mse = (((sc * amp) ** 2 - at ** 2) ** 2)[r0:r0 + rn].mean()
            res[name] = dict(xz=xz[keep].astype(np.float32), I=I.astype(np.float32), eff=eff,
                             psnr=10 * np.log10(1 / mse), snaps=snap_xz, sphi=snap_phi,
                             phi=(phi.detach() % (2 * np.pi)).cpu().numpy())
            print("1D", name, f"alpha {al} eff {eff * 100:.1f}% psnr {10 * np.log10(1 / mse):.1f}")
    # single-pixel sub-hologram cone (one SLM pixel lit), out to 1.1 d*
    u0 = torch.zeros(g, dtype=torch.complex64, device=dev)
    u0[g // 2] = 1
    U = torch.fft.fft(u0)
    zs = torch.linspace(0, 1.1 * ds, 360, device=dev)
    cone = torch.stack([torch.fft.ifft(U * Hz(z)).abs() ** 2 for z in zs], 1).cpu().numpy()
    out = dict(prof=prof.astype(np.float32), cone=cone[keep].astype(np.float32), n=n, g=g, rn=rn, z_img=z_img,
               dstar=ds, snap_iters=np.array(snaps1), snap_xz=np.stack(res["loss"]["snaps"]),
               snap_xz_noloss=np.stack(res["noloss"]["snaps"]), snap_phi=np.stack(res["loss"]["sphi"]),
               snap_phi_noloss=np.stack(res["noloss"]["sphi"]))
    for nm, r in res.items():
        for kk in ("xz", "I", "eff", "psnr", "phi"):
            out[f"{kk}_{nm}"] = r[kk]
    np.savez_compressed(os.path.join(GEN, "toy_1d.npz"), **out)


# ------------------------------------------------------------------ main 2D RGB toy
def run_main():
    out = {}
    for name in ("cat", "coffee"):
        img = rgb_target(name)
        out[f"{name}_target"] = img.astype(np.float32)
        z = ALPHA * dstar(LG)                           # one physical distance for all colours (paper: ~20 cm)
        for tag, gamma in (("noloss", 0.0), ("loss", GAMMA)):
            Is, effs, ss, psnrs = [], [], [], []
            for c, lam in enumerate(LAMS):
                a_t = to_target(img[..., c])
                Hz = transfer(lam, z)
                phi, sn = optimize(a_t, Hz, gamma, snaps=SNAPS if c == 1 and name == "cat" else None)
                with torch.no_grad():
                    I = ideal_fwd(phi, Hz).abs() ** 2
                eff, s, ps = metrics(I, a_t)
                Is.append(I.cpu().numpy().astype(np.float32))
                effs.append(eff)
                ss.append(s)
                psnrs.append(ps)
                if c == 1 and name == "cat":
                    out[f"cat_g_{tag}_snapI"] = np.stack(sn["I"])
                    out[f"cat_g_{tag}_snapeff"] = np.array(sn["eff"])
                    out[f"cat_g_{tag}_snaps"] = np.array(sn["s"])
                    out[f"cat_g_{tag}_phi"] = (phi.cpu().numpy() % (2 * np.pi)).astype(np.float32)
            out[f"{name}_{tag}_I"] = np.stack(Is, -1)
            out[f"{name}_{tag}_eff"] = np.array(effs)
            out[f"{name}_{tag}_s"] = np.array(ss)
            out[f"{name}_{tag}_psnr"] = np.array(psnrs)
            b = 1 / np.sum(np.array(ss) ** 2)
            out[f"{name}_{tag}_bright"] = b
            print(name, tag, "eff", np.round(np.array(effs) * 100, 1), "s", np.round(ss, 4), "bright", round(b, 3),
                  "psnr", np.round(psnrs, 1))
        g = out[f"{name}_loss_bright"] / out[f"{name}_noloss_bright"] - 1
        out[f"{name}_gain"] = g
        print(name, "brightness gain", round(g * 100, 1), "%")
    out["snap_iters"] = np.array(SNAPS)
    np.savez_compressed(os.path.join(GEN, "toy_main.npz"), **out)


# ------------------------------------------------------------------ sparsity sweep (green)
def run_sparsity():
    levels = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.96]
    gray = rgb_target("cat")[..., 1]
    tg = sparse_versions(gray, levels)
    z = ALPHA * dstar(LG)
    Hz = transfer(LG, z)
    rows = []
    Iex = {}
    for sp, im in zip(levels, tg):
        a_t = to_target(im)
        r = [sp]
        for tag, gamma in (("noloss", 0.0), ("loss", GAMMA)):
            phi, _ = optimize(a_t, Hz, gamma, iters=1200)
            with torch.no_grad():
                I = ideal_fwd(phi, Hz).abs() ** 2
            eff, s, ps = metrics(I, a_t)
            r += [eff, 1 / s ** 2, ps]
            if sp in (0.0, 0.6, 0.92):
                Iex[f"I_{tag}_{int(sp * 100)}"] = I.cpu().numpy().astype(np.float16)
        # conventional amplitude display with the same power over the ROI: I_disp = I * maxL / max(I)
        n_roi = RH * RW
        maxL = H * W / n_roi
        conv_eff = im.mean() * maxL / im.max() * n_roi / (H * W)
        conv_bright = maxL / im.max()                     # 1 / s^2 with s^2 = max(I) / max(L)
        r += [conv_eff, conv_bright]
        rows.append(r)
        print("sparsity", sp, "holo eff", round(r[1] * 100, 1), round(r[4] * 100, 1), "bright", round(r[2], 2),
              round(r[5], 2), "conv eff", round(conv_eff * 100, 1), "bright", round(conv_bright, 2))
    np.savez_compressed(os.path.join(GEN, "toy_sparsity.npz"), rows=np.array(rows), levels=np.array(levels),
                        targets=np.stack(tg).astype(np.float32), **Iex)


# ------------------------------------------------------------------ distance sweep (green)
def run_distance():
    alphas = [0.04, 0.08, 0.16, 0.24, 0.32, 0.44, 0.56, 0.64, 0.76, 0.88]
    gray = rgb_target("cat")[..., 1]
    sparse = sparse_versions(gray, [0.85])[0]
    rows = []
    for al in alphas:
        Hz = transfer(LG, al * dstar(LG))
        r = [al]
        for im in (gray, sparse):
            a_t = to_target(im)
            for gamma in (0.0, GAMMA):
                phi, _ = optimize(a_t, Hz, gamma, iters=1200)
                with torch.no_grad():
                    I = ideal_fwd(phi, Hz).abs() ** 2
                eff, s, ps = metrics(I, a_t)
                r += [eff, 1 / s ** 2, ps]
        rows.append(r)
        print("alpha", al, np.round(np.array(r[1:]), 3))
    np.savez_compressed(os.path.join(GEN, "toy_distance.npz"), rows=np.array(rows))


# ------------------------------------------------------------------ camera-in-the-loop / calibrated-model toy
def smooth_field(rng, h, w, sigma, amp):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((h, w)), sigma)
    return amp * f / np.abs(f).max()


def run_citl():
    rng = np.random.default_rng(11)
    lam = LG
    z = ALPHA * dstar(lam)
    Hz = transfer(lam, z)
    yy, xx = np.mgrid[0:H, 0:W]
    ry_, rx_ = (yy - H / 2) / (H / 2), (xx - W / 2) / (W / 2)
    # the toy "real display": Gaussian illumination falloff, off-centre; smooth SLM phase aberration;
    # nonlinear phase response (LUT); 0.5 % distance error
    ill = np.exp(-((rx_ - 0.15) ** 2 / 1.1 + (ry_ + 0.1) ** 2 / 0.9))
    ab = 0.3 * (1.6 * (rx_ ** 2 - ry_ ** 2) * 0.5 + 0.9 * rx_ * (rx_ ** 2 + ry_ ** 2) + smooth_field(rng, H, W, 12, 1.2))
    ill_t = torch.tensor(ill, device=dev, dtype=torch.float32)
    ab_t = torch.tensor(ab, device=dev, dtype=torch.float32)
    Hz_true = transfer(lam, z * 1.005)

    def lut_true(p):
        return p + 0.35 * torch.sin(p) + 0.15 * torch.sin(2 * p)

    def real_fwd(phi):
        u = ill_t * torch.exp(1j * (lut_true(phi) + ab_t))
        return torch.fft.ifft2(torch.fft.fft2(pad(u)) * Hz_true)

    img = rgb_target("cat")[..., 1]
    a_t = to_target(img)
    res = {}
    # --- SGD-ASM (ideal model), captured on the real display
    for tag, gamma in (("noloss", 0.0), ("loss", GAMMA)):
        phi, _ = optimize(a_t, Hz, gamma)
        with torch.no_grad():
            res[f"asm_{tag}"] = (real_fwd(phi).abs() ** 2)
    # --- SGD-CITL: captured amplitudes, ideal-model gradients (Alg. S2)
    for tag, gamma in (("noloss", 0.0), ("loss", GAMMA)):
        phi, _ = optimize(a_t, Hz, gamma, citl=real_fwd)
        with torch.no_grad():
            res[f"citl_{tag}"] = (real_fwd(phi).abs() ** 2)
    # --- dataset of SLM patterns + "captured" images, then fit a toy calibrated model
    print("citl: building dataset")
    names = ["cat", "coffee", "astronaut", "rocket"]
    ims = []
    for nm in names:
        full = rgb_target(nm)
        for c in range(3):
            ims.append(full[..., c])
    phis, caps = [], []
    for i in range(96):
        base = ims[i % len(ims)]
        base = np.roll(base, (rng.integers(0, RH), rng.integers(0, RW)), (0, 1))
        if i % 2:
            base = base[:, ::-1].copy()
        at = to_target(base)
        phi, _ = optimize(at, Hz, 0.0, iters=150, seed=100 + i)
        with torch.no_grad():
            I = real_fwd(phi).abs() ** 2
            I = I * (1 + 0.01 * torch.randn_like(I))       # capture noise
        phis.append(phi)
        caps.append(I.clamp(min=0).sqrt())
    ntr = 80
    # model: learned illumination (low-res), learned SLM phase offset (low-res), learned LUT (32 knots), learned dz
    ill_p = torch.zeros(1, 1, 12, 20, device=dev, requires_grad=True)
    ab_p = torch.zeros(1, 1, 24, 40, device=dev, requires_grad=True)
    lut_p = torch.zeros(33, device=dev, requires_grad=True)
    dz_p = torch.zeros(1, device=dev, requires_grad=True)
    fy = torch.fft.fftfreq(GH, P, device=dev)
    fx = torch.fft.fftfreq(GW, P, device=dev)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    arg = 1 / lam ** 2 - FX ** 2 - FY ** 2
    fzz = torch.sqrt(torch.clamp(arg, min=0))
    band = arg > 0
    knots = torch.linspace(0, 2 * np.pi, 33, device=dev)

    def model_fwd(phi):
        il = torch.nn.functional.interpolate(ill_p, size=(H, W), mode="bicubic", align_corners=True)[0, 0]
        ab_ = torch.nn.functional.interpolate(ab_p, size=(H, W), mode="bicubic", align_corners=True)[0, 0]
        pw = torch.remainder(phi, 2 * np.pi)
        t = pw / (2 * np.pi) * 32
        i0 = torch.clamp(t.floor().long(), 0, 31)
        w_ = t - i0
        off = lut_p[i0] * (1 - w_) + lut_p[i0 + 1] * w_
        u = torch.exp(il) * torch.exp(1j * (pw + off + ab_))
        Hm = torch.exp(1j * 2 * np.pi * fzz * z * (1 + 0.01 * dz_p)) * band
        return torch.fft.ifft2(torch.fft.fft2(pad(u)) * Hm)

    opt = torch.optim.Adam([ill_p, ab_p, lut_p, dz_p], lr=0.02)
    for ep in range(1500):
        i = int(rng.integers(0, ntr))
        amp = model_fwd(phis[i]).abs()
        loss = ((amp - caps[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 300 == 0:
            with torch.no_grad():
                vl = np.mean([((model_fwd(phis[j]).abs() - caps[j]) ** 2).mean().item() for j in range(ntr, 96)])
            print("model epoch", ep, "train", round(loss.item(), 5), "val", round(vl, 5))
    for p_ in (ill_p, ab_p, lut_p, dz_p):
        p_.requires_grad_(False)
    for tag, gamma in (("noloss", 0.0), ("loss", GAMMA)):
        phi, _ = optimize(a_t, Hz, gamma, fwd=model_fwd)
        with torch.no_grad():
            res[f"model_{tag}"] = (real_fwd(phi).abs() ** 2)
    out = {"target": img.astype(np.float32)}
    psrc = float((ill ** 2).sum())
    for k, I in res.items():
        eff, s, ps = metrics(I, a_t)
        eff = eff * H * W / psrc                        # efficiency w.r.t. the real source power
        out[k] = I.cpu().numpy().astype(np.float16)
        out[k + "_m"] = np.array([eff, s, ps])
        print("citl", k, f"eff {eff * 100:.1f}% s {s:.4f} psnr {ps:.2f}")
    for m in ("asm", "citl", "model"):
        g = (out[m + "_noloss_m"][1] / out[m + "_loss_m"][1]) ** 2 - 1
        out[m + "_gain"] = g
        print("citl", m, "brightness gain", round(g * 100, 1), "%")
    out["ill_true"] = ill.astype(np.float32)
    np.savez_compressed(os.path.join(GEN, "toy_citl.npz"), **out)


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["1d", "main", "sparsity", "distance", "citl"]
    for j in jobs:
        t = time.time()
        {"1d": run_1d, "main": run_main, "sparsity": run_sparsity, "distance": run_distance, "citl": run_citl}[j]()
        print(f"== {j} done in {time.time() - t:.0f} s")
