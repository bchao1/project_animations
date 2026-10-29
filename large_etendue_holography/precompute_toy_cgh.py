"""Toy re-implementation of the paper's image formation + pupil-aware light field CGH.

Used only to generate *illustrative* computed visuals for the animation (clearly labelled
"toy simulation" on screen). Not the authors' code; all reported numbers in the video come
from the paper, never from this script.

Model (monochrome, 2-D, SLM-pixel units):
  - phase SLM: N x N pixels, each pixel sampled S x S times (100% fill) -> HDOs appear.
  - Fourier plane = fftshift(fft2(field)); one diffraction-order cell = N samples.
  - J = 3 x 3 mutually incoherent sources, tilted so that each copy lands exactly on the
    neighbouring order cell (paper: tilt matched to the +-1st HDO angle).
  - Fourier amplitude mask P (20 x 20 pixels over the 3x3-cell expanded eyebox).
  - view p = |IFFT(U_j * P * M_p)|^2 summed over sources j and averaged over frames t.
  - loss (paper Eq. 3/4): || s*sqrt(mean_t sum_j |...|^2) - lf_target^(p) ||^2, one random
    pupil p per iteration (stochastic light-field supervision).
Output: data/generated/toy_cgh.npz
"""
import os
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "generated", "toy_cgh.npz")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

N, S = 64, 4
M = N * S                      # 256 field / spectrum samples
C = M // 2                     # spectrum centre index
CELL = N                       # one order cell in spectrum samples
EB = 3 * CELL                  # expanded eyebox width (3 cells) = 192
R_P = int(round(0.2345 * EB))  # pupil radius: 4 mm pupil in 8.53 mm eyebox -> 45
SPAN = EB // 2 - R_P           # pupil centres within +-SPAN
NV = 9                         # 9 x 9 light field
NF = 20                        # 20 x 20 Fourier display (paper's simulation)

yy, xx = np.mgrid[0:M, 0:M]


def scene_view(u, v):
    """Target view for normalized pupil offset (u, v) in [-1, 1]: two depth layers."""
    img = np.zeros((M, M), np.float32)
    # background: vertical bars + soft gradient, shifts opposite to the pupil (far layer)
    bx = xx - 8 * u
    by = yy - 8 * v
    bars = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * bx / 42.0))
    img += 0.18 + 0.22 * bars * (0.6 + 0.4 * by / M)
    # foreground: bright disk + triangle, shift with the pupil (near layer)
    fx = xx + 18 * u
    fy = yy + 18 * v
    disk = ((fx - 100) ** 2 + (fy - 112) ** 2) < 34 ** 2
    tri = (fy > 120) & (fy < 196) & (np.abs(fx - 170) < (fy - 120) * 0.55)
    img[disk] = 0.95
    img[tri] = 0.75
    # only the central SLM-sized window is imaged; soft border
    return img


def pupil_mask(cu, cv, r=R_P):
    return (((xx - (C + cu)) ** 2 + (yy - (C + cv)) ** 2) <= r * r).astype(np.float32)


offs = np.linspace(-SPAN, SPAN, NV)
PUP = [(cu, cv) for cv in offs for cu in offs]
TARGET = np.stack([np.sqrt(scene_view(cu / SPAN, cv / SPAN)) for (cu, cv) in PUP])  # amplitude
PMASK = np.stack([pupil_mask(cu, cv) for (cu, cv) in PUP])

eb = np.zeros((M, M), np.float32)
eb[C - EB // 2:C + EB // 2, C - EB // 2:C + EB // 2] = 1
cell0 = np.zeros((M, M), np.float32)
cell0[C - CELL // 2:C + CELL // 2, C - CELL // 2:C + CELL // 2] = 1

T_TARGET = torch.tensor(TARGET, device=dev)
T_PM = torch.tensor(PMASK, device=dev)
T_EB = torch.tensor(eb, device=dev)
T_CELL = torch.tensor(cell0, device=dev)
SHIFTS = [(dy * CELL, dx * CELL) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def spectrum(phi):
    """phi: (T, N, N) -> centred spectrum (T, M, M) of the pixelated SLM field."""
    f = torch.exp(1j * phi)
    f = f.repeat_interleave(S, -1).repeat_interleave(S, -2)
    return torch.fft.fftshift(torch.fft.fft2(f, norm="ortho"), dim=(-2, -1))


def fourier_mask(logits):
    """(T, NF, NF) logits -> (T, M, M) amplitude in [0, 1] on the expanded eyebox."""
    a = torch.sigmoid(logits)[:, None]
    a = Fn.interpolate(a, size=(EB, EB), mode="nearest")[:, 0]
    P = torch.zeros(a.shape[0], M, M, device=dev)
    P[:, C - EB // 2:C + EB // 2, C - EB // 2:C + EB // 2] = a
    return P


def views(phi, P, pm, multi):
    """Intensity seen through pupil masks pm (K, M, M). Returns (K, M, M)."""
    U = spectrum(phi)                                # (T, M, M)
    out = 0
    srcs = SHIFTS if multi else [(0, 0)]
    for (sy, sx) in srcs:
        Uj = torch.roll(U, shifts=(sy, sx), dims=(-2, -1))
        Uf = (Uj * P)[:, None] * pm[None]            # (T, K, M, M)
        u = torch.fft.ifft2(torch.fft.ifftshift(Uf, dim=(-2, -1)), norm="ortho")
        out = out + (u.abs() ** 2)
    return out.mean(0)


def run(cfg, T, iters=3000):
    multi = cfg in ("III", "IV", "V")
    phi = (torch.rand(T, N, N, device=dev) * 2 * np.pi).requires_grad_(True)
    logs = torch.zeros(1, device=dev, requires_grad=True)
    params = [phi, logs]
    if cfg == "V":
        logits = (torch.randn(T, NF, NF, device=dev) * 0.1).requires_grad_(True)
        params.append(logits)
    elif cfg == "IV":
        g = torch.Generator().manual_seed(3)
        fixed = (torch.rand(1, NF, NF, generator=g) > 0.5).float().to(dev)
    opt = torch.optim.Adam(params, lr=0.05)

    def getP():
        if cfg == "V":
            return fourier_mask(logits)
        if cfg == "IV":
            return fourier_mask(torch.logit(fixed.clamp(1e-4, 1 - 1e-4))).expand(T, M, M)
        if cfg == "III":
            return T_EB.expand(T, M, M)
        return T_CELL.expand(T, M, M)                # I: single source + Fourier filter

    for it in range(iters):
        idx = torch.randint(0, NV * NV, (4,), device=dev)   # a few random views per step
        I = views(phi, getP(), T_PM[idx], multi)
        amp = torch.exp(logs) * torch.sqrt(I + 1e-8)
        loss = ((amp - T_TARGET[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = getP()
        allv = []
        for k in range(0, NV * NV, 9):
            allv.append(views(phi, P, T_PM[k:k + 9], multi))
        allv = torch.cat(allv)
        amp = torch.exp(logs) * torch.sqrt(allv)
        mse = ((amp - T_TARGET) ** 2).mean(dim=(1, 2))
        psnr = (10 * torch.log10(1.0 / mse)).mean().item()
    print(f"{cfg} T={T}: toy PSNR (amplitude) {psnr:.2f} dB")
    return phi.detach(), P.detach(), torch.exp(logs).item(), multi


def path_views(phi, P, s, multi, path):
    pm = torch.tensor(np.stack([pupil_mask(cu, cv) for (cu, cv) in path]), device=dev)
    with torch.no_grad():
        v = views(phi, P, pm, multi)
    return (s * s * v).clamp(0, 1).cpu().numpy()


def total_spectrum(phi, P, multi, use_P=True):
    with torch.no_grad():
        U = spectrum(phi)
        tot = 0
        for (sy, sx) in (SHIFTS if multi else [(0, 0)]):
            Uj = torch.roll(U, shifts=(sy, sx), dims=(-2, -1))
            if use_P:
                Uj = Uj * P
            tot = tot + Uj.abs() ** 2
        return tot.cpu().numpy()          # (T, M, M)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = {}
    # pupil path: left edge -> right edge along the middle row (for the moving-eye demo)
    path = [(float(cu), 0.0) for cu in np.linspace(-SPAN, SPAN, 25)]
    for cfg, T in (("I", 1), ("III", 1), ("IV", 1), ("V", 1), ("V", 6)):
        phi, P, s, multi = run(cfg, T)
        key = f"{cfg}{T}"
        res[f"{key}_path"] = path_views(phi, P, s, multi, path).astype(np.float16)
        res[f"{key}_spec"] = total_spectrum(phi, P, multi).mean(0).astype(np.float32)
        if cfg == "V":
            res[f"{key}_P"] = P[:, C - EB // 2:C + EB // 2, C - EB // 2:C + EB // 2].cpu().numpy()
            res[f"{key}_spec_frames"] = total_spectrum(phi, P, multi).astype(np.float32)
            if T == 6:
                grid = path_views(phi, P, s, multi, PUP)
                res["V6_grid"] = grid[:, ::2, ::2].astype(np.float16)
                res["target_grid"] = (TARGET[:, ::2, ::2] ** 2).astype(np.float16)
            res[f"{key}_spec_nomask"] = total_spectrum(phi, P, multi, use_P=False).mean(0).astype(np.float32)
        if cfg == "I":
            # raw single-source spectrum (no filter) of the optimized phase: shows HDOs
            res["I1_spec_raw"] = total_spectrum(phi, P, False, use_P=False)[0].astype(np.float32)
            res["I1_phase"] = phi[0].cpu().numpy()
        if cfg == "III":
            res["III1_spec_raw"] = total_spectrum(phi, P, True, use_P=False)[0].astype(np.float32)
    res["target_path"] = np.stack([scene_view(cu / SPAN, cv / SPAN) for (cu, cv) in path]).astype(np.float16)
    res["meta"] = np.array([N, S, M, CELL, EB, R_P, SPAN, NF])
    np.savez_compressed(OUT, **res)
    print("wrote", OUT)
