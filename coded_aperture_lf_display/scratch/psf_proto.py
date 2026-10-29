"""Fourier-optics PSF toy: |F{P(ξ) exp(j 2π W20 (ρ/R)^2)}|^2 for an open pupil, 4 corner windows (TDM) and the paper's
Fig. 9 binary 9x9 coded apertures (CTDM), swept over defocus W20."""
import numpy as np
from PIL import Image
import matplotlib

NP = 1024
CELL = 24
D = 9 * CELL
AP = np.load("/home/brianchc/project_animations/coded_aperture_lf_display/data/fig9_apertures_9x9.npy").astype(float)


def pupil_field(P9, w20):
    p = np.kron(P9, np.ones((CELL, CELL)))
    g = np.zeros((NP, NP))
    o = (NP - D) // 2
    g[o:o + D, o:o + D] = p
    y, x = (np.mgrid[0:NP, 0:NP] - NP / 2 + 0.5) / (D / 2)
    return g * np.exp(2j * np.pi * w20 * (x ** 2 + y ** 2) / 2)   # (ρ/R)^2 with R = half-diagonal-ish


def psf(P9, w20, band=(0.85, 0.92, 1.0, 1.08, 1.15)):
    out = 0
    for b in band:
        u = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field(P9, w20 * b))))
        out = out + np.abs(u) ** 2
    return out


if __name__ == "__main__":
    full = np.ones((9, 9))
    corn = np.zeros((9, 9)); corn[0, 0] = corn[0, 8] = corn[8, 0] = corn[8, 8] = 1
    rows = []
    for P9 in (full, corn, AP[0], AP.sum(0) / 1.0):
        row = []
        for w in (0, 4, 10, 18):
            h = psf(P9, w)
            h = h[NP // 2 - 384:NP // 2 + 384, NP // 2 - 384:NP // 2 + 384].reshape(192, 4, 192, 4).mean((1, 3))
            h = (h / h.sum()) ** 0.5
            h = h / np.percentile(h, 99.9)
            row.append((matplotlib.colormaps["inferno"](np.clip(h, 0, 1))[..., :3] * 255).astype(np.uint8))
        rows.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(rows, 0)).save("psf_sheet.png")
