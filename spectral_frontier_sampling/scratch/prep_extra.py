"""Data for the VAE-map scatter and the residual band images (CPU, seconds).

1. data/vae_scatter.npz: log10 coarse (grid 64) vs full-res power at one ring over the 600 round-trip latents
   (/miele/data/austinchi/fractional_expansion/roundtrip_grids.npz), with the rule's fitted line for that ring
   (params/klein_2k_cont.json: lambda, offset_vs_corpus_m; prior m).
2. data/resid_<step>_{true,guess,resid}.png + resid.json: band-pass (in pixels, cycles per image) of the street
   cont2x run's final image and of its clean guess x0hat at that step, on the octave [n/4, n/2) below that step's grid
   boundary, upper-left crop. Grey = 0, contrast set by the true band. resid.json has the pixel new fraction and the
   rule's u.

    python scratch/prep_extra.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
REPO = Path("/home/brianchc/prompt-adaptive-speed-cvpr-2027")
PFILE = REPO / "experiments/fractional_expansion/params/klein_2k_cont.json"
RT = Path("/miele/data/austinchi/fractional_expansion/roundtrip_grids.npz")
RUN = 2348
RING = 8


def scatter():
    d = np.load(RT, allow_pickle=True)
    p = json.load(open(PFILE))
    v = p["vae"]["64"]
    k = RING - 2                                         # vae / prior arrays start at bin 2
    m = p["prior"]["m_ln_power"][k]
    lam, off = v["lambda"][k], v["offset_vs_corpus_m"][k]
    np.savez(HERE / "data" / "vae_scatter.npz", coarse=np.log10(d["PC64"][:, RING]), full=np.log10(d["PF"][:, RING]),
             m=m / np.log(10), lam=lam, off=off / np.log(10), ring=RING)
    print(f"ring {RING}: lambda {lam:.3f}, shift a {off / np.log(10):+.3f} log10")


def band_imgs():
    """the octave below the grid boundary, [n/4, n/2) cycles per image, on an upper-left crop (balloons, facades).
    The rule's own band is narrower ([7n/16, n/2), in the latent); this pixel view is an illustration."""
    cap = HERE / "data" / "capture" / f"cont_{RUN:06d}"
    md = np.load(HERE / "data" / f"method_{RUN:06d}.npz")
    fires = [i for i in range(50) if md["fire"][i]]
    gray = lambda f: np.asarray(Image.open(cap / f).convert("L"), float) / 255  # noqa: E731
    T = np.fft.fft2(gray("final.png"))
    N = T.shape[0]
    f = np.fft.fftfreq(N) * N
    r = np.sqrt(f[:, None] ** 2 + f[None] ** 2)
    out = {}
    for i in (fires[0], fires[-2]):
        n = int(md["n"][i])
        lo, hi = n // 4, n // 2
        mask = (r >= lo) & (r < hi)
        G = np.fft.fft2(gray(f"x0_{i:02d}.png"))
        bt, bg = np.real(np.fft.ifft2(T * mask)), np.real(np.fft.ifft2(G * mask))
        s = 3 * bt.std()
        for name, a in (("true", bt), ("guess", bg), ("resid", bt - bg)):
            c = a[40:296, 0:256]
            Image.fromarray((np.clip(0.5 + c / (2 * s), 0, 1) * 255).astype(np.uint8)).save(
                HERE / "data" / f"resid_{i:02d}_{name}.png")
        frac = float(np.sum(np.abs((T - G)[mask]) ** 2) / np.sum(np.abs(T[mask]) ** 2))
        out[i] = {"n": n, "band": [lo, hi], "new_fraction_pixels": frac, "u_rule": float(md["u"][i])}
        print(f"step {i}, grid {n}, band {lo}-{hi - 1}: pixel new fraction {frac:.2f}, rule u {md['u'][i]:.2f}")
    json.dump(out, open(HERE / "data" / "resid.json", "w"), indent=1)


if __name__ == "__main__":
    scatter()
    band_imgs()
