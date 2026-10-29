"""Toy experiments for the RPWS 3 × 1-min explainer.

Every compositing rule follows the paper (arXiv 2508.17480 v3, Eqs. 1–17) and the supplement pseudocode
(../data/Random_phase_Wave_Splatting (2).pdf, Algs. 1, 3–6, S3, S4, S6.1). Outputs go to data/generated/.

    A  xzA_*      one primitive, smooth vs random phase, x–z side view (sim units, λ = 6 px, SLM pitch p = 10 px)
    C  ab_*       Fig. 2 replica: two translucent Gaussians at one depth; GWS / GWS-NRP / RPWS per pixel
    D  conv_*     Fig. S3 replica: cross-term sample mean vs N for uniform 2π, 4-point and 2-point phases
    E  e_*        Fig. 3 replica: translucent Gaussians over a rear plane, 4 compositing rules, LF views + focal slices
    F  xzF_*      Part 3 side view: rear emitter + occluder, GWS-with-√ (Alg. 5) vs RPWS (Alg. 3), sim units

Run: python sims.py   (≈ 1 min on one GPU; writes data/generated/done_sims)
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "data", "generated")
os.makedirs(GEN, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def gen(n):
    return os.path.join(GEN, n)


# ====================================================================== 1-D side views (sim units)
LAM_S = 6.0        # wavelength in px (exaggerated, as in the multisource explainer)
P_S = 10.0         # SLM pixel pitch in px  → sin θ_max = λ / 2p = 0.3 (θ = 17.5°)
HC = 2048          # transverse compute grid (avoids FFT wrap-around)
HD = 400           # displayed rows
TF = 24            # time-multiplexed frames (the paper's largest setting)
_f1 = np.fft.fftfreq(HC)
_kz1 = 2 * np.pi * np.sqrt(np.maximum(1 / LAM_S ** 2 - _f1 ** 2, 0))
_band1 = np.abs(_f1) <= 1 / (2 * P_S)
_x1 = np.arange(HC) - HC / 2 + 0.5
_rows = slice(HC // 2 - HD // 2, HC // 2 + HD // 2)


def prop1(u0, zs):
    """1-D angular spectrum method: columns 𝒫(u0; z) for z in zs, displayed rows only."""
    A = np.fft.fft(u0) * _band1
    return np.fft.ifft(A[:, None] * np.exp(1j * _kz1[:, None] * zs[None, :]), axis=0)[_rows]


def rand1(rng):
    """i.i.d. 𝒰(−π, π) phase per sample, restricted to the SLM band |f| ≤ 1/2p and renormalised to unit mean power."""
    e = np.fft.ifft(np.fft.fft(np.exp(1j * rng.uniform(-np.pi, np.pi, HC))) * _band1)
    return e / np.sqrt(np.mean(np.abs(e) ** 2))


def sim_A(rng):
    WZ = 480
    zs = np.arange(WZ, dtype=float)
    sig = 24.0
    a = np.exp(-_x1 ** 2 / (2 * sig ** 2))
    Us = prop1(a.astype(complex), zs)
    Ur = np.stack([prop1(a * rand1(rng), zs) for _ in range(TF)])
    # power spectra |û(k)|² (fftshifted, propagating band only)
    fs = np.fft.fftshift(_f1)
    keep = np.abs(fs) < 1 / LAM_S
    Ss = np.abs(np.fft.fftshift(np.fft.fft(a)))[keep] ** 2
    Sr = np.mean([np.abs(np.fft.fftshift(np.fft.fft(a * rand1(rng))))[keep] ** 2 for _ in range(64)], 0)
    # S3.2: σ²(z) ≈ σ²(0) + (z/k)² σ_k²  (normalised moments; k_x = 2π f, k = 2π/λ)
    xs = _x1[_rows]

    def var_z(I):
        w = I / I.sum(0, keepdims=True)
        m = (w * xs[:, None]).sum(0)
        return (w * (xs[:, None] - m) ** 2).sum(0)
    Is = np.abs(Us) ** 2
    Ir = np.mean(np.abs(Ur) ** 2, 0)
    # measured spread on the full compute grid (the displayed window would crop the random beam's tails)
    def var_full(u0s):
        A = np.stack([np.fft.fft(u) * _band1 for u in u0s])
        zz = zs[::8]
        I = np.mean([np.abs(np.fft.ifft(a[:, None] * np.exp(1j * _kz1[:, None] * zz[None, :]), axis=0)) ** 2
                     for a in A], 0)
        w = I / I.sum(0, keepdims=True)
        m = (w * _x1[:, None]).sum(0)
        return np.interp(zs, zz, (w * (_x1[:, None] - m) ** 2).sum(0))
    vs_full = var_full([a.astype(complex)])
    vr_full = var_full([a * rand1(rng) for _ in range(TF)])
    pred = {}
    for name, S, I in (("s", Ss, Is), ("r", Sr, Ir)):
        f = fs[keep]
        sk2 = np.sum((2 * np.pi * f) ** 2 * S) / S.sum()
        v0 = (vs_full if name == "s" else vr_full)[0]
        pred[name] = np.sqrt(v0 + (zs / (2 * np.pi / LAM_S)) ** 2 * sk2)
    np.save(gen("xzA_smooth.npy"), Us.astype(np.complex64))
    np.save(gen("xzA_rand.npy"), Ur.astype(np.complex64))
    np.savez(gen("xzA_meta.npz"), f=fs[keep], Ss=Ss / Ss.max(), Sr=Sr / Ss.max(),
             sig_pred_s=pred["s"], sig_pred_r=pred["r"], sig_meas_s=np.sqrt(vs_full),
             sig_meas_r=np.sqrt(vr_full), a=a[_rows])
    print("A: σ(z_end) smooth pred/meas %.1f/%.1f  random pred/meas %.1f/%.1f px" % (
        pred["s"][-1], np.sqrt(vs_full)[-1], pred["r"][-1], np.sqrt(vr_full)[-1]))


# ====================================================================== star + checkerboard toy scene
STAR_C = np.array([1.00, 0.72, 0.18])     # amber star (front, translucent)
CHK_W = np.array([0.92, 0.94, 1.00])      # checkerboard (behind, opaque): white / blue squares
CHK_B = np.array([0.16, 0.36, 0.92])
STAR_A = 0.72                             # star opacity (interior); soft edge = translucent falloff


def star_alpha(n, r_out, r_in, blur, c=None, npts=5):
    from PIL import Image as _I, ImageDraw as _D, ImageFilter as _F
    c = c or (n / 2, n / 2 + 0.06 * r_out)
    im = _I.new("L", (n, n), 0)
    pts = []
    for i in range(2 * npts):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / npts
        pts.append((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))
    _D.Draw(im).polygon(pts, fill=255)
    return STAR_A * np.asarray(im.filter(_F.GaussianBlur(blur)), dtype=float) / 255.0


def checker(n, sq):
    y, x = np.mgrid[0:n, 0:n]
    k = (((x - n // 2) // sq + (y - n // 2) // sq) % 2).astype(float)[..., None]
    return k * CHK_W + (1 - k) * CHK_B


# ====================================================================== C: Fig. 2 replica (per-pixel blending)
def gauss2(n, cx, cy, su, sv, th, xp=np):
    y, x = xp.meshgrid(xp.arange(n, dtype=xp.float32 if xp is torch else float),
                       xp.arange(n, dtype=xp.float32 if xp is torch else float), indexing="ij")
    if xp is torch:
        y, x = y.to(dev), x.to(dev)
    x, y = x - cx, y - cy
    c, s = (xp.cos(th), xp.sin(th))
    u, v = c * x + s * y, -s * x + c * y
    return xp.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))


def sim_C(rng):
    """Paper Sec. 3.3 / Fig. 2 experiment with simple shapes: a translucent star (1) in front of a checkerboard (2),
    both at one depth.

    Intensity model (RPWS primitives, defines the ground truth, Eq. 15): I = α1′c1′ + (1 − α1′)α2′c2′.
    Amplitude model (GWS primitives, fitted so that its amplitude blend² matches I, as GWS models fit amplitude, S1.4).
    GWS: |w1 + w2|²; GWS-NRP: |w1 e^{iφ1} + w2 e^{iφ2}|² with the amplitude weights (Eq. 12);
    RPWS: |√(α1′c1′) e^{iφ1} + √((1 − α1′)α2′c2′) e^{iφ2}|² (Eq. 17). Phases iid 𝒰(−π, π) per pixel and frame.
    """
    n = 256
    a1 = star_alpha(n, 92, 38, 2.0)                    # front: translucent star (α′ = 0.72 inside, soft edge)
    a2 = np.ones((n, n))                               # behind: opaque checkerboard
    c1 = STAR_C
    c2 = checker(n, 32)
    gt = a1[..., None] * c1 + ((1 - a1) * a2)[..., None] * c2
    # ---- fit the amplitude model (GWS-style primitives: same shapes, own opacity and colours) to √I
    gt_t = torch.tensor(gt, dtype=torch.float32, device=dev)
    shape = torch.tensor(a1 / STAR_A, dtype=torch.float32, device=dev)
    kmask = torch.tensor((c2[..., 0] > 0.5).astype(np.float32), device=dev)[..., None]
    prm = {k: torch.tensor(v, dtype=torch.float32, device=dev, requires_grad=True) for k, v in dict(
        o=1.0, cs=np.sqrt(STAR_C), cw=np.sqrt(CHK_W), cb=np.sqrt(CHK_B)).items()}
    opt = torch.optim.Adam(prm.values(), lr=0.02)

    def amp_model():
        A1 = torch.sigmoid(prm["o"]) * shape
        chk = kmask * prm["cw"] + (1 - kmask) * prm["cb"]
        return A1[..., None] * prm["cs"], (1 - A1)[..., None] * chk
    for it in range(3000):
        w1, w2 = amp_model()
        loss = ((w1 + w2) ** 2 - gt_t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w1, w2 = [w.cpu().numpy().astype(float) for w in amp_model()]
    gws = (w1 + w2) ** 2
    r1 = np.sqrt(a1[..., None] * c1)
    r2 = np.sqrt(((1 - a1) * a2)[..., None] * c2)
    TMAX = 256
    keep_t = list(range(1, 25)) + [32, 48, 64, 96, 128, 192, 256]
    accn = np.zeros_like(gt)
    accr = np.zeros_like(gt)
    cum_n, cum_r = [], []
    for t in range(1, TMAX + 1):
        p1 = np.exp(1j * rng.uniform(-np.pi, np.pi, gt.shape))
        p2 = np.exp(1j * rng.uniform(-np.pi, np.pi, gt.shape))
        accn += np.abs(w1 * p1 + w2 * p2) ** 2
        accr += np.abs(r1 * p1 + r2 * p2) ** 2
        if t in keep_t:
            cum_n.append(accn / t)
            cum_r.append(accr / t)
    cum_n, cum_r = np.stack(cum_n), np.stack(cum_r)
    err = lambda im: ((im - gt) ** 2).mean(-1)
    np.savez_compressed(gen("ab_toy.npz"), gt=gt.astype(np.float32), gws=gws.astype(np.float32),
                        nrp=cum_n.astype(np.float16), rpws=cum_r.astype(np.float16), t=np.array(keep_t),
                        cross=(2 * w1 * w2).mean(-1).astype(np.float32),
                        err_gws=err(gws).astype(np.float32),
                        err_nrp=np.stack([err(c) for c in cum_n]).astype(np.float32),
                        err_rpws=np.stack([err(c) for c in cum_r]).astype(np.float32))
    print("C: fit loss %.2e | MSE GWS %.2e  NRP(T=256) %.2e  RPWS(T=1) %.2e  RPWS(T=24) %.2e  RPWS(T=256) %.2e" % (
        loss.item(), err(gws).mean(), err(cum_n[-1]).mean(), err(cum_r[0]).mean(), err(cum_r[23]).mean(),
        err(cum_r[-1]).mean()))
    print("   max |NRP(256) − GT| %.3f vs max cross term %.3f" % (np.abs(cum_n[-1] - gt).max(), (2 * w1 * w2).max()))


# ====================================================================== D: Fig. S3 replica
def sim_D(rng):
    Ns = np.unique(np.round(np.logspace(0, 4, 90)).astype(int))
    trials = 1000
    dists = {
        "uniform": lambda s: rng.uniform(-np.pi, np.pi, s),
        "four": lambda s: rng.integers(0, 4, s) * (np.pi / 2),
        "two": lambda s: rng.integers(0, 2, s) * np.pi,
    }
    out = {"N": Ns}
    for k, d in dists.items():
        c = np.cos(d((trials, Ns[-1])) - d((trials, Ns[-1])))
        cm = np.cumsum(c, 1)[:, Ns - 1] / Ns
        out[k + "_mean"] = cm.mean(0)
        out[k + "_std"] = cm.std(0)
        out[k + "_trials"] = cm[:60].astype(np.float32)
    np.savez(gen("conv.npz"), **out)
    print("D: std at N=10^4  uniform %.4f (1/√2N %.4f)  four %.4f  two %.4f (1/√N %.4f)" % (
        out["uniform_std"][-1], 1 / np.sqrt(2e4), out["four_std"][-1], out["two_std"][-1], 1e-2))


# ====================================================================== E: Fig. 3 replica (2-D, real units)
LAM = 520e-9
DX = 8e-6
N = 512
CR = slice(96, 416)
ZF, ZB = 4e-3, 12e-3
ZM = 0.5 * (ZF + ZB)     # light-field views are focused between the two layers
FMAX = 1 / (2 * DX)


def _grids_t():
    f = torch.fft.fftfreq(N, DX, device=dev, dtype=torch.float64)
    FY, FX = torch.meshgrid(f, f, indexing="ij")
    FZ = torch.sqrt(torch.clamp(1 / LAM ** 2 - FX ** 2 - FY ** 2, min=0))
    return FX, FY, FZ


FX_, FY_, FZ_ = _grids_t()


def H(z):
    return torch.exp(1j * 2 * np.pi * FZ_ * z)


def P_(z, u):
    return torch.fft.ifft2(torch.fft.fft2(u) * H(z))


def pup(cx, cy, r):
    return (((FX_ / FMAX - cx) ** 2 + (FY_ / FMAX - cy) ** 2) <= r ** 2).to(torch.complex128)


def scene_E():
    """Intensity-model scene (RPWS primitives): a translucent amber star at 4 mm in front of an opaque
    checkerboard plane at 12 mm. Front-to-back order: star, checkerboard."""
    from scipy.ndimage import gaussian_filter
    y, x = np.mgrid[0:N, 0:N]
    win = ((np.abs(x - N / 2 + .5) < 150) & (np.abs(y - N / 2 + .5) < 150)).astype(float)
    win = gaussian_filter(win, 1.5)
    star = star_alpha(N, 100, 42, 2.0)
    return [star, win], [np.broadcast_to(STAR_C, (N, N, 3)), checker(N, 48)], [ZF, ZB]


def shift_np(im, dx_px):
    """Sub-pixel horizontal shift by the Fourier shift theorem (periodic)."""
    k = np.fft.fftfreq(im.shape[1])
    ph = np.exp(-2j * np.pi * k * dx_px)
    return np.real(np.fft.ifft(np.fft.fft(im, axis=1) * ph[None, :, None] if im.ndim == 3
                               else np.fft.fft(im, axis=1) * ph[None, :], axis=1))


def ray_blend(alphas, cols, depths, s, zref, sign):
    """S6.1 ground truth: orthographic view θ = asin(λ f_x), each plane sheared by Δx = −(z − z_ref) tan θ,
    then front-to-back ray alpha blending (intensity model)."""
    th = np.arcsin(LAM * s * FMAX)
    out = 0
    T = 1
    for a, c, z in zip(alphas, cols, depths):
        d = sign * (-(z - zref) * np.tan(th) / DX)
        a_s = shift_np(a, d)
        c_s = shift_np(np.asarray(c), d)
        out = out + (a_s * T)[..., None] * c_s
        T = T * (1 - a_s)
    return out


def sim_E(rng):
    alphas, cols, depths = scene_E()
    K = len(alphas)
    svals = np.linspace(-0.8, 0.8, 25)
    R_SUB, R_EYE = 0.3, 0.5
    # ---- amplitude model for the amplitude-domain rules (GWS-NRP, RPWS w/o √): same geometry, fitted
    # opacities + colours (front) and rear amplitude texture so that (amplitude blend)² matches the GT views.
    fit_s = [-0.8, -0.4, 0.0, 0.4, 0.8]
    gts_fit = [torch.tensor(ray_blend(alphas, cols, depths, s, ZM, 1), device=dev, dtype=torch.float32)
               for s in fit_s]
    shapes = [torch.tensor(a / max(a.max(), 1e-9), device=dev, dtype=torch.float32) for a in alphas[:-1]]
    win = torch.tensor(alphas[-1], device=dev, dtype=torch.float32)
    o_l = torch.tensor([np.log(a.max() / (1 - a.max() + 1e-3)) for a in alphas[:-1]], device=dev,
                       dtype=torch.float32, requires_grad=True)
    c_l = torch.tensor(np.stack([np.sqrt(c[0, 0]) for c in cols[:-1]]), device=dev, dtype=torch.float32,
                       requires_grad=True)
    b_l = torch.tensor(np.sqrt(cols[-1]), device=dev, dtype=torch.float32, requires_grad=True)
    kk = torch.fft.fftfreq(N, device=dev)

    def tshift(im, d):
        ph = torch.exp(-2j * np.pi * kk * d)
        if im.ndim == 3:
            return torch.real(torch.fft.ifft(torch.fft.fft(im, dim=1) * ph[None, :, None], dim=1))
        return torch.real(torch.fft.ifft(torch.fft.fft(im, dim=1) * ph[None, :], dim=1))

    def amp_view(s):
        th = np.arcsin(LAM * s * FMAX)
        out, T = 0, 1
        for k in range(K):
            d = float(-(depths[k] - ZM) * np.tan(th) / DX)
            if k < K - 1:
                a = tshift(torch.sigmoid(o_l[k]) * shapes[k], d)
                c = c_l[k].clamp(min=0)
            else:
                a = tshift(win, d)
                c = tshift(b_l.clamp(min=0), d)
            out = out + (a * T)[..., None] * c
            T = T * (1 - a)
        return out
    opt = torch.optim.Adam([o_l, c_l, b_l], lr=0.01)
    for it in range(800):
        loss = sum(((amp_view(s) ** 2 - g) ** 2).mean() for s, g in zip(fit_s, gts_fit))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        amp_alphas = [torch.sigmoid(o_l[k]).item() * alphas[k] / max(alphas[k].max(), 1e-9) for k in range(K - 1)]
        amp_alphas.append(alphas[-1])
        amp_cols = [np.broadcast_to(c_l[k].clamp(min=0).cpu().numpy(), (N, N, 3)) for k in range(K - 1)]
        amp_cols.append(b_l.clamp(min=0).cpu().numpy())
    print("E: amplitude-model fit loss %.2e" % loss.item())

    T_ = lambda a: torch.tensor(np.ascontiguousarray(a), device=dev, dtype=torch.float64)
    al_i = [T_(a) for a in alphas]
    co_i = [T_(c) for c in cols]
    al_a = [T_(a) for a in amp_alphas]
    co_a = [T_(c) for c in amp_cols]

    def rph():
        return torch.exp(1j * torch.tensor(rng.uniform(-np.pi, np.pi, (N, N)), device=dev))

    def slm(method, ch, phs):
        """u_SLM for one colour channel, phases phs[k] per primitive (shared across methods per frame)."""
        if method in ("nrp", "gwssqrt"):        # Algs. 4, 5: front → back, global transmittance map T
            u = 0
            T = torch.ones((N, N), device=dev, dtype=torch.float64)
            for k in range(K):
                if method == "nrp":
                    a, c = al_a[k], co_a[k][..., ch]
                    u = u + P_(-depths[k], T * a * c * phs[k])          # 𝒫(T·u_k e^{iφ}) · c_k o_k
                else:
                    a, c = al_i[k], co_i[k][..., ch]
                    u = u + P_(-depths[k], torch.sqrt(T * a * c) * phs[k])  # 𝒫(√(T|u_k|) e^{iφ}) √c √o
                T = T * (1 - a)
            return u
        # Algs. 3, 6: back → front, propagate the accumulated field to each primitive's plane, mask, add
        g = 0
        zprev = None
        for k in reversed(range(K)):
            if zprev is not None:
                g = P_(depths[k] - zprev, g)    # from the previous (farther) plane toward the SLM, to plane k
            if method == "rpws":
                a, c = al_i[k], co_i[k][..., ch]
                g = torch.sqrt(1 - a) * g + torch.sqrt(a * c) * phs[k]
            else:                               # "nosqrt"
                a, c = al_a[k], co_a[k][..., ch]
                g = (1 - a) * g + a * c * phs[k]
            zprev = depths[k]
        return P_(-zprev, g)

    Psub = [pup(s, 0, R_SUB) for s in svals]
    Pe = pup(0, 0, R_EYE)
    HF, HB, HM = H(ZF), H(ZB), H(ZM)
    methods = ["nrp", "gwssqrt", "nosqrt", "rpws"]
    acc = {m: dict(lf=torch.zeros((len(svals), 320, 320, 3), device=dev, dtype=torch.float64),
                   ff=torch.zeros((320, 320, 3), device=dev, dtype=torch.float64),
                   fb=torch.zeros((320, 320, 3), device=dev, dtype=torch.float64)) for m in methods}
    rp1 = None
    for t in range(TF):
        for ch in range(3):
            phs = [rph() for _ in range(K)]
            for m in methods:
                F = torch.fft.fft2(slm(m, ch, phs))
                for j, P in enumerate(Psub):
                    acc[m]["lf"][j, ..., ch] += torch.abs(torch.fft.ifft2(F * P * HM))[CR, CR] ** 2
                acc[m]["ff"][..., ch] += torch.abs(torch.fft.ifft2(F * Pe * HF))[CR, CR] ** 2
                acc[m]["fb"][..., ch] += torch.abs(torch.fft.ifft2(F * Pe * HB))[CR, CR] ** 2
        if t == 0:
            rp1 = {k: v.clone() for k, v in acc["rpws"].items()}
    # ---- ground truth (S6.1): views at z_ref = z_F; focal slices by shift-and-add over the eye pupil
    # sign check on the horizontal luminance profile of the pinhole edge view (rear window position dominates)
    prof = lambda im: im.mean((0, 2)) / im.mean()
    rv = prof(acc["rpws"]["lf"][-1].cpu().numpy())
    sgn = min((1, -1), key=lambda sg: np.mean(
        (prof(ray_blend(alphas, cols, depths, svals[-1], ZM, sg)[CR, CR]) - rv) ** 2))
    print("E: shear sign", sgn)
    # S6.1 ground truth through the same aperture: average of sheared orthographic ray-blended views over all
    # directions inside the pupil (shift-and-add), focused at z_ref.
    al_t = [torch.tensor(a_, device=dev, dtype=torch.float32) for a_ in alphas]
    co_t = [torch.tensor(np.ascontiguousarray(c_), device=dev, dtype=torch.float32) for c_ in cols]
    k2 = torch.fft.fftfreq(N, device=dev)
    KY, KX = torch.meshgrid(k2, k2, indexing="ij")

    def sh2(im, dx_, dy_):
        ph = torch.exp(-2j * np.pi * (KX * dx_ + KY * dy_))
        if im.ndim == 3:
            return torch.real(torch.fft.ifft2(torch.fft.fft2(im, dim=(0, 1)) * ph[..., None], dim=(0, 1)))
        return torch.real(torch.fft.ifft2(torch.fft.fft2(im) * ph))

    def gt_ap(cx, r, zref, n=9):
        acc_ = 0
        cnt = 0
        for sx in np.linspace(cx - r, cx + r, n):
            for sy in np.linspace(-r, r, n):
                if (sx - cx) ** 2 + sy ** 2 > r * r:
                    continue
                thx, thy = np.arcsin(LAM * sx * FMAX), np.arcsin(LAM * sy * FMAX)
                out, T = 0, 1
                for a_, c_, z in zip(al_t, co_t, depths):
                    dx_ = sgn * (-(z - zref) * np.tan(thx) / DX)
                    dy_ = sgn * (-(z - zref) * np.tan(thy) / DX)
                    a_s, c_s = sh2(a_, dx_, dy_), sh2(c_, dx_, dy_)
                    out = out + (a_s * T)[..., None] * c_s
                    T = T * (1 - a_s)
                acc_ = acc_ + out
                cnt += 1
        return (acc_ / cnt)[CR, CR].cpu().numpy().astype(float)
    with torch.no_grad():
        gt_lf = np.stack([gt_ap(s_, R_SUB, ZM) for s_ in svals])
        gt_ff = gt_ap(0.0, R_EYE, ZF, 13)
        gt_fb = gt_ap(0.0, R_EYE, ZB, 13)
    # one shared scale per output type: make RPWS match the GT in the unoccluded rear region
    reg = sum(alphas[:-1])[CR, CR] < 0.01
    reg &= alphas[-1][CR, CR] > 0.99
    out = {"gt_lf": gt_lf, "gt_ff": gt_ff, "gt_fb": gt_fb}
    for key, gtv in (("lf", gt_lf[len(svals) // 2]), ("ff", gt_ff), ("fb", gt_fb)):
        r = acc["rpws"][key].cpu().numpy() / TF
        rc = r[len(svals) // 2] if key == "lf" else r
        sc = gtv[reg].mean() / rc[reg].mean()
        for m in methods:
            out[f"{m}_{key}"] = acc[m][key].cpu().numpy() / TF * sc
        out[f"rpws1_{key}"] = rp1[key].cpu().numpy() * sc
    to8 = lambda a: (np.clip(a, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)   # display gamma
    np.savez_compressed(gen("e_views.npz"), svals=svals, **{k: to8(v) for k, v in out.items()})
    for m in methods + ["gt"]:
        lf = out[f"{m}_lf"]
        print("   %-8s LF MSE vs GT (centre / edge) %.4f / %.4f   rear-focus MSE %.4f" % (
            m, ((lf[12] - gt_lf[12]) ** 2).mean(), ((lf[-1] - gt_lf[-1]) ** 2).mean(),
            ((out[f"{m}_fb"] - gt_fb) ** 2).mean()))


# ====================================================================== F: Part 3 side view (sim units)
F_SRC = {"Q": -28.0, "P": 92.0, "R": -150.0}   # rear emitters (x, px): behind the occluder / beside it / far away
F_SIG = 6.0
F_ZO, F_WZ, F_HALF = 200, 440, 55              # occluder plane, SLM plane, occluder half-width (px)


def sim_F(rng):
    """Three small rear primitives (z = 0, colours per source) and a near-opaque occluder (z = F_ZO, |x| < 55,
    α = 0.985) with its own light; SLM at z = F_WZ. Each source is propagated separately (linearity) so it can
    be coloured; cross terms vanish in the time average.
      GWS with √ (Alg. 5): front → back; the occluder's transmittance T = 1 − α_F (a map in SLM-plane x)
        multiplies each rear primitive at its OWN plane, √(T α_k c_k) e^{iφ_k}, then free propagation to the SLM.
      RPWS (Alg. 3): back → front; rear field propagated to the occluder plane, × √(1 − α_F) THERE, plus
        √(α_F c_F) e^{iφ_F}, then propagated to the SLM.
    Eye views: a pupil at angle θ passes |f − sin θ/λ| < w of the SLM spectrum; the eye focuses on the rear plane."""
    WZ, ZO = F_WZ, F_ZO
    smf = lambda x, w: 1 / (1 + np.exp(-x / w))
    aF = 0.985 * smf(F_HALF - np.abs(_x1), 1.5)
    cF = 0.55
    src = {k: np.exp(-(_x1 - x0) ** 2 / (2 * F_SIG ** 2)) for k, x0 in F_SRC.items()}
    z1 = np.arange(ZO, dtype=float)
    z2 = np.arange(WZ - ZO, dtype=float)
    zall = np.arange(WZ, dtype=float)
    keys = list(F_SRC) + ["occ"]
    acc = {m: {k: np.zeros((HD, WZ)) for k in keys} for m in ("r", "g")}
    one = {k: None for k in keys}
    thetas = np.radians(np.array([-13.0, -6.5, 0.0, 6.5, 13.0]))
    w = 0.10 / (2 * P_S)
    views = {m: {k: np.zeros((len(thetas), HC)) for k in keys} for m in ("r", "g")}
    back = np.exp(-1j * _kz1 * WZ)

    def eye_views(u_slm):
        U = np.fft.fft(u_slm)
        return np.stack([np.abs(np.fft.ifft(U * (np.abs(_f1 - np.sin(t) / LAM_S) < w) * back)) ** 2
                         for t in thetas])
    for t in range(TF):
        e = {k: rand1(rng) for k in keys}
        for k in keys:
            if k == "occ":
                ff = np.sqrt(aF * 0.0 + aF * cF) * e[k]
                part = prop1(ff, z2)
                img = np.concatenate([np.zeros((HD, ZO)), np.abs(part) ** 2], 1)
                u_slm = np.fft.ifft(np.fft.fft(ff) * _band1 * np.exp(1j * _kz1 * (WZ - ZO)))
                for m in ("r", "g"):
                    acc[m][k] += img
                    views[m][k] += eye_views(u_slm)
                if t == 0:
                    one[k] = img
                continue
            g0 = src[k] * e[k]                                  # √(α c) e^{iφ} of a rear primitive (α c = src²)
            # RPWS
            f1 = prop1(g0, z1)
            g_at = np.fft.ifft(np.fft.fft(g0) * _band1 * np.exp(1j * _kz1 * ZO))
            gm = np.sqrt(1 - aF) * g_at
            f2 = prop1(gm, z2)
            img_r = np.concatenate([np.abs(f1) ** 2, np.abs(f2) ** 2], 1)
            u_r = np.fft.ifft(np.fft.fft(gm) * _band1 * np.exp(1j * _kz1 * (WZ - ZO)))
            # GWS with √
            gg = np.sqrt(1 - aF) * g0
            img_g = np.abs(prop1(gg, zall)) ** 2
            u_g = np.fft.ifft(np.fft.fft(gg) * _band1 * np.exp(1j * _kz1 * WZ))
            acc["r"][k] += img_r
            acc["g"][k] += img_g
            views["r"][k] += eye_views(u_r)
            views["g"][k] += eye_views(u_g)
            if t == 0:
                one[k] = img_r
    out = {}
    for m in ("r", "g"):
        for k in keys:
            out[f"{m}_{k}"] = (acc[m][k] / TF).astype(np.float32)
            out[f"v{m}_{k}"] = (views[m][k] / TF).astype(np.float32)
    for k in keys:
        out[f"one_{k}"] = one[k].astype(np.float32)
    # what each eye sees, as energy per source (full grid), relative to the source's unoccluded energy.
    # Wave views are divided by the pupil's share of the SLM band (2w · p = 0.1). Ground truth (S6.1): the occluder at
    # depth ZO appears shifted by ±ZO·tan θ relative to the rear plane; sign fixed by matching RPWS's occluder footprint.
    frac = 2 * w * P_S
    e0 = {k: np.sum(src[k] ** 2) for k in F_SRC}
    for sign in (1, -1):
        g = {k: np.array([np.sum((1 - np.interp(_x1, _x1 + sign * ZO * np.tan(th), aF)) * src[k] ** 2) / e0[k]
                          for th in thetas]) for k in F_SRC}
        occ_gt = np.array([np.interp(_x1, _x1 + sign * ZO * np.tan(th), aF) for th in thetas])
        prof = views["r"]["occ"] / TF
        err = np.mean((prof / prof.max() - occ_gt / occ_gt.max()) ** 2)
        if sign == 1 or err < best:
            best, gbest = err, g
    for k in F_SRC:
        out[f"egt_{k}"] = gbest[k]
        for m in ("r", "g"):
            out[f"e{m}_{k}"] = views[m][k].sum(1) / TF / frac / e0[k]
    for m in ("gt", "g", "r"):
        print("F %-2s energies at θ = %s°" % (m, np.round(np.degrees(thetas), 1)),
              {k: np.round(out[f"e{m}_{k}"], 2) for k in F_SRC})
    for m in ("r", "g"):
        for k in keys:
            del out[f"v{m}_{k}"]
    np.savez_compressed(gen("xzF.npz"), thetas=thetas, aF=aF[_rows].astype(np.float32), xs=_x1[_rows], **out)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["A", "C", "D", "E", "F"]
    for k in which:
        globals()["sim_" + k](np.random.default_rng({"A": 1, "C": 2, "D": 3, "E": 4, "F": 5}[k]))
    if len(which) == 5:
        open(gen("done_sims"), "w").write("ok")
