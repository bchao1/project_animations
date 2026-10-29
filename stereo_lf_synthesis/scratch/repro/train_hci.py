"""Short re-run of the ICIP 2021 pipeline on the HCI 4D light field benchmark (paper split: 20 train / 4 test).

Uses the authors' network definitions (DisparityNet, LFRefineNetV2 = PSAS 3D CNN) from github.com/bchao1/lf-flow
(cloned to scratch/lf-flow) and the paper's losses / weights:
    L = w_c L_c + w_t L_t + w_r L_r,  w_c = 1, w_t = 0.001, w_r = 1   (Eqs. 1, 2, 6, 7)
    L_c = |F_l - F_t|_1 + |F_r - F_t|_1,  L_t = TV(D_lr) + TV(D_rl),  L_r = |F_o - F_t|_1
Stereo input = middle row, left- and right-most views (code mode stereo_wide), 135 x 135 random crops, Adam lr 1e-3.
The disparity net predicts flow from the first to the second input image (tanh * max_disparity, HCI: 32), divided by the
8-view baseline -> unit disparity d; view (i, j) samples its source view s at (x + d (j - j_s), y + d (i - i_s)).
F_c = A_l F_l + (1 - A_l) F_r with A_l = K_r / (K_l + K_r) (Eqs. 3-5).

Writes to data/generated/repro/: ckpt.pt, log.json, metrics.json and origami outputs (see export()).
"""
import os, sys, json, time, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scratch", "lf-flow"))
from models.lf_net import DisparityNet, LFRefineNetV2  # noqa: E402

HCI = os.path.join(ROOT, "data", "hci", "full_data")
OUT = os.path.join(ROOT, "data", "generated", "repro")
os.makedirs(OUT, exist_ok=True)
N = 9
ROW, JL, JR = N // 2, 0, N - 1
MAXD = 32.0
BASE = float(JR - JL)
W_C, W_T, W_R = 1.0, 0.001, 1.0
CROP = 135
BATCH = 8
ITERS = int(os.environ.get("ITERS", 6000))
dev = torch.device("cuda")
torch.manual_seed(0)
np.random.seed(0)

TRAIN = sorted(glob.glob(os.path.join(HCI, "additional", "*"))) + sorted(glob.glob(os.path.join(HCI, "training", "*")))
TRAIN = [d for d in TRAIN if os.path.isdir(d)]
TEST = [os.path.join(HCI, "test", s) for s in ("bedroom", "bicycle", "herbs", "origami")]


def load_lf(d):
    v = [np.asarray(Image.open(os.path.join(d, f"input_Cam{k:03d}.png")).convert("RGB")) for k in range(N * N)]
    return np.stack(v).reshape(N, N, *v[0].shape)          # (9, 9, 512, 512, 3) uint8


def to_t(lf):                                              # -> (81, 3, H, W) in [-1, 1]
    t = torch.from_numpy(lf).to(dev).float().div(127.5).sub(1)
    return t.reshape(N * N, *t.shape[2:]).permute(0, 3, 1, 2).contiguous()


def warp(img, d, js):
    """img (b, 3, h, w), d (b, h, w) unit disparity -> light field (b, 81, 3, h, w) by backward warping."""
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    out = []
    for i in range(N):
        for j in range(N):
            gx = xs + d * (j - js)
            gy = ys + d * (i - ROW)
            grid = torch.stack([2 * gx / (w - 1) - 1, 2 * gy / (h - 1) - 1], -1)
            out.append(F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True))
    return torch.stack(out, 1)


IJ = np.stack(np.meshgrid(np.arange(N), np.arange(N), indexing="ij"), -1).astype(np.float32)   # (i, j)
K_L = np.linalg.norm(IJ - np.array([ROW, JL], np.float32), axis=-1)
K_R = np.linalg.norm(IJ - np.array([ROW, JR], np.float32), axis=-1)
A_L = K_R / (K_L + K_R)                                                                      # Eqs. 3-5
A_T = torch.from_numpy(A_L.reshape(1, N * N, 1, 1, 1)).to(dev)


def tv(d):
    return (d[:, 1:, :] - d[:, :-1, :]).abs().mean() + (d[:, :, 1:] - d[:, :, :-1]).abs().mean()


def forward(gnet, rnet, lf):
    """lf (b, 81, 3, h, w) target. Returns dict of all intermediate tensors."""
    sl = lf[:, ROW * N + JL]
    sr = lf[:, ROW * N + JR]
    f_lr, _ = gnet(torch.cat([sl, sr], 1))
    f_rl, _ = gnet(torch.cat([sr, sl], 1))
    d_lr = (f_lr[:, 0] * MAXD) / BASE
    d_rl = -(f_rl[:, 0] * MAXD) / BASE
    Fl = warp(sl, d_lr, JL)
    Fr = warp(sr, d_rl, JR)
    Fc = A_T * Fl + (1 - A_T) * Fr
    Fo = rnet(Fc)
    return dict(d_lr=d_lr, d_rl=d_rl, Fl=Fl, Fr=Fr, Fc=Fc, Fo=Fo)


def psnr(a, b):          # a, b in [-1, 1]
    mse = ((a - b) / 2).pow(2).mean().item()
    return 10 * math.log10(1 / max(mse, 1e-12))


def train():
    lfs = [to_t(load_lf(d)) for d in TRAIN]
    print(len(lfs), "training light fields", flush=True)
    gnet = DisparityNet(N * N).to(dev)
    rnet = LFRefineNetV2(N * N, N * N, N * N).to(dev)
    print("params g", sum(p.numel() for p in gnet.parameters()), "r", sum(p.numel() for p in rnet.parameters()), flush=True)
    opt = torch.optim.Adam(list(gnet.parameters()) + list(rnet.parameters()), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    log = []
    t0 = time.time()
    snaps = sorted(set([0] + [int(round(x)) for x in np.geomspace(10, ITERS, 24)]))
    for it in range(ITERS + 1):
        if it in snaps:
            snapshot(gnet, rnet, it)
            gnet.train(); rnet.train()
        if it == ITERS:
            break
        batch = []
        for _ in range(BATCH):
            lf = lfs[np.random.randint(len(lfs))]
            y = np.random.randint(0, lf.shape[2] - CROP + 1)
            x = np.random.randint(0, lf.shape[3] - CROP + 1)
            batch.append(lf[:, :, y:y + CROP, x:x + CROP])
        tgt = torch.stack(batch)
        o = forward(gnet, rnet, tgt)
        Lc = (o["Fl"] - tgt).abs().mean() + (o["Fr"] - tgt).abs().mean()
        Lt = tv(o["d_lr"]) + tv(o["d_rl"])
        Lr = (o["Fo"] - tgt).abs().mean()
        loss = W_C * Lc + W_T * Lt + W_R * Lr
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            rec = dict(it=it, loss=loss.item(), Lc=Lc.item(), Lt=Lt.item(), Lr=Lr.item(),
                       psnr_o=psnr(o["Fo"].detach(), tgt), psnr_c=psnr(o["Fc"].detach(), tgt), t=time.time() - t0)
            log.append(rec)
            if it % 200 == 0:
                print(rec, flush=True)
                json.dump(log, open(os.path.join(OUT, "log.json"), "w"))
    json.dump(log, open(os.path.join(OUT, "log.json"), "w"))
    torch.save(dict(g=gnet.state_dict(), r=rnet.state_dict()), os.path.join(OUT, "ckpt.pt"))
    return gnet, rnet


ORI = None


def origami():
    global ORI
    if ORI is None:
        lf = load_lf(os.path.join(HCI, "test", "origami"))[:, :, 4:508, 4:508]     # 504 = 56 * 9 (PSAS needs N | H, W)
        ORI = to_t(np.ascontiguousarray(lf))[None]
    return ORI


@torch.no_grad()
def snapshot(gnet, rnet, it):
    """Centre-row outputs on origami during training (for the 'training' beat)."""
    gnet.eval(); rnet.eval()
    o = forward(gnet, rnet, origami())
    u8 = lambda t: ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    k = ROW * N + ROW
    np.save(os.path.join(OUT, f"snap_{it:05d}.npy"),
            np.stack([u8(o["Fl"][0, [k]])[0], u8(o["Fc"][0, [k]])[0], u8(o["Fo"][0, [k]])[0]]))
    np.save(os.path.join(OUT, f"snapd_{it:05d}.npy"), o["d_lr"][0].cpu().numpy().astype(np.float16))


@torch.no_grad()
def evaluate(gnet, rnet):
    gnet.eval(); rnet.eval()
    res = {}
    for d in TEST:
        name = os.path.basename(d)
        lf = to_t(np.ascontiguousarray(load_lf(d)[:, :, 4:508, 4:508]))[None]
        o = forward(gnet, rnet, lf)
        avg = 0.5 * (o["Fl"] + o["Fr"])
        r = {k: psnr(o[k], lf) for k in ("Fl", "Fr", "Fc", "Fo")}
        r["avg"] = psnr(avg, lf)
        # per-view MSE maps (Fig. 2 style), in [0, 1] intensity units
        for k in ("Fl", "Fr", "Fc", "Fo"):
            r["mse_" + k] = (((o[k] - lf) / 2) ** 2).mean(dim=(2, 3, 4))[0].reshape(N, N).cpu().numpy().tolist()
        res[name] = r
        print(name, {k: round(v, 2) for k, v in r.items() if not k.startswith("mse")}, flush=True)
        if name == "origami":
            export(o, lf)
    keys = ("Fl", "Fr", "avg", "Fc", "Fo")
    res["mean"] = {k: float(np.mean([res[os.path.basename(d)][k] for d in TEST])) for k in keys}
    for k in ("Fl", "Fr", "Fc", "Fo"):
        res["mean"]["mse_" + k] = np.mean([res[os.path.basename(d)]["mse_" + k] for d in TEST], 0).tolist()
    print("mean", {k: round(v, 2) for k, v in res["mean"].items() if not k.startswith("mse")}, flush=True)
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"))


def export(o, lf):
    u8 = lambda t: ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()[0]
    for k in ("Fl", "Fr", "Fc", "Fo"):
        np.save(os.path.join(OUT, f"origami_{k}.npy"), u8(o[k]))
    np.save(os.path.join(OUT, "origami_gt.npy"), u8(lf))
    np.save(os.path.join(OUT, "origami_d_lr.npy"), o["d_lr"][0].cpu().numpy())
    np.save(os.path.join(OUT, "origami_d_rl.npy"), o["d_rl"][0].cpu().numpy())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        ck = torch.load(os.path.join(OUT, "ckpt.pt"))
        g = DisparityNet(N * N).to(dev); g.load_state_dict(ck["g"])
        r = LFRefineNetV2(N * N, N * N, N * N).to(dev); r.load_state_dict(ck["r"])
        evaluate(g, r)
    else:
        g, r = train()
        evaluate(g, r)
