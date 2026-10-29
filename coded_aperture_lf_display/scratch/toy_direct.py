import numpy as np, torch, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from optics_proto import *
from toy_train import showcase, prep, save_img
import toy_train
OUT = "/home/brianchc/project_animations/coded_aperture_lf_display/scratch/proto_direct_" + sys.argv[5]
toy_train.OUT = OUT
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
V = render_views(showcase()).astype(np.float32)
E = bases()
cn, Fg, W, Fb = prep(V, E)
W = focus_weights(Fg, float(os.environ.get("BETA", "2")))
K = int(os.environ.get("K", "4")); NORM = os.environ.get("NORM", "sum")
print("baseline", psnr(Fb, Fg), time.time() - t0)
IT = int(sys.argv[1]); lrI = float(sys.argv[2]); lrP = float(sys.argv[3]); init = sys.argv[4]
if init == "base":
    P0 = torch.full((4, L, L), -float(sys.argv[6]))
    for i, (v, u) in enumerate(CORN):
        P0[i, v, u] = float(sys.argv[6])
    I0 = torch.logit(cn.clamp(0.01, 0.99))
else:
    torch.manual_seed(0)
    P0 = torch.randn(K, L, L) * 0.5
    I0 = torch.logit(cn.mean(0, keepdim=True).repeat(K, 1, 1, 1).clamp(0.01, 0.99))
Pp = torch.nn.Parameter(P0.clone()); Ip = torch.nn.Parameter(I0.clone())
opt = torch.optim.Adam([{"params": [Pp], "lr": lrP}, {"params": [Ip], "lr": lrI}])
for it in range(IT):
    Pc = torch.sigmoid(Pp)
    Fc = ctdm(torch.sigmoid(Ip), Pc, E) * (Pc.sum() / float(os.environ.get('FIXN', '81')) if NORM == 'fixed' else 1.0)
    loss = (crop(W) * (crop(Fc) - crop(Fg)).abs()).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 200 == 0 or it == IT - 1:
        print(it, round(float(loss), 4), round(psnr(Fc.detach(), Fg), 2), round(time.time() - t0), flush=True)
with torch.no_grad():
    for z in (0, 2, 4, 6, 8):
        save_img(Fc[z], f"c_{z}.png"); save_img(Fg[z], f"gt_{z}.png"); save_img(Fb[z], f"b_{z}.png")
    for i in range(K):
        save_img(torch.sigmoid(Ip)[i], f"enc_{i}.png")
    np.save(OUT + "/P.npy", torch.sigmoid(Pp).numpy())
