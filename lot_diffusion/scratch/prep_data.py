"""Build lot_diffusion/data/ from the paper PDF rasters and the mirrored supplementary site (web/).

Outputs
  data/layouts/*.json      exact token layouts: {"H":..,"W":..,"rects":[[y0,y1,x0,x1],...]} (image) or
                           {"H","W","slices":[[rects], ...]} (video, one list per latent time slice)
  data/img/*.png|jpg       stills (paper figure crops, website results)
  data/clips/<key>/*.jpg   video frames at the source fps (see data/clips/index.json)
Run: python scratch/prep_data.py   (imp_tome env). Idempotent; delete a file to rebuild it.
"""
import base64
import glob
import gzip
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
D = os.path.join(ROOT, "data")
for sub in ("layouts", "img", "clips"):
    os.makedirs(os.path.join(D, sub), exist_ok=True)
AMAP = json.load(open(os.path.join(WEB, "asset_map.json")))


def W(p):
    """website path (named or assets/) -> local file"""
    if p.startswith("assets/"):
        return os.path.join(WEB, p)
    if p in AMAP:
        return os.path.join(WEB, AMAP[p])
    return os.path.join(WEB, p)


def save_layout(name, obj):
    json.dump(obj, open(os.path.join(D, "layouts", name + ".json"), "w"))


# ------------------------------------------------------------------ layout parsing
def parse_grid_image(path, n_h, n_w, line="dark"):
    """Recover axis-aligned token rectangles from a boundary image on an n_h x n_w lattice.

    line="dark": black borders over gray fills (images.json grids); line="red": red borders over black (painted).
    Two neighbouring lattice cells belong to the same token when the pixels on their shared edge are not border.
    """
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    H, Wp = a.shape[:2]
    ch, cw = H / n_h, Wp / n_w

    def is_line(px):
        r, g, b = px
        return r < 20 and g < 20 and b < 20

    def contrast_line(y, x0, x1, horizontal):
        # painted grids: coloured 1-px lines over a smooth map -> the edge pixel stands out from both sides
        hits = 0
        for t in np.linspace(0.25, 0.75, 5):
            best = 0
            for d in (-1, 0, 1):
                if horizontal:
                    yy, xx = int(round(y)) + d, int(x0 + t * (x1 - x0))
                    s1, s2 = a[min(max(yy - 4, 0), H - 1), xx], a[min(yy + 4, H - 1), xx]
                else:
                    xx, yy = int(round(y)) + d, int(x0 + t * (x1 - x0))
                    s1, s2 = a[yy, min(max(xx - 4, 0), Wp - 1)], a[yy, min(xx + 4, Wp - 1)]
                yy, xx = min(max(yy, 0), H - 1), min(max(xx, 0), Wp - 1)
                best = max(best, np.abs(a[yy, xx] - (s1 + s2) / 2).sum())
            hits += best > 60
        return hits >= 3

    def edge_is_line(y, x0, x1, horizontal):
        if line != "dark":
            return contrast_line(y, x0, x1, horizontal)
        # sample a few pixels along a shared edge (widened by +-1 px)
        hits = 0
        tot = 0
        for t in np.linspace(0.3, 0.7, 5):
            for d in (-1, 0, 1):
                if horizontal:
                    yy, xx = int(round(y)) + d, int(x0 + t * (x1 - x0))
                else:
                    xx, yy = int(round(y)) + d, int(x0 + t * (x1 - x0))
                yy = min(max(yy, 0), H - 1)
                xx = min(max(xx, 0), Wp - 1)
                tot += 1
                hits += is_line(a[yy, xx])
        return hits >= 3

    right = np.zeros((n_h, n_w), bool)   # border between (i,j) and (i,j+1)
    down = np.zeros((n_h, n_w), bool)    # border between (i,j) and (i+1,j)
    for i in range(n_h):
        for j in range(n_w - 1):
            right[i, j] = edge_is_line((j + 1) * cw, i * ch, (i + 1) * ch, horizontal=False)
    for i in range(n_h - 1):
        for j in range(n_w):
            down[i, j] = edge_is_line((i + 1) * ch, j * cw, (j + 1) * cw, horizontal=True)
    lab = -np.ones((n_h, n_w), int)
    rects = []
    for i in range(n_h):
        for j in range(n_w):
            if lab[i, j] >= 0:
                continue
            # grow a rectangle: extend right while no border, then down while the whole row is open
            x1 = j
            while x1 + 1 < n_w and not right[i, x1] and lab[i, x1 + 1] < 0:
                x1 += 1
            y1 = i
            while y1 + 1 < n_h and all(not down[y1, x] for x in range(j, x1 + 1)):
                y1 += 1
            lab[i:y1 + 1, j:x1 + 1] = len(rects)
            rects.append([i, y1 + 1, j, x1 + 1])
    return rects


def parse_quadtree(path, n=64, top=8, thr=45):
    """Painted layouts are lattice-aligned quadtrees (paper Eq. 23): split a b x b block when its two mid-lines
    are drawn. Line test = contrast of the mid-line pixel against pixels 3 px to either side."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    H = a.shape[0]
    c = H / n

    def line_frac(fixed, lo, hi, horizontal):
        hits, tot = 0, 0
        for t in np.linspace(lo, hi, max(4, int((hi - lo) / c) * 3))[1:-1]:
            best = 0
            for d in (-1, 0, 1):
                f = int(round(fixed)) + d
                v = int(t)
                if horizontal:
                    e, s1, s2 = a[f, v], a[max(f - 3, 0), v], a[min(f + 3, H - 1), v]
                else:
                    e, s1, s2 = a[v, f], a[v, max(f - 3, 0)], a[v, min(f + 3, H - 1)]
                best = max(best, np.abs(e - (s1 + s2) / 2).sum())
            hits += best > thr
            tot += 1
        return hits / max(tot, 1)

    rects = []

    def rec(y, x, b):
        if b > 1:
            ym, xm = (y + b / 2) * c, (x + b / 2) * c
            fh = line_frac(ym, x * c, (x + b) * c, True)
            fv = line_frac(xm, y * c, (y + b) * c, False)
            if fh > 0.5 and fv > 0.5:
                h = b // 2
                for dy in (0, h):
                    for dx in (0, h):
                        rec(y + dy, x + dx, h)
                return
        rects.append([y, y + b, x, x + b])

    for y in range(0, n, top):
        for x in range(0, n, top):
            rec(y, x, top)
    return rects


def decode_leaves(tg):
    raw = np.frombuffer(gzip.decompress(base64.b64decode(tg["leaves"])), dtype=np.uint8).reshape(-1, 4)
    offs = list(tg["offsets"])
    if offs[-1] != len(raw):
        offs = offs + [len(raw)]
    return [raw[offs[k]:offs[k + 1]].tolist() for k in range(len(offs) - 1)]


# ------------------------------------------------------------------ video frames
CLIPS = {}


def frames(key, src, fps=None, scale_w=1280, start=0.0, dur=None, crop=None):
    out = os.path.join(D, "clips", key)
    if not os.path.exists(src):
        print("missing video", key, src)
        return
    if not glob.glob(os.path.join(out, "*.jpg")):
        os.makedirs(out, exist_ok=True)
        vf = []
        if crop:
            vf.append("crop=%s" % crop)
        if fps:
            vf.append("fps=%s" % fps)
        vf.append("scale=%d:-2:flags=lanczos" % scale_w)
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start)] + (["-t", str(dur)] if dur else []) + [
            "-i", src, "-vf", ",".join(vf), "-q:v", "2", os.path.join(out, "%04d.jpg")]
        subprocess.run(cmd, check=True)
    n = len(glob.glob(os.path.join(out, "*.jpg")))
    if fps is None:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                            "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
        a, b = r.split("/")
        fps = float(a) / float(b)
    CLIPS[key] = {"fps": float(fps), "n": n, "src": os.path.relpath(src, ROOT)}


def main():
    # ---------------- paper figures (rasters are stored vertically flipped in the PDF)
    PI = os.path.join(ROOT, "paper", "imgs")
    fig1 = ImageOps.flip(Image.open(os.path.join(PI, "p01_00_2450x2159.png")).convert("RGB"))
    fig1.save(os.path.join(D, "img", "fig1_full.png"))
    # ---------------- Fig. 10 budget sweep (deer row) from the flipped raster; 6 columns x (image, grid)
    f10 = ImageOps.flip(Image.open(os.path.join(PI, "p23_00_2570x3751.png")).convert("RGB"))
    f10.save(os.path.join(D, "img", "fig10_full.png"))
    # ---------------- Fig. 19 controllability rows from the 250 dpi page render
    p34 = Image.open(os.path.join(ROOT, "paper", "pages", "p34_hi.png")).convert("RGB")
    p34.save(os.path.join(D, "img", "fig19_page.png"))
    p39 = Image.open(os.path.join(ROOT, "paper", "pages", "p39_hi.png")).convert("RGB")
    p39.save(os.path.join(D, "img", "fig22_page.png"))

    # ---------------- images.json: same prompt across 4 sources (exact layouts from grid images)
    ij = json.load(open(os.path.join(WEB, "images.json")))
    meta = []
    for it in ij["items"]:
        iid = it["image_id"]
        rec = {"id": iid, "prompt": it["prompt"], "sources": {}}
        for s, v in it["sources"].items():
            m = v["media"]
            name = "img_%s_%s" % (iid, s)
            lp = os.path.join(D, "layouts", name + ".json")
            if not os.path.exists(lp):
                g = Image.open(W(m["grid"]))
                n = g.size[0] // 16
                rects = parse_grid_image(W(m["grid"]), g.size[1] // 16, n)
                save_layout(name, {"H": g.size[1] // 16, "W": n, "rects": rects, "tokens": v["tokens"]})
            for k in ("cue", "generated", "generated_4b"):
                if k in m:
                    dst = os.path.join(D, "img", "%s_%s.png" % (name, k))
                    if not os.path.exists(dst):
                        Image.open(W(m[k])).convert("RGB").save(dst)
            rec["sources"][s] = {"tokens": v["tokens"], "compression": v["compression"],
                                 "parsed": len(json.load(open(lp))["rects"])}
        meta.append(rec)
    json.dump(meta, open(os.path.join(D, "img", "images_index.json"), "w"), indent=1)

    # ---------------- painted image demos (Result / Token grid / Painting)
    pj = json.load(open(os.path.join(WEB, "painted_demos.json")))
    for k, e in enumerate(pj["examples"]):
        name = "paint%d" % k
        for view, fn in (("Result", "result"), ("Painting", "painting"), ("Grid overlay", "overlay")):
            dst = os.path.join(D, "img", "%s_%s.png" % (name, fn))
            if not os.path.exists(dst):
                Image.open(W(e["views"][view])).convert("RGB").save(dst)
        lp = os.path.join(D, "layouts", name + ".json")
        if not os.path.exists(lp):
            g = Image.open(W(e["views"]["Token grid"]))
            want = e["config"]["token_count"]
            cands = [parse_quadtree(W(e["views"]["Token grid"]), thr=t) for t in range(40, 76, 3)]
            rects = min(cands, key=lambda r: abs(len(r) - want))
            save_layout(name, {"H": 64, "W": 64, "rects": rects, "tokens": e["config"]["token_count"],
                               "title": e["title"]})
    for k, r in enumerate(pj["recordings"]):
        frames("rec%d" % k, W(r["video"]), fps=30, scale_w=1600)

    # ---------------- showcase videos (layout + generated) with exact per-slice layouts
    sj = json.load(open(os.path.join(WEB, "showcase.json")))
    sidx = []
    for k, it in enumerate(sj["items"]):
        key = "show%02d" % k
        frames(key + "_gen", W(it["generated"]))
        if it.get("filled_layout"):
            frames(key + "_lay", W(it["filled_layout"]))
        lp = os.path.join(D, "layouts", key + ".json")
        if not os.path.exists(lp):
            tg = it["token_grid"]
            save_layout(key, {"H": tg["height"], "W": tg["width"], "slices": decode_leaves(tg), "frames": tg["frames"]})
        sidx.append({"key": key, "modality": it["modality"], "domain": it["domain"], "compression": it["compression"],
                     "prompt": it["prompt"], "legend": it.get("layout_legend")})
    json.dump(sidx, open(os.path.join(D, "clips", "showcase_index.json"), "w"), indent=1)

    # ---------------- 3D-scene and painted video demos
    for key in ("train", "restaurant", "platformer", "corgi", "dress", "basketball"):
        for part in ("detail-map", "generated-grid", "generated-video", "generated-video-grid-overlay"):
            p = os.path.join(WEB, "video_demo_media", key, part + ".mp4")
            if not os.path.exists(p):
                p = W("video_demo_media/%s/%s.mp4" % (key, part))
            if os.path.exists(p):
                frames("vd_%s_%s" % (key, part.replace("generated-video-grid-overlay", "overlay")
                                     .replace("generated-video", "gen").replace("generated-grid", "grid")
                                     .replace("detail-map", "map")), p)
        rp = W("video_demo_media/%s/recording.mp4" % key)
        if os.path.exists(rp):
            frames("vd_%s_rec" % key, rp, fps=30, scale_w=1600)

    # ---------------- video baseline comparisons (5 synced videos each)
    cj = json.load(open(os.path.join(WEB, "comparisons.json")))
    cidx = []
    for k, it in enumerate(cj["items"]):
        rec = {"key": "cmp%d" % k, "compression": it["compression"], "prompt": it["prompt"],
               "stats": it["method_stats"], "videos": {}}
        for meth, p in it["videos"].items():
            tag = {"Layout cue": "cue", "Dense Wan2.1": "dense", "ToMe-SD": "tome",
                   "Foveated Diffusion": "fov", "Ours": "ours"}[meth]
            frames("cmp%d_%s" % (k, tag), W(p), scale_w=960)
            rec["videos"][meth] = "cmp%d_%s" % (k, tag)
        cidx.append(rec)
    json.dump(cidx, open(os.path.join(D, "clips", "comparisons_index.json"), "w"), indent=1)

    # ---------------- bbox (Ideogram-style) and typography stills
    bj = json.load(open(os.path.join(WEB, "bbox_examples.json")))
    for e in bj["examples"]:
        for k, p in e["media"].items():
            dst = os.path.join(D, "img", "bbox_%s_%s.png" % (e["id"], k))
            if not os.path.exists(dst):
                Image.open(os.path.join(WEB, p)).convert("RGB").save(dst)
    json.dump(bj, open(os.path.join(D, "img", "bbox_index.json"), "w"), indent=1)
    tj = json.load(open(os.path.join(WEB, "typography.json")))
    for s in tj["samples"]:
        for k in ("ours", "dense", "overlay"):
            p = os.path.join(WEB, s[k]["full"])
            dst = os.path.join(D, "img", "typo%03d_%s.png" % (s["number"], k))
            if os.path.exists(p) and not os.path.exists(dst):
                im = Image.open(p).convert("RGB")
                im.thumbnail((1400, 1400))
                im.save(dst)
        gp = os.path.join(WEB, s["grid"]["full"])
        dst = os.path.join(D, "img", "typo%03d_grid.png" % s["number"])
        if os.path.exists(gp) and not os.path.exists(dst):
            Image.open(gp).convert("RGB").save(dst)
    json.dump(tj, open(os.path.join(D, "img", "typography_index.json"), "w"), indent=1)

    json.dump(CLIPS, open(os.path.join(D, "clips", "index.json"), "w"), indent=1)
    print("clips", len(CLIPS))


if __name__ == "__main__":
    main()


def crop_figs():
    """Fig. 10 budget sweep (flipped raster) and Fig. 19 controllability panels (450 dpi page render)."""
    import fitz
    f10 = Image.open(os.path.join(D, "img", "fig10_full.png"))
    cols = [(10, 430), (436, 856), (862, 1282), (1288, 1708), (1714, 2134), (2140, 2560)]
    rows = [(51, 471), (477, 897), (979, 1399), (1405, 1825), (1907, 2327), (2333, 2753), (2835, 3255), (3261, 3681)]
    names = ["deer", "alps", "palace", "falls"]
    tags = ["ref", "B1024", "B1792", "B2560", "B3328", "B4096"]
    for s, nm in enumerate(names):
        for c, tg in enumerate(tags):
            for r, kind in enumerate(("img", "grid")):
                y0, y1 = rows[2 * s + r]
                f10.crop((cols[c][0], y0, cols[c][1], y1)).save(os.path.join(D, "img", "f10_%s_%s_%s.png" % (nm, tg, kind)))
    doc = fitz.open(os.path.join(ROOT, "7014_Level_of_Token_Diffusion.pdf"))
    k = 450 / 250.0
    pg = os.path.join(ROOT, "paper", "pages", "p34_450.png")
    if not os.path.exists(pg):
        doc[33].get_pixmap(dpi=450).save(pg)
    P = Image.open(pg).convert("RGB")
    cols = [(587, 774), (778, 965), (969, 1156), (1160, 1347), (1351, 1538)]
    rows = [(393, 580), (607, 795), (822, 1010), (1037, 1224), (1252, 1439), (1467, 1654), (1682, 1869), (1897, 2084),
            (2112, 2299)]
    ctags = ["gridA", "oursA", "gridB", "oursB", "dense"]
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(cols):
            P.crop((int(x0 * k) + 2, int(y0 * k) + 2, int(x1 * k) - 2, int(y1 * k) - 2)).save(
                os.path.join(D, "img", "f19_%s_%s.png" % ("abcdefghi"[r], ctags[c])))


if __name__ == "__main__" and os.environ.get("CROP_ONLY"):
    crop_figs()
