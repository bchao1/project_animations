"""Extract cropped demo-clip frames and paper-figure crops for the 10 s world-model clip.

Run from world_models/:  python scratch/prep_assets.py
Writes data/clips/<name>/%04d.jpg (30 fps, pre-cropped, pre-scaled to card pixel size) and data/figcrops/*.png.
"""
import os
import subprocess

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "data", "videos")
F = os.path.join(ROOT, "data", "figs")
CLIPS = os.path.join(ROOT, "data", "clips")
CROPS = os.path.join(ROOT, "data", "figcrops")

CARD_W = 1920  # full-bleed width; VideoCard downsizes per style

# name: (source, start s, duration s, crop w:h:x:y or None)
CLIP_SPECS = {
    "camctrl2": ("cc2_citywalk1.mp4", 0.0, 4.2, "1231:695:680:144"),
    "genreal": ("gr_teaser_final2.mp4", 22.25, 2.35, None),
    "bullettime": ("bt_intro_public_1.mp4", 13.25, 3.0, "1800:760:0:200"),
    "spmem": ("sm_demo_page.mp4", 5.6, 3.2, None),
}

# name: (figure file, crop box in figure px)
FIG_SPECS = {
    "cc1_row": ("cameractrl_fig1.png", (150, 5, 605, 122)),
    "cc2_rows": ("cameractrl2_fig1.png", (236, 66, 962, 332)),
    "gr_inputs": ("gr_fig1.png", (28, 18, 432, 548)),
    "bt_fig1": ("bt_fig1.png", (0, 0, 1412, 690)),
    "bt_bullet": ("bt_fig5.png", (690, 0, 1418, 228)),
    "sm_teaser": ("sm_teaser.jpg", (0, 0, 1545, 1306)),
}


def extract_clips():
    for name, (src, t0, dur, crop) in CLIP_SPECS.items():
        out = os.path.join(CLIPS, name)
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
        vf = (f"crop={crop}," if crop else "") + f"fps=30,scale={CARD_W}:-2:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(dur),
                        "-i", os.path.join(V, src), "-vf", vf, "-q:v", "2",
                        os.path.join(out, "%04d.jpg")], check=True)
        print(name, len(os.listdir(out)), "frames")


def round_corners(im, r):
    im = im.convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], r, fill=255)
    im.putalpha(mask)
    return im


def crop_figs():
    os.makedirs(CROPS, exist_ok=True)
    for name, (src, box) in FIG_SPECS.items():
        im = Image.open(os.path.join(F, src)).convert("RGB")
        if name == "sm_teaser":  # paint out the grey "next" arrows that poke in from the right column
            d = ImageDraw.Draw(im)
            for y0, y1 in ((600, 660), (1240, 1306)):
                d.rectangle([1480, y0, 1545, y1], fill="white")
        im = im.crop(box)
        # pad with white so the paper figure sits on a clean card
        pad = max(8, im.size[0] // 60)
        card = Image.new("RGB", (im.size[0] + 2 * pad, im.size[1] + 2 * pad), "white")
        card.paste(im, (pad, pad))
        round_corners(card, 3 * pad).save(os.path.join(CROPS, name + ".png"))
        print(name, card.size)


if __name__ == "__main__":
    extract_clips()
    crop_figs()
