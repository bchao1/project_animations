import sys, glob
from PIL import Image, ImageDraw
tag = sys.argv[1]
fs = sorted(glob.glob(f'scratch/still_{tag}_*.png'))
cw, ch = 640, 360
rows = (len(fs) + 2) // 3
sheet = Image.new('RGB', (cw * 3, ch * rows), 'white')
for k, f in enumerate(fs):
    im = Image.open(f).resize((cw, ch)); ImageDraw.Draw(im).text((8, 8), f[-9:-4], fill='red')
    sheet.paste(im, ((k % 3) * cw, (k // 3) * ch))
for i in range(0, rows, 3):
    sheet.crop((0, i * ch, cw * 3, min(rows, i + 3) * ch)).save(f'scratch/sheet_{tag}_{i // 3}.png')
