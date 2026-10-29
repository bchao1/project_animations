import glob, sys
from PIL import Image, ImageDraw
tag = sys.argv[1]
fs = sorted(glob.glob(f'scratch/still_{tag}_*.png'))
cw, ch = 640, 360
for part in range(0, len(fs), 9):
    sub = fs[part:part + 9]
    sheet = Image.new('RGB', (cw * 3, ch * 3), 'black')
    for k, f in enumerate(sub):
        im = Image.open(f).resize((cw, ch)); ImageDraw.Draw(im).text((8, 8), f[-9:-4], fill='red')
        sheet.paste(im, ((k % 3) * cw, (k // 3) * ch))
    sheet.save(f'scratch/sheet_{tag}_{part // 9}.png')
