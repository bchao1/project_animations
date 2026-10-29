import subprocess, sys, os, glob, json
from PIL import Image, ImageDraw
V = sys.argv[1]; OUT = sys.argv[2]; STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
os.makedirs(OUT, exist_ok=True)
for mp4 in sorted(glob.glob(os.path.join(V, "S*.mp4"))):
    name = os.path.basename(mp4)[:-4]
    dur = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",mp4]).decode())
    tdir = os.path.join(OUT, name); os.makedirs(tdir, exist_ok=True)
    for f in glob.glob(tdir + "/*.png"): os.remove(f)
    subprocess.run(["ffmpeg","-v","error","-i",mp4,"-vf",f"fps=1/{STEP}","-s","427x240",tdir+"/f_%03d.png"])
    fr = sorted(glob.glob(tdir + "/f_*.png"))
    cols = 4; rows = (len(fr) + cols - 1)//cols
    for page in range(0, rows, 5):
        sub = fr[page*cols:(page+5)*cols]
        sheet = Image.new("RGB", (cols*431, ((len(sub)+cols-1)//cols)*244), (60,60,60))
        for k, f in enumerate(sub):
            im = Image.open(f); x, y = (k%cols)*431, (k//cols)*244
            sheet.paste(im, (x+2, y+2))
            ImageDraw.Draw(sheet).text((x+6, y+4), f"{(page*cols+k)*STEP:.0f}s", fill=(255,255,0))
        sheet.save(os.path.join(OUT, f"{name}_p{page//5}.png"))
    print(name, f"{dur:.1f}s", len(fr))
