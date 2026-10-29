import json,gzip,base64,numpy as np
from PIL import Image, ImageDraw
d=json.load(open('../web/showcase.json'))
COL={1:(68,138,255),2:(0,200,83),4:(27,127,59)}
def decode(it):
    tg=it['token_grid']; raw=np.frombuffer(gzip.decompress(base64.b64decode(tg['leaves'])),np.uint8).reshape(-1,4)
    off=tg['offsets']; return [raw[off[t]:off[t+1]] for t in range(len(off)-1)]
def lay(L,s=4):
    im=Image.new('RGB',(80*s,45*s)); dr=ImageDraw.Draw(im)
    for y0,y1,x0,x1 in L.astype(int):
        e=max(y1-y0,x1-x0); dr.rectangle([x0*s,y0*s,x1*s-1,y1*s-1],fill=COL.get(e,(14,90,42)),outline=(0,0,0))
    return im
cands=[2,4,10,12,14,16,17,31,35,3]
rows=[]
for k in cands:
    fr=decode(d['items'][k]); row=Image.new('RGB',(320*6,180),'black')
    for j,(lt,pf) in enumerate([(0,1),(10,41),(20,81)]):
        g=Image.open(f'../data/clips/show{k:02d}_gen/{pf:04d}.jpg').convert('RGB').resize((320,180))
        row.paste(g,(j*640,0)); row.paste(lay(fr[lt]),(j*640+320,0))
    ImageDraw.Draw(row).text((5,5),f"{k} {d['items'][k]['domain']} {d['items'][k]['modality']}",fill='white')
    rows.append(row)
sh=Image.new('RGB',(1920,180*len(rows)))
for i,r in enumerate(rows): sh.paste(r,(0,180*i))
sh.save('scratch/cands.png')
