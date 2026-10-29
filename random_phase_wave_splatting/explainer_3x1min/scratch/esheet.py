import numpy as np
from PIL import Image, ImageDraw
d=np.load('data/generated/e_views.npz')
ms=['gt','nrp','gwssqrt','nosqrt','rpws']
rows=[]
lin=lambda a:(a/255.)**2.2
for key in ['lf12','lf24','lf0','fb']:
    ims=[];errs=[]
    for m in ms:
        k = f'{m}_lf' if key.startswith('lf') else f'{m}_{key}'
        a=d[k]; g=d['gt_lf' if key.startswith('lf') else 'gt_'+key]
        if key.startswith('lf'): a=a[int(key[2:])]; g=g[int(key[2:])]
        ims.append(a)
        e=np.abs(lin(a)-lin(g)).mean(-1); e=np.clip(e/0.35,0,1)
        errs.append((np.stack([e,e*0.6,e*0.2],-1)*255).astype(np.uint8))
    rows.append(np.concatenate(ims,1)); rows.append(np.concatenate(errs,1))
sheet=np.concatenate(rows,0)
im=Image.fromarray(sheet).resize((sheet.shape[1]*2//5,sheet.shape[0]*2//5))
dr=ImageDraw.Draw(im)
for i,m in enumerate(ms): dr.text((i*128+4,2),m,fill=(255,255,0))
im.save('scratch/e_sheet.png')
