import numpy as np
from PIL import Image
d=np.load('data/generated/xzF.npz')
COL={'Q':np.array([0.22,0.89,0.49]),'P':np.array([0.27,0.75,1.0]),'R':np.array([0.88,0.25,0.98]),'occ':np.array([1.0,0.79,0.16])}
def comp(m,g=0.5,sc=0.6):
    out=0
    for k,c in COL.items():
        v=np.clip(d[f'{m}_{k}']/sc,0,1)**g
        out=out+v[...,None]*c
    return (np.clip(out,0,1)*255).astype(np.uint8)
sep=np.full((d['r_Q'].shape[0],6,3),80,np.uint8)
im=np.concatenate([comp('g'),sep,comp('r'),sep,comp('one')],1)
Image.fromarray(im).save('scratch/f_xz.png')
