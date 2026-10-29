import sys; sys.argv=['x']
exec(open('helpers_part.py').read())
T=32
def crop(path, box):
    im=Image.open(path).convert('RGB').crop(box).resize((T,T),Image.BOX); return np.asarray(im)/255.
fl=Image.open(os.path.join(DATA,'fig1_flower_full.png')); W,H=fl.size
cands={'paint':crop(gen('toy_target_hi.png'),(190,40,340,190)),
 'flower':np.clip(crop(os.path.join(DATA,'fig1_flower_full.png'),(int(W*.47),int(H*.3),int(W*.47)+150,int(H*.3)+150))*1.35,0,1),
 'flower2':np.clip(crop(os.path.join(DATA,'fig1_flower_full.png'),(int(W*.47),int(H*.3),int(W*.47)+260,int(H*.3)+260))*1.35,0,1),
 'art':crop(os.path.join(DATA,'fig1_art_full.png'),(600,400,850,650)),
 'target':crop(gen('toy_target_hi.png'),(0,0,384,384))}
base=np.array([1.0,0.6,0.3]); st=star_mask(T)
outs=[]
for k,t in cands.items():
    a,c=splat_layers(240,1.9,0.62,0.52,0.3,base,1.0,st,t-base)
    a2,c2=splat_layers(240,1.9,0.62,0.52,0.3,base,1.0,None,t-base)
    outs.append(np.hstack([over_black(a2,c2),over_black(a,c),(np.asarray(Image.fromarray((t*255).astype(np.uint8)).resize((240,240),Image.NEAREST)))]))
Image.fromarray(np.vstack(outs)).save('peek_tex.png')
