import json,gzip,base64,numpy as np,collections
d=json.load(open('../web/showcase.json'))
def decode(it):
    tg=it['token_grid']; raw=np.frombuffer(gzip.decompress(base64.b64decode(tg['leaves'])),np.uint8).reshape(-1,4)
    off=tg['offsets']; return [raw[off[t]:off[t+1]] for t in range(len(off)-1)], tg['height'], tg['width']
for k,it in enumerate(d['items']):
    fr,H,W=decode(it)
    fine=[]; cents=[]; static=0
    ext=collections.Counter()
    for L in fr:
        m=np.zeros((H,W),bool)
        for y0,y1,x0,x1 in L:
            a=(y1-y0)*(x1-x0); ext[(y1-y0,x1-x0)]+=1
            if a==1: m[y0:y1,x0:x1]=True
        fine.append(m.mean()); ys,xs=np.nonzero(m); cents.append((xs.mean() if len(xs) else 0, ys.mean() if len(ys) else 0))
    cents=np.array(cents); path=np.abs(np.diff(cents,axis=0)).sum()
    # temporal change of the leaf set
    sets=[set(map(tuple,L.tolist())) for L in fr]; persist=np.mean([len(sets[t]&sets[t+1])/len(sets[t]) for t in range(len(sets)-1)])
    tot=sum(len(L) for L in fr)
    print(f"{k:2d} {it['domain']:10s} {it['modality']:4s} tok={tot} comp={75600/tot:.2f} fine%={np.mean(fine)*100:4.1f} path={path:5.1f} persist={persist:.2f} ext={dict(ext.most_common(6))}")
