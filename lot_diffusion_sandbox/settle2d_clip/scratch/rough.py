import sys, numpy as np
d=np.load(sys.argv[1]); P=d['pos']; tile=d['tile']
def rough(f):
    r=[]
    for i in range(6):
        x,y=P[f,tile==i].T
        top=np.array([y[(x>=c)&(x<c+1)].max()+0.5 for c in range(64)])
        r.append((top.std(), top.max()-top.min()))
    return np.array(r)
for t in (8.0, 10.0, 12.0, 14.4):
    r=rough(int((t-4.6)*60)); print(f"t={t}: std {r[:,0].mean():.2f} (max {r[:,0].max():.2f})  range {r[:,1].mean():.2f} (max {r[:,1].max():.1f})")
