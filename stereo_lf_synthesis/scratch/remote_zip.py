import urllib.request, io, zipfile, sys, os, time
U = "http://lightfield-analysis.net/benchmark/downloads/full_data.zip"
class RF(io.RawIOBase):
    def __init__(s, u):
        s.u = u; s.p = 0
        r = urllib.request.urlopen(urllib.request.Request(u, method="HEAD")); s.n = int(r.headers["Content-Length"])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.p
    def seek(s, o, w=0):
        s.p = o if w == 0 else (s.p + o if w == 1 else s.n + o); return s.p
    def read(s, k=-1):
        if k < 0 or s.p + k > s.n: k = s.n - s.p
        if k <= 0: return b""
        for t in range(20):
            try:
                r = urllib.request.urlopen(urllib.request.Request(s.u, headers={"Range": f"bytes={s.p}-{s.p+k-1}"}), timeout=60)
                d = r.read(); break
            except Exception:
                time.sleep(3 + 2 * t)
        s.p += len(d); return d
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)
z = zipfile.ZipFile(io.BufferedReader(RF(U), buffer_size=1 << 20))
if len(sys.argv) == 1:
    for n in z.namelist():
        print(n, z.getinfo(n).file_size)
else:
    out = sys.argv[1]
    for n in z.namelist():
        if any(p in n for p in sys.argv[2:]) and not n.endswith("/"):
            dst = os.path.join(out, n)
            if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(n).file_size: continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(z.read(n)); print("got", n)
