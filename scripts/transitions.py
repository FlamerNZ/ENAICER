import json, subprocess, numpy as np, librosa, os
SR=11025; HOP=256; FPS=SR/HOP
SP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
def load(p): return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True).stdout,dtype=np.float32)
def mel(y): return librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=HOP,n_mels=40,fmin=40,fmax=5500)+1e-10)
flux=lambda M: np.diff(M,axis=1,prepend=M[:,:1]).clip(min=0)
def stretch(M,rate):
    n=M.shape[1]; m=int(n/rate); x=np.arange(m)*rate
    return np.stack([np.interp(x,np.arange(n),b) for b in M])
def pearson(A,B):
    A=A-A.mean(axis=1,keepdims=True); B=B-B.mean(axis=1,keepdims=True)
    d=np.sqrt((A**2).sum(axis=1)*(B**2).sum(axis=1)); ok=d>1e-9
    return float(((A*B).sum(axis=1)[ok]/d[ok]).mean()) if ok.any() else 0.0
recs={"live":os.path.expanduser("~/Music/rekordbox/Recording/EN4CER/Dub Live/Dub Live 2026-10-17 (XDJ-AZ REC001).wav"),
      "practice":os.path.expanduser("~/Music/rekordbox/Recording/EN4CER/Dub(2)/01 Dub Practice 2.wav")}
cache={}
for label,rec in recs.items():
    al=json.load(open(f"{SP}/align_{label}.json"))
    R=flux(mel(load(rec))); T=R.shape[1]; w=int(4*FPS)
    for it in al:
        if it["path"] not in cache: cache[it["path"]]=mel(load(it["path"]))
        F=flux(stretch(cache[it["path"]],it["rate"])); N=F.shape[1]; off=it["offset"]
        pres=[]
        for tau in np.arange(max(4,off+4.5), min(T/FPS-4, off+N/FPS-4.5), 1.0):
            i=int(tau*FPS); j=int((tau-off)*FPS)
            best=-1
            for dl in range(-13,14):
                jj=j-dl
                if jj-w<0 or jj+w>N: continue
                best=max(best,pearson(R[:,i-w:i+w],F[:,jj-w:jj+w]))
            pres.append((float(tau),best))
        it["presence"]=pres; it["rec_end"]=off+N/FPS
    # transitions
    TH=0.35; trans=[]
    def first_on(p,after=-1e9):
        for k in range(len(p)-2):
            if p[k][0]>=after and p[k][1]>TH and p[k+1][1]>TH and p[k+2][1]>TH: return p[k][0]-4
    def last_on(p):
        for k in range(len(p)-1,1,-1):
            if p[k][1]>TH and p[k-1][1]>TH: return p[k][0]+4
    for A,B in zip(al,al[1:]):
        b_in=first_on(B["presence"]); a_out=last_on(A["presence"])
        trans.append(dict(a=A["name"],b=B["name"],a_title=A["title"],b_title=B["title"],b_in=b_in,a_out=a_out))
    for it in al:
        p=np.array([v for _,v in it["presence"]]); it["on_s"]=first_on(it["presence"]); it["off_s"]=last_on(it["presence"]); it["peak"]=float(p.max()) if len(p) else 0
    json.dump(dict(tracks=al,trans=trans),open(f"{SP}/trans_{label}.json","w"))
    print(f"\n==== {label}")
    fm=lambda s: "  -  " if s is None else f"{int(s//60):2d}:{int(s%60):02d}"
    for t in trans:
        d=None if (t['b_in'] is None or t['a_out'] is None) else t['a_out']-t['b_in']
        print(f"{t['a'][:15]:15} → {t['b'][:15]:15} | B audible {fm(t['b_in'])} | A gone {fm(t['a_out'])} | overlap {'' if d is None else f'{d:5.0f}s'}")
