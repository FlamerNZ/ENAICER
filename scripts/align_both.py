import json, subprocess, numpy as np, librosa, os
SR=11025; HOP=256; FPS=SR/HOP
SP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
def load(p): return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True).stdout,dtype=np.float32)
def mel(y): return librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=HOP,n_mels=40,fmin=40,fmax=5500)+1e-10)
flux=lambda M: np.diff(M,axis=1,prepend=M[:,:1]).clip(min=0)
def stretch(M,rate):
    n=M.shape[1]; m=int(n/rate); x=np.arange(m)*rate
    return np.stack([np.interp(x,np.arange(n),b) for b in M])
def ncc(R,S):
    T=R.shape[1]; L=S.shape[1]; n=1<<int(np.ceil(np.log2(T+L)))
    Sz=S-S.mean(axis=1,keepdims=True); sd=Sz.std(axis=1,keepdims=True); ok=sd[:,0]>1e-6; Sz=Sz[ok]/sd[ok]; R=R[ok]
    cs=np.cumsum(np.pad(R,((0,0),(1,0))),axis=1); cs2=np.cumsum(np.pad(R**2,((0,0),(1,0))),axis=1)
    num=np.fft.irfft(np.fft.rfft(R,n=n,axis=1)*np.fft.rfft(Sz[:,::-1],n=n,axis=1),n=n,axis=1)[:,L-1:T]
    m=(cs[:,L:]-cs[:,:-L])/L; v=(cs2[:,L:]-cs2[:,:-L])/L-m**2
    return (num/(L*np.sqrt(np.maximum(v,1e-9)))).mean(axis=0)
def pearson(A,B):
    A=A-A.mean(axis=1,keepdims=True); B=B-B.mean(axis=1,keepdims=True)
    d=np.sqrt((A**2).sum(axis=1)*(B**2).sum(axis=1)); ok=d>1e-9
    return float(((A*B).sum(axis=1)[ok]/d[ok]).mean()) if ok.any() else 0.0
pl=json.load(open(SP+"/pl_tracks.json"))["Dub"]; bpm=json.load(open(SP+"/bpm.json"))
cache={}
def track_feats(path,b):
    if path not in cache:
        M=mel(load(path)); rate=140.0/b; cache[path]=(flux(stretch(M,rate)),rate,M.shape[1]/FPS)
    return cache[path]
def run(label,rec,items):
    R=flux(mel(load(rec))); T=R.shape[1]; res=[]
    for it in items:
        F,rate,dur=track_feats(it["path"],it["bpm"]); N=F.shape[1]; L=int(8*FPS)
        a0=max(0,int(it["a"]*FPS)); b0=min(T,int(it["b"]*FPS)); W=R[:,a0:b0]
        offs=[]
        for s in range(int(2*FPS),N-L,int(4*FPS)):
            if W.shape[1]<=L: break
            c=ncc(W,F[:,s:s+L]); k=int(np.argmax(c))
            if c[k]>0.3: offs.append((a0+k-s)/FPS)
        if not offs: res.append(dict(**it,offset=None)); print(f"[{label}] {it['name']:15} NOT FOUND",flush=True); continue
        h=np.round(np.array(offs)*4)/4; vals,cnt=np.unique(h,return_counts=True); off=float(vals[np.argmax(cnt)])
        # presence curve every 0.5 s with 4 s windows
        pres=[]; w=int(2*FPS)
        for tau in np.arange(max(0,off), min(T/FPS, off+N/FPS), 0.5):
            i=int(tau*FPS); j=int((tau-off)*FPS)
            if i-w<0 or i+w>T or j-w<0 or j+w>N: continue
            pres.append((float(tau),pearson(R[:,i-w:i+w],F[:,j-w:j+w])))
        res.append(dict(**it,offset=off,rate=rate,dur=dur,n_votes=int(cnt.max()),n_snip=len(offs),presence=pres))
        p=np.array([v for _,v in pres]); tt=np.array([t for t,_ in pres]); on=tt[p>0.3]
        print(f"[{label}] {it['name']:15} offset {off:8.2f}s rate {rate:.4f} votes {cnt.max()}/{len(offs)} | audible {on.min()/60 if len(on) else -1:.2f}–{on.max()/60 if len(on) else -1:.2f} min",flush=True)
    json.dump(res,open(f"{SP}/align_{label}.json","w"))
    return res
find=lambda name: next(x for x in pl if x["title"].startswith(name) or name in x["title"])
# LIVE: order from XDJ history; windows from earlier pass
live_w=[("Energy",-60,5),("They Follow",0,5),("Don't Stop",2,7.5),("I Run",5,10),("Swing King",7,12),("Push",10,14),("I Miss U",12,17),("Hush",14,18),("Patterns",15.5,19.5),("dontwaiT",17,22),("GITP",20,24),("THE FIT",21.5,27.5),("i wanna dance",24,29),("ROUND N' ROUND",26,30),("Leave",28,33),("TAKA",31,35.5),("Easy for them",33,37),("Control",35,38.5),("LUNCH",36.5,40),("Do My Ting",38,42),("Espresso",40.5,44),("Criminal",42.5,46.5),("Skanka",45,49.5),("Badman Season",47.5,51.5),("Drop In",49.5,53.3)]
items=[]
for n,a,b in live_w:
    r=find(n); items.append(dict(name=n,title=r["title"],artist=r["artist"],path=r["path"],bpm=bpm[r["id"]],a=max(0,a*60),b=b*60))
LIVE=os.path.expanduser("~/Music/rekordbox/Recording/EN4CER/Dub Live/Dub Live 2026-10-17 (XDJ-AZ REC001).wav")
run("live",LIVE,items)
# PRACTICE: order from Rekordbox history; windows around load time
ph=json.load(open(SP+"/practice_hist.json")); items=[]
for h in ph:
    if h["dup"]: continue
    items.append(dict(name=h["title"].split(" (")[0][:15],title=h["title"],artist=h["artist"],path=h["path"],bpm=h["bpm"],a=max(0,h["t_load"]-240),b=h["t_load"]+360))
PRAC=os.path.expanduser("~/Music/rekordbox/Recording/EN4CER/Dub(2)/01 Dub Practice 2.wav")
run("practice",PRAC,items)
