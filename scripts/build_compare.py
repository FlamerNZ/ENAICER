import json, subprocess, os, numpy as np, soundfile as sf
SP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","out","compare"); os.makedirs(OUT+"/clips",exist_ok=True)  # was the Dub Edit folder; this project must not write there
REC={"live":os.path.expanduser("~/Music/rekordbox/Recording/EN4CER/Dub Live/Dub Live 2026-10-17 (XDJ-AZ REC001).wav"),
     "practice":os.path.expanduser("~/Music/rekordbox/Recording/EN4CER/Dub(2)/01 Dub Practice 2.wav")}
GAIN={"live":0.0,"practice":-0.5}   # practice is 0.5 LU louder overall → match for fair A/B
D={l:json.load(open(f"{SP}/trans_{l}.json")) for l in REC}
TR={l:{t["title"]:t for t in D[l]["tracks"]} for l in REC}
short=lambda t: t.split(" (")[0].replace(" feat. Lexi Shanley","")
def tr_info(l,a,b):
    t=next((x for x in D[l]["trans"] if x["a_title"]==a and x["b_title"]==b),None)
    if not t or t["b_in"] is None or t["a_out"] is None: return None
    A=TR[l][a]; B=TR[l][b]; b_in=t["b_in"]; a_out=max(t["a_out"],b_in+2)
    a_out=min(a_out,b_in+75)  # cap implausible overlaps
    return dict(rec=l,b_in=b_in,a_out=a_out,dur=a_out-b_in,
                a_tt=(b_in-A["offset"])*A["rate"], b_tt=(a_out-B["offset"])*B["rate"], rate_a=A["rate"], rate_b=B["rate"])
info=sf.info(REC["live"]); SR=info.samplerate
def stats(l,s,e):
    x,_=sf.read(REC[l],start=int(s*SR),stop=int(e*SR),dtype="int16")
    a=np.abs(x.astype(np.int32)).max(axis=1); full=(a>=32767).astype(np.int8)
    runs=np.diff(np.concatenate([[0],full,[0]])); st=np.where(runs==1)[0]; en=np.where(runs==-1)[0]
    clips=int(sum(1 for p,q in zip(st,en) if q-p>=3))
    return clips
def clip(l,b_in,a_out,key):
    s=max(0,b_in-20); e=a_out+15; f=f"clips/{key}_{l}.mp3"
    if not os.path.exists(f"{OUT}/{f}"):
        subprocess.run(["ffmpeg","-v","error","-y","-ss",f"{s:.3f}","-t",f"{e-s:.3f}","-i",REC[l],"-af",f"volume={GAIN[l]}dB,afade=t=in:d=0.3,afade=t=out:st={e-s-0.6:.3f}:d=0.6","-c:a","libmp3lame","-b:a","192k",f"{OUT}/{f}"],check=True)
    def lvl(a,b):
        x,_=sf.read(REC[l],start=int(a*SR),stop=int(b*SR),dtype="float32"); return 20*np.log10(np.sqrt((x**2).mean())+1e-9)
    bump=lvl(b_in,a_out)-(lvl(max(0,b_in-15),b_in)+lvl(a_out,a_out+15))/2
    fm=lambda t: f"{int(t//60)}:{int(t%60):02d}"
    return dict(file=f,start=s,label=f"{fm(b_in)}–{fm(a_out)}",clips=stats(l,s,e),bump=round(float(bump),1))
order_p=[t["title"] for t in D["practice"]["tracks"]]; order_l=[t["title"] for t in D["live"]["tracks"]]
T=lambda n: next(t for t in order_p+order_l if t.startswith(n))
def step_pair(a,b):
    a,b=T(a),T(b); key=f"{short(a)}__{short(b)}".replace(" ","_").replace("'","")
    opts={}
    for l in ("practice","live"):
        i=tr_info(l,a,b)
        if i: i.update(clip(l,i["b_in"],i["a_out"],key)); opts[l]=i
    return dict(kind="pair",a=a,b=b,title=f"{short(a)} → {short(b)}",opts=opts)
def route_tr(l,a,b):
    a,b=T(a),T(b); key=f"{short(a)}__{short(b)}".replace(" ","_").replace("'",""); i=tr_info(l,a,b); i.update(clip(l,i["b_in"],i["a_out"],key))
    return dict(a=a,b=b,title=f"{short(a)} → {short(b)}",**i)
steps=[]
steps.append(dict(kind="fixed",title="Opening (practice only — live recording starts mid-Energy)",trs=[route_tr("practice","PFTD","Give It To Me"),route_tr("practice","Give It To Me","They Follow")]))
for a,b in [("They Follow","Don't Stop"),("Don't Stop","I Run"),("I Run","Swing King"),("Swing King","Push"),("Push","I Miss U"),("I Miss U","Hush")]: steps.append(step_pair(a,b))
steps.append(dict(kind="route",id="hush",title="After Hush",routes=[
  dict(id="practice",name="Practice: Hush → dontwaiT",trs=[route_tr("practice","Hush","dontwaiT")]),
  dict(id="live",name="Live: Hush → Patterns → dontwaiT",trs=[route_tr("live","Hush","Patterns"),route_tr("live","Patterns","dontwaiT")])]))
for a,b in [("dontwaiT","GITP"),("GITP","THE FIT"),("THE FIT","i wanna dance"),("i wanna dance","ROUND N"),("ROUND N","Leave")]: steps.append(step_pair(a,b))
steps.append(dict(kind="route",id="leave",title="After Leave",routes=[
  dict(id="practice",name="Practice: Leave → THINGS SHE SAID → TAKA",trs=[route_tr("practice","Leave","THINGS SHE SAID"),route_tr("practice","THINGS SHE SAID","TAKA")]),
  dict(id="live",name="Live: Leave → TAKA",trs=[route_tr("live","Leave","TAKA")])]))
for a,b in [("TAKA","Easy for them"),("Easy for them","Control"),("Control","LUNCH"),("LUNCH","Do My Ting"),("Do My Ting","Espresso"),("Espresso","Criminal")]: steps.append(step_pair(a,b))
steps.append(dict(kind="route",id="ending",title="Ending (after Criminal)",routes=[
  dict(id="practice",name="Practice: We're Not Alone → Innocence → I Feel Like → Badman Season → Skanka → Drop In",trs=[route_tr("practice","Criminal","We're Not Alone"),route_tr("practice","We're Not Alone","Innocence"),route_tr("practice","Innocence","I Feel Like"),route_tr("practice","I Feel Like","Badman"),route_tr("practice","Badman","Skanka"),route_tr("practice","Skanka","Drop In")]),
  dict(id="live",name="Live: Skanka → Badman Season → Drop In",trs=[route_tr("live","Criminal","Skanka"),route_tr("live","Skanka","Badman"),route_tr("live","Badman","Drop In")]),
  dict(id="hybrid",name="Hybrid: practice through I Feel Like → Badman Season, then live Badman Season → Drop In (no Skanka)",trs=[route_tr("practice","Criminal","We're Not Alone"),route_tr("practice","We're Not Alone","Innocence"),route_tr("practice","Innocence","I Feel Like"),route_tr("practice","I Feel Like","Badman"),route_tr("live","Badman","Drop In")])]))
# edges: first track from practice rec start; last track to end of Drop In in the chosen recording
P0=TR["practice"][T("PFTD")]
edges=dict(start_tt=(0-P0["offset"])*P0["rate"],
           end_tt={l:(min(TR[l][T("Drop In")]["rec_end"], sf.info(REC[l]).duration)-TR[l][T("Drop In")]["offset"])*TR[l][T("Drop In")]["rate"] for l in REC})
rates={t["title"]:t["rate"] for l in REC for t in D[l]["tracks"]}
json.dump(dict(steps=steps,edges=edges,rates=rates),open(OUT+"/compare_data.json","w"),indent=1)
open(OUT+"/compare_data.js","w").write("window.CMP="+json.dumps(dict(steps=steps,edges=edges,rates=rates))+";")
n=sum(len(s.get("opts",{})) for s in steps)+sum(len(r["trs"]) for s in steps if s["kind"]=="route" for r in s["routes"])+2
print("steps",len(steps),"| clips",len(os.listdir(OUT+"/clips")))
for s in steps:
    if s["kind"]=="pair": print(f"  {s['title']:34} " + " | ".join(f"{l}: {o['label']} ({o['dur']:.0f}s, bump {o['bump']:+.1f} dB, clip runs {o['clips']})" for l,o in s["opts"].items()))
