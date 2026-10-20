"""Blind A/B clips of the blends EN4CER drilled on the DDJ, to choose between model settings by ear.

For each good drill blend: EN4CER's own take, cut from the drill recording, as a reference, and three model
renders of the same two tracks. They enter at the same musical point and differ only in how the blend is done:

- smooth: the template averaged from the recorded sets (what the v5 render did), with v6's beat alignment
- moves_quick: EN4CER's knob and fader moves from the drills (enaicer.drills), bass swapped within a beat
- moves_gradual: the same moves, bass blended over 8 bars

The page shows the three as A/B/C in a fixed random order per pair; picks save to out/disco/feedback/ab_drills.json
(ab_drills_<round>.json for later rounds).

    uv run python -m enaicer.abtest     # renders into out/disco/ab/, then open http://127.0.0.1:8767/ab/
    uv run python -m enaicer.abtest --round 2 --pairs p2 p5   # a new round of just those pairs, fresh picks
"""

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from enaicer import render
from enaicer.catalog import DATA, ROOT
from enaicer.plan import load_track
from enaicer.profile import AUDIBLE, good_drills, moves_templates
from enaicer.rekordbox import open_copy

OUT = ROOT / "out" / "disco" / "ab"
PAGE = Path(__file__).with_name("ab_page.html")
LEAD, TAIL = 8, 8  # bars of the outgoing alone before the blend, and of the incoming alone after it
LUFS = -14.0  # every clip, so louder doesn't sound better
VARIANTS = ("smooth", "moves_quick", "moves_gradual")


def pair_plan(d: dict, A: dict, B: dict, template: dict, span: int, a_out: int, shift: int, b_len: int) -> dict:
    """Two-track plan: B cued `shift` bars after EN4CER's cue, so the model's fade-in starts where theirs did, and
    played for `b_len` bars (the same for every version, so clip length doesn't give one away)."""
    a_bar = round(d["a_bar_at_cue"] - 1) + shift
    b_bar = d["b_cue_bar"] - 1 + shift
    A = A | {"first_bar": a_bar - LEAD, "last_bar": min(a_bar + span + 1, len(A["down"]) - 1)}
    B = B | {"first_bar": b_bar, "last_bar": min(b_bar + b_len, len(B["down"]) - 1)}
    x = {"i": 0, "a": A["title"], "b": B["title"], "a_bar": a_bar, "b_bar": b_bar, "span_bars": span,
         "tempo": d["bpm"], "bass_swap_at": None, "fade_highs_early": False}  # fmt: skip
    return {"profile": {"template": template, "habits": {"a_out_bars": a_out}}, "tracks": [A, B], "transitions": [x]}


def user_clip(d: dict, rec_path: str, shift: int, span: int, out: Path) -> None:
    """EN4CER's take of the same stretch: from LEAD bars before the model's cue to TAIL bars after the blend."""
    bar = d["bar_s"]
    t0 = d["cue_rec"] + (shift - LEAD) * bar
    t1 = d["cue_rec"] + (shift + span + TAIL) * bar
    info = sf.info(rec_path)
    a, b = max(0, int(t0 * info.samplerate)), min(info.frames, int(t1 * info.samplerate))
    x, sr = sf.read(rec_path, start=a, stop=b, dtype="float32", always_2d=True)
    if sr != render.SR:
        import soxr

        x = soxr.resample(x, sr, render.SR).astype(np.float32)
    x *= 10 ** ((LUFS - render.loudness(x)) / 20)
    render.limiter(x)
    wav = out.with_suffix(".wav")
    sf.write(wav, x, render.SR, subtype="PCM_24")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "256k", str(out)], check=True)
    wav.unlink()


def build(round_: int = 1, only: list[str] | None = None) -> Path:
    drills = good_drills()
    quick, gradual, gone = moves_templates(drills)
    smooth_plan = json.loads((ROOT / "out" / "disco" / "plan_full_v5.json").read_text())["profile"]
    smooth_out = round(smooth_plan["habits"]["a_out_bars"])
    sets = {s["id"]: s for s in json.loads((DATA / "sets.json").read_text())["sets"]}
    db = open_copy()
    by_title = {}
    for s in sets.values():
        for t in s["tracks"]:
            by_title.setdefault(t["title"], t["content_id"])
    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, d in enumerate(drills):
        A = load_track(db.get_content(ID=by_title[d["a"]]))
        B = load_track(db.get_content(ID=by_title[d["b"]]))
        shift = max(0, 4 * round((d["b_fader_up_bar"] - AUDIBLE) / 4))  # where EN4CER waited, the model waits too
        pid = f"p{i + 1}"
        if only and pid not in only:
            continue
        versions, b_len = {}, max(gone, smooth_out) + TAIL
        for v in VARIANTS:
            versions[v] = f"clips/{pid}_{v}.mp3"
            if (OUT / versions[v]).exists():  # delete a clip to render it again
                continue
            if v == "smooth":
                plan = pair_plan(d, A, B, smooth_plan["template"], smooth_out, smooth_out, shift, b_len)
            else:
                plan = pair_plan(d, A, B, quick if v == "moves_quick" else gradual, gone, gone, shift, b_len)
            tmp = OUT / "render" / f"{pid}_{v}"
            render.render(plan, tmp, LUFS)
            shutil.copy(tmp / "mix.mp3", OUT / versions[v])
            print(f"{pid} {v}: {d['a'][:28]} -> {d['b'][:28]}", flush=True)
        user_clip(d, sets[d["set"]]["file"], shift, max(gone, smooth_out), clips / f"{pid}_you.mp3")
        order = list(VARIANTS)
        random.Random(i if round_ == 1 else f"{round_}:{i}").shuffle(order)
        pair = {"id": pid, "a": d["a"], "b": d["b"], "set": d["set"], "you": f"clips/{pid}_you.mp3",
                "versions": [{"letter": "ABC"[k], "variant": v, "audio": versions[v]} for k, v in enumerate(order)],
                "bar_s": d["bar_s"], "lead_s": LEAD * d["bar_s"]}  # fmt: skip
        pairs.append(pair)
    data = {"id": "ab_drills" if round_ == 1 else f"ab_drills_{round_}", "round": round_, "pairs": pairs,
            "swap_bar": quick["swap_bar"], "gone_bar": gone}  # fmt: skip
    (OUT / "ab_data.js").write_text("window.AB = " + json.dumps(data) + ";\n")
    shutil.copy(PAGE, OUT / "index.html")
    shutil.rmtree(OUT / "render", ignore_errors=True)
    return OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--pairs", nargs="*", help="pair ids (p1..p8) to include; default all")
    a = ap.parse_args()
    print(f"built {build(a.round, a.pairs)}/index.html -> http://127.0.0.1:8767/ab/")


if __name__ == "__main__":
    main()
