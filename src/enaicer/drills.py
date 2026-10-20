"""What EN4CER's hands did in a drill: the DDJ-400 log (enaicer.midilog) lined up with its recording.

Per transition: both decks' channel fader and EQ knobs through the blend, in bars from the incoming track's bar 1
(knobs 0..1, 0.5 = flat), when the incoming deck was started and cued in the headphones, the jog-wheel nudges,
and how far apart the two beat grids sat (ms, + = incoming late) as the blend went on.

New drill recordings are registered and measured first (catalog, alignment, key lock, transitions).

    uv run python -m enaicer.drills [set_id ...]     # default: every drill; writes data/drills/<id>.json
"""

import json
import os
import sys
from bisect import bisect_right

import numpy as np
import soundfile as sf

from enaicer.catalog import DATA, drill_sets
from enaicer.midilog import DIR as MIDI_DIR
from enaicer.midilog import decode
from enaicer.rekordbox import beat_grid, open_copy
from enaicer.transitions import Placement

OUT = DATA / "drills"
KNOBS = ("fader", "eq_high", "eq_mid", "eq_low", "trim")
JOG_GAP = 0.3  # s between jog messages that still count as one nudge
BEFORE, AFTER, STEP = 32, 64, 0.25  # bars around the incoming bar 1, and the sampling step


def events_for(start: float, end: float) -> list[tuple]:
    """Decoded MIDI events of every logger session that overlaps [start, end] (unix), whole sessions so the
    state the DDJ reported when Rekordbox connected is included."""
    out = []
    for f in sorted(MIDI_DIR.glob("*.jsonl")):
        ev = decode(f)
        if ev and ev[0][0] <= end and ev[-1][0] >= start:
            out += ev
    return sorted(out, key=lambda e: e[0])


def starts(events: list[tuple], deck: int) -> list[float]:
    """Unix times the deck started playing: a play press, or the cue press before it if cue was held into play
    (holding CUE plays from the cue point, and pressing PLAY while holding keeps it going)."""
    out, cue_down = [], None
    for t, d, name, v in events:
        if d != deck:
            continue
        if name == "cue":
            cue_down = t if v else None
        elif name == "play" and v:
            out.append(cue_down if cue_down is not None else t)
    return out


def audio_onset(path: str) -> float:
    """Seconds of digital silence at the start of the recording."""
    y, sr = sf.read(path, frames=44100 * 10, dtype="float32", always_2d=True)
    loud = np.flatnonzero(np.abs(y).max(axis=1) > 1e-3)
    return loud[0] / sr if len(loud) else 0.0


def clock(path: str, events: list[tuple]) -> tuple[float, str]:
    """Unix time of recording time 0. The file's birth time is late: Rekordbox keeps ~1 s of audio from before it
    creates the file. When the recording opens on silence, the play press that ended it is exact."""
    birth = os.stat(path).st_birthtime
    onset = audio_onset(path)
    presses = [t for d in (1, 2) for t in starts(events, d) if birth - 5 <= t <= birth + onset + 1]
    if onset > 0.05 and presses:
        return min(presses, key=lambda t: abs(t - birth - onset + 1)) - onset, "first play press"
    return birth - 1.0, "file birth time - 1 s"


def series(events: list[tuple], deck: int, name: str, times: np.ndarray) -> np.ndarray:
    """A control's value at each unix time (the last value sent before it; NaN before the first)."""
    ev = [(t, v) for t, d, n, v in events if d == deck and n == name]
    ts = [t for t, _ in ev]
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        k = bisect_right(ts, t)
        if k:
            out[i] = ev[k - 1][1]
    return out


def jog_bursts(events: list[tuple], deck: int, t0: float, t1: float) -> list[dict]:
    """Jog-wheel nudges on a deck in [t0, t1]: messages less than JOG_GAP apart make one nudge."""
    out = []
    for t, d, n, v in events:
        if d != deck or not n.startswith("jog") or n == "jog_touch" or not t0 <= t <= t1:
            continue
        if out and t - out[-1]["t1"] < JOG_GAP:
            b = out[-1]
            b["t1"], b["ticks"] = t, b["ticks"] + int(v)
            b["kinds"].add(n)
        else:
            out.append({"t0": t, "t1": t, "ticks": int(v), "kinds": {n}})
    return out


def grid_phase(beats: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """Fractional beat position of track times on the track's beat grid."""
    k = np.clip(np.searchsorted(beats, tt) - 1, 0, len(beats) - 2)
    return k + (tt - beats[k]) / (beats[k + 1] - beats[k])


def grid_offset(A: Placement, B: Placement, ga: np.ndarray, gb: np.ndarray, rec: np.ndarray, beat_s: float):
    """How far the incoming track's beat grid sits behind the outgoing one's in the recording (ms, + = late)."""
    d = grid_phase(ga, A.track(rec)) - grid_phase(gb, B.track(rec))
    return ((d + 0.5) % 1 - 0.5) * beat_s * 1000


def cue_point(B: Placement, grid: tuple, start_rec: float) -> tuple[int, float]:
    """(bar of the incoming track it was started from, 1-based; recording time of that bar's downbeat).
    The cue sits on a downbeat when EN4CER cues on a phrase; the nearest one is taken."""
    times, beats = grid[0], grid[1]
    down = times[beats == 1]
    k = int(np.argmin(np.abs(down - float(B.track(start_rec)))))
    return k + 1, float(B.rec(down[k]))


def analyze(set_id: str) -> dict:
    cat = {s["id"]: s for s in json.loads((DATA / "sets.json").read_text())["sets"]}
    s = cat[set_id]
    al = {t["n"]: t for t in json.loads((DATA / "align" / f"{set_id}.json").read_text())["tracks"]}
    trans = json.loads((DATA / "transitions" / f"{set_id}.json").read_text())["transitions"]
    birth = os.stat(s["file"]).st_birthtime
    events = events_for(birth - 3600, birth + s["duration"] + 60)
    if not events:
        sys.exit(f"{set_id}: no DDJ log covers the recording")
    t0, how = clock(s["file"], events)
    db = open_copy()
    grids = {t["n"]: beat_grid(db.get_content(ID=t["content_id"])) for t in s["tracks"]}

    # which deck played which track: decks alternate (a blend needs both), starting on whichever deck fits best,
    # i.e. had its fader up while the track was clearly heard (the other deck's fader is up only during blends).
    # Then when it was started: the last start on that deck before the middle of its matched audio.
    found = sorted((n for n, t in al.items() if t["found"]), key=lambda n: np.median([x[0] for x in al[n]["anchors"]]))
    heard = {n: np.array([x[0] for x in al[n]["anchors"] if x[2] >= 0.5]) for n in found}
    up = {(n, d): float(np.nanmean(series(events, d, "fader", t0 + heard[n]))) for n in found for d in (1, 2)}
    first = max((1, 2), key=lambda d0: sum(up[n, d0 if i % 2 == 0 else 3 - d0] for i, n in enumerate(found)))
    decks = {n: first if i % 2 == 0 else 3 - first for i, n in enumerate(found)}
    began = {}
    for n in found:
        cands = [x for x in starts(events, decks[n]) if x <= t0 + float(np.median(heard[n]))]
        began[n] = cands[-1] - t0 if cands else None

    out = {"id": set_id, "clock": how, "rec0_unix": round(t0, 3), "decks": decks, "transitions": []}
    for x in trans:
        out["transitions"].append(transition(x, al, grids, events, t0, decks, began))
    OUT.mkdir(exist_ok=True)
    (OUT / f"{set_id}.json").write_text(json.dumps(out, indent=1))
    return out


def transition(x: dict, al: dict, grids: dict, events: list, t0: float, decks: dict, began: dict) -> dict:
    """One blend, in bars from the incoming track's cue point (where EN4CER started it from)."""
    a, b = x["a_n"], x["b_n"]
    da, db_ = decks[a], decks[b]
    A, B = Placement(al[a]), Placement(al[b])
    bar = 240 / x["bpm_b"]
    if began[b] is not None:
        cue_bar, zero = cue_point(B, grids[b], began[b])
    else:
        cue_bar, zero = x["b_cue_bar"], x["b_bar1_rec"]
    bars = np.arange(-BEFORE, AFTER + 1e-9, STEP)
    rec = zero + bars * bar
    knobs = {f"{w}_{k}": series(events, d, k, t0 + rec) for w, d in (("a", da), ("b", db_)) for k in KNOBS}
    lo, hi = t0 + rec[0], t0 + rec[-1]

    def to_bar(u: float) -> float:
        return round(float((u - t0 - zero) / bar), 2)

    cue = {
        w: [(to_bar(t), v) for t, d, n, v in events if d == dk and n == "headphone_cue" and lo <= t <= hi]
        for w, dk in (("a", da), ("b", db_))
    }
    jogs = {
        w: [
            {"bar": to_bar(j["t0"]), "s": round(j["t1"] - j["t0"], 2), "ticks": j["ticks"], "kinds": sorted(j["kinds"])}
            for j in jog_bursts(events, dk, lo, hi)
        ]
        for w, dk in (("a", da), ("b", db_))
    }
    ob = np.arange(0, AFTER + 1e-9, 1.0)
    orec = zero + ob * bar
    both = (orec >= max(A.first, B.first)) & (orec <= min(A.last, B.last))
    off = grid_offset(A, B, grids[a][0], grids[b][0], orec, 60 / x["bpm_b"])
    bf, af = knobs["b_fader"], knobs["a_fader"]
    up = np.flatnonzero((bf > 0.05) & (bars >= 0))
    full = np.flatnonzero((bf >= 0.95 * np.nanmax(bf)) & (bars >= (bars[up[0]] if len(up) else 0)))
    down = np.flatnonzero((af > 0.05) & (bars >= 0))
    return {
        "a": x["a"], "b": x["b"], "deck_a": da, "deck_b": db_, "bar_s": round(bar, 4), "bpm": x["bpm_b"],
        "b_cue_bar": cue_bar,
        "cue_rec": round(zero, 3),
        "a_bar_at_cue": round(float(grid_phase(grids[a][0][grids[a][1] == 1], A.track(zero))) + 1, 2),
        "b_started_bar": None if began[b] is None else to_bar(began[b] + t0),
        "b_fader_up_bar": float(bars[up[0]]) if len(up) else None,
        "b_fader_full_bar": float(bars[full[0]]) if len(full) else None,
        "a_fader_gone_bar": float(bars[down[-1]] + STEP) if len(down) and down[-1] < len(bars) - 1 else None,
        "b_entry_eq": {k: round(float(knobs[f"b_eq_{k}"][up[0]]), 2) for k in ("high", "mid", "low")} if len(up) else None,
        "headphone_cue": cue,
        "jog": jogs,
        "grid_offset_ms": [[float(o), round(float(v), 1)] for o, v, k in zip(ob, off, both) if k],
        "bars": bars.tolist(),
        "knobs": {k: np.round(v, 3).tolist() for k, v in knobs.items()},
    }  # fmt: skip


def _row(k: dict, w: str, i: int) -> str:
    vals = (k[f"{w}_{c}"][i] for c in ("fader", "eq_high", "eq_mid", "eq_low"))
    return " ".join("  - " if np.isnan(v) else f"{v:4.2f}" for v in vals)


def report(d: dict) -> None:
    print(f"== {d['id']}  (clock: {d['clock']})")
    for t in d["transitions"]:
        print(f"\n  {t['a'][:40]} (deck {t['deck_a']}) -> {t['b'][:40]} (deck {t['deck_b']})  {t['bpm']:.2f} BPM")
        print(
            f"  incoming cued at its bar {t['b_cue_bar']}, on outgoing bar {t['a_bar_at_cue']}; started at bar"
            f" {t['b_started_bar']}  fader up {t['b_fader_up_bar']}  full {t['b_fader_full_bar']}"
            f"  outgoing fader gone {t['a_fader_gone_bar']}"
        )
        jb = t["jog"]["b"]
        hp = t["headphone_cue"]
        print(
            f"  entry EQ {t['b_entry_eq']}  jog on incoming: {len(jb)} nudges, {sum(abs(j['ticks']) for j in jb)} ticks"
            f" ({sum(j['bar'] >= (t['b_fader_up_bar'] or 0) for j in jb)} after fader up)"
            f"  headphone cue on: in {[c[0] for c in hp['b'] if c[1]]} out {[c[0] for c in hp['a'] if c[1]]}"
        )
        off = dict(map(tuple, t["grid_offset_ms"]))
        bars = np.array(t["bars"])
        print("     bar | in: fader  hi  mid  low | out: fader  hi  mid  low | grids ms")
        for b in range(-4, 65, 4):
            i = int(np.argmin(np.abs(bars - b)))
            o = off.get(float(b))
            print(
                f"    {b:4d} | {_row(t['knobs'], 'b', i)}  | {_row(t['knobs'], 'a', i)}  | {'' if o is None else f'{o:+6.1f}'}"
            )


def prepare(ids: list[str]) -> None:
    """Register and measure the drills that haven't been yet."""
    from enaicer import align, catalog, keylock, transitions

    known = {s["id"] for s in json.loads((DATA / "sets.json").read_text())["sets"]}
    if set(ids) - known:
        catalog.main()
    for sid in ids:
        if not (DATA / "transitions" / f"{sid}.json").exists():
            align.run(sid, True)
            keylock.annotate(sid, keylock.check(sid))
            (DATA / "transitions").mkdir(exist_ok=True)
            print(transitions.run(sid))


def main() -> None:
    ids = sys.argv[1:] or [s["id"] for s in drill_sets()]
    prepare(ids)
    for sid in ids:
        report(analyze(sid))


if __name__ == "__main__":
    main()
