"""EN4CER's blend template: median gain curves per deck and band, in bars from the incoming track's start, plus
timing habits. Learned either from measured transitions in a chosen set of recordings (e.g. the disco practice
for a disco set), excluding transitions labelled as mistakes, or from their knob and fader moves on the DDJ
drills (enaicer.drills; profile "drills").
"""

import json

import numpy as np

from enaicer.catalog import DATA
from enaicer.gains import BANDS, FLOOR_DB

GRID = np.arange(-8, 64.01, 0.25)  # bars from incoming bar 1

AUDIBLE = 8  # template bar where the incoming fader comes up (the drills: 6.5..13, unless EN4CER waited)
# EQ knob position -> dB, from the drills: knob moves against the band levels the recordings show
KNOB_POS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
KNOB_DB = [-26.0, -14.0, -8.5, -6.5, -4.0, 0.0, 6.0]


def knob_db(k: np.ndarray) -> np.ndarray:
    return np.interp(k, KNOB_POS, KNOB_DB)


def fader_db(f: np.ndarray) -> np.ndarray:
    return np.where(f <= 0.02, FLOOR_DB, 20 * np.log10(np.clip(f, 0.02, 1.0)))


def good_drills() -> list[dict]:
    bad = {(x["set"], x["a"], x["b"]) for x in json.loads((DATA / "labels.json").read_text()).get("bad_takes", [])}
    out = []
    for f in sorted((DATA / "drills").glob("*.json")):
        d = json.loads(f.read_text())
        out += [t | {"set": d["id"]} for t in d["transitions"] if (d["id"], t["a"], t["b"]) not in bad]
    return out


def moves_templates(drills: list[dict]) -> tuple[dict, dict, int]:
    """(quick-swap template, gradual template, bar the outgoing is gone) from EN4CER's knob and fader moves: the
    median dB per deck and band, each drill registered so its incoming fader comes up at bar AUDIBLE. The lows
    are drawn explicitly (a median of quick and gradual swaps would be neither), centred on their median swap bar."""
    cols, fad, swaps = {}, {"a": [], "b": []}, []
    for d in drills:
        bars, k = np.array(d["bars"]), d["knobs"]
        shift = d["b_fader_up_bar"] - AUDIBLE
        u = GRID + shift
        for w in "ab":
            f = fader_db(np.array(k[f"{w}_fader"], float))
            fad[w].append(np.interp(u, bars, f))
            for band in BANDS:
                v = np.maximum(knob_db(np.array(k[f"{w}_eq_{band}"], float)) + f, FLOOR_DB)
                cols.setdefault(f"{w}_{band}", []).append(np.interp(u, bars, v))
        low = np.array(k["b_eq_low"], float)
        half = np.flatnonzero((low >= 0.25) & (bars >= d["b_fader_up_bar"]))
        if len(half):
            swaps.append(bars[half[0]] - shift)
    med = {key: np.nanmedian(rows, axis=0) for key, rows in cols.items()}
    fa, fb = np.nanmedian(fad["a"], axis=0), np.nanmedian(fad["b"], axis=0)
    gone = int(np.ceil(GRID[np.flatnonzero((fa <= FLOOR_DB + 1) & (GRID > AUDIBLE))[0]]))
    j = float(np.median(swaps))
    out = []
    for width in (0.25, 8.0):  # quick: within a beat; gradual: over 8 bars
        kb = np.clip((GRID - (j - width / 2)) / width, 0, 1) * 0.5  # incoming low knob 0 -> flat
        t = {"bars": GRID.tolist()} | {key: v.tolist() for key, v in med.items()}
        t["b_low"] = np.maximum(fb + knob_db(kb), FLOOR_DB).tolist()
        t["a_low"] = np.maximum(fa + knob_db(0.5 - kb), FLOOR_DB).tolist()
        t["swap_bar"] = j
        out.append(t)
    return out[0], out[1], gone


def transitions(sets: list[str]) -> list[dict]:
    labels = json.loads((DATA / "labels.json").read_text())
    bad = {(x["set"], x["a"], x["b"]) for x in labels["ran_out_of_time"] + labels.get("bad_takes", [])}
    out = []
    for s in sets:
        for t in json.loads((DATA / "transitions" / f"{s}.json").read_text())["transitions"]:
            if t.get("blend_bars") is not None and (s, t["a"], t["b"]) not in bad and not t.get("too_much_intro"):
                out.append(t | {"set": s})
    return out


def template(ts: list[dict]) -> dict:
    """Median dB curve per deck/band on GRID (bars from the incoming track's first bar), across transitions.

    Each band is registered before averaging: a transition's curves for that band are shifted so its
    crossover (incoming overtakes outgoing) lands on the median crossover for the band. Averaging raw curves
    blurs the staggered swaps (mids, then highs, then lows) into one crossover; registering keeps the timing.
    """
    cross = {
        b: float(np.median([t[f"cross_{b}_bars"] for t in ts if t.get(f"cross_{b}_bars") is not None])) for b in BANDS
    }
    curves = {f"{d}_{b}": [] for d in "ab" for b in BANDS}
    for t in ts:
        bars = np.array(t["curves"]["t"]) * t["bpm_a"] / 240
        for k, acc in curves.items():
            band = k[2:]
            shift = cross[band] - t[f"cross_{band}_bars"] if t.get(f"cross_{band}_bars") is not None else 0.0
            y = np.array(t["curves"][k], float)
            v = np.interp(GRID, bars + shift, y, left=np.nan, right=np.nan)
            if k.startswith("b_"):
                v[GRID < shift] = FLOOR_DB  # incoming isn't playing before its first bar
            if k.startswith("a_"):
                v[GRID > t["a_out_bars"] + shift + 0.5] = FLOOR_DB  # outgoing is gone
            acc.append(v)
    out = {"bars": GRID.tolist(), "n": len(ts), "cross_bars": cross}
    for k, vs in curves.items():
        m = np.nanmedian(np.array(vs), axis=0)
        # beyond the measured windows, hold the deck's settled state
        m = np.where(np.isnan(m), FLOOR_DB if k.startswith("a_") else 0.0, m)
        out[k] = np.round(m, 1).tolist()
    return out


def habits(ts: list[dict]) -> dict:
    med = lambda k: float(np.median([t[k] for t in ts if t.get(k) is not None]))
    return {
        "b_in_bars": med("b_in_bars"),
        "a_out_bars": med("a_out_bars"),
        "blend_bars": med("blend_bars"),
        "n": len(ts),
    }


def from_drills() -> dict:
    """The profile from the drills: EN4CER's knob and fader moves with a gradual bass blend (the planner places
    quick swaps per blend). Every A/B pick (2026-10-20, 7 blends) went to these moves over the template from the
    recorded sets."""
    drills = good_drills()
    _, gradual, gone = moves_templates(drills)
    g = np.array(gradual["bars"])
    over = lambda b: g[np.flatnonzero((g > AUDIBLE) & (np.array(gradual[f"b_{b}"]) >= np.array(gradual[f"a_{b}"])))[0]]
    cross = {b: float(over(b)) for b in BANDS}
    return {"sets": ["drills"], "template": gradual | {"n": len(drills), "cross_bars": cross},
            "habits": {"a_out_bars": float(gone), "n": len(drills)}}  # fmt: skip


def build(sets: list[str]) -> dict:
    if sets == ["drills"]:
        return from_drills()
    ts = transitions(sets)
    return {"sets": sets, "template": template(ts), "habits": habits(ts)}
