"""Transition timing for each pair of consecutive tracks in a set.

- Where the incoming track's bar 1 lands in the outgoing track: bar, phrase position, and nearest drop.
- The blend itself (enaicer.blend): when each deck came in and left, the order the bands swapped in, and so on.
- Beatmatch: the lag between the two tracks' hi-hat onsets, and separately their kick onsets, during
  the overlap (EN4CER matches on the hats).

    uv run python -m enaicer.transitions [set_id ...]
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from enaicer import blend
from enaicer.align import heard_from
from enaicer.catalog import DATA
from enaicer.rekordbox import beat_grid, open_copy
from enaicer.structure import FPS, bar_db, bass_contrast, breaks, drops, features


class Placement:
    """One track as placed in the recording: track time <-> recording time, including measured drift."""

    def __init__(self, a: dict):
        self.offset, self.rate = a["offset"], a["rate"]
        an = np.array([x for x in a["anchors"] if x[2] >= 0.5])
        drift = an[:, 0] - (self.offset + an[:, 1] / self.rate)
        k = 2  # running median over 5 anchors (~10 s) smooths out single bad anchors
        self.knots = an[:, 1]
        self.drift = np.array([np.median(drift[max(0, i - k) : i + k + 1]) for i in range(len(drift))])
        self.first, self.last = float(an[0, 0]), float(an[-1, 0])  # audible span (roughly)

    def rec(self, tt):
        return self.offset + np.asarray(tt) / self.rate + np.interp(tt, self.knots, self.drift)

    def track(self, rec):
        tt = (np.asarray(rec) - self.offset) * self.rate
        return (np.asarray(rec) - self.offset - np.interp(tt, self.knots, self.drift)) * self.rate


def onset_lag(A: Placement, B: Placement, fa: dict, fb: dict, band: str, t0: float, t1: float, win: float = 8.0):
    """Lag (ms, + = incoming late) between the two tracks' `band` onsets in the recording, per `win` s window."""
    lags = []
    step = 1 / FPS
    max_lag = int(0.06 * FPS)  # +-60 ms: well inside half a 16th note at these tempos
    for w0 in np.arange(t0, t1 - win + 1e-9, win):
        tau = np.arange(w0, w0 + win, step)
        ea = np.interp(A.track(tau) * FPS, np.arange(len(fa[f"{band}_onset"])), fa[f"{band}_onset"])
        eb = np.interp(B.track(tau) * FPS, np.arange(len(fb[f"{band}_onset"])), fb[f"{band}_onset"])
        ea, eb = ea - ea.mean(), eb - eb.mean()
        if ea.std() < 1e-6 or eb.std() < 1e-6:
            continue
        c = np.array(
            [
                np.dot(ea[max(0, -k) : len(ea) - max(0, k)], eb[max(0, k) : len(eb) - max(0, -k)])
                for k in range(-max_lag, max_lag + 1)
            ]
        )
        c /= np.linalg.norm(ea) * np.linalg.norm(eb)
        i = int(np.argmax(c))
        if 0 < i < len(c) - 1 and c[i] > 0.2:
            d = c[i - 1] - 2 * c[i] + c[i + 1]
            frac = 0.5 * (c[i - 1] - c[i + 1]) / d if d < 0 else 0.0
            lags.append(round((i - max_lag + frac) / FPS * 1000, 1))
    return lags


def analyze(set_id: str) -> dict:
    db = open_copy()
    al = json.loads((DATA / "align" / f"{set_id}.json").read_text())
    cat = next(s for s in json.loads((DATA / "sets.json").read_text())["sets"] if s["id"] == set_id)
    cid = {t["n"]: t["content_id"] for t in cat["tracks"]}
    cue = {
        t["n"]: t.get("first_bar", 0) for t in cat["tracks"]
    }  # bar a track was started from (renders can cue in later)
    played = sorted((t for t in al["tracks"] if t.get("found") and not t["repeat"]), key=heard_from)
    info = []
    for t in played:
        times, beats, _ = beat_grid(db.get_content(ID=cid[t["n"]]))
        down = times[beats == 1]
        feat = features(t["path"])
        low = bar_db(feat, down)
        P = Placement(t)
        info.append(
            {
                "t": t,
                "P": P,
                "down": down,
                "feat": feat,
                "drops": drops(low),
                "contrast": bass_contrast(low),
                "drops_rec": [float(P.rec(down[d])) for d in drops(low)],
                "breaks_rec": [float(P.rec(down[d])) for d in breaks(low)],
                "gain": {"P": P, "path": t["path"], "rate": t["rate"], "keylock": t.get("keylock")},
            }
        )
    out = []
    for A, B in itertools.pairwise(info):
        a, b = A["t"], B["t"]
        b_bar1 = float(B["P"].rec(B["down"][cue[b["n"]]]))  # incoming track's first bar played, in recording time
        a_bar = float(np.interp(A["P"].track(b_bar1), A["down"], np.arange(len(A["down"])))) + 1  # 1-based, fractional
        nearest = min(A["drops"], key=lambda d: abs(d + 1 - a_bar), default=None)
        bpm_a, bpm_b = a["bpm"] * a["rate"], b["bpm"] * b["rate"]
        x = {
            "a": a["title"],
            "b": b["title"],
            "a_n": a["n"],
            "b_n": b["n"],
            "a_keylock": a.get("keylock"),
            "b_keylock": b.get("keylock"),
            "b_bar1_rec": round(b_bar1, 2),
            "a_bar_at_b_bar1": round(a_bar, 2),
            "a_drops": [d + 1 for d in A["drops"]],
            "b_drops": [d + 1 for d in B["drops"]],
            "b_cue_bar": cue[b["n"]] + 1,
            "a_bass_contrast_db": round(A["contrast"], 1),
            "from_nearest_drop_bars": None if nearest is None else round(a_bar - (nearest + 1), 2),
            # rates are only good to ~+-0.02 bpm (whole-frame snippet fit), so this can't tell Sync from ear
            "bpm_a": round(bpm_a, 3),
            "bpm_b": round(bpm_b, 3),
        }
        x |= blend.analyze(
            cat["file"], x, A["gain"], B["gain"], float(A["P"].rec(a["dur"])), B["drops_rec"], B["breaks_rec"]
        )
        if x.get("blend_bars"):  # beatmatch over the blend, from both tracks as placed in the recording
            bar = 240 / bpm_a
            t0, t1 = b_bar1 + x["b_in_bars"] * bar, b_bar1 + x["a_out_bars"] * bar
            x["hat_lag_ms"] = onset_lag(A["P"], B["P"], A["feat"], B["feat"], "hat", t0, t1)
            x["kick_lag_ms"] = onset_lag(A["P"], B["P"], A["feat"], B["feat"], "kick", t0, t1)
        out.append(x)
    return {"id": set_id, "transitions": out}


def run(set_id: str) -> str:
    res = analyze(set_id)
    (DATA / "transitions" / f"{set_id}.json").write_text(json.dumps(res))
    lines = [f"== {set_id}"]
    for t in res["transitions"]:
        d = t["from_nearest_drop_bars"]
        lines.append(
            f"  {t['a'][:18]:18} -> {t['b'][:18]:18} start@A bar {t['a_bar_at_b_bar1']:6.1f}"
            f" (drop {'' if d is None else f'{d:+.0f}':>4})  in {t.get('b_in_bars')}  out {t.get('a_out_bars')}"
            f"  blend {t.get('blend_bars')} bars  order {t.get('swap_order')}  entry {t.get('b_entry_db')}"
            f"  ran out {t.get('a_played_out')}  gone<quiet {t.get('gone_before_quiet')}"
        )
    return "\n".join(lines)


def main() -> None:
    ids = sys.argv[1:] or [
        s["id"]
        for s in json.loads((DATA / "sets.json").read_text())["sets"]
        if (DATA / "align" / f"{s['id']}.json").exists()
    ]
    (DATA / "transitions").mkdir(exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        for text in ex.map(run, ids):
            print(text, flush=True)


if __name__ == "__main__":
    main()
