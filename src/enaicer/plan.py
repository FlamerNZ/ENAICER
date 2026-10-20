"""Plan a set from playlists: track order, tempo, the part of each track to play, and where each transition sits.

- Order: cheapest path through each playlist (in playlist order of playlists), by Camelot key distance and
  tempo distance. Exact (Held-Karp) for playlists up to ~16 tracks.
- Tempo: a straight-line climb across transitions; constant through each blend so both decks stay locked,
  ramping only while a track plays alone.
- Windows: tracks start at bar 1 like EN4CER does, except long ones, which are cut to their best middle part,
  cued in on a phrase before a drop.
- Transitions: the incoming track starts on the outgoing track's 8-bar phrase grid, timed so the outgoing
  track is gone (per the template) by the end of its window and before the incoming track's first breakdown.
  The bass blends gradually, or swaps in a beat where the mix goes quiet (drop_swap).

    uv run python -m enaicer.plan out.json <playlist id> [<playlist id> ...] [--tempo 125:125 125:128]
    (a playlist id can also be a track list from enaicer.pick: a .json path)
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from enaicer import memory, profile, vocals
from enaicer.catalog import DATA
from enaicer.rekordbox import beat_grid, is_local, open_copy
from enaicer.structure import bar_db, bass_on, breaks, drops, features, phrase_offset, runs, true_beats

LONG_MIN = 10.0  # tracks longer than this (minutes) are cut to their best part: "really long tracks like The Bomb"
WINDOW_BARS = 128  # length of that part (~4 min at 125): EN4CER wanted more of The Bomb cut
OUTRO_BARS = 8  # the outgoing is gone with this much of a full-length track left (EN4CER: ~15 s)
PHRASE_CONF = 1.3  # trust a track's own section grid above this confidence
MIN_SPAN = 24  # shortest blend: 16 bars sounded abrupt (Play Your Cards -> Flower Child dipped 6 dB)


def camelot(k: str) -> tuple[int, str]:
    return int(k[:-1]), k[-1].upper()


def key_cost(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 3.0
    (na, la), (nb, lb) = camelot(a), camelot(b)
    dn = min((nb - na) % 12, (na - nb) % 12)
    if la == lb:
        return {0: 0.0, 1: 1.0, 2: 3.0}.get(dn, 6.0)
    return {0: 1.0, 1: 2.0}.get(dn, 6.0)


def key_step(a: str | None, b: str | None) -> int:
    """+1 if b is clockwise from a on the Camelot wheel (up to 2 steps), -1 if anticlockwise, else 0."""
    if not a or not b:
        return 0
    d = (camelot(b)[0] - camelot(a)[0]) % 12
    return 1 if d in (1, 2) else -1 if d in (10, 11) else 0


def preferences() -> dict:
    """EN4CER's explicit pair preferences from review feedback (data/preferences.json)."""
    f = DATA / "preferences.json"
    p = json.loads(f.read_text()) if f.exists() else {}
    return {
        k: {(x["a"], x["b"]) for x in p.get(k, [])}
        for k in ("quick_bass_swap", "avoid_pairs", "fade_outgoing_highs_early")
    }


PREFS = preferences()
MEMORY = memory.load()


def pair_cost(a: dict, b: dict, direction: int = 1) -> float:
    """Key distance, tempo distance, and a small push to keep moving round the wheel in `direction`.

    EN4CER's prepared disco hours climb the wheel (with tempo); the Dub set descends: pass the direction.
    Pairs EN4CER said don't fit get a cost no reasonable alternative reaches."""
    against = key_step(a["key"], b["key"]) == -direction
    avoid = 50.0 if (a["title"], b["title"]) in PREFS["avoid_pairs"] else 0.0
    return key_cost(a["key"], b["key"]) + 0.15 * abs(a["bpm"] - b["bpm"]) + (0.75 if against else 0.0) + avoid


def best_path(tracks: list[dict], start_cost=None, direction: int = 1) -> tuple[float, list[int]]:
    """Held-Karp: cheapest order visiting every track once. start_cost[i]: cost of opening with track i."""
    n = len(tracks)
    C = [[pair_cost(a, b, direction) for b in tracks] for a in tracks]
    best = {(1 << i, i): ((start_cost or [0.0] * n)[i], None) for i in range(n)}
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = sum(1 << i for i in subset)
            for j in subset:
                prev = mask & ~(1 << j)
                best[(mask, j)] = min(
                    ((best[(prev, i)][0] + C[i][j], i) for i in subset if i != j and (prev, i) in best),
                    default=(np.inf, None),
                )
    full = (1 << n) - 1
    cost, j = min((best[(full, j)][0], j) for j in range(n))
    path, mask = [], full
    while j is not None:
        path.append(j)
        mask, j = mask & ~(1 << j), best[(mask, j)][1]
    return cost, path[::-1]


def track_list(db, pid: str) -> tuple[str, list]:
    """(name, contents) of a Rekordbox playlist by ID, or of a track list from enaicer.pick (a .json path)."""
    if pid.endswith(".json"):
        d = json.loads(Path(pid).read_text())
        return d["name"], [db.get_content(ID=i) for i in d["content_ids"]]
    pl = db.get_playlist(ID=pid)
    return pl.Name, [s.Content for s in sorted(pl.Songs, key=lambda s: s.TrackNo)]


def load_track(c) -> dict:
    times, beats, bpms = beat_grid(c)
    feat = features(c.FolderPath)
    # where the beats actually are, not the fixed-tempo grid: the time map locks these, so drifting tracks stay
    # on the beat (Music Is The Answer's grid is up to a quarter beat out)
    down = true_beats(feat, times, float(np.median(bpms)))[beats == 1]
    low, mid, high = bar_db(feat, down), bar_db(feat, down, "mid"), bar_db(feat, down, "high")
    phrase, conf = phrase_offset(feat, down)
    mem = MEMORY.get(c.ID)
    return {
        "content_id": c.ID, "title": c.Title, "artist": c.Artist.Name if c.Artist else None, "path": c.FolderPath,
        "bpm": (c.BPM or 0) / 100, "key": c.Key.ScaleName if c.Key else None, "length": c.Length,
        "down": down.round(4).tolist(), "drops": drops(low), "breaks": breaks(low),
        "phrase": phrase if conf >= PHRASE_CONF else None,
        "memory_anchor": mem["exit_anchor"] if mem else None,
        "_low": low, "_mid": mid, "_high": high, "_level": 10 * np.log10(10 ** (low / 10) + 10 ** (mid / 10)),
        "_vocal": vocals.active(vocals.bar_levels(c.FolderPath, down)),
    }  # fmt: skip


def on_grid(t: dict, bar: int) -> bool:
    """Is `bar` a phrase start of this track? Where EN4CER has mixed out of it before, their grid (track memory);
    else its own section grid if clear (matches 75% of EN4CER's exits), else 8 bars from its first drop / bar 1."""
    if t.get("memory_anchor") is not None:
        anchor = t["memory_anchor"]
    elif t["phrase"] is not None:
        anchor = t["phrase"]
    else:
        anchor = t["drops"][0] if t["drops"] else 0
    return (bar - anchor) % 8 == 0


def bass_gap(A: dict, B: dict, x: int, span: int, a_until: float, b_from: float) -> int:
    """Bars in the blend where neither deck has bass: A's bass counts until `a_until` (bars from B's start),
    B's from `b_from`, and only where the track itself has bass there. EN4CER: "the bass of Help Me Lord drops
    out before the piano kicks in... feels a little dead between"."""
    la = A["_low"] - np.nanpercentile(A["_low"], 85)
    lb = B["_low"] - np.nanpercentile(B["_low"], 85)
    has = lambda low, k: 0 <= k < len(low) and low[k] >= -12
    gap = run = 0
    for j in range(span):
        if (j < a_until and has(la, x + j)) or (j >= b_from and has(lb, B["first_bar"] + j)):
            run = 0
        else:
            run += 1
            gap += run >= 2  # one quiet bar is a fill; two or more is dead air
    return gap


def bass_entry(B: dict, start: int, span: int) -> int | None:
    """First bar (from B's start, >= `start`) where B's own bass is in, if it comes in later than that."""
    lb = B["_low"] - np.nanpercentile(B["_low"], 85)
    for j in range(start, span - 2):
        k = B["first_bar"] + j
        if k + 1 < len(lb) and lb[k] >= -12 and lb[k + 1] >= -12:
            return j
    return None


def vocal_clash(A: dict, B: dict, x: int, span: int) -> float:
    """Bars where both tracks sing while both are audible: the incoming track from ~8 bars in, A until `span`."""
    va = vocal_bars(A, x + 8, x + span)
    vb = vocal_bars(B, B["first_bar"] + 8, B["first_bar"] + span)
    n = min(len(va), len(vb))
    return float(np.sum(va[:n] & vb[:n]))


def drop_swap(A: dict, B: dict, x: int, span: int, usual: float, scale: float) -> int | None:
    """Bar (from B's start) for a quick bass swap, or None to blend the bass gradually. EN4CER: "prefer quick swaps
    when the outgoing track goes quiet at the same time the incoming track drops". In the A/B tests of their drill
    blends (7 picks, 2026-10-20) quick won exactly where, at a 4-bar step near the usual swap, the outgoing's mids
    and highs fell >= 5 dB (Body Move's breakdown; Good Measure as Sun is Shining's bass dropped in) or the
    incoming's bass fell away for a moment (>= 6 dB, Santa Maria): the mix goes quiet. Gradual won where the
    incoming dropped while the outgoing carried on (The Bomb -> All Good), or where nothing changed."""

    def step(lv: np.ndarray, k: int) -> float:  # dB, the 4 bars from k against the 4 before
        return float(np.nanmean(lv[k : k + 4]) - np.nanmean(lv[k - 4 : k])) if 4 <= k <= len(lv) - 4 else 0.0

    near = [j for j in range(4, span, 4) if usual - 8 * scale <= j <= usual + 4 * scale]  # scale: span / a_out
    a_quiet = [j for j in near if (step(A["_mid"], x + j) + step(A["_high"], x + j)) / 2 <= -5]
    b_out = [j for j in near if step(B["_low"], B["first_bar"] + j) <= -6]
    for c in (a_quiet, b_out):  # their own rule first
        if c:
            return min(c, key=lambda j: abs(j - usual))
    return None


def quiet_swap(A: dict, B: dict, x: int, span: int, usual: float) -> int | None:
    """Bar (from B's start) to put a quick bass swap: where either track's bass drops out for a moment, nearest
    the usual crossover. EN4CER: "sometimes I swap the bass quickly on a quiet part"; in their recordings a quick
    swap (<= 4 bars) happens 24% of the time after such a drop-out, 6% otherwise. That's too weak to decide
    *whether* to swap quickly (applied as a rule it would have changed blends EN4CER rated good), so it only
    picks *where*, for pairs they asked for (data/preferences.json). None: no quiet part nearby."""
    la = A["_low"] - np.nanpercentile(A["_low"], 85)
    lb = B["_low"] - np.nanpercentile(B["_low"], 85)
    # a real quiet part, not every fill: deep (-20 dB) and near where the bass would swap anyway. Disco has short
    # fills everywhere; a looser test fired on 8 of 13 transitions, against EN4CER's ~1 in 5 quick swaps.
    out = lambda low, k: k - 2 >= 0 and k < len(low) and np.nanmin(low[k - 2 : k]) < -20
    near = range(max(16, round(usual) - 6), min(span - 3, round(usual) + 7))
    cands = [j for j in near if out(la, x + j) or out(lb, B["first_bar"] + j)]
    return min(cands, key=lambda j: abs(j - usual)) if cands else None


def vocal_bars(t: dict, a: int, b: int) -> np.ndarray:
    v = t.get("_vocal")
    return np.zeros(max(0, b - a), bool) if v is None else np.asarray(v[max(0, a) : max(0, b)], bool)


def choose_window(t: dict, span: int) -> tuple[int, int]:
    """(first bar, last bar) to play, 0-based bar indices into t["down"]. `span`: bars the outgoing track stays
    in once the next one starts."""
    nbars = len(t["down"]) - 1
    if t["length"] / 60 <= LONG_MIN:
        return 0, max(0, nbars - OUTRO_BARS)
    main, mem = main_section(t, nbars), MEMORY.get(t["content_id"])
    if main is not None or mem:
        if main is not None:
            # Play the main full-energy section and mix out where it ends and the track drops back: EN4CER left
            # The Bomb at bar 320, just after its bassline-and-vocal section (153-312); v5 played bars 35-163,
            # "only the intro".
            b, e = main
            x = next((x for x in range(e, e - 9, -1) if on_grid(t, x)), e)
            lo = max(b - 16, x - WINDOW_BARS)
        else:
            # no clear main section, but EN4CER has mixed out of it before: out at the same place (they cut long
            # tracks in the same place every time). Can't Let You Go: out at 296; scored, it got the dub's intro.
            x = round(float(np.median(mem["exits"])))
            lo = x - WINDOW_BARS
        # enter up to WINDOW_BARS before that, on a phrase start with the fewest vocals
        starts = [s for s in range(max(0, lo), min(x - 32, lo + 9)) if on_grid(t, s)] or [max(0, lo)]
        s = min(starts, key=lambda s: vocal_bars(t, s, s + 24).mean())
        return s, min(nbars, x + span)
    # no clear main section (a flat track): the best-scoring window.
    # good bars: bass in, plus how busy the mids are (vocals, horns, keys) relative to the track's own range.
    # Disco and house keep the bass in nearly throughout, so the mids carry most of the structure.
    on = bass_on(t["_low"]).astype(float)
    mid = t["_mid"]
    z = (mid - np.nanpercentile(mid, 20)) / max(np.nanpercentile(mid, 80) - np.nanpercentile(mid, 20), 1e-6)
    good = 0.5 * on + np.clip(np.nan_to_num(z), 0, 1.5)
    # Cue in on one of the track's own phrase starts (cueing mid-section put Music Is The Answer 4 bars out of
    # phrase: "really rough"), preferring an entry without vocals (they clash with the outgoing track's).
    starts = [s for s in range(max(1, nbars - WINDOW_BARS + 1)) if on_grid(t, s)]
    best = None
    for s in starts:
        e = min(nbars, s + WINDOW_BARS)
        core = good[s + 16 : e - 16]
        score = (core.mean() if len(core) else -1) - 0.75 * vocal_bars(t, s, s + 24).mean()
        if best is None or score > best[0] + 1e-6:
            best = (score, s, e)
    return best[1], best[2]


def main_section(t: dict, nbars: int) -> tuple[int, int] | None:
    """(start, end) bars of the track's main full-energy section: the longest stretch whose level (8-bar mean)
    stays within 2 dB of the track's upper quartile, dips of up to 8 bars bridged, outro excluded. None if the
    track is flat (the stretch covers nearly all of it) or has no stretch of 32+ bars."""
    lv = np.nan_to_num(t["_level"], nan=float(np.nanmin(t["_level"])))
    sm = np.convolve(lv, np.ones(8) / 8, mode="same")
    end = nbars - OUTRO_BARS
    hot = sm >= np.percentile(sm[:end], 75) - 2.0
    hot[end:] = False
    for v, b, e in runs(hot):
        if not v and 0 < b and e < end and e - b <= 8:
            hot[b:e] = True
    best = max(((e - b, b, e) for v, b, e in runs(hot) if v), default=None)
    if best is None or best[0] < 32 or best[0] > 0.8 * end:
        return None
    return best[1], best[2]


def place(A: dict, B: dict, a_out: int, cross_low: float) -> dict:
    """Where B starts (bar x of A) and how long the blend runs, with the usual bass handover for that length (bars
    from B's start) and the bars of dead bass it leaves. `cross_low`: the template's bass handover for a full blend."""
    # EN4CER's rule: the outgoing fader is fully down before the incoming track's first quiet part.
    # If that breakdown comes sooner than the usual blend, compress the blend (not below MIN_SPAN bars).
    q = next((b for b in B["breaks"] if b > B["first_bar"] + 8), None)
    span = a_out if q is None else int(np.clip(q - B["first_bar"], MIN_SPAN, a_out))
    # Start the incoming track on one of A's phrase starts, as late as A's window allows, where A is still
    # near full level (EN4CER's blends sit on the outgoing groove and come out ~2 dB louder than either side;
    # blending over a thin outro left the mix dipping), and where the two tracks' vocals overlap least
    # while both are audible (EN4CER: "vocals on the two tracks did collide").
    level = A["_level"] - np.nanpercentile(A["_level"], 80)
    lo = A["first_bar"] + 32
    latest = A["last_bar"] - (a_out if A["length"] / 60 > LONG_MIN else span)  # long: out where its window says
    cands = [x for x in range(latest, max(lo, latest - 48) - 1, -1) if on_grid(A, x)] or [max(lo, latest)]
    full = [x for x in cands if np.nanmean(level[x : x + 16]) >= -3] or cands[:1]
    # where the template hands the bass over (bars from B's start), scaled to this blend's length
    usual = cross_low * span / a_out
    a_until, b_from = usual + 2 * span / a_out, usual - 1 * span / a_out
    # Score each exit: clashing vocals, dead bass, and leaving early (8 bars earlier is worth it only
    # for 2+ fewer bars of trouble).
    x = min(
        full,
        key=lambda x: (
            vocal_clash(A, B, x, span) + 2 * bass_gap(A, B, x, span, a_until, b_from) + 0.5 * (full[0] - x) / 8
        ),
    )
    return {"x": int(x), "span": span, "usual": usual, "a_until": a_until, "b_from": b_from,
            "gap": bass_gap(A, B, x, span, a_until, b_from)}  # fmt: skip


def plan(
    playlist_ids: list[str], tempos_by_playlist: list[tuple[float, float]], profile_sets: list[str], direction: int = 1
) -> dict:
    """tempos_by_playlist: (start, end) tempo for each playlist's transitions; the bridge into a playlist is its
    first transition. Tempo changes are spread evenly, and only happen while a track plays alone."""
    db = open_copy()
    prof = profile.build(profile_sets)
    hab = prof["habits"]
    a_out = round(hab["a_out_bars"])  # outgoing gone this many bars after the incoming track starts
    order, prev_end = [], None
    for pid in playlist_ids:
        name, contents = track_list(db, pid)
        tr = [load_track(c) for c in contents if is_local(c)]
        start = None if prev_end is None else [pair_cost(prev_end, t, direction) for t in tr]
        _, path = best_path(tr, start, direction)
        order += [tr[i] | {"playlist": name, "playlist_id": pid} for i in path]
        prev_end = order[-1]
    n = len(order)
    tempos = []  # tempo of each transition, by the playlist of its incoming track
    for k, (pid, (ta, tb)) in enumerate(zip(playlist_ids, tempos_by_playlist)):
        m = sum(1 for i in range(1, n) if order[i]["playlist_id"] == pid)
        tempos += list(np.linspace(ta, tb, max(m, 1))[:m])
    tempos = np.array(tempos if tempos else [tempos_by_playlist[0][0]])
    for i, t in enumerate(order):
        t["first_bar"], t["last_bar"] = choose_window(t, a_out)
        t["tempo_in"] = float(tempos[i - 1]) if i else float(tempos[0])
        t["tempo_out"] = float(tempos[i]) if i < n - 1 else float(tempos[-1])
    transitions = []
    for i, (A, B) in enumerate(itertools.pairwise(order)):
        # EN4CER waits for the incoming track to get going: Santa Maria -> Body Move, they started Body Move at
        # bar 1 but faded it in only at its bar 25, as its bass came back at 33. Where B's bass would leave the
        # blend without any, start B one or more phrases later (the same as holding its fader down), if that
        # clears 4+ bars of dead bass.
        first, cross = B["first_bar"], prof["template"]["cross_bars"]["low"]
        opts = []
        for wait in range(0, 33, 8):
            B["first_bar"] = first + wait
            opts.append((place(A, B, a_out, cross), wait))
        p, wait = min(opts, key=lambda o: (o[0]["gap"], o[1]))
        if p["gap"] > opts[0][0]["gap"] - 4:
            p, wait = opts[0]
        B["first_bar"] = first + wait
        x, span, usual, a_until, b_from = p["x"], p["span"], p["usual"], p["a_until"], p["b_from"]
        swap = drop_swap(A, B, x, span, usual, span / a_out)
        if swap is not None and bass_gap(A, B, x, span, swap, swap) > bass_gap(A, B, x, span, a_until, b_from):
            swap = None  # B's bass isn't in yet: swapping there would leave dead air
        if swap is None and (A["title"], B["title"]) in PREFS["quick_bass_swap"]:  # EN4CER asked: nearest quiet part
            swap = quiet_swap(A, B, x, span, usual) or round(usual)
        late = bass_entry(B, round(b_from), span)
        if swap is None and late is not None and late > b_from + 2 and bass_gap(A, B, x, span, a_until, b_from):
            swap = late  # B's bass only arrives later: keep A's bass until then and swap on B's entry
        transitions.append({
            "i": i, "a": A["title"], "b": B["title"], "a_bar": int(x), "b_bar": B["first_bar"],
            "span_bars": span, "wait_bars": wait, "tempo": float(tempos[i]), "key_cost": key_cost(A["key"], B["key"]),
            "vocal_clash_bars": vocal_clash(A, B, x, span),
            "bass_swap_at": swap,
            "fade_highs_early": (A["title"], B["title"]) in PREFS["fade_outgoing_highs_early"],
            "bass_gap_bars": bass_gap(A, B, x, span, swap or a_until, swap or b_from),
        })  # fmt: skip
        A["last_bar"] = int(x) + span  # nothing of A is heard after it's gone
    for t in order:
        for k in ("_low", "_mid", "_high", "_level", "_vocal"):
            t.pop(k)
    return {"profile": prof, "tracks": order, "transitions": transitions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("playlists", nargs="+")
    ap.add_argument("--tempo", nargs="+", default=["125:125"], help="start:end per playlist, e.g. 125:125 125:128")
    ap.add_argument("--profile", nargs="+", default=["drills"], help="drills, or recorded set ids to learn from")
    a = ap.parse_args()
    tempos = [tuple(float(v) for v in x.split(":")) for x in a.tempo]
    tempos += [tempos[-1]] * (len(a.playlists) - len(tempos))
    p = plan(a.playlists, tempos, a.profile)
    with open(a.out, "w") as f:
        json.dump(p, f)
    for t, x in itertools.zip_longest(p["tracks"], p["transitions"]):
        print(
            f"{t['key']:>4} {t['bpm']:6.1f}  {t['title'][:44]:44} bars {t['first_bar']:3d}-{t['last_bar']:3d} of {len(t['down']) - 1:3d}"
            + (
                f"   -> next starts at bar {x['a_bar']} @ {x['tempo']:.2f} bpm (key cost {x['key_cost']:.0f})"
                if x
                else ""
            )
        )


if __name__ == "__main__":
    main()
