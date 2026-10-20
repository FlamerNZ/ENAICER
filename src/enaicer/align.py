"""Align every track of a recorded set to the recording (alignment v2).

For each track: measure its playback rate (EN4CER beatmatches by ear, no Sync, so rates aren't exact),
find its offset, then track drift every 2 s so jog nudges show up.
Mapping: recording_time = offset + track_time / rate (+ drift at that point, from `anchors`).

    uv run python -m enaicer.align [set_id ...] [--varispeed] [--jobs N]
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from enaicer.audio import FPS, SR, decode, flux, mel, track_flux
from enaicer.catalog import DATA

SNIP = int(8 * FPS)  # snippet length used to find the track
ANCHOR = int(6 * FPS)  # shorter window for local drift
MIN_NCC = 0.3
TOL = 2  # frames: snippet offsets within this count as the same vote
COARSE = np.arange(0.90, 1.1001, 0.004)


class Window:
    """A slice of the recording's flux, FFT'd once so many snippets can be correlated against it."""

    def __init__(self, R: np.ndarray, a: int, b: int, L: int):
        self.a, self.L = a, L
        W = R[:, a:b]
        self.T = W.shape[1]
        self.n = 1 << int(np.ceil(np.log2(self.T + L)))
        self.F = np.fft.rfft(W, n=self.n, axis=1)
        cs = np.cumsum(np.pad(W, ((0, 0), (1, 0))), axis=1)
        cs2 = np.cumsum(np.pad(W**2, ((0, 0), (1, 0))), axis=1)
        m = (cs[:, L:] - cs[:, :-L]) / L
        self.sd = np.sqrt(np.maximum((cs2[:, L:] - cs2[:, :-L]) / L - m**2, 1e-9))

    def ncc(self, S: np.ndarray) -> np.ndarray | None:
        """Normalized cross-correlation of snippet S at every position, averaged over bands."""
        Sz = S - S.mean(axis=1, keepdims=True)
        sd = Sz.std(axis=1, keepdims=True)
        ok = sd[:, 0] > 1e-6
        if ok.sum() < 10 or self.T <= self.L:
            return None
        Sz = Sz[ok] / sd[ok]
        num = np.fft.irfft(self.F[ok] * np.fft.rfft(Sz[:, ::-1], n=self.n, axis=1), n=self.n, axis=1)
        return (num[:, self.L - 1 : self.T] / (self.L * self.sd[ok])).mean(axis=0)


def hits(F: np.ndarray, win: Window, starts) -> tuple[list[tuple[int, int, float]], list[tuple[int, np.ndarray]]]:
    """Each snippet's best match (track frame, recording frame, ncc), plus its whole NCC curve for rescoring."""
    out, curves = [], []
    for s in starts:
        c = win.ncc(F[:, s : s + win.L])
        if c is None:
            continue
        curves.append((s, c.astype(np.float32)))
        k = int(np.argmax(c))
        if c[k] > MIN_NCC:
            out.append((s, win.a + k, float(c[k])))
    return out, curves


def vote(h, curves, win: Window, top: int = 5, band: tuple[float, float] | None = None) -> tuple[int, float, int]:
    """(votes, evidence, offset in frames) of the best candidate offset, optionally within `band` (frames).

    Candidates are the most-voted snippet offsets. Each is scored by every snippet's correlation at that
    offset, so a track that repeats in 16-bar blocks lands where the whole track fits, not on a repeat.
    """
    if band:
        h = [x for x in h if band[0] <= x[1] - x[0] <= band[1]]
    if not h:
        return 0, 0.0, 0
    offs = np.array([r - s for s, r, _ in h])
    cands = []
    for o in np.unique(offs):
        m = np.abs(offs - o) <= TOL
        cands.append((int(m.sum()), int(np.median(offs[m]))))
    picked = []
    for n, o in sorted(cands, reverse=True):
        if all(abs(o - p) > TOL for _, p in picked):
            picked.append((n, o))
        if len(picked) == top:
            break
    best = (0, -1.0, 0)
    for n, o in picked:
        ev = 0.0
        for s, c in curves:
            i = o + s - win.a
            if i - TOL >= 0 and i + TOL < len(c):
                ev += max(0.0, float(c[i - TOL : i + TOL + 1].max()))
        if ev > best[1]:
            best = (n, ev, o)
    return best


def spread(n: int, k: int) -> list[int]:
    return [int(s) for s in np.linspace(2 * FPS, max(2 * FPS, n - SNIP - FPS), k)]


def dense(n: int) -> range:
    return range(int(2 * FPS), n - SNIP, int(2 * FPS))


def search_rate(path: str, win: Window, rates, keylock: bool, band=None) -> tuple[float, tuple]:
    best = (0.0, (0, -1.0, 0))
    for r in rates:
        F = track_flux(path, r, keylock)
        v = vote(*hits(F, win, spread(F.shape[1], 10)), win, band=band)
        if v[1] > best[1][1]:
            best = (float(r), v)
    return best


def fit_line(h, off0: int, rate0: float) -> tuple[float, float, np.ndarray]:
    """Robust fit of recording frame = A + s * k over snippet hits near the voted offset -> (offset s, rate, mask)."""
    s = np.array([x[0] for x in h], float)
    r = np.array([x[1] for x in h], float)
    w = np.array([x[2] for x in h])
    keep = np.abs(r - s - off0) <= 3 * FPS
    A, k = float(off0), 1.0
    for _ in range(4):
        if keep.sum() < 3:
            break
        k, A = np.polyfit(s[keep], r[keep], 1, w=w[keep])
        keep = np.abs(r - (A + k * s)) <= max(TOL, 1.5)
    return A / FPS, rate0 / k, keep


def anchors(F: np.ndarray, win: Window, offset_f: float, beat: float) -> list[tuple[float, float, float]]:
    """Local drift every 2 s: (recording s, played-track s, ncc).

    Lag is searched within +-0.35 beat of the line: half a beat away the off-beat hi-hats match too.
    """
    out, lag = [], max(2, int(0.35 * beat * FPS))
    for s in range(0, F.shape[1] - win.L, int(2 * FPS)):
        c = win.ncc(F[:, s : s + win.L])
        if c is None:
            continue
        p = int(np.rint(offset_f + s)) - win.a
        lo, hi = max(1, p - lag), min(len(c) - 1, p + lag + 1)
        if hi - lo < 3:
            continue
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] < 0.25 or k in (lo, hi - 1):
            continue
        d = c[k - 1] - 2 * c[k] + c[k + 1]
        frac = 0.5 * (c[k - 1] - c[k + 1]) / d if d < 0 else 0.0  # parabolic peak
        # centre of the window: the drift measured applies there
        out.append(((win.a + k + frac + win.L / 2) / FPS, (s + win.L / 2) / FPS, round(float(c[k]), 3)))
    return out


def align_track(t: dict, R: np.ndarray, prior: float | None, keylock: bool, band=None) -> dict:
    """Align one track. `band`: allowed offsets (recording s of track time 0), from its history timestamp."""
    T = R.shape[1]
    if t.get("t_hist") is None:
        a, b = 0, T
    else:  # Rekordbox logs a track ~43 s after it starts playing; long tracks run up to ~12 min
        a, b = max(0, int((t["t_hist"] - 120) * FPS)), min(T, int((t["t_hist"] + 720) * FPS))
    win = Window(R, a, b, SNIP)
    tried = "prior"
    if prior:
        rate, v = search_rate(t["path"], win, prior * (1 + np.arange(-0.016, 0.0161, 0.002)), keylock, band)
    if not prior or v[0] < 4:
        rate, v = search_rate(t["path"], win, COARSE, keylock, band)
        tried = "coarse"
    if 1 <= v[0] < 4:  # only part of the track is in the recording: check every 2 s at the best rate
        F = track_flux(t["path"], rate, keylock)
        dv = vote(*hits(F, win, dense(F.shape[1])), win, band=band)
        if dv[0] >= 5:
            v, tried = (4, dv[1], dv[2]), tried + "+dense"
    res = {
        "n": t["n"],
        "title": t["title"],
        "artist": t["artist"],
        "path": t["path"],
        "bpm": t["bpm"],
        "t_hist": t.get("t_hist"),
        "repeat": t["repeat"],
        "search": tried,
        "votes": v[0],
    }
    if v[0] < 4:
        return res | {"found": False}
    for _ in range(2):  # refine rate from the slope of snippet positions, then re-measure at that rate
        F = track_flux(t["path"], rate, keylock)
        h, curves = hits(F, win, dense(F.shape[1]))
        v = vote(h, curves, win, band=band)
        offset, new_rate, keep = fit_line(h, v[2], rate)
        done = abs(new_rate / rate - 1) < 2e-4
        rate = new_rate
        if done:
            break
    F = track_flux(t["path"], rate, keylock)
    aw = Window(R, a, b, ANCHOR)
    an = anchors(F, aw, offset * FPS, 60 / (t["bpm"] * rate) if t["bpm"] else 0.43)
    # anchors are in played-track time; convert to track time
    an = [(rr, round(tt * rate, 3), c) for rr, tt, c in an]
    # Real matches have dozens of 2 s points correlating >= 0.6 (15 for a partial, clipped live track);
    # chance alignments have none. A track heard alone only briefly (end of a 2 min drill) has fewer, but
    # they come in an unbroken run, which chance doesn't produce.
    ok = [c >= 0.6 for _, _, c in an]
    strong = sum(ok)
    run = max((len(list(g)) for k, g in itertools.groupby(ok) if k), default=0)
    if strong < 8 and run < 6:
        return res | {"found": False, "rejected": f"only {strong} strong drift points"}
    drift = np.array([rr - (offset + tt / rate) for rr, tt, _ in an]) * 1000
    return res | {
        "found": True,
        "n_strong": strong,
        "rate": rate,
        "bpm_played": round(t["bpm"] * rate, 3),
        "offset": offset,
        "n_snip": len(h),
        "n_fit": int(keep.sum()),
        "dur": round(F.shape[1] / FPS * rate, 2),
        "drift_ms": {
            "median_abs": round(float(np.median(np.abs(drift))), 1) if len(drift) else None,
            "p95_abs": round(float(np.percentile(np.abs(drift), 95)), 1) if len(drift) else None,
        },
        "anchors": an,
    }


def heard_from(t: dict) -> float:
    """Recording time a track is first clearly heard. Sort play order by this, not by `offset` (where its bar 1
    would be): a track cued in mid-way has its bar 1 long before it actually plays."""
    strong = [a[0] for a in t.get("anchors", []) if a[2] >= 0.5]
    return min(strong) if strong else t["offset"]


def recording_flux(s: dict) -> np.ndarray:
    cache = DATA / "cache" / f"{s['id']}_flux.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(exist_ok=True)
    R = flux(mel(decode(s["file"], SR))).astype(np.float32)
    np.save(cache, R)
    return R


def report(set_id: str, res: dict) -> None:
    if res["found"]:
        print(
            f"[{set_id}] {res['title'][:34]:34} rate {res['rate']:.5f} ({res['bpm_played']:7.3f} bpm) "
            f"offset {res['offset']:8.2f}s votes {res['votes']:2d} fit {res['n_fit']}/{res['n_snip']} "
            f"drift {res['drift_ms']['median_abs']} ms [{res['search']}]",
            flush=True,
        )
    else:
        print(
            f"[{set_id}] {res['title'][:34]:34} not found ({res.get('rejected', f'votes {res["votes"]}')})", flush=True
        )


def hist_lag(out: list[dict]) -> float | None:
    """The set's constant gap between a track starting and Rekordbox logging it (offset - t_hist, s)."""
    d = np.array([r["offset"] - r["t_hist"] for r in out if r["found"] and r["t_hist"] is not None])
    if len(d) < 4:
        return None
    m = np.median(d)
    d = d[np.abs(d - m) < 3]
    return float(np.median(d)) if len(d) >= 4 else None


def align_set(s: dict, keylock: bool = True) -> dict:
    R = recording_flux(s)
    out, prior, prev_bpm = [], None, None
    for t in s["tracks"]:
        res = align_track(t, R, prior * prev_bpm / t["bpm"] if prior and t["bpm"] else None, keylock)
        if res["found"]:
            prior, prev_bpm = res["rate"], t["bpm"]
        report(s["id"], res)
        out.append(res)
    # Second pass: looping tracks can still land on a repeat. Re-search tracks that are missing or off the
    # set's history pattern, only at offsets their history time allows (start up to 40 s into the track).
    K = hist_lag(out)
    for i, (t, res) in enumerate(zip(s["tracks"], out)):
        if K is None or t.get("t_hist") is None:
            continue
        if res["found"] and abs(res["offset"] - t["t_hist"] - K) <= 2:
            continue
        lo, hi = t["t_hist"] + K - 40, t["t_hist"] + K + 2
        prev = next((r for r in reversed(out[:i]) if r["found"]), None)
        p = prev["rate"] * prev["bpm"] / t["bpm"] if prev and t["bpm"] else None
        new = align_track(t, R, p, keylock, band=(lo * FPS, hi * FPS))
        if new["found"]:
            new["search"] += "+hist"
            new["pass1_offset"] = res.get("offset")
            out[i] = new
            report(s["id"], new)
        elif res["found"]:
            res["off_pattern"] = round(res["offset"] - t["t_hist"] - K, 1)
    # Third pass: a track still missing may have been replayed, with its only history row from an earlier
    # play (often before the recording started). Search the whole recording.
    for i, (t, res) in enumerate(zip(s["tracks"], out)):
        if res["found"] or t["repeat"] or t.get("t_hist") is None:
            continue
        new = align_track(t | {"t_hist": None}, R, None, keylock)
        if new["found"]:
            new["t_hist"], new["search"] = t["t_hist"], new["search"] + "+whole"
            out[i] = new
            report(s["id"], new)
    played = sorted((r for r in out if r["found"]), key=heard_from)
    return {
        "id": s["id"],
        "file": s["file"],
        "keylock": keylock,
        "hist_lag": K,
        "tracks": out,
        "order": [r["n"] for r in played],
    }


def run(set_id: str, keylock: bool) -> str:
    cat = json.loads((DATA / "sets.json").read_text())
    s = next(x for x in cat["sets"] if x["id"] == set_id)
    res = align_set(s, keylock)
    out = DATA / "align" / f"{set_id}{'' if keylock else '_varispeed'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res))
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--varispeed", action="store_true", help="model key lock off (pitch follows tempo)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cat = json.loads((DATA / "sets.json").read_text())
    ids = a.sets or [s["id"] for s in cat["sets"] if not any(t.get("candidate") for t in s["tracks"])]
    with ProcessPoolExecutor(a.jobs) as ex:
        for p in ex.map(run, ids, [not a.varispeed] * len(ids)):
            print("wrote", Path(p).relative_to(DATA.parent))


if __name__ == "__main__":
    main()
