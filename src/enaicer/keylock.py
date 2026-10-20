"""Was key lock (Master Tempo) on? Measure the pitch shift of tracks played well off their native tempo.

Key lock off: a track played at `rate` sounds 12*log2(rate) semitones higher. Key lock on: no shift.
Compares long-term average spectra of the recording and the original track over the track's solo middle,
on a log-frequency axis, and finds the shift that lines them up.

    uv run python -m enaicer.keylock [set_id ...]
"""

import json
import subprocess
import sys

import numpy as np

from enaicer.catalog import DATA

SR = 22050
NFFT = 8192
BINS_PER_ST = 10  # 0.1 semitone resolution
F_LO, F_HI = 40.0, 4000.0


def segment(path: str, start: float, dur: float) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def log_spectrum(y: np.ndarray) -> np.ndarray:
    """Long-term average log spectrum on a log-frequency grid, with the broad tilt removed."""
    frames = np.lib.stride_tricks.sliding_window_view(y, NFFT)[:: NFFT // 4] * np.hanning(NFFT)
    ltas = np.log(np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0) + 1e-9)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    grid = F_LO * 2 ** (np.arange(int(12 * BINS_PER_ST * np.log2(F_HI / F_LO))) / (12 * BINS_PER_ST))
    s = np.interp(grid, f, ltas)
    k = 12 * BINS_PER_ST  # one octave
    return s - np.convolve(s, np.ones(k) / k, mode="same")


def shift_semitones(rec: np.ndarray, orig: np.ndarray, max_st: float = 2.0) -> tuple[float, float]:
    """Shift (semitones) that best maps the original track's spectrum onto the recording's, and its correlation."""
    m = int(max_st * BINS_PER_ST)
    lags = np.arange(-m, m + 1)
    core = slice(m, len(orig) - m)
    c = [np.corrcoef(rec[core], np.roll(orig, lag)[core])[0, 1] for lag in lags]
    k = int(np.argmax(c))
    return lags[k] / BINS_PER_ST, float(c[k])


def check(set_id: str, min_dev: float = 0.012) -> list[dict]:
    al = json.loads((DATA / "align" / f"{set_id}.json").read_text())  # either model: only offset and rate are used
    out = []
    for t in al["tracks"]:
        if not t.get("found") or abs(t["rate"] - 1) < min_dev or len(t["anchors"]) < 20:
            continue
        rec_times = [a[0] for a in t["anchors"]]
        a, b = np.percentile(rec_times, 30), np.percentile(rec_times, 70)  # middle of the audible span: no blend
        dur = min(b - a, 60.0)
        mid = (a + b) / 2
        rec = log_spectrum(segment(al["file"], mid - dur / 2, dur))
        orig = log_spectrum(segment(t["path"], (mid - dur / 2 - t["offset"]) * t["rate"], dur * t["rate"]))
        shift, corr = shift_semitones(rec, orig)
        expect = 12 * np.log2(t["rate"])
        out.append(
            {
                "n": t["n"],
                "title": t["title"],
                "rate": round(t["rate"], 4),
                "expected_if_off": round(expect, 2),
                "measured": shift,
                "corr": round(corr, 3),
                "verdict": "off" if abs(shift - expect) < abs(shift) else "on",
            }
        )
    return out


def annotate(set_id: str, res: list[dict]) -> str:
    """Write key lock per track into data/align/<set>.json: measured where the tempo change is big enough,
    otherwise inferred from deck alternation (tracks alternate decks in play order; key lock is per deck)."""
    path = DATA / "align" / f"{set_id}.json"
    al = json.loads(path.read_text())
    played = sorted((t for t in al["tracks"] if t.get("found") and not t["repeat"]), key=lambda t: t["offset"])
    measured = {r["n"]: r["verdict"] == "on" for r in res}
    by_deck = [{measured[t["n"]] for t in played[d::2] if t["n"] in measured} for d in (0, 1)]
    alternating = all(len(s) <= 1 for s in by_deck)
    for i, t in enumerate(played):
        if t["n"] in measured:
            t["keylock"], t["keylock_source"] = measured[t["n"]], "measured"
        elif alternating and by_deck[i % 2]:
            t["keylock"], t["keylock_source"] = next(iter(by_deck[i % 2])), "deck alternation"
        else:
            t["keylock"], t["keylock_source"] = None, None
    path.write_text(json.dumps(al))
    deck = ["?" if not s else ("on" if s == {True} else "off" if s == {False} else "mixed") for s in by_deck]
    return f"decks (by play order) {deck[0]} / {deck[1]}" + ("" if alternating else "  (not alternating)")


def main() -> None:
    for set_id in sys.argv[1:]:
        res = check(set_id)
        for r in res:
            print(
                f"[{set_id}] {r['title'][:34]:34} rate {r['rate']:.4f}  key lock off -> {r['expected_if_off']:+.2f} st"
                f"  measured {r['measured']:+.2f} st (corr {r['corr']:.2f})  => key lock {r['verdict']}"
            )
        votes = [r["verdict"] for r in res]
        print(f"[{set_id}] key lock: off {votes.count('off')}, on {votes.count('on')}; {annotate(set_id, res)}")


if __name__ == "__main__":
    main()
