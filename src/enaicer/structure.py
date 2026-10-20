"""Per-track features for transition analysis: band onset envelopes, per-bar energy, and drop detection.

Drops are found from low-end energy per bar: a drop starts on a phrase boundary where the next 8 bars
are much louder in the lows than the 8 before (intro or breakdown -> drop).
"""

import hashlib
import itertools
from pathlib import Path

import numpy as np

from enaicer.audio import decode
from enaicer.catalog import DATA

SR = 22050
HOP = 110  # 5 ms
FPS = SR / HOP
NFFT = 1024
TRACK_BANDS = ("mid",)  # true_beats: whose onsets mark the beat
GRID_NEAR_MS, FAR_WEIGHT = 60.0, 0.7  # true_beats: attacks further than this from the grid count for less
JUMP_COST = 0.15  # true_beats: cost per ms a beat's offset changes (0.4 held MITA through a real jump; 0.05 wandered)
BANDS = {"kick": (40, 150), "low": (20, 200), "mid": (200, 2500), "high": (2500, 11000), "hat": (6000, 11000)}


def _cache(path: str) -> Path:
    return DATA / "cache" / "tracks" / (hashlib.sha1(path.encode()).hexdigest()[:16] + ".npz")


def band_features(y: np.ndarray, bands=None) -> dict[str, np.ndarray]:
    """Band power (dB) and positive-flux onset envelopes at 5 ms frames, for mono audio at SR."""
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (NFFT // 2, NFFT // 2)), NFFT)[::HOP]
    P = np.abs(np.fft.rfft(frames * np.hanning(NFFT), axis=1)) ** 2
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    out = {}
    for name, (lo, hi) in (bands or BANDS).items():
        band = P[:, (freqs >= lo) & (freqs < hi)]
        out[f"{name}_db"] = (10 * np.log10(band.sum(axis=1) + 1e-10)).astype(np.float32)
        # onset envelope: positive change of log power, summed over the band's bins
        lb = np.log(band + 1e-10)
        out[f"{name}_onset"] = np.diff(lb, axis=0, prepend=lb[:1]).clip(min=0).sum(axis=1).astype(np.float32)
    return out


def features(path: str) -> dict[str, np.ndarray]:
    """band_features of a track file, in the track's own time (cached)."""
    f = _cache(path)
    if f.exists():
        return dict(np.load(f))
    out = band_features(decode(path, SR))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return out


def bar_db(feat: dict, downbeats: np.ndarray, band: str = "low") -> np.ndarray:
    """Mean band power (dB) of each bar, bars delimited by consecutive downbeats (track seconds)."""
    x = feat[f"{band}_db"]
    idx = np.clip((downbeats * FPS).astype(int), 0, len(x))
    return np.array([x[a:b].mean() if b > a else np.nan for a, b in itertools.pairwise(idx)])


def drops(low: np.ndarray, min_on: int = 8, min_off: int = 4, gap_db: float = 15.0) -> list[int]:
    """Bar indices (0-based) where a drop starts: the bass comes in for `min_on`+ bars after being out `min_off`+.

    "Bass in" = within `gap_db` of the track's full bass level. Starts snap to a 4-bar boundary when within
    a bar of one. Checked against Rekordbox phrase analysis on the Dub tracks that have it. Tracks whose bass
    never drops out (e.g. Patterns) give no drops: see `bass_contrast`.
    """
    on = bass_on(low, gap_db)
    out, b = [], 0
    while b < len(on):
        e = b
        while e < len(on) and on[e] == on[b]:
            e += 1
        if on[b] and e - b >= min_on and b >= min_off and not on[max(0, b - min_off) : b].any():
            s = 4 * round(b / 4)
            out.append(s if abs(s - b) <= 1 else b)
        b = e
    return out


def bass_on(low: np.ndarray, gap_db: float = 15.0) -> np.ndarray:
    """Per bar: is the bass in? Fills of up to 2 bars are bridged and one-bar blips dropped."""
    on = np.nan_to_num(low, nan=-200) >= np.nanpercentile(low, 85) - gap_db
    rs = runs(on)
    for i, (v, b, e) in enumerate(rs):
        if not v and e - b <= 2 and 0 < i < len(rs) - 1:
            on[b:e] = True
    for v, b, e in runs(on):
        if v and e - b == 1:
            on[b:e] = False
    return on


def breaks(low: np.ndarray, min_off: int = 4) -> list[int]:
    """Bar indices (0-based) where a breakdown starts: the bass goes out for `min_off`+ bars after being in."""
    return [b for v, b, e in runs(bass_on(low)) if not v and b > 0 and e - b >= min_off and e < len(low)]


def true_beats(feat: dict, times: np.ndarray, bpm: float) -> np.ndarray:
    """Beat times where the beats actually are (warp markers), following the track's own tempo.

    The beat is marked by the mid-band attack (200-2500 Hz: the click of the kick, claps, snares), which is one
    sharp peak per beat in every track checked, ~35 ms before Rekordbox's grid line. The kick band is muddy:
    bass notes land in it too (Need 2 Freak has one 140 ms before each beat). Lining up mid attacks reproduces
    EN4CER's own drill blends to within 9 ms (8 blends); their kicks sit up to 16 ms apart, since they match by ear
    on the hats.

    Rekordbox grids are a fixed tempo, and older productions drift against them: Music Is The Answer's beat wanders
    from -110 to +150 ms off its grid. Each beat's offset from its grid line (within 0.4 beat) is chosen by Viterbi:
    the path through the beats that lands on the strongest attacks while changing offset as little as possible.
    Real drift is slow, so in breakdowns the path holds its offset instead of wandering, and it can't slip a beat.
    Attacks more than GRID_NEAR_MS off the grid count for less, so the path leaves the grid only where nothing is
    near it. Tracks without enough clear attacks keep the grid, shifted by their median attack offset."""
    env = sum(feat[f"{b}_onset"] / (np.median(feat[f"{b}_onset"]) + 1e-9) for b in TRACK_BANDS)
    p0 = float(np.median(np.diff(times)))
    R = int(0.4 * p0 * FPS)
    offs = np.arange(-R, R + 1)
    idx = np.clip(np.round(times * FPS).astype(int)[:, None] + offs[None, :], 0, len(env) - 1)
    E = env[idx]
    clear = E.max(axis=1) >= 4 * (np.median(E, axis=1) + 1e-9)
    # reward: onset strength against the track's typical clear attack, so weaker hits in breakdowns don't pull
    # the path off the beat
    S = np.log1p(E / (np.median(E.max(axis=1)[clear]) / 4 + 1e-9)) if clear.any() else np.zeros_like(E)
    if clear.sum() < 32:
        return times + float(np.median(offs[S.argmax(axis=1)][clear])) / FPS if clear.sum() >= 8 else times
    P = JUMP_COST * np.abs(offs[:, None] - offs[None, :]) * (1000 / FPS)  # per ms of offset change
    # an attack near the grid beats a stronger hit further off; far-off hits only win where nothing is near the
    # grid (Music Is The Answer's drifted stretches)
    S = S * np.where(np.abs(offs) * (1000 / FPS) <= GRID_NEAR_MS, 1.0, FAR_WEIGHT)[None, :]
    D, back = S[0].copy(), np.zeros(E.shape, dtype=np.int32)
    for j in range(1, len(times)):
        m = D[:, None] - P  # from offset l (rows) to k (columns)
        back[j] = m.argmax(axis=0)
        D = S[j] + m.max(axis=0)
    path = np.empty(len(times), dtype=int)
    path[-1] = int(D.argmax())
    for j in range(len(times) - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    return times + offs[path] / FPS


def phrase_offset(feat: dict, down: np.ndarray, span: int = 8) -> tuple[int, float]:
    """(bar offset 0..span-1 where sections change, confidence). Sections change where band levels jump;
    confidence is the winning offset's mean change over the median offset's."""
    lvl = np.vstack([bar_db(feat, down, b) for b in ("low", "mid", "high")])
    nov = np.r_[0, np.nansum(np.abs(np.diff(lvl, axis=1)), axis=0)]
    sc = np.array([np.nanmean(nov[o::span]) for o in range(span)])
    return int(np.argmax(sc)), float(sc.max() / max(np.median(sc), 1e-9))


def runs(x: np.ndarray) -> list[tuple[bool, int, int]]:
    """(value, start, end) of each run of equal values."""
    edges = np.flatnonzero(np.diff(x.astype(int))) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(x)]
    return [(bool(x[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def bass_contrast(low: np.ndarray) -> float:
    """dB between the track's full-bass bars and its quiet-bass bars; under ~10 dB, drops aren't detectable from bass."""
    return float(np.nanpercentile(low, 85) - np.nanpercentile(low, 15))
