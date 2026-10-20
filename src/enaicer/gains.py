"""Per-band gain curves of both decks through a transition, fitted from the recording.

In each band, the recording's power spectrum is modelled as gA * outgoing + gB * incoming, both placed by
their alignment (and pitch-shifted with tempo where key lock was off). Gains are fitted every ~0.5 s by
minimizing the Itakura-Saito divergence (scale-invariant, so a deck 40 dB down is still fitted properly;
plain least squares flips between on and off there), smoothed with a running median, then expressed in dB
against each track's solo level: 0 dB = as loud as when that track plays alone.
"""

import subprocess
from functools import lru_cache

import numpy as np

from enaicer.audio import decode

SR = 22050
NFFT = 2048
HOP = 512
FPS = SR / HOP
BANDS = {"low": (20, 200), "mid": (200, 2500), "high": (2500, 11000)}
BLOCK = 21  # frames per gain estimate (~0.5 s)
SMOOTH = 5  # running median over this many blocks
FLOOR_DB = -60.0


def stft_power(y: np.ndarray) -> np.ndarray:
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (NFFT // 2, NFFT // 2)), NFFT)[::HOP]
    return (np.abs(np.fft.rfft(frames * np.hanning(NFFT), axis=1)) ** 2).astype(np.float32)


@lru_cache(maxsize=8)
def track_power(path: str) -> np.ndarray:
    return stft_power(decode(path, SR))


def recording_power(path: str, start: float, dur: float) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(0, start):.3f}",
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
    return stft_power(np.frombuffer(raw, dtype=np.float32))


def placed_power(P, path: str, rate: float, keylock: bool | None, tau: np.ndarray) -> np.ndarray:
    """The track's power spectrum as heard at recording times `tau` (zeros where it isn't playing)."""
    S = track_power(path)
    idx = P.track(tau) * FPS
    ok = (idx >= 0) & (idx <= len(S) - 1)
    i0 = np.clip(np.floor(idx).astype(int), 0, len(S) - 2)
    w = np.clip(idx - i0, 0, 1)[:, None]
    out = (1 - w) * S[i0] + w * S[i0 + 1]
    out[~ok] = 0
    if keylock is False:  # pitch follows tempo: what's heard at f was at f / rate in the file
        bins = np.arange(S.shape[1])
        out = np.stack([np.interp(bins / rate, bins, row, right=0) for row in out])
    return out


def is_gains(R, A, B, g0=(1.0, 1.0), iters=40):
    """(gA, gB) >= 0 minimizing the Itakura-Saito divergence between R and gA A + gB B (multiplicative updates)."""
    eps = 1e-6 * R.mean() + 1e-12
    R, A, B = R + eps, A + eps * 1e-3, B + eps * 1e-3
    ga, gb = max(g0[0], 1e-6), max(g0[1], 1e-6)
    if A.sum() <= eps * len(A):
        ga = 0.0
    if B.sum() <= eps * len(B):
        gb = 0.0
    for _ in range(iters):
        V = ga * A + gb * B + eps
        q = R / V**2
        ga *= (A * q).sum() / (A / V).sum() if ga else 0
        gb *= (B * q).sum() / (B / V).sum() if gb else 0
    return ga, gb


def running_median(x: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    h = k // 2
    return np.array([np.median(x[max(0, i - h) : i + h + 1]) for i in range(len(x))])


def is_div(R, V) -> float:
    q = (R + 1e-12) / (V + 1e-12)
    return float((q - np.log(q) - 1).mean())


def marginal(R, A, B, ga, gb) -> tuple[float, float]:
    """How much each deck improves the fit over the other deck alone (0..1). Present if > PRESENT."""
    d = is_div(R, ga * A + gb * B)
    a_only = is_div(R, np.mean(R / (A + 1e-12)) * A) if A.any() else np.inf
    b_only = is_div(R, np.mean(R / (B + 1e-12)) * B) if B.any() else np.inf
    ma = 1 - d / b_only if np.isfinite(b_only) and b_only > 0 and ga > 0 else 0.0
    mb = 1 - d / a_only if np.isfinite(a_only) and a_only > 0 and gb > 0 else 0.0
    return max(ma, 0.0), max(mb, 0.0)


# A deck counts as audible when adding it improves the fit by more than this. Checked on Dub Practice 2:
# outgoing tail 12 dB down: 87-94%; after it's gone: 1%; an unrelated track as a control: 6%.
PRESENT = 0.25


def fit(rec_file: str, t0: float, t1: float, A: dict, B: dict) -> dict:
    """Gain curves over recording times [t0, t1]. A/B: {"P": Placement, "path", "rate", "keylock"}.

    Per band: a_<band>, b_<band> (linear gain) and a_<band>_m, b_<band>_m (marginal fit improvement)."""
    R = recording_power(rec_file, t0, t1 - t0)
    tau = t0 + np.arange(len(R)) / FPS
    SA = placed_power(A["P"], A["path"], A["rate"], A["keylock"], tau)
    SB = placed_power(B["P"], B["path"], B["rate"], B["keylock"], tau)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    n = len(R) // BLOCK
    out = {"t": (t0 + (np.arange(n) * BLOCK + BLOCK / 2) / FPS).round(3)}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        g, prev = [], (1.0, 1.0)
        for k in range(n):
            blk = [X[k * BLOCK : (k + 1) * BLOCK, m].ravel() for X in (R, SA, SB)]
            prev = is_gains(*blk, g0=prev)
            g.append((*prev, *marginal(*blk, *prev)))
        g = np.array(g)
        for j, deck in enumerate("ab"):
            out[f"{deck}_{name}"] = running_median(g[:, j])
            out[f"{deck}_{name}_m"] = running_median(g[:, 2 + j]).round(3)
        # the sources' own band level (dB) per block: a gain only means something where the track has sound
        for deck, S in (("a", SA), ("b", SB)):
            p = S[: n * BLOCK, m].sum(axis=1).reshape(n, BLOCK).mean(axis=1)
            out[f"{deck}_{name}_src"] = (10 * np.log10(p + 1e-12)).round(1)
    return out


def to_db(curves: dict, a_solo: np.ndarray, b_solo: np.ndarray) -> dict:
    """Gains in dB relative to each track's median gain over its solo blocks (boolean masks over `t`)."""
    out = {k: v for k, v in curves.items() if k.endswith(("_m", "_src"))} | {"t": curves["t"]}
    for name in BANDS:
        for deck, solo in (("a", a_solo), ("b", b_solo)):
            g = curves[f"{deck}_{name}"]
            ref = np.median(g[solo]) if solo.sum() >= 5 else np.percentile(g, 90)
            out[f"{deck}_{name}"] = (
                np.maximum(10 * np.log10(g / ref + 1e-12), FLOOR_DB).round(1) if ref > 0 else np.full(len(g), FLOOR_DB)
            )
    return out
