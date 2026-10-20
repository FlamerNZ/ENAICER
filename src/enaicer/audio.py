"""Audio loading and the onset-flux features used to find tracks inside a recorded mix.

Spectral shape (log-mel cosine) can't tell dub tracks apart; per-band positive spectral flux (rhythm and
melody patterns over time) can, so that's the feature everything matches on.
"""

import subprocess
from functools import lru_cache

import librosa
import numpy as np
import soxr

SR = 11025  # feature sample rate
HOP = 256  # ~23 ms frames
FPS = SR / HOP
DECODE_SR = 22050  # tracks are decoded once at this rate, then resampled per playback rate


def decode(path: str, sr: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


@lru_cache(maxsize=64)
def _decoded(path: str) -> np.ndarray:
    return decode(path, DECODE_SR)


def played(path: str, rate: float = 1.0) -> np.ndarray:
    """The track as heard when played at `rate` x speed with key lock off (pitch moves with tempo), at SR."""
    return soxr.resample(_decoded(path), DECODE_SR, SR / rate).astype(np.float32)


def mel(y: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=HOP, n_mels=40, fmin=40, fmax=5500) + 1e-10
    )


def flux(m: np.ndarray) -> np.ndarray:
    return np.diff(m, axis=1, prepend=m[:, :1]).clip(min=0)


def stretch(m: np.ndarray, rate: float) -> np.ndarray:
    """Time-scale features as if played at `rate` x speed with key lock on (pitch unchanged)."""
    n = m.shape[1]
    x = np.arange(int(n / rate)) * rate
    return np.stack([np.interp(x, np.arange(n), b) for b in m])


@lru_cache(maxsize=64)
def _native_mel(path: str) -> np.ndarray:
    return mel(played(path))


def track_flux(path: str, rate: float, keylock: bool = True) -> np.ndarray:
    """Flux of the track as heard at `rate` x speed. EN4CER plays with key lock on, so that's the default."""
    if keylock:
        return flux(stretch(_native_mel(path), rate))
    return flux(mel(played(path, rate)))
