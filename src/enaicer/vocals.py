"""Where a track's vocals are, bar by bar, from its Demucs vocal stem.

Stems are made with an isolated tool environment (PyTorch stays out of the project):

    uv tool run --python 3.11 --from demucs --with numpy --with "torch<2.9" --with "torchaudio<2.9" \\
        --with soundfile demucs -n htdemucs --two-stems vocals -d mps -o data/cache/stems <files>
"""

import itertools
from pathlib import Path

import numpy as np
import soundfile as sf

from enaicer.catalog import DATA

STEMS = DATA / "cache" / "stems" / "htdemucs"
ACTIVE_DB = 12.0  # a bar has vocals if its vocal level is within this of the track's loud vocal level


def stem(path: str) -> Path:
    return STEMS / Path(path).stem / "vocals.wav"


def bar_levels(path: str, down: np.ndarray) -> np.ndarray | None:
    """Vocal RMS (dB) per bar, bars between consecutive downbeats; None if the stem hasn't been made."""
    f = stem(path)
    if not f.exists():
        return None
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    p = (y**2).mean(axis=1)
    idx = np.clip((down * sr).astype(int), 0, len(p))
    return np.array([10 * np.log10(p[a:b].mean() + 1e-12) if b > a else -120.0 for a, b in itertools.pairwise(idx)])


def active(levels: np.ndarray | None) -> np.ndarray | None:
    """Per bar: does it have vocals (relative to the track's own loud vocal parts)?"""
    if levels is None:
        return None
    return levels >= np.percentile(levels, 90) - ACTIVE_DB
