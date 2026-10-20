import numpy as np

from enaicer.align import TOL, Window, hits, vote
from enaicer.keylock import SR, log_spectrum, shift_semitones


def test_window_ncc_finds_planted_snippet():
    rng = np.random.default_rng(0)
    R = rng.random((40, 5000)).astype(np.float32)
    S = R[:, 3210:3290].copy()
    c = Window(R, 1000, 5000, S.shape[1]).ncc(S)
    assert int(np.argmax(c)) + 1000 == 3210
    assert c.max() > 0.99


def test_vote_band_restricts_offsets():
    rng = np.random.default_rng(1)
    track = rng.random((40, 600)).astype(np.float32)
    R = rng.random((40, 4000)).astype(np.float32)
    R[:, 500:1100] = track  # played twice: at frames 500 and 2500
    R[:, 2500:3100] = track
    win = Window(R, 0, R.shape[1], 80)
    h, curves = hits(track, win, range(0, 500, 50))
    assert abs(vote(h, curves, win, band=(2400, 2600))[2] - 2500) <= TOL
    assert abs(vote(h, curves, win, band=(400, 600))[2] - 500) <= TOL


def test_keylock_detects_varispeed_shift():
    t = np.arange(SR * 20) / SR
    y = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((110, 165, 220, 277, 330, 440, 554, 660)))
    y = (y + 0.01 * np.random.default_rng(2).standard_normal(len(t))).astype(np.float32)
    rate = 1.0294  # 136 -> 140 bpm without key lock: +0.50 semitones
    fast = np.interp(np.arange(0, len(y) - 1, rate), np.arange(len(y)), y).astype(np.float32)
    shift, _ = shift_semitones(log_spectrum(fast), log_spectrum(y))
    assert abs(shift - 12 * np.log2(rate)) <= 0.1
    assert abs(shift_semitones(log_spectrum(y), log_spectrum(y))[0]) <= 0.05
