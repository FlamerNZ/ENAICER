import numpy as np

from enaicer.structure import FPS, true_beats


def synthetic(drift_ms: np.ndarray, n_frames: int, beat_s: float, decoy_ms: float | None = None) -> dict:
    """Onset envelopes for a track whose beat sits drift_ms off a fixed grid; optional bass-note decoy in the
    kick band decoy_ms from each real beat, stronger than the kick."""
    rng = np.random.default_rng(0)
    mid, kick = rng.random(n_frames) * 0.2, rng.random(n_frames) * 0.2
    times = np.arange(len(drift_ms)) * beat_s + 1.0
    for t, d in zip(times, drift_ms):
        mid[round((t + d / 1000) * FPS)] += 5.0
        kick[round((t + d / 1000) * FPS)] += 3.0
        if decoy_ms is not None:
            kick[round((t + (d + decoy_ms) / 1000) * FPS)] += 6.0
    return {"mid_onset": mid, "kick_onset": kick}, times


def test_true_beats_follows_a_drifting_beat_past_a_quarter_beat():
    beat_s = 60 / 125
    n = 600
    drift = np.concatenate([np.linspace(-100, 150, n // 2), np.full(n - n // 2, 150.0)])  # MITA-like
    feat, times = synthetic(drift, int((n * beat_s + 3) * FPS), beat_s)
    got = (true_beats(feat, times, 125.0) - times) * 1000
    assert np.max(np.abs(got - drift)) <= 10


def test_true_beats_ignores_a_bass_note_before_the_beat():
    beat_s = 60 / 133
    n = 400
    feat, times = synthetic(np.zeros(n), int((n * beat_s + 3) * FPS), beat_s, decoy_ms=-140)  # Need 2 Freak
    got = (true_beats(feat, times, 133.0) - times) * 1000
    assert np.max(np.abs(got)) <= 10
