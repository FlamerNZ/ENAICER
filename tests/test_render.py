import numpy as np
import soundfile as sf

from enaicer import render


def test_stretch_puts_the_beats_where_the_time_map_says(tmp_path):
    # a track at 132 BPM played at 127: a hit on every beat over a noise bed, so there's sound from sample 0
    beat, ratio, sr = 60 / 132, 132 / 127, render.SR
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(40 * 4 * beat * sr)) * 0.02).astype(np.float32)
    hits = np.arange(40 * 4) * beat
    for s in hits:
        i = int(s * sr)
        x[i : i + 300] += rng.standard_normal(300).astype(np.float32) * np.hanning(300).astype(np.float32)
    sf.write(tmp_path / "t.wav", np.stack([x, x], 1), sr, subtype="FLOAT")
    t = {"down": (np.arange(40) * 4 * beat).tolist(), "first_bar": 4, "last_bar": 30, "content_id": "t",
         "path": str(tmp_path / "t.wav")}  # fmt: skip
    bars = np.arange(27) * 4 * beat * ratio
    y, lead = render.stretch(t, bars, tmp_path)
    e = np.abs(y[:, 0])
    err = []
    for s in hits[4 * 4 + 4 : 28 * 4] - hits[4 * 4]:  # source time from bar 4 -> mix time
        w = lead + s * ratio
        i0 = int((w - 0.1) * sr)
        seg = e[i0 : i0 + int(0.2 * sr)]
        err.append((i0 + np.argmax(seg > 0.3 * seg.max())) / sr - w)
    assert abs(np.median(err)) < 0.005  # a key frame at source frame 0 put these 70 ms early
