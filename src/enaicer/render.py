"""Render a planned set (enaicer.plan) to audio.

Each track is time-stretched to the tempo plan with Rubber Band (R3 engine; key lock on, like EN4CER's decks),
using a time map so the tempo only ramps between blends and both decks stay bar-locked through them. Tracks
are loudness-matched first (the DJ's trims), then EN4CER's blend template is applied as 3-band EQ + fader
automation in the STFT domain, with exactly the bands it was measured in (a filter crossover leaks: the incoming
bass came through at mid-band level, early), everything is summed, and the mix is mastered
to a target loudness with a peak limiter.

    uv run python -m enaicer.render plan.json out_dir [--lufs -10.5]
"""

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
import soxr
from scipy.ndimage import maximum_filter1d
from scipy.signal import istft, stft

from enaicer.gains import BANDS, FLOOR_DB
from enaicer.structure import FPS as ONSET_FPS
from enaicer.structure import SR as ONSET_SR
from enaicer.structure import band_features

SR = 44100
CTRL = 100  # gain automation points per second
TRACK_LUFS = -14.0  # every track is matched to this before mixing
XOVER = (BANDS["low"][1], BANDS["mid"][1])  # 200 Hz, 2500 Hz
ONSET_BANDS = {"hat": (6000, 11000), "kick": (40, 150)}
SWAP_BARS = 0.25  # a quick bass swap takes one beat: a one-bar ramp was "not quick enough"
HAT_FADE = (12.0, 22.0)  # bars from B's start over which A's highs go when the hats clash


def bar_times(t: dict, tin: float, tout: float, span_in: int, x_out: int) -> np.ndarray:
    """Mix-time offset (s, from the track's first bar) of each bar boundary first_bar..last_bar.

    Bars inside the incoming blend run at `tin`, bars from the next transition on at `tout`, linear between."""
    k0, k1 = t["first_bar"], t["last_bar"]
    ks = np.arange(k0, k1)
    ramp_a, ramp_b = k0 + span_in, max(x_out, k0 + span_in + 1)
    tempo = np.where(
        ks < ramp_a, tin, np.where(ks >= ramp_b, tout, tin + (tout - tin) * (ks - ramp_a) / (ramp_b - ramp_a))
    )
    return np.concatenate([[0.0], np.cumsum(240.0 / tempo)])


def stretch(t: dict, bars: np.ndarray, tmp: Path) -> tuple[np.ndarray, float]:
    """Stretch the track's window so bar k lands at bars[k - first_bar]. Returns (stereo audio, lead-in s).

    Lead-in: when the window starts at bar 1, the pickup before the first downbeat plays before mix time 0."""
    down = np.array(t["down"])
    k0, k1 = t["first_bar"], t["last_bar"]
    s0 = 0.0 if k0 == 0 else down[k0]
    s1 = down[k1]
    src = tmp / f"src_{t['content_id']}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.6f}", "-t", f"{s1 - s0:.6f}", "-i", t["path"],
                    "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(src)], check=True)  # fmt: skip
    first_ratio = (bars[1] - bars[0]) / (down[k0 + 1] - down[k0])
    lead = (down[k0] - s0) * first_ratio
    src_frames = np.round((down[k0 : k1 + 1] - s0) * SR).astype(int)
    tgt_frames = np.round((lead + bars) * SR).astype(int)
    # no key frame at source frame 0 (the start is implicit): with one there, R3 shifts the whole output, by an
    # amount that grows with the stretch (Body Move at 132 -> 127 BPM came out 70 ms off its beat)
    keys = [(a, b) for a, b in zip(src_frames, tgt_frames) if a > 0]
    tm = tmp / f"map_{t['content_id']}.txt"
    tm.write_text("".join(f"{a} {b}\n" for a, b in keys))
    out = tmp / f"st_{t['content_id']}.wav"
    subprocess.run(["rubberband", "-3", "-q", "--timemap", str(tm), "-D", f"{tgt_frames[-1] / SR:.6f}", str(src), str(out)],
                   check=True, capture_output=True)  # fmt: skip
    x, _ = sf.read(out, dtype="float32", always_2d=True)
    return x, lead


NPS, HOPR = 2048, 512  # STFT for the EQ: same resolution class as the measurement


def band_weights(freqs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bin weight of each band. Edges as measured (200 Hz, 2.5 kHz), softened over 1/3 octave; sums to 1."""
    ramp = lambda edge: np.clip((np.log2(np.maximum(freqs, 1) / edge) + 1 / 6) * 3, 0, 1)
    low, high = 1 - ramp(XOVER[0]), ramp(XOVER[1])
    return {"low": low, "mid": 1 - low - high, "high": high}


def eq(x: np.ndarray, gain_at) -> np.ndarray:
    """Apply time-varying 3-band gains in the STFT domain, so the bands are the ones the template was measured in.

    gain_at(times s from the audio's start) -> {band: linear gain per time}. With all gains 1 the output is x."""
    f, tt, Z = stft(x.T, fs=SR, window="hann", nperseg=NPS, noverlap=NPS - HOPR)
    w, g = band_weights(f), gain_at(tt)
    G = sum(w[b][:, None] * g[b][None, :] for b in BANDS)
    _, y = istft(Z * G[None].astype(np.float32), fs=SR, window="hann", nperseg=NPS, noverlap=NPS - HOPR)
    return y.T[: len(x)].astype(np.float32)


def curve(template: dict, deck: str, band: str, u: np.ndarray, scale: float) -> np.ndarray:
    """Template dB for `deck` at template bars u (bars from incoming bar 1), time-scaled by `scale`."""
    g = np.array(template["bars"])
    return np.interp(u / scale, g, np.array(template[f"{deck}_{band}"]), left=np.nan, right=np.nan)


def stretch_track(args) -> dict:
    """Stage 1: stretch and loudness-match one track, and take its onsets (before any gain) for locking.
    The audio goes to a temp file rather than back through the process pool."""
    t, tin, tout, span_in, x_out, tmp = args
    bars = bar_times(t, tin, tout, span_in, x_out)
    x, lead = stretch(t, bars, Path(tmp))
    x *= 10 ** ((TRACK_LUFS - loudness(x)) / 20)
    f = Path(tmp) / f"matched_{t['content_id']}.npy"
    np.save(f, x)
    on = band_features(soxr.resample(x.mean(axis=1), SR, ONSET_SR).astype(np.float32), ONSET_BANDS)
    return {"file": str(f), "n": len(x), "lead": lead, "bars": bars, "hat": on["hat_onset"], "kick": on["kick_onset"]}


def gain_curves(t: dict, part: dict, incoming: dict | None, outgoing: dict | None, template: dict, a_out: int):
    """dB automation per band for one track, at CTRL points from its first sample (s from its first bar)."""
    bars, lead = part["bars"], part["lead"]
    tctrl = np.arange(-lead, part["n"] / SR - lead, 1 / CTRL)
    kb = np.arange(t["first_bar"], t["last_bar"] + 1)
    bar_at = np.interp(tctrl, bars, kb, left=kb[0] - 1, right=kb[-1] + 1)  # fractional bar index at each point
    gains = {}
    for band in BANDS:
        db = np.zeros(len(tctrl))
        if incoming:  # this track is B: template bars counted from its cue bar
            u = bar_at - incoming["b_bar"]
            v = curve(template, "b", band, u, incoming["span_bars"] / a_out)
            if band == "low" and incoming.get("bass_swap_at") is not None:  # quick swap: held down, then straight in
                j = incoming["bass_swap_at"]
                v = np.where(
                    u < j - SWAP_BARS,
                    np.minimum(v, -20.0),
                    np.where(u < j, -20.0 + 20.0 * (u - j + SWAP_BARS) / SWAP_BARS, 0.0),
                )
            db = np.where(np.isnan(v), db, v)
        if outgoing:  # this track is A: template bars counted from where B starts
            u = bar_at - outgoing["a_bar"]
            v = curve(template, "a", band, u, outgoing["span_bars"] / a_out)
            if band == "low" and outgoing.get("bass_swap_at") is not None:  # quick swap: full until the swap, then out
                j = outgoing["bass_swap_at"]
                v = np.where(u < j - SWAP_BARS, 0.0, np.where(u < j, -30.0 * (u - j + SWAP_BARS) / SWAP_BARS, FLOOR_DB))
            if band == "high" and outgoing.get("fade_highs_early"):
                # EN4CER heard the two hat patterns collide ("colliding half hats"): take A's highs out by the time
                # B's are full, instead of leaving them at -7 dB for the rest of the blend. Only where they asked:
                # as a rule on low hat correlation it would also have hit blends they rated good.
                h0, h1 = HAT_FADE
                v = np.minimum(v, np.where(u < h0, 0.0, np.where(u < h1, -24.0 * (u - h0) / (h1 - h0), FLOOR_DB)))
            db = np.where(np.isnan(v), db, np.minimum(db, v))
            db[bar_at >= outgoing["a_bar"] + outgoing["span_bars"]] = FLOOR_DB
        gains[band] = np.where(db <= FLOOR_DB + 1, 0.0, 10 ** (db / 20))
    return tctrl, gains


def shape_track(args) -> np.ndarray:
    """Stage 3: apply the blend automation (EQ + fader) to a stretched track."""
    t, part, incoming, outgoing, template, a_out = args
    x = np.load(part["file"])
    tctrl, gains = gain_curves(t, part, incoming, outgoing, template, a_out)
    lead = part["lead"]
    return eq(x, lambda tt: {b: np.interp(tt - lead, tctrl, gains[b]) for b in BANDS})


def lock(A: dict, B: dict, a0: float, b0: float, t0: float, t1: float) -> dict:
    """Nudge for B (s) that lines it up with A over mix times [t0, t1], like EN4CER on the jog wheel.

    The time map already puts both tracks' beats (mid-band attacks, structure.true_beats) together, which matches
    EN4CER's own blends to within ~10 ms. So this only fine-tunes on the hats, within +-10 ms: a wider search
    (+-150 ms, v5) found false matches between different hat patterns and pushed Body Move 55 ms early.
    a0/b0: mix time of each part's first sample. Returns the nudge and the hat/kick correlation (-1..1) with no
    nudge and with it, so a bad choice shows up."""
    fps = ONSET_FPS
    n = int((t1 - t0) * fps)
    if n < 4 * fps:
        return {"nudge": 0.0}

    def env(p, start, key, shift=0.0):
        i = round((t0 - start + shift) * fps)
        e = p[key][max(0, i) : max(0, i) + n]
        e = np.pad(e, (0, n - len(e)))
        return (e - e.mean()) / (e.std() + 1e-9)

    ha, ka = env(A, a0, "hat"), env(A, a0, "kick")
    corr = lambda key, ref, k: float(np.dot(ref, env(B, b0, key, k / fps)) / n)
    near = np.arange(-int(0.01 * fps), int(0.01 * fps) + 1)
    fine = int(near[np.argmax([corr("hat", ha, k) for k in near])])
    # B read `fine` frames later lines up: B is late by fine/fps, so it moves earlier by that. Only trust it when
    # the hats clearly line up there (with little rhythm in common, an a cappella intro, the peak is noise) and
    # the best shift isn't on the edge of the search.
    edge = abs(fine) >= near[-1]
    trusted = not edge and corr("hat", ha, fine) >= 0.2 and corr("hat", ha, fine) - corr("hat", ha, 0) >= 0.05
    return {"nudge": -fine / fps if trusted else 0.0, "trusted": trusted, "hat_corr_before": round(corr("hat", ha, 0), 3), "kick_corr_before": round(corr("kick", ka, 0), 3),
            "hat_corr_after": round(corr("hat", ha, fine), 3), "kick_corr_after": round(corr("kick", ka, fine), 3)}  # fmt: skip


def limiter(x: np.ndarray, ceiling_db: float = -1.0, look_ms: float = 5.0, recover_db_s: float = 30.0) -> np.ndarray:
    """Lookahead peak limiter, in place. Reduction is held through each peak (lookahead) and recovers at a fixed
    dB/s, computed without a per-sample loop: r_out[n] = max_k<=n (r[k] - d (n - k))."""
    n = int(SR * look_ms / 1000)
    d = recover_db_s / SR
    carry, chunk = 0.0, SR * 600
    for a in range(0, len(x), chunk):
        b = min(len(x), a + chunk)
        lo, hi = max(0, a - n), min(len(x), b + n)
        peak = maximum_filter1d(np.abs(x[lo:hi]).max(axis=1), size=2 * n + 1)[a - lo : a - lo + (b - a)]
        r = np.maximum(0.0, 20 * np.log10(np.maximum(peak, 1e-9)) - ceiling_db)
        i = np.arange(b - a)
        r = np.maximum(np.maximum.accumulate(r + d * i) - d * i, carry - d * (i + 1))
        carry = r[-1]
        x[a:b] *= (10 ** (-r / 20)).astype(np.float32)[:, None]
    return x


def loudness(x: np.ndarray) -> float:
    """Integrated loudness (LUFS), measured at half rate to keep long mixes quick (within ~0.2 dB)."""
    return pyloudnorm.Meter(SR // 2).integrated_loudness(x[::2].astype(np.float64))


def render(plan: dict, out_dir: Path, lufs: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks, trans = plan["tracks"], plan["transitions"]
    template = plan["profile"]["template"]
    a_out = round(plan["profile"]["habits"]["a_out_bars"])
    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        for i, t in enumerate(tracks):
            inc = trans[i - 1] if i else None
            outg = trans[i] if i < len(trans) else None
            tin = inc["tempo"] if inc else outg["tempo"]
            tout = outg["tempo"] if outg else tin
            jobs.append((t, tin, tout, inc["span_bars"] if inc else 0, outg["a_bar"] if outg else t["last_bar"], tmp))
        with ProcessPoolExecutor(6) as ex:
            parts = list(ex.map(stretch_track, jobs))
        # Stage 2: place tracks. B's cue bar lands on A's transition bar, then B is nudged so the hi-hats line up.
        starts = [parts[0]["lead"]]  # mix time of each track's first bar
        nudges = []
        for i, x in enumerate(trans):
            A, bar = tracks[i], 240 / x["tempo"]
            s = starts[i] + parts[i]["bars"][x["a_bar"] - A["first_bar"]]
            # both decks clearly audible: from ~8 bars after B starts until A starts leaving
            t0, t1 = s + 8 * bar, s + max(12.0, x["span_bars"] - 8) * bar
            lk = lock(parts[i], parts[i + 1], starts[i] - parts[i]["lead"], s - parts[i + 1]["lead"], t0, t1)
            starts.append(s + lk["nudge"])
            nudges.append({"i": i, "nudge_ms": round(lk.pop("nudge") * 1000, 1)} | lk)
        # Stage 3: the blend automation, now that hat clashes are known
        with ProcessPoolExecutor(6) as ex:
            jobs = [(t, parts[i], trans[i - 1] if i else None, trans[i] if i < len(trans) else None, template, a_out)
                    for i, t in enumerate(tracks)]  # fmt: skip
            audio = list(ex.map(shape_track, jobs))
    for p, y in zip(parts, audio):
        p["audio"] = y
    total = max(s - p["lead"] + len(p["audio"]) / SR for s, p in zip(starts, parts))
    mix = np.zeros((int(total * SR) + SR, 2), np.float32)
    for s, p in zip(starts, parts):
        a = round((s - p["lead"]) * SR)
        mix[a : a + len(p["audio"])] += p["audio"]
    mix *= 10 ** ((lufs - loudness(mix)) / 20)
    limiter(mix)
    wav = out_dir / "mix.wav"
    sf.write(wav, mix, SR, subtype="PCM_24")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "320k", str(out_dir / "mix.mp3")], check=True
    )
    # tracklist: a track is "on" once its mids have taken over (template crossover, ~16 bars after its cue)
    lines, listing = [], []
    for i, (t, s) in enumerate(zip(tracks, starts)):
        on = 0.0 if i == 0 else s + parts[i]["bars"][min(16, len(parts[i]["bars"]) - 1)]
        lines.append(f"{int(on // 60):02d}:{int(on % 60):02d}  {t['artist']} - {t['title']}")
        listing.append(
            {
                "title": t["title"],
                "artist": t["artist"],
                "first_bar": t["first_bar"],
                "path": t["path"],
                "start_s": round(s, 3),
                "on_s": round(on, 3),
            }
        )
    (out_dir / "tracklist.txt").write_text("\n".join(lines) + "\n")
    meta = {"length_s": round(len(mix) / SR, 1), "lufs": lufs, "tracks": listing, "nudges": nudges}
    (out_dir / "render.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out_dir")
    ap.add_argument("--lufs", type=float, default=-10.5)
    a = ap.parse_args()
    meta = render(json.loads(Path(a.plan).read_text()), Path(a.out_dir), a.lufs)
    print(f"{meta['length_s'] / 60:.1f} min -> {a.out_dir}/mix.wav")
    print((Path(a.out_dir) / "tracklist.txt").read_text())


if __name__ == "__main__":
    main()
