"""What happened in a blend, read off the fitted gain curves: when each deck came in and left, the order
the bands swapped in, the incoming EQ on entry, the bass swap, and checks for EN4CER's own rules
(outgoing fader fully down before the incoming track's first quiet part; no long exposed intro)."""

import numpy as np

from enaicer.gains import BANDS, PRESENT, fit, to_db

SUSTAIN = 4  # blocks (~2 s) a condition must hold


def _first(mask: np.ndarray, start: int = 0) -> int | None:
    """First index >= start where mask holds for SUSTAIN blocks."""
    for i in range(start, len(mask) - SUSTAIN + 1):
        if mask[i : i + SUSTAIN].all():
            return i
    return None


def _last(mask: np.ndarray) -> int | None:
    """Last index where mask held for SUSTAIN blocks."""
    for i in range(len(mask) - SUSTAIN, -1, -1):
        if mask[i : i + SUSTAIN].all():
            return i + SUSTAIN - 1
    return None


def analyze(
    rec_file: str, x: dict, A: dict, B: dict, a_end: float, b_drops_rec: list[float], b_breaks_rec: list[float]
) -> dict:
    """x: the transition's timing record; A/B: placement dicts for gains.fit; times in recording seconds."""
    bar = 240 / x["bpm_a"]
    t0, t1 = x["b_bar1_rec"] - 20, min(a_end + 15, x["b_bar1_rec"] + 240)
    if t1 - t0 < 30:  # outgoing ended long before the incoming track started: a track is missing in between
        return {"b_in_bars": None, "a_out_bars": None, "gap": True}
    c = fit(rec_file, t0, t1, A, B)
    t = c["t"]
    a_pres = np.any([c[f"a_{b}_m"] > PRESENT for b in BANDS], axis=0)
    b_pres = (c["b_mid_m"] > PRESENT) | (c["b_high_m"] > PRESENT)
    i_in = _first(b_pres & (t >= x["b_bar1_rec"]))
    after = np.arange(len(t)) > (i_in if i_in is not None else len(t))
    i_out = _last(a_pres)
    db = to_db(c, t < x["b_bar1_rec"] - 0.5, after & ~a_pres if i_in is not None else np.zeros(len(t), bool))
    rel = lambda i: None if i is None else round((t[i] - x["b_bar1_rec"]) / bar, 1)
    ev = {"b_in_bars": rel(i_in), "a_out_bars": rel(i_out)}
    if i_in is not None and i_out is not None:
        ev["blend_bars"] = round((t[i_out] - t[i_in]) / bar, 1)
        ev["a_played_out"] = bool(t[i_out] >= a_end - 2)  # still audible at the end of its file: ran out of track
        entry = slice(i_in, min(i_in + 8, len(t)))
        # entry EQ per band, only where the incoming track has sound in that band (within 20 dB of its typical level)
        ev["b_entry_db"] = {}
        for b in BANDS:
            src = db[f"b_{b}_src"]
            ok = src[entry] >= np.median(src[after]) - 20 if after.any() else np.ones(entry.stop - entry.start, bool)
            ev["b_entry_db"][b] = round(float(np.median(db[f"b_{b}"][entry][ok])), 1) if ok.sum() >= 3 else None
        for b in BANDS:  # when the incoming deck overtakes the outgoing one in this band
            j = _first((db[f"b_{b}"] >= db[f"a_{b}"]) & (c[f"b_{b}_m"] > PRESENT), i_in)
            ev[f"cross_{b}_bars"] = rel(j)
        k = _first(db["b_low"] >= -3, i_in)
        ev["b_low_full_bars"] = rel(k)
        k20 = _first(db["b_low"] >= -20, i_in)
        ev["bass_swap_bars"] = None if k is None or k20 is None else round((t[k] - t[k20]) / bar, 1)
        order = sorted((ev[f"cross_{b}_bars"], b) for b in BANDS if ev[f"cross_{b}_bars"] is not None)
        ev["swap_order"] = [b for _, b in order]
        # EN4CER's rules
        q = next((r for r in b_breaks_rec if r > t[i_in]), None)
        ev["gone_before_quiet"] = None if q is None else bool(t[i_out] <= q)
        d1 = next((r for r in b_drops_rec if r > x["b_bar1_rec"]), None)
        ev["exposed_intro_bars"] = None if d1 is None else round(max(0.0, d1 - t[i_out]) / bar, 1)
        ev["too_much_intro"] = bool(ev["a_played_out"] and (ev["exposed_intro_bars"] or 0) >= 8)
    keep = ["t"] + [f"{d}_{b}" for d in "ab" for b in BANDS]
    ev["curves"] = {k: np.asarray(db[k]).round(1).tolist() for k in keep}
    ev["curves"]["t"] = [round(v - x["b_bar1_rec"], 2) for v in t]  # seconds from incoming bar 1
    ev["presence"] = {"a": a_pres.astype(int).tolist(), "b": b_pres.astype(int).tolist()}
    return ev
