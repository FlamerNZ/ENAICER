"""Check a rendered mix against a real recording by running it through the same analysis: register the render
as a set, align it, measure its transitions, and compare with the reference set.

    uv run python -m enaicer.replay <render dir> <set id for the render> <reference set id>
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from enaicer.align import align_set, heard_from
from enaicer.catalog import DATA, track_info
from enaicer.rekordbox import open_copy
from enaicer.transitions import analyze


def register(render_dir: Path, set_id: str) -> dict:
    """Add (or replace) the render in data/sets.json, tracklist from its render.json."""
    meta = json.loads((render_dir / "render.json").read_text())
    db = open_copy()
    by_path = {c.FolderPath: c for c in db.get_content() if c.FolderPath}
    tracks = []
    for i, t in enumerate(meta["tracks"]):
        c = by_path[t["path"]]
        # where the render placed it, as a history-style hint (real sets get this from Rekordbox's play history)
        hint = round(t["start_s"] + 60, 1)
        tracks.append(
            track_info(c)
            | {
                "source": "collection",
                "path": t["path"],
                "t_hist": hint,
                "n": i,
                "repeat": False,
                "first_bar": t.get("first_bar", 0),
            }
        )
    wav = str((render_dir / "mix.wav").resolve())
    entry = {"id": set_id, "file": wav, "device": "render", "duration": round(sf.info(wav).duration, 2), "start": None,
             "note": f"rendered by enaicer.render from {render_dir}", "tracks": tracks}  # fmt: skip
    cat = json.loads((DATA / "sets.json").read_text())
    cat["sets"] = [s for s in cat["sets"] if s["id"] != set_id] + [entry]
    tmp = DATA / "sets.json.tmp"
    tmp.write_text(json.dumps(cat, indent=1))
    tmp.replace(DATA / "sets.json")
    return entry


def kendall_tau(a: list, b: list) -> float:
    common = [x for x in a if x in b]
    pa, pb = {x: i for i, x in enumerate(common)}, {x: i for i, x in enumerate(x for x in b if x in common)}
    pairs = [(x, y) for i, x in enumerate(common) for y in common[i + 1 :]]
    s = sum(np.sign(pa[x] - pa[y]) * np.sign(pb[x] - pb[y]) for x, y in pairs)
    return float(s / len(pairs)) if pairs else float("nan")


def played_order(set_id: str) -> list[str]:
    al = json.loads((DATA / "align" / f"{set_id}.json").read_text())
    return [t["title"] for t in sorted((t for t in al["tracks"] if t.get("found") and not t["repeat"]), key=heard_from)]


def stats(set_id: str) -> dict:
    ts = [
        t
        for t in json.loads((DATA / "transitions" / f"{set_id}.json").read_text())["transitions"]
        if t.get("blend_bars")
    ]
    med = lambda k: float(np.median([t[k] for t in ts if t.get(k) is not None]))
    order = sum(t["swap_order"] == ["mid", "high", "low"] for t in ts) / len(ts)
    entry = {
        b: float(np.median([t["b_entry_db"][b] for t in ts if t["b_entry_db"].get(b) is not None]))
        for b in ("low", "mid", "high")
    }
    return {"n": len(ts), "b_in_bars": med("b_in_bars"), "a_out_bars": med("a_out_bars"), "blend_bars": med("blend_bars"),
            "bass_swap_bars": med("bass_swap_bars"), "mid_high_low": order, "entry_db": entry,
            "cross_mid": med("cross_mid_bars"), "cross_high": med("cross_high_bars"), "cross_low": med("cross_low_bars")}  # fmt: skip


def main() -> None:
    render_dir, set_id, ref = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    s = register(render_dir, set_id)
    res = align_set(s)
    (DATA / "align" / f"{set_id}.json").write_text(json.dumps(res))
    (DATA / "transitions" / f"{set_id}.json").write_text(json.dumps(analyze(set_id)))
    a, b = played_order(set_id), played_order(ref)
    print(f"\norder agreement with {ref} (Kendall tau, 1 = same order): {kendall_tau(a, b):+.2f}")
    print(
        f"length: render {s['duration'] / 60:.1f} min, {ref} {next(x for x in json.loads((DATA / 'sets.json').read_text())['sets'] if x['id'] == ref)['duration'] / 60:.1f} min"
    )
    sr, sf_ = stats(set_id), stats(ref)
    print(f"{'':16} {'render':>10} {ref:>22}")
    for k in sr:
        fmt = lambda v: json.dumps({kk: round(vv, 1) for kk, vv in v.items()}) if isinstance(v, dict) else f"{v:.2f}"
        print(f"{k:16} {fmt(sr[k]):>10}  {fmt(sf_[k]):>22}")


if __name__ == "__main__":
    main()
