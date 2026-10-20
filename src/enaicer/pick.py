"""Pick tracks for a set from the local library: EN4CER's colour label (ground truth for genre), a tempo range,
and a smooth key path, until the set would run about as long as asked. Writes a track list that enaicer.plan takes
in place of a Rekordbox playlist ID (the library stays read-only, so no playlist is made). Streaming entries have
no audio file and are never picked.

    uv run python -m enaicer.pick out/techhouse/tracks.json --color "Tech House" --bpm 124:128 --minutes 60
"""

import argparse
import json
from pathlib import Path

from enaicer.plan import OUTRO_BARS, pair_cost
from enaicer.profile import from_drills
from enaicer.rekordbox import is_local, open_copy

BEAM = 400  # partial paths kept per length


def pick(color: str, bpm: tuple[float, float], minutes: float, direction: int = 1) -> dict:
    """The cheapest key path (plan.pair_cost per transition, on average) through tracks with this colour label and
    tempo that fills `minutes`. A track carries the set for its length less the outro left unplayed and the blend
    into the next one."""
    db = open_copy()
    pool = [
        c
        for c in db.get_content()
        if is_local(c) and c.Color and c.Color.Commnt == color and bpm[0] <= (c.BPM or 0) / 100 <= bpm[1]
    ]
    t = [{"title": c.Title, "key": c.Key.ScaleName if c.Key else None, "bpm": c.BPM / 100} for c in pool]
    bar_s = 240 / (sum(bpm) / 2)
    dur = [c.Length - (OUTRO_BARS + from_drills()["habits"]["a_out_bars"]) * bar_s for c in pool]
    beams, done = [(0.0, [i], dur[i]) for i in range(len(pool))], []
    while beams:
        grown = []
        for cost, path, d in beams:
            if d >= minutes * 60:
                done.append((cost / (len(path) - 1), path))
                continue
            grown += [(cost + pair_cost(t[path[-1]], t[j], direction), path + [j], d + dur[j])
                      for j in range(len(pool)) if j not in path]  # fmt: skip
        beams = sorted(grown, key=lambda b: b[0])[:BEAM]
    _, path = min(done)
    return {
        "name": f"{color} {minutes:g} min",
        "color": color,
        "bpm": list(bpm),
        "content_ids": [pool[i].ID for i in path],
        "titles": [f"{pool[i].Artist.Name if pool[i].Artist else ''} - {pool[i].Title}" for i in path],
        "pool": len(pool),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--color", required=True, help="Rekordbox colour label, e.g. 'Tech House'")
    ap.add_argument("--bpm", default="124:128")
    ap.add_argument("--minutes", type=float, default=60.0)
    a = ap.parse_args()
    p = pick(a.color, tuple(float(v) for v in a.bpm.split(":")), a.minutes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(p, indent=1))
    print(f"{len(p['content_ids'])} of {p['pool']} tracks -> {a.out}")
    print("\n".join(p["titles"]))


if __name__ == "__main__":
    main()
