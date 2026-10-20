"""What EN4CER's own sets say about individual tracks: where they mix out of each one, on which phrase grid.

Built from every measured transition in their recordings (not renders). A track's exits, taken mod 8, give
the phrase grid EN4CER uses for it; that beats the audio estimate where the two disagree (the estimate put
Music Is The Answer 4 bars off EN4CER's own exit, which sounded "off").

    uv run python -m enaicer.memory      # writes data/track_memory.json
"""

import json
from collections import Counter

from enaicer.catalog import DATA

PATH = DATA / "track_memory.json"


def build() -> dict:
    cat = {s["id"]: s for s in json.loads((DATA / "sets.json").read_text())["sets"]}
    exits: dict[str, list[int]] = {}
    for sid, s in cat.items():
        p = DATA / "transitions" / f"{sid}.json"
        if s["device"] == "render" or not p.exists():
            continue
        cid = {t["n"]: t["content_id"] for t in s["tracks"]}
        # drills know where the incoming track was cued (enaicer.drills); elsewhere it's taken as its bar 1
        dp = DATA / "drills" / f"{sid}.json"
        cued = (
            {(d["a"], d["b"]): d["a_bar_at_cue"] for d in json.loads(dp.read_text())["transitions"]}
            if dp.exists()
            else {}
        )
        for t in json.loads(p.read_text())["transitions"]:
            x = cued.get((t["a"], t["b"]), t["a_bar_at_b_bar1"]) - 1  # 0-based outgoing bar where the incoming started
            if abs(x - round(x)) <= 0.15:  # on a downbeat
                exits.setdefault(cid[t["a_n"]], []).append(round(x))
    out = {}
    for c, xs in exits.items():
        anchor, n = Counter(x % 8 for x in xs).most_common(1)[0]
        out[c] = {"exit_anchor": anchor, "agree": n, "exits": sorted(xs)}
    return out


def load() -> dict:
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def main() -> None:
    m = build()
    PATH.write_text(json.dumps(m, indent=1))
    print(f"{len(m)} tracks with EN4CER exits -> {PATH.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
