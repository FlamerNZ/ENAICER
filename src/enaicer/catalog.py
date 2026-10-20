"""Build data/sets.json: every recorded set, its tracklist in play order, and an audio file for each track.

Tracklists come from Rekordbox song-history rows created during the recording (by time window, since a
history can span several days), from the XDJ-AZ USB history, from the .cue sheet Rekordbox's recorder writes
next to the WAV, or from a candidate pool to search. Drill recordings (in the TRAINING folders, each with its
.cue) are picked up automatically.
Streaming entries (FolderPath /v4/catalog/...) are resolved to files the user owns, matched by name.

    uv run python -m enaicer.catalog
"""

import datetime as dt
import json
import re
import unicodedata
from pathlib import Path

import soundfile as sf
from pyrekordbox.masterdb.models import DjmdSongHistory
from pyrekordbox.onelibrary import OneLibrary

from enaicer.rekordbox import is_local, open_copy

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REC = Path("~/Music/rekordbox/Recording/EN4CER").expanduser()
HOME = Path.home()
FILE_ROOTS = [
    HOME / "Music/Tracks",
    HOME / "Music/Music",
    HOME / "Library/CloudStorage/OneDrive-Personal/Music",
    HOME / "Library/CloudStorage/OneDrive-Personal/Music 1",
]
AUDIO = {".mp3", ".wav", ".aif", ".aiff", ".m4a", ".flac"}
TRAINING = "M*d*l Training*"  # drills on the DDJ-400 (enaicer.midilog); Rekordbox adds "(2)"..., one is "Mdoel"

# device: what the set was played on. tracklist: "history" (Rekordbox recorder), "usb:<OneLibrary db>",
# "file:<json with tracks[].path in play order>", "cue" (the recording's .cue), or "search" ("pool": playlists).
SETS = [
    {
        "id": "deep_end_1_practice",
        "file": "deep end/deep end .1 (practice).wav",
        "device": "rekordbox",
        "tracklist": "history",
    },
    {"id": "deep_end_1", "file": "deep end(2)/deep end .1.wav", "device": "rekordbox", "tracklist": "history"},
    {
        "id": "dumb_n_basic_1",
        "file": "dumb n basic/01 dumb n basic .1.wav",
        "device": "rekordbox",
        "tracklist": "history",
    },
    {"id": "en4fricer", "file": "EN4FRICER/01 EN4FRICER.wav", "device": "rekordbox", "tracklist": "history"},
    {"id": "melody_techno_1", "file": "Melody Techno/01 .1.wav", "device": "rekordbox", "tracklist": "history"},
    {
        "id": "dnb_practice_1",
        "file": "Drum n Bassings Practice/01 Drum n Bassings Practice 1.wav",
        "device": "rekordbox",
        "tracklist": "history",
    },
    {
        "id": "liquid_4_kristi",
        "file": "Liquid DnB/01 4 Kristi.wav",
        "device": "rekordbox",
        "tracklist": "history",
        "note": "Published on Mixcloud as dumb n basic .2",
    },
    {"id": "deep_end_02", "file": "Deep End(3)/01 Deep End .02.wav", "device": "rekordbox", "tracklist": "history"},
    {
        "id": "funky_frank_practice",
        "file": "Unknown Album/01 Funky Frank - Practice.wav",
        "device": "rekordbox",
        "tracklist": "history",
    },
    {"id": "dub_practice_1", "file": "Dub/01 Dub Practice.wav", "device": "rekordbox", "tracklist": "history"},
    # End silence was trimmed in place, so the file's birth time is lost; start = untrimmed copy's mtime - duration.
    {
        "id": "dub_practice_2",
        "file": "Dub(2)/01 Dub Practice 2.wav",
        "device": "rekordbox",
        "tracklist": "history",
        "start": "2026-10-17T15:10:21.8",
    },
    {
        "id": "dub_live",
        "file": "Dub Live/Dub Live 2026-10-17 (XDJ-AZ REC001).wav",
        "device": "xdj-az",
        "tracklist": "usb:data/sources/xdj_az_usb_2026-10-17/exportLibrary.db",
        "note": "Recording missed the first ~6 min (starts mid-Energy)",
    },
    {
        "id": "miss_melody_techno",
        "file": "Miss Melody Techno/miss melody techno .1.WAV",
        "device": "xdj-rx3",
        "tracklist": "file:data/sources/miss_melody_techno_tracklist.json",
        "note": "Probably recorded on the XDJ-RX3 (clock unset). Tracklist assembled by another session from the "
        "2025-09-03..05 practice histories plus an audio search of the Melodic/Techno/Deep pool",
    },
]


def drill_sets() -> list[dict]:
    """Every recording in the TRAINING folders that has its .cue sheet, e.g. "01 Keylog 1.wav" -> keylog_1."""
    out = []
    for w in sorted(REC.glob(f"{TRAINING}/*.wav")):
        if w.with_suffix(".cue").exists():
            sid = "_".join(norm(re.sub(r"^\d+\s+", "", w.stem)))
            out.append({"id": sid, "file": str(w.relative_to(REC)), "device": "rekordbox", "tracklist": "cue"})
    return out


def read_cue(path: Path) -> list[tuple[str, float]]:
    """(track file, INDEX time in s) per TRACK of a Rekordbox recorder .cue. Rekordbox writes INDEX as
    HH:MM:SS, not the standard MM:SS:frames (a 10 min recording has its last track at 00:09:09)."""
    out, f = [], None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("FILE"):  # the first one, before any TRACK, is the recording itself
            f = line[5:-5].strip().strip('"')
        elif line.startswith("INDEX 01"):
            h, m, sec = (int(x) for x in line.split()[2].split(":"))
            out.append((f, 3600 * h + 60 * m + sec))
    return out


def norm(s: str | None) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def core_title(title: str) -> list[str]:
    """Title words without the (mix) / [edit] part."""
    return norm(re.sub(r"[(\[].*?[)\]]", " ", title or ""))


def artist_words(name: str | None) -> set[str]:
    parts = re.split(r",|&| x | feat\.? | ft\.? | vs\.? ", (name or "").lower())
    return {w for p in parts for w in norm(p) if len(w) > 1}


def track_info(c) -> dict:
    return {
        "content_id": c.ID,
        "title": c.Title,
        "artist": c.Artist.Name if c.Artist else None,
        "remixer": c.Remixer.Name if c.Remixer else None,
        "genre": c.Genre.Name if c.Genre else None,
        "bpm": (c.BPM or 0) / 100,
        "length": c.Length,
        "key": c.Key.ScaleName if c.Key else None,
    }


class FileIndex:
    """Owned audio files outside the collection, matched by artist and title words in their path."""

    def __init__(self, collection_local: list):
        self.items = [(Path(c.FolderPath), c.ID) for c in collection_local]
        seen = {p for p, _ in self.items}
        for root in FILE_ROOTS:
            if root.exists():
                self.items += [(p, None) for p in root.rglob("*") if p.suffix.lower() in AUDIO and p not in seen]
        self.words = [(set(norm(p.stem)), set(norm(" ".join(p.parts[-3:-1])))) for p, _ in self.items]

    def find(self, title: str, artist: str | None, remixer: str | None, limit: int = 4) -> list[dict]:
        need = set(core_title(title))
        arts = artist_words(artist)
        extra = set(norm(title)) - need | artist_words(remixer)
        hits = []
        for (path, cid), (stem, dirs) in zip(self.items, self.words):
            if not need or not need <= stem:
                continue
            a = len(arts & (stem | dirs))
            if arts and not a:
                continue
            score = a + len(extra & stem) + (2 if cid else 0) - 0.1 * len(stem - need - arts - extra)
            hits.append((score, str(path), cid))
        hits.sort(key=lambda h: -h[0])
        return [{"path": p, "content_id": cid, "score": round(s, 1)} for s, p, cid in hits[:limit]]


def recording_times(path: Path, start: str | None) -> tuple[dt.datetime, float]:
    dur = sf.info(path).duration
    if start:
        return dt.datetime.fromisoformat(start), dur
    st = path.stat()
    born, mod = dt.datetime.fromtimestamp(st.st_birthtime), dt.datetime.fromtimestamp(st.st_mtime)
    return (born if mod - born > dt.timedelta(seconds=60) else mod - dt.timedelta(seconds=dur)), dur


def resolve(entry: dict, c, index: FileIndex) -> dict:
    if is_local(c):
        entry.update(source="collection", path=c.FolderPath)
        return entry
    cands = index.find(c.Title, entry["artist"], entry["remixer"])
    entry.update(
        source="file" if cands else "unresolved",
        path=cands[0]["path"] if cands else None,
        stream=True,
        candidates=cands,
    )
    if cands and cands[0]["content_id"]:
        entry.update(source="collection", content_id=cands[0]["content_id"], streamed_as=c.ID)
    return entry


def build() -> dict:
    db = open_copy()
    local = [c for c in db.get_content() if is_local(c)]
    by_id = {c.ID: c for c in db.get_content()}
    index = FileIndex(local)
    history = db.query(DjmdSongHistory).all()
    out = []
    for s in SETS + drill_sets():
        path = REC / s["file"]
        start, dur = (
            (None, sf.info(path).duration) if s["device"] != "rekordbox" else recording_times(path, s.get("start"))
        )
        rec = {
            "id": s["id"],
            "file": str(path),
            "device": s["device"],
            "duration": round(dur, 2),
            "start": start.isoformat(timespec="seconds") if start else None,
            "note": s.get("note"),
            "tracks": [],
        }
        if s["tracklist"] == "history":
            # t_hist: when Rekordbox logged the track, ~43 s after play started. Rows up to 15 min before
            # the recording started can be the opening track.
            rows = sorted(
                (
                    r
                    for r in history
                    if start - dt.timedelta(minutes=15) <= r.created_at <= start + dt.timedelta(seconds=dur)
                ),
                key=lambda r: r.created_at,
            )
            rec["histories"] = sorted({r.History.Name for r in rows})
            for r in rows:
                e = track_info(r.Content) | {"t_hist": round((r.created_at - start).total_seconds(), 1)}
                rec["tracks"].append(resolve(e, r.Content, index))
        elif s["tracklist"].startswith("usb:"):
            lib = OneLibrary(str(ROOT / s["tracklist"][4:]))
            h = next(iter(lib.get_history()))
            for hc in sorted(h.songs, key=lambda x: x.sequenceNo):
                c = by_id[str(hc.content.masterContentId)]
                rec["tracks"].append(resolve(track_info(c) | {"t_hist": None}, c, index))
        elif s["tracklist"] == "cue":
            # t_hist: the .cue's INDEX time, when the recorder logged the track (same clock as history rows)
            by_path = {c.FolderPath: c for c in local}
            for f, t in read_cue(path.with_suffix(".cue")):
                c = by_path[f]
                rec["tracks"].append(resolve(track_info(c) | {"t_hist": float(t)}, c, index))
        elif s["tracklist"].startswith("file:"):
            by_path = {c.FolderPath: c for c in local}
            for x in json.loads((ROOT / s["tracklist"][5:]).read_text())["tracks"]:
                c = by_path[x["path"]]
                rec["tracks"].append(resolve(track_info(c) | {"t_hist": None}, c, index))
        else:
            pls = [p for p in db.get_playlist() if p.Name in s["pool"] and p.Songs]
            pool = {sp.Content.ID: sp.Content for p in pls for sp in p.Songs if sp.Content and is_local(sp.Content)}
            rec["pool"] = {"playlists": s["pool"], "size": len(pool)}
            rec["tracks"] = [
                track_info(c) | {"source": "collection", "path": c.FolderPath, "t_hist": None, "candidate": True}
                for c in pool.values()
            ]
        seen = set()
        for i, t in enumerate(rec["tracks"]):
            t["n"] = i
            t["repeat"] = t["content_id"] in seen
            seen.add(t["content_id"])
        out.append(rec)
    return {"built": dt.datetime.now().isoformat(timespec="seconds"), "sets": out}


def main() -> None:
    cat = build()
    old = DATA / "sets.json"
    if old.exists():  # renders are registered by enaicer.replay, not built here: keep them
        cat["sets"] += [s for s in json.loads(old.read_text())["sets"] if s["device"] == "render"]
    tmp = DATA / "sets.json.tmp"
    tmp.write_text(json.dumps(cat, indent=1))
    tmp.replace(DATA / "sets.json")  # atomic: alignment jobs may be reading it
    for s in cat["sets"]:
        tr = [t for t in s["tracks"] if not t["repeat"]]
        src = {k: sum(t["source"] == k for t in tr) for k in ("collection", "file", "unresolved")}
        print(f"{s['id']:22} {s['duration'] / 60:5.1f} min  {len(tr):3d} tracks  {src}  {s.get('histories', '')}")
        for t in tr:
            if t.get("stream"):
                c = t.get("candidates") or []
                print(
                    f"    stream: {t['artist']} - {t['title']}  ->  {Path(c[0]['path']).name if c else 'NOT FOUND'}"
                    + (f"  (+{len(c) - 1} more)" if len(c) > 1 else "")
                )


if __name__ == "__main__":
    main()
