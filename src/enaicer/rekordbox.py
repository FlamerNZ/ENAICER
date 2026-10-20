"""Read-only access to the Rekordbox library, always through a copy of master.db."""

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from pyrekordbox.anlz import AnlzFile
from pyrekordbox.masterdb import MasterDatabase

RB_DIR = Path("~/Library/Pioneer/rekordbox").expanduser()
SHARE = RB_DIR / "share"

logging.getLogger("pyrekordbox").setLevel(logging.ERROR)  # copies lack masterPlaylists6.xml; harmless


def open_copy(dest: Path | None = None) -> MasterDatabase:
    """Copy master.db (+ -wal/-shm) into `dest` (default: a new temp dir) and open the copy."""
    dest = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="enaicer-rb-"))
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        src = RB_DIR / f"master.db{suffix}"
        if src.exists():
            shutil.copy2(src, dest / src.name)
    return MasterDatabase(path=dest / "master.db")


def is_local(content) -> bool:
    """True for a track backed by an audio file on disk (not a streaming entry like /v4/catalog/...)."""
    p = content.FolderPath or ""
    return p.startswith("/Users/") and Path(p).exists()


def beat_grid(content) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(beat times s, beat-in-bar 1..4, bpm) from the track's ANLZ .DAT PQTZ tag, or None.

    Parses the .DAT alone: db.read_anlz_files() crashes on newer .EXT/.2EX tags.
    """
    if not content.AnalysisDataPath:
        return None
    f = SHARE / content.AnalysisDataPath.lstrip("/")
    if not f.exists():
        return None
    q = AnlzFile.parse_file(f).get_tag("PQTZ")
    return np.asarray(q.get_times()), np.asarray(q.get_beats()), np.asarray(q.get_bpms())
