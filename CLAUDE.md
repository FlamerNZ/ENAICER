# ENAICER

Learn EN4CER's DJ mixing style from their recorded sets, then render new mixes in it. Background, measurements so far, related research and the plan are in `docs/HANDOFF.md`.

## Rules

- `~/Music/rekordbox/Recording/EN4CER/Dub Edit/` is read-only. Another session is assembling the Mixcloud edit there.
- Never write to `~/Library/Pioneer/rekordbox/master.db`. Copy `master.db` (plus `-wal`/`-shm` if present) to a temp dir and read the copy.
- Never commit audio, `data/` or DB copies. They describe the user's personal library and purchased music.
- The user's manual color labels in Rekordbox are ground truth.

## Environment

- `uv run python ...` (Python 3.13). `ffmpeg` is at `/opt/homebrew/bin`; it has no rubberband filter.
- Open the DB copy with `pyrekordbox.masterdb.MasterDatabase(path=copy)`. pyrekordbox 0.5.0 has the key built in.
- Read an XDJ-AZ USB DB (`PIONEER/rekordbox/exportLibrary.db`, copied first) with `pyrekordbox.onelibrary.OneLibrary(copy)`, using `get_history()`.
- Beat grids: `AnlzFile.parse_file(~/Library/Pioneer/rekordbox/share + content.AnalysisDataPath).get_tag("PQTZ")`. `db.read_anlz_files()` crashes on `.EXT`/`.2EX`.

## Data

- Recordings: all 13 sets (under `~/Music/rekordbox/Recording/EN4CER/`) and their tracklists are in `data/sets.json`, built by `enaicer.catalog`. `~/Music/rekordbox/Recording/REC001.WAV` is a duplicate of Miss Melody Techno.
- Three playlists are named "Dub". The one for these sets is root-level (ID 263764427).
- Alignment: `recording_time = offset + track_time / rate`, with `offset` and `rate` per track in `data/align/<set>.json` (v2). The files `data/align_{practice,live}.json` are the first session's v1 output. v1 placed Skanka (practice) and Energy (live) on repeated sections.
- `t_hist` in `data/sets.json` is when Rekordbox logged the track, not when it was loaded. Rows land a constant ~59 s after the track starts playing, ~43 s for Dub Practice 2 because its start time comes from a different source. A track far off that constant is worth a second look.
- Key lock is per deck and was not always on: one deck had it off in every set up to July 2026; both on from October 2026. `uv run python -m enaicer.keylock <set>` measures it and writes `keylock` per track into the alignment. Render sources accordingly: time-stretch where it's on, resample (pitch follows tempo) where it's off.
- `data/sources/xdj_az_usb_2026-10-17/` is the XDJ-AZ USB export DB the live track order came from.
