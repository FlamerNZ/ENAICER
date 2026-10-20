# ENAICER

Learn how one DJ (EN4CER) mixes, from their own recorded sets and their hands on the controller, then render new mixes in that style.

- **Recordings:** every track is found inside a recorded set, and each blend is measured: where it starts on the phrase grid, and each deck's fader and 3-band EQ gains over time.
- **Controller:** a logger records every knob and fader move on a Pioneer DDJ-400 while they practise. Those moves are lined up with the recording, so the blend template comes from what their hands did.
- **Mixes:** sets are planned from Rekordbox playlists (order by key and tempo, the part of each track to play, where each blend sits), rendered with Rubber Band, and reviewed by ear on local pages, including blind A/B tests.

What has been learned so far is in [`docs/STYLE.md`](docs/STYLE.md).

This is a personal project built around one library. It reads a local Rekordbox library and recordings, and the recordings it learns from are configured in `catalog.py`. Everything derived from the library (`data/`, `out/`) stays out of git.

## Requirements

- macOS with Rekordbox 7. The library is only ever read, through a copy of `master.db`.
- [uv](https://docs.astral.sh/uv/), plus `ffmpeg` and `rubberband` on `PATH` (`brew install ffmpeg rubberband`).
- Optional: [Demucs](https://github.com/adefossez/demucs) vocal stems, used to avoid clashing vocals. They're made in an isolated tool environment; the command is in `vocals.py`.
- Optional: a Pioneer DDJ-400 for the controller logger.

```sh
uv sync
uv run pytest
```

## Pipeline

Analyse the recorded sets:

1. `uv run python -m enaicer.catalog` builds `data/sets.json`: each recorded set, its tracklist in play order, and the audio file for each track.
2. `uv run python -m enaicer.align [set_id ...]` finds each track in the recording: playback rate, offset, and drift every 2 s.
3. `uv run python -m enaicer.keylock [set_id ...]` checks from the audio whether key lock was on.
4. `uv run python -m enaicer.transitions [set_id ...]` measures each blend: phrase position, per-deck band gains, bass swap.
5. `uv run python -m enaicer.memory` records where they mix out of each track.

Learn from the controller:

6. `uv run python -m enaicer.midilog --install` runs a launchd agent that logs the DDJ-400 whenever it's plugged in. It only listens, so Rekordbox works as normal.
7. `uv run python -m enaicer.drills` lines each logged practice recording up with its log: fader and EQ moves per blend, in bars.

Make a mix:

8. `uv run python -m enaicer.plan plan.json <playlist id> [...] --tempo 125:128` plans a set. A playlist can also be a track list from `enaicer.pick`, which selects local tracks by colour label, tempo and key.
9. `uv run python -m enaicer.render plan.json out_dir` renders it.
10. `uv run python -m enaicer.replay` and `enaicer.review` measure the render the same way as a real recording and build a review page with per-transition feedback.
11. `uv run python -m enaicer.abtest` makes blind A/B clips of drilled blends, to choose between model settings by ear.

## Layout

- `src/enaicer/`
  - `rekordbox.py`: read-only Rekordbox access and beat grids
  - `audio.py`, `structure.py`, `vocals.py`: features, true beat positions, bar energy, drops, vocals
  - `catalog.py`, `align.py`, `keylock.py`, `gains.py`, `blend.py`, `transitions.py`, `memory.py`: analysis of recordings
  - `midilog.py`, `drills.py`: the controller logger and drill analysis
  - `profile.py`, `pick.py`, `plan.py`, `render.py`: the blend template, track picking, planning and rendering
  - `replay.py`, `review.py`, `abtest.py`, `*_page.html`: review and A/B pages
- `scripts/`: the first session's one-off scripts, kept as they ran
- `docs/STYLE.md`: the mixing style, in EN4CER's words and as measured
- `docs/HANDOFF.md`: background, related research and the original plan
- `CLAUDE.md`: working rules for Claude Code sessions on this project

## License

[MIT](LICENSE)
