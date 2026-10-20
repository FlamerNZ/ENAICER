# AI DJ Style: handoff from the Dub-edit session (2026-10-18)

**Goal:** analyze how EN4CER (the user) actually mixes, using their own recorded sets, and eventually render new mixes that follow that style. This is a separate thread from the Mixcloud Dub edit, which is still in progress in another Claude session. **Treat `~/Music/rekordbox/Recording/EN4CER/Dub Edit/` as read-only**: the user is making picks there and the other session will assemble the edit from it.

Background on the user's library, Rekordbox DB access (pyrekordbox 0.4.4 + key), and safety rules is in memory: `rekordbox_library_analysis.md` and `dub_playlist_taste.md`.

## What exists today

### Recordings (both Dub sets, UK 140 dub, everything synced to 140 BPM)
| | Path | Length | Tracks | Source |
|---|---|---|---|---|
| Practice 2 | `~/Music/rekordbox/Recording/EN4CER/Dub(2)/01 Dub Practice 2.wav` | 62:43 (end silence trimmed; untrimmed copy in `~/Music/rekordbox_backups_claude/recordings/`) | 29 | Rekordbox recorder, 17 Oct 15:10. Order = Rekordbox history 4076677843 (`data/practice_hist.json`) |
| Live | `~/Music/rekordbox/Recording/EN4CER/Dub Live/Dub Live 2026-10-17 (XDJ-AZ REC001).wav` | 53:13, missed the first ~6 min (starts mid-Energy) | 27 (25 recorded) | AlphaTheta XDJ-AZ USB recording; order from the XDJ-AZ history in the backup USB's `exportLibrary.db` |

- Practice is −10.7 LUFS with **no clipping**. Live is −11.2 LUFS, peaks 0 dBFS, with ~700 short clip runs (master in the red, worst at 15–17 and 38–41 min).
- Native BPMs that differ from 140, so tracks play at rate = 140/BPM: Don't Stop 142, I Run 138, I Miss U 144, dontwaiT 136, THE FIT 136, Criminal 142, Badman Season 145, Drop In 145, PFTD 139.95.

### Alignment data (`data/`)
- `align_{live,practice}.json`: per track: `offset` (recording time of stretched-track t=0), `rate`, votes, `path`, `presence` (curve).
  - Mapping: recording_time = offset + original_track_time / rate.
- `trans_{live,practice}.json`: per transition: `b_in` (incoming audible), `a_out` (outgoing gone), in recording seconds. Also includes tracks with presence curves.
- `pl_tracks.json` (Dub playlist with local file paths), `bpm.json` (content ID → BPM), `practice_hist.json`.

### Scripts (`scripts/`)
These hard-code the old session's scratchpad paths in `SP=`, so repoint them to `data/`.
- `align_both.py`:
  - Features: 11025 Hz mono; log-mel with 40 bands (40–5500 Hz) and hop 256 (≈23 ms); then positive spectral flux per band, which captures rhythm/melody patterns.
  - For each track: time-stretch its features by 140/BPM, cut 8 s snippets every 4 s, and run normalized cross-correlation (averaged per band) against a window of the recording. The offset is the mode of the snippet offsets.
- `transitions.py`: presence = per-band Pearson correlation in 8 s windows, searching ±13 frames of lag; threshold 0.35. `b_in` / `a_out` come from where the curve stays above the threshold.
- `build_compare.py`: clips and stats for the comparison page (blend length, level change during the blend versus either side, clip runs).

**Lessons:**
- Comparing spectral shape (cosine of log-mel frames) does NOT tell dub tracks apart. Use onset-flux time patterns.
- Rounding offsets to 0.25 s ruins flux correlation. Search a small lag window instead.
- Tracks not at 140 must be time-stretched before matching.

### Style facts measured so far
- Everything is synced to 140; tracks off 140 are played at their stretched rate, so there are no tempo changes or ramps.
- **Blend length**:
  - Practice: median 23.1 s (IQR 15–32 s).
  - Live: median 23.4 s (IQR 13–29 s).
  - That's about 13.5 bars at 140 (1 bar = 1.714 s). These are probably 16-bar blends whose quiet edges fall under the detection threshold. Confirm this against beat grids.
- Live blends are slightly shorter and tighter. A few blends are long, 45–75 s: TAKA→Easy for them, Leave→THINGS SHE SAID (practice), Skanka→Badman (live).
- Playlist order follows the Camelot wheel DESCENDING (5A→3B→2A→1A→12A→9A→8A→7B→7A→6A→5B). The user treats the playlist as a pool.

## Research findings (what others have published)
No one has published using Claude or another LLM to learn one DJ's style from their own recordings and render mixes in it. The closest work:
- **Kim, Choi, Sacks, Yang, Nam: "A Computational Analysis of Real-World DJ Mixes using Mix-To-Track Subsequence Alignment"**, ISMIR 2020.
  - Aligns 1,557 mixes (20,765 transitions) to their source tracks using subsequence DTW. Measures cue points, blend lengths, and tempo/key changes.
  - Same idea as our alignment. https://arxiv.org/pdf/2008.10267 · https://mir-aidj.github.io/djmix-analysis/ · https://mac.kaist.ac.kr/ai_dj.html
- **Kim, Yang, Nam: "Joint Estimation of Fader and Equalizer Gains of DJ Mixers using Convex Optimization"**, DAFx 2022.
  - Estimates time-varying fader and EQ gains per transition from the aligned sources. Dataset of 5,040 mixes / 50,742 transitions.
  - **This is the natural next step for us.** https://mac.kaist.ac.kr/pubs/KimYangNam-dafx2022.pdf · dataset: https://github.com/mir-aidj/djmix-dataset
- **DJ Mix Transcription with Multi-Pass NMF** (2024): another gain/EQ estimation approach. https://arxiv.org/pdf/2410.04198
- **DJtransGAN** (Chen et al., ICASSP 2022): a GAN that generates transitions using differentiable EQ and fader, learned from many DJs' mixes (a crowd style, not one DJ). https://arxiv.org/abs/2110.06525 · https://github.com/ChenPaulYu/DJtransGAN
- **Cue point estimation using object detection** (2024): predicts where DJs put cue points. https://arxiv.org/pdf/2407.06823
- **DJ-AI** (ACM, Dec 2025): playlist ordering, plus transitions generated with MERT embeddings and MusicGen. https://dl.acm.org/doi/10.1145/3771594.3771640
- Vande Veire & De Bie: an automated DJ system for drum & bass (2018). https://www.researchgate.net/publication/327851916
- **Claude projects:**
  - claude-dj: Claude Code drives a browser DJ rig with fixed transition types (blend/bass swap/echo/filter…). https://github.com/tomek-i/claude-dj
  - DJ AI System: library organization with Claude + pyrekordbox. https://dj-ai-system.vercel.app/
  - A Claude Code skill that organizes a DJ library: https://gist.github.com/djhallaaaa/cd68f7f3458384f43aea920c3793d2e5
  - AIDJSets (set building): https://aidjsets.com/
- **Commercial:** DJ.Studio uses rule-based transition presets on the beat grid. https://dj.studio/blog/ai-mixing-apps
- Pop articles such as djing.ai's "Train AI to match your mixing style" are speculative; they name no real tool or study.

## Original plan (2026-10-18)

All four steps were built: `transitions.py` (1), `gains.py` and `blend.py` (2), `profile.py` (3), and `plan.py` and `render.py` (4). From 2026-10-19, a DDJ-400 logger (`midilog.py`, `drills.py`) replaced the fitted curves as the source of the blend template. See the README for the pipeline as it is now, and `docs/STYLE.md` for the findings.

1. **Beat/phrase-level timing.**
   - Read each track's Rekordbox beat grid by parsing the ANLZ .DAT `PQTZ` directly. `db.read_anlz_files` crashes, so use `AnlzFile.parse_file(~/Library/Pioneer/rekordbox/share + content.AnalysisDataPath)`.
   - Express each mix-in and mix-out as a bar/phrase number in both tracks: does the user start on 16/32-bar phrases, how far into the incoming track, and where in the outgoing track?
   - Measure blend length in bars.
2. **Fader + 3-band EQ gain curves per transition**, following DAFx 2022.
   - With both source tracks aligned, fit time-varying gains (low <~200 Hz, mid, high) per deck in the STFT domain by constrained least squares.
   - That shows bass-swap timing (the key dub-mixing move), whether highs are pre-cut, and fade shapes. Validate by reconstruction error.
   - Live has clipping, so prefer practice for fitting, or mask clipped frames.
3. **Style profile:** distributions, not single numbers, e.g. "bass swap at bar 8–9 of a 16-bar blend; incoming highs cut until bar 4".
4. **Render:** take a new ordered list, pick mix points by the profile on phrase boundaries, time-stretch to 140 (rubberband/ffmpeg), and apply the gain curves offline. Produce a WAV for the user to judge by ear.
   - Be honest that about 50 transitions captures habits, not judgment.

## Rules (from memory; still apply)
- Never write to master.db while Rekordbox is running. pgrep right before writing, back up master.db + -wal + -shm, and run an integrity check after any force-quit.
- Read DB copies when Rekordbox is open.
- Don't put credentials in chat. The user buys their own music and prefers legal purchase.
- The user's manual color labels are ground truth.
