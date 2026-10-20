# EN4CER mixing style

## In their own words (2026-10-18)

- **Tempo:** no Sync. EN4CER matches the incoming track's tempo to the playing one by ear, listening to the hi-hats rather than the bass.
- **Mix-in point:** they pick the phrase to mix into, usually the start of the playing track's second drop.
  - They get the incoming track roughly in place with the cue button, hit play on that phrase, then correct with the jog wheel. They listen in the headphones to the incoming track with its highs turned up.
  - Then they cue the playing track too, to check that the two match and sound good together.
- **Blend**, in order:
  1. Incoming EQ starts with lows off, mids at medium and highs at about 2/3. They bring the channel fader up slowly.
  2. Mids swap: incoming goes from half to full, outgoing from full to half.
  3. Highs swap the same way, to about 2/3 each.
  4. Lows: incoming goes from 0 to 100, usually as a gradual blend like the mids and highs. Sometimes they swap the bass quickly on a quiet part of the tracks.
  5. The outgoing track leaves by its channel fader (they never use the crossfader). It must be fully down before any quiet part of the incoming track.
- **Key lock (Master Tempo):** they say "the pitch doesn't change when I change the tempo" and that they've never touched the setting. The audio says it's per deck:
  - In every set up to July 2026, one deck had key lock off. Tracks on that deck shift pitch with tempo, measured on 41 tracks, and a listening A/B confirms it.
  - In the October 2026 sets, both decks had it on.
  - This isn't a style choice, so it's modelled per track (`keylock` in `data/align/<set>.json`), not learned.
- **Melody Techno sets (Melody Techno .1, Miss Melody Techno):** an experiment in mixing without the faders, using only the high/mid/low EQ. In the data, the incoming track is audible from its first bar and the outgoing track tends to play to its end. Analyse these separately from the usual fader style.
- **Sync:** most older mixes used Beat Sync. The Oct 2026 Dub sets were beatmatched by ear, matching the hi-hats rather than the bass.
- **Selection:** mostly by key. They also test pairs in practice sessions, because some pairs just don't work well together.
- **Waiting (2026-10-19):** if the outgoing track is still in a good part and the incoming one hasn't reached a high-energy bit yet, they hold the incoming back. Santa Maria → Body Move: Body Move was started from bar 1 but faded in only at its bar 25, with the fader full at bar 33 as its bass dropped. Its bars 1–16 are little more than a kick.
- **Bass swap (2026-10-19):** "I don't always quick swap. We should prefer quick swaps when the outgoing track goes quiet at the same time the incoming track drops." Otherwise the lows blend gradually like the mids and highs. In the drills, swaps took from a quarter bar to 9 bars, usually about 20 bars after the incoming fader came up. In blind A/B tests of the drilled blends (2026-10-20, 7 picks), quick won exactly where the mix goes quiet at a 4-bar step: the outgoing's mids and highs fall 5+ dB (Body Move's breakdown; Good Measure as Sun is Shining's bass drops in), or the incoming's bass falls away for a moment (Santa Maria). Gradual won where the incoming drops while the outgoing carries on (The Bomb → All Good), or nothing changes.
- **Long tracks (2026-10-19):** mix out at the final drop before the outro, not the second drop. The Bomb's good part (full bassline and the vocal hook, bars 145–312) comes after the second drop.
- **Known mistake (don't learn it as style):** sometimes they run out of time on the outgoing track before the blend is finished, so the incoming track's intro plays on its own.

## Measured so far (Dub practice and live, 2026-10-17)

- **Track use:** the incoming track becomes audible about 18 s in (22 s live). The outgoing track is gone about 92% of the way through, with about 15 s left. About 82% of each track gets played.
- **Exits:** when they cut a long track early, they cut it in the same place on both nights. Swing King goes at about 200 of 269 s, TAKA at 192 of 209 s. Entry points vary more.
- **Blend length:** median about 23 s, probably 16 bars at 140. Live blends are slightly tighter. A few run 45–75 s.
- **Loudness:** practice blends come out about 1 dB quieter than the parts on either side, which suggests EQ cuts during the overlap. In the live set, 96% of blends clipped.
- **Order:** the set moves down the Camelot wheel. The live route nearly repeated the practice route.

## Measured on the DDJ-400 (drills, 2026-10-19)

From the controller log (`enaicer.midilog`) lined up with the recordings (`enaicer.drills`). Knob positions are 0–1 with 0.5 flat; bars are counted from where the incoming track was cued.

- **Cueing:** the incoming track is started on its phrase (bar 1, or bar 49 for Little More), with its highs up around 0.65–0.78 for the headphones.
- **Fade-in:** just before the fader comes up, the incoming highs go down to about 0.3 (their "2/3" means two-thirds of the way up to flat). Mids sit around 0.10–0.16 and lows at 0. The fader goes up at about bar 8 and is full by 13–16, unless they're waiting (see above).
- **Swaps:** mids at about bar 20 (incoming to 0.5, outgoing to about 0.2), highs at 20–24 (outgoing to about 0.3–0.45), lows at 21–32 (incoming 0 → 0.5 while outgoing 0.5 → 0), quick in four blends and over 4–9 bars in four. The outgoing fader is down about 32 bars after the incoming one came up.
- **Knob to dB** (knob positions against the band levels the recordings show): 0.4 ≈ −4 dB, 0.3 ≈ −6.5, 0.2 ≈ −8.5, 0.1 ≈ −14. The measurement bottoms out around −15 to −25 dB, so where a knob at 0 actually lands isn't known yet.
- **The blend itself:** in the blind A/B tests every pick went to these knob and fader moves over the template averaged from the recorded sets, and on the five pairs of round 2 a model version beat their own take.
- **EQ shape** (measured from the drill audio, each deck's level per third-octave over stretches where no knob moved): the bands are broad and overlap. Lows at 0 cut about −25 dB at 30–50 Hz, −10 at 125 Hz, −5 at 200 Hz and −2 at 300 Hz. The mids knob reaches up to 3–5 kHz, and the highs knob mostly acts above 5 kHz. Mids at 0.15 are about −9 dB.
- **Beatmatching:** in their own blends the two tracks' mid-band attacks (claps, snares, the click of the kick) land within 9 ms of each other in all 8 blends; their kicks can sit up to 16 ms apart, since they match on the hats. Music Is The Answer's Rekordbox grid is wrong: its real beat wanders up to about 180 ms off the grid, and they just follow the real beat.
