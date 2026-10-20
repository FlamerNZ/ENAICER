"""Log every control EN4CER touches on the DDJ-400, as ground truth for how they mix.

It only listens: macOS lets several programs read one MIDI device, so Rekordbox works as normal. Each time the
controller is connected a new file starts, data/midi/<local start time>.jsonl: a header line, then one line per
MIDI message, {"t": unix time, "m": [status, data1, data2]}. Raw bytes are the record; `decode` names them.

    uv run python -m enaicer.midilog --install    # run as a launchd agent: logs whenever the DDJ is plugged in
    uv run python -m enaicer.midilog --watch      # show what the agent is logging, live
    uv run python -m enaicer.midilog --uninstall

Recordings give the play position; play presses in the log tie its clock to the recording's.

CoreMIDI only reports devices that were connected when the process first touched it (no run loop here), so the
agent waits for the DDJ on USB before opening MIDI, and exits when it's unplugged; launchd starts it again.
"""

import argparse
import datetime as dt
import json
import os
import plistlib
import queue
import subprocess
import sys
import time
from pathlib import Path

from enaicer.catalog import DATA, ROOT

DIR = DATA / "midi"
DEVICE = "DDJ-400"
LABEL = "com.enaicer.midilog"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"

# DDJ-400 MIDI map, from Mixxx's res/controllers/Pioneer-DDJ-400.midi.xml.
# Deck controls and the channel strip are on MIDI channel 0 (deck 1) and 1 (deck 2), beat FX on 4, the mixer on 6.
# 14-bit controls send the MSB on the CC below, then the LSB on CC + 0x20. When Rekordbox connects it asks the DDJ
# (SysEx) for its state, and the DDJ answers with every knob and fader's position, so a session starts known.
STRIP = {0x00: "tempo", 0x04: "trim", 0x07: "eq_high", 0x0B: "eq_mid", 0x0F: "eq_low", 0x13: "fader"}
MIXER = {0x1F: "crossfader", 0x17: "filter_1", 0x18: "filter_2", 0x0D: "headphone_level"}
FX = {0x02: "fx_level"}
CC14 = {0: STRIP, 1: STRIP, 4: FX, 6: MIXER}
JOGS = {0x21: "jog_side", 0x22: "jog_top_vinyl", 0x23: "jog_top", 0x29: "jog_search"}  # relative, 64 = no move
BUTTONS = {
    0x0B: "play", 0x0C: "cue", 0x10: "loop_in", 0x11: "loop_out", 0x36: "jog_touch", 0x3F: "shift",
    0x47: "shift_play", 0x48: "shift_cue", 0x4D: "reloop", 0x54: "headphone_cue", 0x58: "sync",
    0x5C: "sync_hold", 0x60: "tempo_range", 0x67: "shift_jog_touch", 0x68: "quantize",
}  # fmt: skip
LOAD = {0x46: 1, 0x47: 2}  # notes on channel 6
FX_BUTTONS = {
    0x10: "fx_to_deck1", 0x11: "fx_to_deck2", 0x14: "fx_to_master", 0x47: "fx_on", 0x43: "shift_fx_on",
    0x4A: "fx_beat_left", 0x4B: "fx_beat_right", 0x63: "fx_select", 0x64: "shift_fx_select",
}  # fmt: skip
PAD_CH = {7: (1, ""), 8: (1, "shift_"), 9: (2, ""), 10: (2, "shift_")}
PAD_MODES = {0x00: "hotcue", 0x20: "beatjump", 0x30: "sampler", 0x60: "beatloop"}  # 8 pads from each base note


class Decoder:
    """Raw messages -> (deck, control, value) events; deck 0 is the mixer section.

    Faders and knobs are 0..1. The tempo slider too: 0 is the top, which slows the track (checked with EN4CER); its
    range is a Rekordbox setting. Jogs are signed ticks; buttons 1 on press and 0 on release (the FX channel switch:
    1 where it's set). SysEx and unknown messages come back as ("sysex" or "?", hex)."""

    def __init__(self):
        self.msb: dict[tuple[int, int], int] = {}

    def __call__(self, m: list[int]) -> tuple[int, str, float | str]:
        kind, ch = m[0] & 0xF0, m[0] & 0x0F
        if kind == 0xB0 and len(m) == 3:
            cc, v = m[1], m[2]
            names = CC14.get(ch, {})
            if cc in names:
                self.msb[ch, cc] = v
                return self._deck(ch), names[cc], (v << 7) / 16383
            if cc - 0x20 in names:
                return self._deck(ch), names[cc - 0x20], (self.msb.get((ch, cc - 0x20), 0) << 7 | v) / 16383
            if ch in (0, 1) and cc in JOGS:
                return ch + 1, JOGS[cc], v - 64
        if kind in (0x90, 0x80) and len(m) == 3:
            note, on = m[1], int(kind == 0x90 and m[2] > 0)
            if ch in (0, 1) and note in BUTTONS:
                return ch + 1, BUTTONS[note], on
            if ch == 6 and note in LOAD:
                return LOAD[note], "load", on
            if ch == 4 and note in FX_BUTTONS:
                return 0, FX_BUTTONS[note], on
            if ch in PAD_CH:
                deck, shift = PAD_CH[ch]
                base = max((b for b in PAD_MODES if b <= note), default=None)
                if base is not None and note - base < 8:
                    return deck, f"{shift}{PAD_MODES[base]}_{note - base + 1}", on
        return 0, "sysex" if m[0] == 0xF0 else "?", " ".join(f"{b:02X}" for b in m)

    @staticmethod
    def _deck(ch: int) -> int:
        return ch + 1 if ch in (0, 1) else 0


def decode(path: Path) -> list[tuple[float, int, str, float | str]]:
    """(unix time, deck, control, value) for every message in a session file."""
    d = Decoder()
    out = []
    for line in path.read_text().splitlines()[1:]:
        r = json.loads(line)
        out.append((r["t"], *d(r["m"])))
    return out


def plugged_in() -> bool:
    usb = subprocess.run(["ioreg", "-p", "IOUSB", "-w0"], check=False, capture_output=True, text=True).stdout
    return DEVICE.lower() in usb.lower()


def log() -> None:
    """Wait for the DDJ, log it until it's unplugged, then exit (launchd restarts the agent)."""
    print(f"{dt.datetime.now():%F %T} waiting for {DEVICE}", flush=True)
    while not plugged_in():
        time.sleep(2)
    time.sleep(2)  # let CoreMIDI register it before the first (and only) look
    import mido  # first CoreMIDI contact: only now, so the device list includes the DDJ

    port = next((n for n in mido.get_input_names() if DEVICE.lower() in n.lower()), None)
    if port is None:
        sys.exit(f"{DEVICE} is on USB but has no MIDI input; inputs: {mido.get_input_names()}")
    q: queue.SimpleQueue = queue.SimpleQueue()
    DIR.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.now()
    path = DIR / f"{start:%Y-%m-%d_%H%M%S}.jsonl"
    n, last_check, missing = 0, time.time(), 0
    with mido.open_input(port, callback=lambda msg: q.put((time.time(), msg.bytes()))), path.open("a") as f:
        f.write(json.dumps({"session": start.isoformat(timespec="seconds"), "port": port, "t": time.time()}) + "\n")
        f.flush()
        print(f"{start:%F %T} logging {port} -> {path.relative_to(ROOT)}", flush=True)
        while True:
            try:
                t, m = q.get(timeout=0.25)
                f.write(json.dumps({"t": round(t, 6), "m": m}) + "\n")
                n += 1
                while not q.empty():
                    t, m = q.get_nowait()
                    f.write(json.dumps({"t": round(t, 6), "m": m}) + "\n")
                    n += 1
                f.flush()
            except queue.Empty:
                pass
            if time.time() - last_check > 5:
                last_check = time.time()
                missing = 0 if plugged_in() else missing + 1
                if missing >= 2:
                    break
    print(f"{dt.datetime.now():%F %T} unplugged: {n} messages in {path.name}", flush=True)


def watch() -> None:
    """Follow the newest session file and print what moves (reads the file only; the agent owns the device)."""
    files = sorted(DIR.glob("*.jsonl"))
    if not files:
        sys.exit(f"no sessions in {DIR.relative_to(ROOT)} yet: is the agent installed and the DDJ plugged in?")
    path = files[-1]
    print(f"watching {path.relative_to(ROOT)} (Ctrl-C to stop)")
    d, state, jog = Decoder(), {}, {}
    with path.open() as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                for deck, (ticks, t0, t1) in list(jog.items()):
                    if time.time() - t1 > 0.4:
                        _say(f"D{deck[0]} {deck[1]} {ticks:+d} ticks over {t1 - t0:.1f} s", state)
                        del jog[deck]
                newer = max(DIR.glob("*.jsonl"))
                if newer != path:
                    return watch()
                time.sleep(0.05)
                continue
            r = json.loads(line)
            deck, name, v = d(r["m"])
            if name.startswith("jog") and name != "jog_touch":
                ticks, t0, _ = jog.get((deck, name), (0, r["t"], 0))
                jog[deck, name] = (ticks + int(v), t0, r["t"])
            elif isinstance(v, float):
                state[deck, name] = v
                _say(None, state)
            elif name in ("?", "sysex"):
                _say(f"{name} {v[:48]}", state)
            elif v == 1 or name in ("headphone_cue", "jog_touch", "shift"):
                _say(f"D{deck} {name}{'' if v == 1 else ' off'}" if deck else f"{name}", state)


def _say(event: str | None, state: dict) -> None:
    def pct(deck, name):
        v = state.get((deck, name))
        return " --" if v is None else f"{round(100 * v):3d}"

    strip = [
        f"D{k} fader{pct(k, 'fader')} hi{pct(k, 'eq_high')} mid{pct(k, 'eq_mid')} lo{pct(k, 'eq_low')}" for k in (1, 2)
    ]
    line = f"{strip[0]} | xf{pct(0, 'crossfader')} | {strip[1]}"
    if event:
        print(f"\r\033[K{dt.datetime.now():%T} {event}")
    print(f"\r\033[K{line}", end="", flush=True)


def install() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "enaicer.midilog"],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",  # timestamps matter: no background timer coalescing
        "StandardOutPath": str(DIR / "agent.log"),
        "StandardErrorPath": str(DIR / "agent.log"),
    }
    uninstall(quiet=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_bytes(plistlib.dumps(plist))
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)], check=True)
    print(f"installed {PLIST}; log: {(DIR / 'agent.log').relative_to(ROOT)}")


def uninstall(quiet: bool = False) -> None:
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], check=False, capture_output=True)
    if PLIST.exists():
        PLIST.unlink()
    if not quiet:
        print(f"removed {LABEL}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--install", action="store_true", help="run as a launchd agent whenever you're logged in")
    g.add_argument("--uninstall", action="store_true")
    g.add_argument("--watch", action="store_true", help="print what the agent is logging, live")
    a = ap.parse_args()
    if a.install:
        install()
    elif a.uninstall:
        uninstall()
    elif a.watch:
        try:
            watch()
        except KeyboardInterrupt:
            print()
    else:
        log()


if __name__ == "__main__":
    main()
