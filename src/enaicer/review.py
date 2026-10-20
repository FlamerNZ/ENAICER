"""Build a local review page for a rendered mix: player, tracklist, every transition with the model's blend
curves (measured back from the render) next to EN4CER's own blend into the same track in a reference
recording, quick A/B buttons, and per-transition feedback that exports as JSON.

    uv run python -m enaicer.review <render dir> <render set id> [<reference set id>]
    python3 <render dir>/review/serve.py   # prints the URL to open
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from enaicer.catalog import DATA

PAGE = Path(__file__).with_name("review_page.html")
SERVE = """# Local server for the review page, with HTTP Range support (needed to seek in long audio).
import http.server, json, os, re, socketserver
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))  # serve the folder holding all renders, so references are reachable
os.chdir(root)
class H(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path); rng = self.headers.get("Range")
        if not rng or not os.path.isfile(path): return super().send_head()
        m = re.match(r"bytes=(\\d*)-(\\d*)", rng); size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = min(int(m.group(2)) if m.group(1) and m.group(2) else size - 1, size - 1)
        f = open(path, "rb"); f.seek(start)
        self.send_response(206); self.send_header("Content-Type", self.guess_type(path)); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}"); self.send_header("Content-Length", str(end - start + 1)); self.end_headers()
        self._remaining = end - start + 1; return f
    def copyfile(self, src, dst):
        n = getattr(self, "_remaining", None)
        if n is None: return super().copyfile(src, dst)
        while n > 0:
            b = src.read(min(65536, n))
            if not b: break
            dst.write(b); n -= len(b)
    def do_POST(self):
        # the page saves feedback here as you click: <root>/feedback/<id>.json
        m = re.fullmatch(r"/feedback/([A-Za-z0-9_-]{1,80})", self.path)
        n = int(self.headers.get("Content-Length") or 0)
        if not m or n > 2_000_000:
            self.send_error(400); return
        body = self.rfile.read(n)
        try: json.loads(body)
        except ValueError: self.send_error(400); return
        os.makedirs("feedback", exist_ok=True)
        with open(os.path.join("feedback", m.group(1) + ".json"), "wb") as f: f.write(body)
        self.send_response(204); self.end_headers()
    def log_message(self, *a): pass
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True
print("open http://127.0.0.1:8767/" + os.path.relpath(here, root) + "/")
S(("127.0.0.1", 8767), H).serve_forever()
"""


def rms_db(path: str) -> np.ndarray:
    """1 s RMS (dB) of a recording."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    n = len(x) // sr * sr
    return 20 * np.log10(np.sqrt((x[:n].reshape(-1, sr, x.shape[1]) ** 2).mean(axis=(1, 2))) + 1e-9)


def level_change(r: np.ndarray, c: float) -> float:
    """Blend level vs the average of 30 s windows either side, around takeover time c (s)."""
    c = int(c)
    seg = lambda a, b: float(np.mean(r[max(0, a) : max(max(0, a) + 1, b)]))
    return round(seg(c - 20, c + 10) - (seg(c - 60, c - 30) + seg(c + 30, c + 60)) / 2, 1)


def slim(t: dict) -> dict:
    """Measured transition, cut down for the page (curves every ~0.5 s)."""
    keep = ("b_in_bars", "a_out_bars", "blend_bars", "swap_order", "b_entry_db", "bass_swap_bars", "a_played_out",
            "gone_before_quiet", "b_bar1_rec", "bpm_a")  # fmt: skip
    out = {k: t.get(k) for k in keep}
    c = t.get("curves")
    if c:
        out["curves"] = {k: v for k, v in c.items()}
    return out


def build(render_dir: Path, set_id: str, ref: str | None) -> Path:
    plan = json.loads((render_dir.parent / f"plan_{render_dir.name}.json").read_text())
    meta = json.loads((render_dir / "render.json").read_text())
    mf = DATA / "transitions" / f"{set_id}.json"  # measured back from the render; the page works without it
    measured = json.loads(mf.read_text())["transitions"] if mf.exists() else []
    by_b = {t["b"]: t for t in measured}
    r_mix = rms_db(str(render_dir / "mix.wav"))
    page = render_dir / "review"
    page.mkdir(exist_ok=True)
    data = {"title": f"Model mix: {render_dir.name}", "mix": "../mix.mp3", "length_s": meta["length_s"], "id": set_id,
            "built": datetime.now().strftime("%H:%M")}  # fmt: skip
    data["tracks"] = [
        {"title": t["title"], "artist": t["artist"], "key": t["key"], "bpm": t["bpm"], "start_s": m["start_s"], "on_s": m["on_s"],
         "window": [t["first_bar"], t["last_bar"]], "bars": len(t["down"]) - 1, "tempo": [t["tempo_in"], t["tempo_out"]]}
        for t, m in zip(plan["tracks"], meta["tracks"])
    ]  # fmt: skip
    ref_data, ref_by_b, r_ref = None, {}, None
    if ref:
        rs = next(s for s in json.loads((DATA / "sets.json").read_text())["sets"] if s["id"] == ref)
        mp3 = render_dir.parent / f"ref_{ref}.mp3"
        if not mp3.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", rs["file"], "-b:a", "192k", str(mp3)], check=True)
        ref_by_b = {t["b"]: t for t in json.loads((DATA / "transitions" / f"{ref}.json").read_text())["transitions"]}
        r_ref = rms_db(rs["file"])
        ref_data = {"id": ref, "audio": f"../../{mp3.name}"}
    data["ref"] = ref_data
    data["transitions"] = []
    for i, x in enumerate(plan["transitions"]):
        m = by_b.get(x["b"])
        start = meta["tracks"][i + 1]["start_s"]
        bar = 240 / x["tempo"]
        item = {"i": i, "a": x["a"], "b": x["b"], "a_bar": x["a_bar"], "b_bar": x["b_bar"], "span": x["span_bars"],
                "tempo": x["tempo"], "key_cost": x["key_cost"], "start_s": start, "bar_s": bar,
                "level_db": level_change(r_mix, start + 16 * bar), "measured": slim(m) if m else None,
                "hat_ms": round(float(np.median(np.abs(m["hat_lag_ms"]))), 1) if m and m.get("hat_lag_ms") else None}  # fmt: skip
        rt = ref_by_b.get(x["b"])
        if rt:
            rb = 240 / rt["bpm_a"]
            item["ref"] = slim(rt) | {"a": rt["a"], "start_s": rt["b_bar1_rec"], "bar_s": rb,
                                      "level_db": level_change(r_ref, rt["b_bar1_rec"] + 16 * rb)}  # fmt: skip
        data["transitions"].append(item)
    (page / "review_data.js").write_text("window.REV = " + json.dumps(data) + ";\n")
    shutil.copy(PAGE, page / "index.html")
    (page / "serve.py").write_text(SERVE)
    return page


def main() -> None:
    page = build(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"built {page}/index.html\nserve: python3 {page}/serve.py")


if __name__ == "__main__":
    main()
