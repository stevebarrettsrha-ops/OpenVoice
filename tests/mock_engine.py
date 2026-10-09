"""
mock_engine.py - stands in for engine.py so the server can be tested with no
PyTorch, no checkpoints and no card.

Speaks the same JSON-lines protocol. `speak` writes short tone bursts (one per
line, pitch by line index) joined with the requested pause, so a test can
check the join without a model. Set MOCK_ENGINE_FAIL=speak to make that
command fail, or MOCK_ENGINE_CRASH=1 to exit mid-speak.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

_PROTO = sys.__stdout__       # same channel as engine.py
sys.stdout = sys.stderr
SR = 22050


def send(obj):
    _PROTO.write(json.dumps(obj) + "\n")
    _PROTO.flush()


def tone(path: Path, seconds: float, freq: float):
    n = int(SR * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / SR)))
                               for i in range(n)))
    return n


def main():
    send({"id": "", "event": "ready", "pid": os.getpid()})
    print("[mock] started", file=sys.stderr, flush=True)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        rid, cmd = req.get("id", ""), req.get("cmd")
        fail = os.environ.get("MOCK_ENGINE_FAIL", "")
        if cmd == fail:
            send({"id": rid, "ok": False, "error": f"mock failure in {cmd}"})
            continue
        if cmd == "hello":
            # MOCK_ENGINE_SLOW_HELLO=<seconds>: a PyTorch import that crawls
            time.sleep(float(os.environ.get("MOCK_ENGINE_SLOW_HELLO", "0") or 0))
            send({"id": rid, "ok": True, "result": {"torch": "0.0-mock", "cuda": False,
                                                    "device": "cpu", "gpu": "", "melo": True,
                                                    "device_override": os.environ.get("OPENVOICE_DEVICE", "")}})
        elif cmd == "status":
            send({"id": rid, "ok": True, "result": {"device": "cpu", "converter": "v2",
                                                    "vram_used": 0}})
        elif cmd == "load":
            send({"id": rid, "ok": True, "result": {"converter": req.get("version")}})
        elif cmd == "embed":
            cache = Path(req["cache_dir"])
            cache.mkdir(parents=True, exist_ok=True)
            se = cache / (Path(req["clip"]).stem + f"_{req.get('version')}_mock.pth")
            cached = se.exists()
            se.write_bytes(b"mock")
            send({"id": rid, "ok": True, "result": {"se": str(se), "cached": cached,
                                                    "speech_seconds": 4.2}})
        elif cmd == "speak":
            out = Path(req["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            files, total = [], 0
            lines = req.get("lines", [])
            pause = float(req.get("opts", {}).get("pause", 0.35))
            for i, ln in enumerate(lines):
                if os.environ.get("MOCK_ENGINE_CRASH") and i == 1:
                    print("[mock] crashing on purpose", file=sys.stderr, flush=True)
                    os._exit(3)
                send({"id": rid, "event": "progress", "line": i, "total": len(lines),
                      "stage": "tts", "msg": f"Line {i + 1}: base voice"})
                f = out / f"line{i:03d}.wav"
                n = tone(f, 0.3, 440 + 110 * i)
                files.append(str(f))
                total += n + (int(SR * pause) if i else 0)
                send({"id": rid, "event": "progress", "line": i, "total": len(lines),
                      "stage": "done", "msg": f"Line {i + 1}: 0.3s", "file": str(f),
                      "seconds": 0.3})
                time.sleep(0.02)
            with wave.open(str(out / "take.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                for i, f in enumerate(files):
                    if i:
                        w.writeframes(b"\x00\x00" * int(SR * pause))
                    with wave.open(f, "rb") as r:
                        w.writeframes(r.readframes(r.getnframes()))
            send({"id": rid, "ok": True, "result": {"file": str(out / "take.wav"),
                                                    "lines": files, "seconds": round(total / SR, 2),
                                                    "sample_rate": SR, "elapsed": 0.1}})
        elif cmd == "unload":
            send({"id": rid, "ok": True, "result": {"device": "cpu", "converter": ""}})
        elif cmd == "cancel":
            send({"id": rid, "ok": True, "result": {"cancelling": True}})
        elif cmd == "quit":
            send({"id": rid, "ok": True, "result": {}})
            break
        else:
            send({"id": rid, "ok": False, "error": f"unknown command {cmd}"})


if __name__ == "__main__":
    main()
