"""
server.py - OpenVoice Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7811)

The page in web/index.html talks to this over /api/*. The models never run in
this process: engine.py is started as a child and spoken to over pipes (see
EngineProcess), so it can be stopped to free the card, restarted after an
install, or lost to an out-of-memory without taking the page down.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import manager
from manager import APP_DIR, DATA_DIR, MODELS, TASKS, load_config, save_config

WEB_DIR = APP_DIR / "web"
VOICES_DIR = DATA_DIR / "voices"
VOICES_PATH = DATA_DIR / "voices.json"
SE_DIR = DATA_DIR / "se"
TAKES_DIR = DATA_DIR / "takes"
TAKES_PATH = DATA_DIR / "takes.json"
PORT = int(os.environ.get("OPENVOICE_STUDIO_PORT", "7811"))
ENGINE_SCRIPT = Path(os.environ.get("OPENVOICE_STUDIO_ENGINE") or (APP_DIR / "engine.py"))
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".webm"}
MAX_LINE_CHARS = 600
MAX_LINES = 200

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
cfg = load_config()
manager.ensure_tools_path()      # the ffmpeg the Engine page installed, if any

# The base voices each version offers, before any model is loaded. V2 reads the
# live speaker list from MeloTTS once a language is loaded, but the page needs
# the catalogue first. Keys match engine.py.
V1_BASES = {
    "EN": {"label": "English", "styles": ["default", "whispering", "shouting", "excited",
                                          "cheerful", "terrified", "angry", "sad", "friendly"]},
    "ZH": {"label": "Chinese", "styles": ["default"]},
}
V2_BASES = {
    "EN": {"label": "English", "speakers": {"EN-Default": "Default", "EN-US": "American",
                                            "EN-BR": "British", "EN_INDIA": "Indian",
                                            "EN-AU": "Australian"}},
    "EN_NEWEST": {"label": "English (newest model)", "speakers": {"EN-Newest": "Newest"}},
    "ES": {"label": "Spanish", "speakers": {"ES": "Spanish"}},
    "FR": {"label": "French", "speakers": {"FR": "French"}},
    "ZH": {"label": "Chinese", "speakers": {"ZH": "Chinese"}},
    "JP": {"label": "Japanese", "speakers": {"JP": "Japanese"}},
    "KR": {"label": "Korean", "speakers": {"KR": "Korean"}},
}


# --------------------------------------------------------------------------- #
# the engine process
# --------------------------------------------------------------------------- #
class EngineError(Exception):
    pass


def advice_for(text: str) -> str:
    """One sentence for the failures people actually hit."""
    t = text or ""
    if "WinError 126" in t or "c10.dll" in t or "shm.dll" in t or "fbgemm.dll" in t:
        return ("PyTorch's DLLs would not load: install the Microsoft Visual C++ "
                "Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) and "
                "restart the engine.")
    if "No module named 'torch'" in t:
        return "PyTorch is not installed in this app's environment: Engine page, PyTorch, Install."
    if "No module named" in t:
        return "A package is missing: press Install on 'OpenVoice packages' in the Engine page."
    if "CUDA driver version is insufficient" in t or "cudaErrorInsufficientDriver" in t:
        return ("The NVIDIA driver is older than this PyTorch build needs: update the "
                "driver, or reinstall PyTorch picking an older CUDA build.")
    if "out of memory" in t.lower():
        return "The card ran out of memory: close other GPU programs or turn on 'Free GPU memory after each take'."
    return ""


class EngineProcess:
    """engine.py as a child, spoken to one command at a time."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.state = "stopped"          # stopped | starting | ready | error
        self.error = ""
        self.hello: dict = {}
        self.console: list[str] = []
        self.pending: dict[str, dict] = {}
        self.lock = threading.RLock()       # one command in flight
        self.write_lock = threading.Lock()
        self.started = 0.0

    # -- console ------------------------------------------------------------
    def note(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.console.append(line)
        if len(self.console) > 3000:
            del self.console[:1500]
        print(f"[engine] {msg}", flush=True)

    def log_view(self, since: int = 0) -> dict:
        return {"state": self.state, "error": self.error, "cursor": len(self.console),
                "lines": self.console[since:], "hello": self.hello,
                "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
                "uptime": round(time.time() - self.started) if self.proc else 0}

    # -- lifecycle ----------------------------------------------------------
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        with self.lock:
            if self.alive():
                return
            if not ENGINE_SCRIPT.is_file():
                raise EngineError(f"engine script missing: {ENGINE_SCRIPT}")
            self.state = "starting"
            self.error = ""
            self.hello = {}
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            dev = cfg.get("device", "auto")
            if dev in ("cpu", "cuda"):
                env["OPENVOICE_DEVICE"] = dev
            else:
                env.pop("OPENVOICE_DEVICE", None)
            self.note(f"Starting the engine: {sys.executable} {ENGINE_SCRIPT.name}")
            self.proc = subprocess.Popen(
                [sys.executable, str(ENGINE_SCRIPT)], cwd=str(APP_DIR), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            self.started = time.time()
            self._stderr_thread = threading.Thread(target=self._read_stderr, args=(self.proc,),
                                                   daemon=True)
            self._stderr_thread.start()
            threading.Thread(target=self._read_stdout, args=(self.proc,), daemon=True).start()
            deadline = time.time() + 30
            while time.time() < deadline and self.state == "starting" and self.alive():
                time.sleep(0.1)
            if not self.alive():
                self.state = "error"
                self.error = self.error or "The engine exited while starting. See the console."
                raise EngineError(self.error)
            try:
                self.hello = self.call("hello", timeout=600)
            except EngineError as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            self.note("Engine ready: PyTorch " + str(self.hello.get("torch")) +
                      (f" on {self.hello.get('gpu')}" if self.hello.get("cuda")
                       else " (CPU)"))

    def stop(self) -> None:
        with self.lock:
            p = self.proc
            if p is None:
                self.state = "stopped"
                return
            if p.poll() is None:
                try:
                    self._write({"id": "q", "cmd": "quit"})
                    p.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    p.kill()
                    p.wait(timeout=5)
            for pipe in (p.stdin, p.stdout, p.stderr):
                try:
                    if pipe:
                        pipe.close()
                except OSError:
                    pass
            self.proc = None
            self.state = "stopped"
            self.hello = {}
            for fut in list(self.pending.values()):
                fut["error"] = "The engine stopped"
                fut["event"].set()
            self.note("Engine stopped")

    def restart(self) -> None:
        self.stop()
        self.start()

    def ensure(self) -> None:
        if not self.alive() or self.state != "ready":
            if self.alive():
                return
            self.start()

    # -- protocol -----------------------------------------------------------
    def _write(self, obj: dict) -> None:
        if not self.alive() or self.proc is None or self.proc.stdin is None:
            raise EngineError("The engine is not running")
        with self.write_lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                self.note(raw)
                continue
            if msg.get("event") == "ready":
                self.state = "ready"
                continue
            fut = self.pending.get(msg.get("id", ""))
            if fut is None:
                continue
            if msg.get("event") == "progress":
                cb = fut.get("on_progress")
                if cb:
                    try:
                        cb(msg)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if msg.get("ok"):
                fut["result"] = msg.get("result")
            else:
                fut["error"] = msg.get("error") or "unknown error"
                fut["cancelled"] = bool(msg.get("cancelled"))
            fut["event"].set()
        if self.proc is proc:
            code = proc.poll()
            if self.state != "stopped":
                self.state = "error"
                self.error = f"The engine exited (code {code}). {self.last_complaint()}".strip()
                self.note(self.error)
            for fut in list(self.pending.values()):
                fut["error"] = self.error or "The engine stopped"
                fut["event"].set()

    def last_complaint(self) -> str:
        """The engine's own last words, so the error says *why* and not just *that*.

        A crash on import prints a traceback to stderr and exits; the last
        non-empty line of it names the problem (a DLL that would not load, a
        missing module). Common Windows ones get a sentence of advice.
        """
        t = getattr(self, "_stderr_thread", None)
        if t is not None:
            t.join(timeout=2)
        # Skip the app's own notes (they start with a [HH:MM:SS] stamp); keep
        # everything the engine itself printed.
        lines = [l for l in self.console[-40:]
                 if l.strip() and not re.match(r"^\[\d\d:\d\d:\d\d\]", l)]
        tail = lines[-1].strip() if lines else ""
        for l in reversed(lines):
            if "Error" in l or "error:" in l.lower():
                tail = l.strip()
                break
        return (tail + " " + advice_for(tail)).strip() if tail else "See the console on the Engine page."

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            raw = raw.rstrip()
            if raw:
                self.console.append(raw)
                if len(self.console) > 3000:
                    del self.console[:1500]

    def call(self, cmd: str, timeout: float = 3600, on_progress=None, **kw) -> dict:
        rid = uuid.uuid4().hex[:10]
        fut = {"event": threading.Event(), "on_progress": on_progress}
        with self.lock:
            if not self.alive():
                raise EngineError("The engine is not running")
            self.pending[rid] = fut
            try:
                self._write({"id": rid, "cmd": cmd, **kw})
                if not fut["event"].wait(timeout):
                    raise EngineError(f"The engine did not answer '{cmd}' within {timeout:.0f}s")
            finally:
                self.pending.pop(rid, None)
        if "error" in fut:
            raise EngineError(fut["error"])
        return fut.get("result") or {}

    def cancel(self) -> None:
        if self.alive():
            try:
                self._write({"id": "c", "cmd": "cancel"})
            except EngineError:
                pass


ENGINE = EngineProcess()


# --------------------------------------------------------------------------- #
# stores
# --------------------------------------------------------------------------- #
_store_lock = threading.Lock()


def _read_json(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _write_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_voices() -> list[dict]:
    with _store_lock:
        return _read_json(VOICES_PATH)


def write_voices(items: list[dict]) -> None:
    with _store_lock:
        _write_json(VOICES_PATH, items)


def read_takes() -> list[dict]:
    with _store_lock:
        return _read_json(TAKES_PATH)


def write_takes(items: list[dict]) -> None:
    with _store_lock:
        _write_json(TAKES_PATH, items)


def find(items: list[dict], item_id: str) -> dict | None:
    for it in items:
        if it.get("id") == item_id:
            return it
    return None


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / float(w.getframerate() or 1), 2)
    except (OSError, wave.Error, EOFError):
        return 0.0


# --------------------------------------------------------------------------- #
# script parsing
# --------------------------------------------------------------------------- #
SPEAKER_RE = re.compile(r"^\s*([A-Za-z0-9 _.'\-]{1,40}?)\s*[:：]\s*(.+?)\s*$")


def parse_script(text: str) -> list[dict]:
    """Turn pasted text into [{speaker, text}].

    `Name: words` starts a line for Name. Lines without a speaker join the
    previous one, or go to an unnamed speaker ("") when there is none yet.
    Blank lines separate paragraphs into separate lines.
    """
    out: list[dict] = []
    current: dict | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            current = None
            continue
        m = SPEAKER_RE.match(line)
        if m and not re.match(r"^\d+$", m.group(1).strip()) and len(m.group(2)) > 0 \
                and not _looks_like_time(m.group(1)):
            current = {"speaker": m.group(1).strip(), "text": m.group(2).strip()}
            out.append(current)
        elif current is not None:
            current["text"] = (current["text"] + " " + line).strip()
        else:
            current = {"speaker": "", "text": line}
            out.append(current)
    return out


def _looks_like_time(s: str) -> bool:
    return bool(re.match(r"^\d{1,2}$", s.strip()))


def speakers_in(lines: list[dict]) -> list[str]:
    seen: list[str] = []
    for ln in lines:
        s = ln.get("speaker", "")
        if s not in seen:
            seen.append(s)
    return seen


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
JOBS: dict[str, dict] = {}
JOB_QUEUE: list[str] = []
_job_lock = threading.Lock()
_job_wake = threading.Event()


def job_view(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "payload"}


def validate_lines(payload: dict) -> list[dict]:
    lines = payload.get("lines") or []
    if not isinstance(lines, list) or not lines:
        raise ValueError("Nothing to read")
    if len(lines) > MAX_LINES:
        raise ValueError(f"At most {MAX_LINES} lines per take")
    voices = {v["id"]: v for v in read_voices()}
    version = payload.get("version") or cfg.get("version", "v2")
    if version not in MODELS:
        raise ValueError("version must be v1 or v2")
    out = []
    for i, ln in enumerate(lines):
        text = str(ln.get("text", "")).strip()
        if not text:
            continue
        if len(text) > MAX_LINE_CHARS:
            raise ValueError(f"Line {i + 1} is longer than {MAX_LINE_CHARS} characters; "
                             "split it")
        base = dict(ln.get("base") or {})
        if version == "v1":
            lang = base.get("language", "EN")
            if lang not in V1_BASES:
                raise ValueError(f"Line {i + 1}: V1 has no {lang} base speaker")
            style = base.get("style", "default")
            if style not in V1_BASES[lang]["styles"]:
                raise ValueError(f"Line {i + 1}: {lang} has no style {style!r}")
            base = {"language": lang, "style": style}
        else:
            lang = base.get("language", "EN")
            if lang not in V2_BASES:
                raise ValueError(f"Line {i + 1}: V2 has no {lang} base")
            spk = base.get("speaker") or next(iter(V2_BASES[lang]["speakers"]))
            if spk not in V2_BASES[lang]["speakers"]:
                raise ValueError(f"Line {i + 1}: {lang} has no speaker {spk!r}")
            base = {"language": lang, "speaker": spk}
        vid = ln.get("voice") or ""
        if vid and vid not in voices:
            raise ValueError(f"Line {i + 1}: voice {vid!r} does not exist")
        try:
            speed = float(ln.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        out.append({"text": text, "base": base, "voice": vid,
                    "speaker": str(ln.get("speaker", ""))[:40],
                    "speed": max(0.5, min(2.0, speed))})
    if not out:
        raise ValueError("Every line is empty")
    return out


def submit_job(payload: dict) -> dict:
    lines = validate_lines(payload)
    version = payload.get("version") or cfg.get("version", "v2")
    opts = payload.get("opts") or {}
    job = {"id": uuid.uuid4().hex[:10], "state": "queued", "created": time.time(),
           "version": version, "lines": len(lines), "done": 0, "stage": "",
           "msg": "Waiting", "take": None, "error": "", "pct": 0,
           "payload": {"lines": lines, "version": version, "opts": {
               "pause": float(opts.get("pause", cfg.get("pause", 0.35))),
               "tau": float(opts.get("tau", cfg.get("tau", 0.3))),
               "watermark": bool(opts.get("watermark", cfg.get("watermark", False))),
               "free_after": bool(opts.get("free_after", cfg.get("free_after", False))),
           }, "title": str(payload.get("title") or "")[:80]}}
    with _job_lock:
        JOBS[job["id"]] = job
        JOB_QUEUE.append(job["id"])
        finished = [j for j in JOBS.values() if j["state"] not in ("queued", "running")]
        for old in sorted(finished, key=lambda j: j["created"])[:-30]:
            JOBS.pop(old["id"], None)
    _job_wake.set()
    return job


def take_title(lines: list[dict], given: str) -> str:
    if given:
        return given
    first = lines[0]["text"]
    return (first[:48] + "…") if len(first) > 48 else first


def run_job(job: dict) -> None:
    payload = job["payload"]
    lines = payload["lines"]
    version = payload["version"]
    job["state"] = "running"
    job["msg"] = "Starting the engine"
    ENGINE.ensure()
    voices = {v["id"]: v for v in read_voices()}
    SE_DIR.mkdir(parents=True, exist_ok=True)
    # Embeddings first, once per voice; they are cached by the clip's hash.
    se_for: dict[str, str] = {}
    for ln in lines:
        vid = ln["voice"]
        if vid and vid not in se_for:
            v = voices[vid]
            job["msg"] = f"Listening to {v['name']}"
            job["stage"] = "embed"
            res = ENGINE.call("embed", clip=str(VOICES_DIR / v["file"]), version=version,
                              cache_dir=str(SE_DIR), timeout=1800)
            se_for[vid] = res["se"]
            if res.get("speech_seconds") is not None:
                v["speech_seconds"] = res["speech_seconds"]
                items = read_voices()
                it = find(items, vid)
                if it is not None:
                    it["speech_seconds"] = res["speech_seconds"]
                    write_voices(items)
    take_id = uuid.uuid4().hex[:10]
    out_dir = TAKES_DIR / take_id
    engine_lines = [{"text": ln["text"], "base": ln["base"], "speed": ln["speed"],
                     "se": se_for.get(ln["voice"], "")} for ln in lines]

    def on_progress(msg: dict) -> None:
        job["stage"] = msg.get("stage", "")
        job["msg"] = msg.get("msg", "")
        if msg.get("stage") == "done":
            job["done"] = int(msg.get("line", 0)) + 1
        total = max(1, len(lines))
        job["pct"] = round(min(99, (job["done"] + (0.5 if msg.get("stage") == "convert" else 0))
                               / total * 100))

    job["msg"] = "Reading"
    result = ENGINE.call("speak", version=version, lines=engine_lines, opts=payload["opts"],
                         out_dir=str(out_dir), timeout=7200, on_progress=on_progress)
    take = {"id": take_id, "title": take_title(lines, payload.get("title", "")),
            "created": time.time(), "version": version,
            "seconds": result.get("seconds", wav_seconds(out_dir / "take.wav")),
            "sample_rate": result.get("sample_rate"), "elapsed": result.get("elapsed"),
            "vram_peak": result.get("vram_peak"), "pause": payload["opts"]["pause"],
            "lines": [{"text": ln["text"], "speaker": ln["speaker"], "voice": ln["voice"],
                       "voice_name": voices.get(ln["voice"], {}).get("name", ""),
                       "base": ln["base"], "file": Path(f).name,
                       "seconds": wav_seconds(Path(f))}
                      for ln, f in zip([l for l in lines if l["text"]], result.get("lines", []))]}
    items = read_takes()
    items.insert(0, take)
    write_takes(items)
    job["take"] = take_id
    job["pct"] = 100
    job["done"] = len(lines)
    job["state"] = "done"
    job["msg"] = f"Done in {result.get('elapsed', 0)}s"


def job_worker() -> None:
    while True:
        _job_wake.wait(1.0)
        _job_wake.clear()
        while True:
            with _job_lock:
                if not JOB_QUEUE:
                    break
                jid = JOB_QUEUE.pop(0)
            job = JOBS.get(jid)
            if job is None or job["state"] != "queued":
                continue
            try:
                run_job(job)
            except EngineError as exc:
                job["state"] = "cancelled" if "Cancelled" in str(exc) else "error"
                job["error"] = str(exc)
                job["msg"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                job["state"] = "error"
                job["error"] = str(exc)
                job["msg"] = str(exc)
                ENGINE.note(f"Job {job['id']} failed: {exc}")


# --------------------------------------------------------------------------- #
# routes: page
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_asset(name: str):
    return send_from_directory(WEB_DIR, name)


@app.get("/resources/<path:name>")
def resource(name: str):
    return send_from_directory(APP_DIR / "resources", name)


@app.errorhandler(413)
def too_big(_e):
    return jsonify({"error": "That file is larger than 64 MB. Trim the clip first."}), 413


# --------------------------------------------------------------------------- #
# routes: status, config, deps, tasks
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    fast = request.args.get("fast") == "1"
    gpu = manager.gpu_info()
    tf = manager.torch_facts_cached(force=False) if not fast else manager._torch_cache["value"] or {}
    models = {k: manager.model_state(k) for k in MODELS}
    running = [j for j in JOBS.values() if j["state"] in ("queued", "running")]
    live = {}
    if ENGINE.alive() and ENGINE.state == "ready" and not running:
        try:
            live = ENGINE.call("status", timeout=5)
        except EngineError:
            live = {}
    fits = None
    if gpu.get("present") and gpu.get("vram_total"):
        fits = gpu["vram_total"] >= 3 * 1024 ** 3
    return jsonify({
        "python": manager.python_facts(), "gpu": gpu, "torch": tf or {},
        "models": models, "engine": {"state": ENGINE.state, "error": ENGINE.error,
                                     "hello": ENGINE.hello, "live": live},
        "config": {k: v for k, v in cfg.items() if k != "hf_token"},
        "has_token": bool(cfg.get("hf_token")),
        "jobs_running": len(running), "fits": fits,
        "ready": bool(tf.get("installed")) and not tf.get("error")
                 and any(m["present"] for m in models.values()),
        "port": PORT, "data_dir": str(DATA_DIR), "app_dir": str(APP_DIR),
    })


@app.get("/api/config")
def api_config():
    return jsonify({k: v for k, v in cfg.items() if k != "hf_token"})


@app.post("/api/config")
def api_config_save():
    body = request.get_json(silent=True) or {}
    changed_device = False
    for k, v in body.items():
        if k not in manager.DEFAULT_CONFIG:
            continue
        if k == "zip_urls" and isinstance(v, dict):
            cfg["zip_urls"].update({kk: str(vv) for kk, vv in v.items() if kk in MODELS})
        elif k == "device":
            if v in ("auto", "cpu", "cuda") and v != cfg.get("device"):
                cfg[k] = v
                changed_device = True
        elif k == "version":
            if v in MODELS:
                cfg[k] = v
        elif k in ("pause", "tau"):
            try:
                cfg[k] = max(0.0, min(5.0 if k == "pause" else 1.0, float(v)))
            except (TypeError, ValueError):
                pass
        elif isinstance(manager.DEFAULT_CONFIG[k], bool):
            cfg[k] = bool(v)
        else:
            cfg[k] = v
    save_config(cfg)
    if changed_device and ENGINE.alive():
        ENGINE.note("Device setting changed; restart the engine for it to apply")
    return jsonify({"ok": True, "config": {k: v for k, v in cfg.items() if k != "hf_token"}})


@app.get("/api/deps")
def api_deps():
    fast = request.args.get("fast") == "1"
    return jsonify({"deps": manager.deps(cfg, fast=fast),
                    "builds": manager.TORCH_BUILDS,
                    "installing": [t.meta.get("dep") for t in TASKS.running("install")]})


@app.post("/api/deps/<dep_id>/install")
def api_dep_install(dep_id: str):
    body = request.get_json(silent=True) or {}
    if dep_id.startswith("model_"):
        return api_model_download(dep_id[len("model_"):])
    fn = manager.INSTALLERS.get(dep_id)
    if fn is None:
        return jsonify({"error": f"nothing installs {dep_id}"}), 404
    if TASKS.running("install"):
        return jsonify({"error": "An install is already running; wait for it"}), 409
    if dep_id == "torch":
        build = body.get("build") or cfg.get("torch_build", "auto")
        cfg["torch_build"] = build
        save_config(cfg)
        fn2 = lambda task: manager.install_torch(task, build)  # noqa: E731
    else:
        fn2 = fn
    # Installing into the interpreter the engine runs on while it runs is how
    # you get a half-imported torch. Stop it first; the next take restarts it.
    if ENGINE.alive():
        ENGINE.note(f"Stopping the engine to install {dep_id}")
        ENGINE.stop()

    def wrapped(task):
        fn2(task)
        tf = manager.torch_facts_cached(force=True)
        # The engine was stopped for the install; bring it back when there is
        # a PyTorch to bring it back on, so the page does not sit on "stopped".
        if tf.get("installed") and not tf.get("error"):
            task.log("Starting the engine again")
            try:
                ENGINE.start()
                task.log("Engine ready")
            except EngineError as exc:
                task.log(f"The engine did not start: {exc}")

    task = manager.spawn("install", f"Install {dep_id}", wrapped, {"dep": dep_id})
    return jsonify({"task": task.view()})


@app.get("/api/tasks")
def api_tasks():
    return jsonify({"tasks": [t.view() for t in TASKS.list()[:20]]})


@app.get("/api/tasks/<task_id>")
def api_task(task_id: str):
    t = TASKS.get(task_id)
    if t is None:
        return jsonify({"error": "no such task"}), 404
    since = int(request.args.get("since", 0))
    return jsonify({"task": t.view(since)})


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    t = TASKS.get(task_id)
    if t is None:
        return jsonify({"error": "no such task"}), 404
    t.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# routes: models
# --------------------------------------------------------------------------- #
@app.get("/api/models")
def api_models():
    return jsonify({"models": [manager.model_state(k) for k in ("v2", "v1")],
                    "hf_endpoint": cfg.get("hf_endpoint"), "has_token": bool(cfg.get("hf_token")),
                    "zip_urls": cfg.get("zip_urls", {})})


@app.post("/api/models/<key>/download")
def api_model_download(key: str):
    if key not in MODELS:
        return jsonify({"error": "no such model"}), 404
    if any(t.meta.get("model") == key for t in TASKS.running("download")):
        return jsonify({"error": "already downloading"}), 409
    task = manager.spawn("download", f"Download {MODELS[key]['label']}",
                         lambda t: manager.download_model(t, key, cfg), {"model": key})
    return jsonify({"task": task.view()})


@app.post("/api/models/<key>/import")
def api_model_import(key: str):
    if key not in MODELS:
        return jsonify({"error": "no such model"}), 404
    body = request.get_json(silent=True) or {}
    path = str(body.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path is required"}), 400
    task = manager.spawn("download", f"Import {MODELS[key]['label']}",
                         lambda t: manager.import_local(t, key, path), {"model": key})
    return jsonify({"task": task.view()})


@app.delete("/api/models/<key>")
def api_model_delete(key: str):
    if key not in MODELS:
        return jsonify({"error": "no such model"}), 404
    if ENGINE.alive():
        ENGINE.stop()
    manager.delete_model(key)
    return jsonify({"ok": True})


@app.post("/api/hf/settings")
def api_hf_settings():
    body = request.get_json(silent=True) or {}
    if "hf_endpoint" in body:
        ep = str(body["hf_endpoint"]).strip() or manager.HF_DEFAULT
        if not ep.startswith("http"):
            return jsonify({"error": "endpoint must start with http"}), 400
        cfg["hf_endpoint"] = ep.rstrip("/")
    if "hf_token" in body:
        cfg["hf_token"] = str(body["hf_token"]).strip()
    if isinstance(body.get("zip_urls"), dict):
        cfg["zip_urls"].update({k: str(v).strip() for k, v in body["zip_urls"].items()
                                if k in MODELS})
    save_config(cfg)
    return jsonify({"ok": True, "hf_endpoint": cfg["hf_endpoint"],
                    "has_token": bool(cfg["hf_token"]), "zip_urls": cfg["zip_urls"]})


# --------------------------------------------------------------------------- #
# routes: engine
# --------------------------------------------------------------------------- #
@app.get("/api/engine/log")
def api_engine_log():
    since = int(request.args.get("since", 0))
    return jsonify(ENGINE.log_view(since))


@app.post("/api/engine/start")
def api_engine_start():
    try:
        ENGINE.start()
    except EngineError as exc:
        return jsonify({"error": str(exc), "state": ENGINE.state}), 500
    return jsonify({"ok": True, "state": ENGINE.state, "hello": ENGINE.hello})


@app.post("/api/engine/stop")
def api_engine_stop():
    if any(j["state"] == "running" for j in JOBS.values()):
        ENGINE.cancel()
    ENGINE.stop()
    return jsonify({"ok": True, "state": ENGINE.state})


@app.post("/api/engine/restart")
def api_engine_restart():
    try:
        ENGINE.restart()
    except EngineError as exc:
        return jsonify({"error": str(exc), "state": ENGINE.state}), 500
    return jsonify({"ok": True, "state": ENGINE.state, "hello": ENGINE.hello})


@app.post("/api/engine/free")
def api_engine_free():
    if not ENGINE.alive():
        return jsonify({"ok": True, "state": ENGINE.state, "live": {}})
    try:
        live = ENGINE.call("unload", timeout=60)
    except EngineError as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "live": live})


@app.post("/api/selftest")
def api_selftest():
    """One line, every step reported, so 'it does not work' has a place to point."""
    if TASKS.running("selftest"):
        return jsonify({"error": "a self-test is already running"}), 409
    body = request.get_json(silent=True) or {}
    version = body.get("version") or cfg.get("version", "v2")
    if version not in MODELS:
        return jsonify({"error": "version must be v1 or v2"}), 400

    def run(task):
        steps = []

        def step(ok, name, detail=""):
            steps.append({"ok": ok, "name": name, "detail": detail})
            task.meta["steps"] = list(steps)
            task.log(("ok    " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))

        tf = manager.torch_facts_cached(force=True)
        if not tf.get("installed") or tf.get("error"):
            step(False, "PyTorch imports", tf.get("error") or "not installed")
            raise RuntimeError("PyTorch is not usable")
        step(True, "PyTorch imports", f"{tf['version']}" +
             (f", CUDA on {tf.get('gpu')}" if tf.get("cuda") else ", CPU"))
        ms = manager.model_state(version)
        if not ms["present"]:
            step(False, f"{ms['label']} is on disk", "missing " + ", ".join(ms["missing"]))
            raise RuntimeError("checkpoints missing")
        step(True, f"{ms['label']} is on disk", f"{ms['files']} files, {ms['bytes'] / 1e6:.0f} MB")
        try:
            ENGINE.ensure()
        except EngineError as exc:
            step(False, "The engine starts", str(exc))
            raise
        step(True, "The engine starts", f"pid {ENGINE.proc.pid if ENGINE.proc else '?'}")
        try:
            ENGINE.call("load", version=version, watermark=bool(cfg.get("watermark")),
                        timeout=600)
        except EngineError as exc:
            step(False, "The converter loads", str(exc))
            raise
        step(True, "The converter loads", version.upper())
        out_dir = DATA_DIR / "selftest"
        shutil.rmtree(out_dir, ignore_errors=True)
        base = {"language": "EN", "style": "default"} if version == "v1" \
            else {"language": "EN", "speaker": "EN-Default"}
        ref = APP_DIR / "resources" / "example_reference.mp3"
        se = ""
        if ref.is_file():
            try:
                res = ENGINE.call("embed", clip=str(ref), version=version,
                                  cache_dir=str(SE_DIR), timeout=900)
                se = res["se"]
                step(True, "A reference clip becomes an embedding",
                     "cached" if res.get("cached") else
                     f"{res.get('speech_seconds', '?')}s of speech")
            except EngineError as exc:
                step(False, "A reference clip becomes an embedding", str(exc))
                raise
        t0 = time.time()
        try:
            res = ENGINE.call("speak", version=version, out_dir=str(out_dir),
                              lines=[{"text": "This audio is generated by OpenVoice.",
                                      "base": base, "speed": 1.0, "se": se}],
                              opts={"pause": 0.2, "tau": 0.3,
                                    "watermark": bool(cfg.get("watermark"))}, timeout=1800)
        except EngineError as exc:
            step(False, "Speech comes back", str(exc))
            raise
        secs = res.get("seconds", 0)
        quiet = False
        try:
            with wave.open(str(out_dir / "take.wav"), "rb") as w:
                frames = w.readframes(min(w.getnframes(), 200000))
            import array
            samples = array.array("h", frames) if w.getsampwidth() == 2 else None
            quiet = samples is not None and max(abs(s) for s in samples) < 50
        except Exception:  # noqa: BLE001
            pass
        detail = (f"{secs}s of audio at {res.get('sample_rate')} Hz in "
                  f"{time.time() - t0:.1f}s")
        if res.get("vram_peak"):
            detail += f" · peak {res['vram_peak'] / 1e9:.2f} GB on the card"
        step(not quiet and secs > 0.3, "Speech comes back",
             detail + (" — but it is silent" if quiet else ""))
        task.meta["file"] = str(out_dir / "take.wav")
        task.set(detail="All good" if all(s["ok"] for s in steps) else "Something failed")

    task = manager.spawn("selftest", f"Self-test {version.upper()}", run,
                         {"version": version, "steps": []})
    return jsonify({"task": task.view()})


@app.get("/api/selftest/audio")
def api_selftest_audio():
    f = DATA_DIR / "selftest" / "take.wav"
    if not f.is_file():
        return jsonify({"error": "no self-test audio yet"}), 404
    return send_file(f, mimetype="audio/wav")


# --------------------------------------------------------------------------- #
# routes: voices (reference clips)
# --------------------------------------------------------------------------- #
@app.get("/api/bases")
def api_bases():
    v2 = {}
    ses = MODELS["v2"]["dir"] / "base_speakers" / "ses"
    for lang, info in V2_BASES.items():
        v2[lang] = {"label": info["label"], "speakers": [
            {"key": k, "label": v, "ready": (ses / f"{k.lower().replace('_', '-')}.pth").is_file()}
            for k, v in info["speakers"].items()]}
    v1 = {lang: {"label": info["label"], "styles": info["styles"],
                 "ready": (MODELS["v1"]["dir"] / "base_speakers" / lang / "checkpoint.pth").is_file()}
          for lang, info in V1_BASES.items()}
    return jsonify({"v1": v1, "v2": v2})


@app.get("/api/voices")
def api_voices():
    return jsonify({"voices": read_voices()})


@app.post("/api/voices")
def api_voice_add():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in AUDIO_EXTS:
        return jsonify({"error": f"{ext or 'that'} is not an audio format I can read; "
                        "use wav, mp3, flac, ogg or m4a"}), 400
    name = (request.form.get("name") or Path(f.filename).stem).strip()[:60] or "Voice"
    vid = uuid.uuid4().hex[:10]
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{vid}{ext}"
    f.save(VOICES_DIR / fname)
    voice = {"id": vid, "name": name, "file": fname, "original": Path(f.filename).name,
             "created": time.time(), "bytes": (VOICES_DIR / fname).stat().st_size,
             "seconds": wav_seconds(VOICES_DIR / fname) if ext == ".wav" else None}
    items = read_voices()
    items.insert(0, voice)
    write_voices(items)
    return jsonify({"voice": voice})


@app.post("/api/voices/<vid>")
def api_voice_rename(vid: str):
    items = read_voices()
    v = find(items, vid)
    if v is None:
        return jsonify({"error": "no such voice"}), 404
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()[:60]
    if name:
        v["name"] = name
    write_voices(items)
    return jsonify({"voice": v})


@app.delete("/api/voices/<vid>")
def api_voice_delete(vid: str):
    items = read_voices()
    v = find(items, vid)
    if v is None:
        return jsonify({"error": "no such voice"}), 404
    items.remove(v)
    write_voices(items)
    (VOICES_DIR / v["file"]).unlink(missing_ok=True)
    stem = Path(v["file"]).stem
    if SE_DIR.is_dir():
        for p in SE_DIR.glob(f"{stem}_*.pth"):
            p.unlink(missing_ok=True)
    return jsonify({"ok": True})


@app.get("/api/voices/<vid>/audio")
def api_voice_audio(vid: str):
    v = find(read_voices(), vid)
    if v is None:
        return jsonify({"error": "no such voice"}), 404
    return send_from_directory(VOICES_DIR, v["file"])


# --------------------------------------------------------------------------- #
# routes: speaking, jobs, takes
# --------------------------------------------------------------------------- #
@app.post("/api/parse")
def api_parse():
    body = request.get_json(silent=True) or {}
    lines = parse_script(str(body.get("text") or ""))
    return jsonify({"lines": lines, "speakers": speakers_in(lines)})


@app.post("/api/speak")
def api_speak():
    body = request.get_json(silent=True) or {}
    try:
        job = submit_job(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"job": job_view(job)})


@app.get("/api/jobs")
def api_jobs():
    items = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
    return jsonify({"jobs": [job_view(j) for j in items[:20]]})


@app.get("/api/jobs/<jid>")
def api_job(jid: str):
    j = JOBS.get(jid)
    if j is None:
        return jsonify({"error": "no such job"}), 404
    return jsonify({"job": job_view(j)})


@app.post("/api/jobs/<jid>/cancel")
def api_job_cancel(jid: str):
    j = JOBS.get(jid)
    if j is None:
        return jsonify({"error": "no such job"}), 404
    if j["state"] == "queued":
        j["state"] = "cancelled"
        j["msg"] = "Cancelled"
        with _job_lock:
            if jid in JOB_QUEUE:
                JOB_QUEUE.remove(jid)
    elif j["state"] == "running":
        ENGINE.cancel()
        j["msg"] = "Stopping after this line"
    return jsonify({"job": job_view(j)})


@app.get("/api/takes")
def api_takes():
    return jsonify({"takes": read_takes()})


@app.get("/api/takes/<tid>/audio")
def api_take_audio(tid: str):
    t = find(read_takes(), tid)
    if t is None:
        return jsonify({"error": "no such take"}), 404
    f = TAKES_DIR / tid / "take.wav"
    if not f.is_file():
        return jsonify({"error": "audio missing on disk"}), 404
    return send_file(f, mimetype="audio/wav", download_name=f"{t['title'][:40] or tid}.wav",
                     as_attachment=request.args.get("download") == "1")


@app.get("/api/takes/<tid>/line/<int:i>")
def api_take_line(tid: str, i: int):
    t = find(read_takes(), tid)
    if t is None or i < 0 or i >= len(t["lines"]):
        return jsonify({"error": "no such line"}), 404
    return send_from_directory(TAKES_DIR / tid, t["lines"][i]["file"])


@app.post("/api/takes/<tid>")
def api_take_rename(tid: str):
    items = read_takes()
    t = find(items, tid)
    if t is None:
        return jsonify({"error": "no such take"}), 404
    body = request.get_json(silent=True) or {}
    title = str(body.get("title") or "").strip()[:80]
    if title:
        t["title"] = title
    write_takes(items)
    return jsonify({"take": t})


@app.delete("/api/takes/<tid>")
def api_take_delete(tid: str):
    items = read_takes()
    t = find(items, tid)
    if t is None:
        return jsonify({"error": "no such take"}), 404
    items.remove(t)
    write_takes(items)
    shutil.rmtree(TAKES_DIR / tid, ignore_errors=True)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #
def sweep_orphans() -> None:
    """Take folders without a record (a crash mid-take) are removed at boot."""
    if not TAKES_DIR.is_dir():
        return
    known = {t["id"] for t in read_takes()}
    for d in TAKES_DIR.iterdir():
        if d.is_dir() and d.name not in known:
            shutil.rmtree(d, ignore_errors=True)


def start_engine_at_boot() -> None:
    tf = manager.torch_facts_cached(force=True)
    if tf.get("installed") and not tf.get("error"):
        try:
            ENGINE.start()
        except EngineError as exc:
            print(f"[engine] not started: {exc}", flush=True)
    else:
        ENGINE.note("PyTorch is not installed yet — open the Engine page to set it up")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sweep_orphans()
    threading.Thread(target=job_worker, daemon=True).start()
    threading.Thread(target=start_engine_at_boot, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"OpenVoice Studio at {url}  (data in {DATA_DIR})", flush=True)
    if not os.environ.get("OPENVOICE_STUDIO_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
    finally:
        ENGINE.stop()


if __name__ == "__main__":
    main()
