"""
manager.py - what the Engine and Models pages need to get the machine ready.

Dependencies: Python, ffmpeg, PyTorch (and whether it can see the card), the
OpenVoice packages, MeloTTS for V2, the optional extras — checked in the
interpreter this server runs on, which is the one engine.py runs on, and
installed into it on request with the output streamed to a task.

Models: the V1 and V2 checkpoints. The S3 bucket the upstream docs link
(`myshell-public-repo-host`) no longer exists, so they are pulled from the
HuggingFace mirrors file by file with resume, or imported from a zip someone
already has.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from importlib import metadata
from pathlib import Path

import requests

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("OPENVOICE_STUDIO_DATA") or (APP_DIR / "data")).resolve()
CKPT = {"v1": APP_DIR / "checkpoints", "v2": APP_DIR / "checkpoints_v2"}
CONFIG_PATH = DATA_DIR / "config.json"
HF_DEFAULT = "https://huggingface.co"
PYTORCH_INDEX = "https://download.pytorch.org/whl/"
MELO_GIT = "git+https://github.com/myshell-ai/MeloTTS.git"
TOOLS_DIR = APP_DIR / "tools"          # ffmpeg lands here, see install_ffmpeg
PIP_RAW_MIN = (24, 1)                  # first pip with --progress-bar raw


def ensure_tools_path() -> None:
    """Put tools/ first on PATH so shutil.which and child processes find
    the ffmpeg the app installed, whatever the system has."""
    if TOOLS_DIR.is_dir():
        cur = os.environ.get("PATH", "")
        if str(TOOLS_DIR) not in cur.split(os.pathsep):
            os.environ["PATH"] = str(TOOLS_DIR) + os.pathsep + cur

DEFAULT_CONFIG = {
    "version": "v2",            # which OpenVoice the Create page uses
    "watermark": False,         # wavmark watermark on every take
    "free_after": False,        # drop the models from the card after each take
    "pause": 0.35,              # seconds between lines
    "tau": 0.3,                 # converter temperature
    "device": "auto",           # auto | cuda | cpu
    "torch_build": "auto",      # auto | cu128 | cu126 | cu121 | cpu
    "hf_endpoint": HF_DEFAULT,
    "hf_token": "",
    "zip_urls": {               # a zip for each version, used before HuggingFace
        "v1": "",
        "v2": "",
    },
}

# Each version is one HuggingFace repo. `strip` is a top-level folder the repo
# carries that the local layout does not — V1's repo nests everything under
# checkpoints/, V2's does not — and `key_files` are what "downloaded" means.
MODELS = {
    "v2": {
        "label": "OpenVoice V2",
        "repo": "myshell-ai/OpenVoiceV2",
        "strip": "checkpoints_v2/",
        "dir": CKPT["v2"],
        "key_files": ["converter/checkpoint.pth", "converter/config.json",
                      "base_speakers/ses/en-default.pth"],
        "about": "Tone colour converter plus the base-speaker embeddings for "
                 "English (5 accents), Spanish, French, Chinese, Japanese and "
                 "Korean. The base voices themselves come from MeloTTS.",
        "size": "about 140 MB",
        "vram": "under 1 GB with one MeloTTS voice loaded",
    },
    "v1": {
        "label": "OpenVoice V1",
        "repo": "myshell-ai/OpenVoice",
        "strip": "checkpoints/",
        "dir": CKPT["v1"],
        "key_files": ["converter/checkpoint.pth", "converter/config.json",
                      "base_speakers/EN/checkpoint.pth", "base_speakers/EN/config.json",
                      "base_speakers/EN/en_default_se.pth", "base_speakers/EN/en_style_se.pth",
                      "base_speakers/ZH/checkpoint.pth", "base_speakers/ZH/config.json",
                      "base_speakers/ZH/zh_default_se.pth"],
        "about": "The original release: an English base speaker with nine "
                 "emotion styles (whispering, cheerful, sad...), a Chinese base "
                 "speaker, and the V1 converter. Needs no MeloTTS.",
        "size": "about 550 MB",
        "vram": "under 1 GB",
    },
}
SKIP_FILES = {".gitattributes", "README.md", "LICENSE", "logo.jpg"}

TORCH_BUILDS = [
    ("auto", "Pick for me"),
    ("cu128", "NVIDIA, CUDA 12.8 (driver 570 or newer)"),
    ("cu126", "NVIDIA, CUDA 12.6 (driver 560 or newer)"),
    ("cu121", "NVIDIA, CUDA 12.1 (older drivers)"),
    ("cpu", "CPU only (no NVIDIA card)"),
]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
_cfg_lock = threading.Lock()


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            for k, v in saved.items():
                if k == "zip_urls" and isinstance(v, dict):
                    cfg["zip_urls"].update({kk: str(vv) for kk, vv in v.items()})
                elif k in cfg:
                    cfg[k] = v
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    with _cfg_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        os.replace(tmp, CONFIG_PATH)


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, kind: str, title: str, meta: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.meta = meta or {}
        self.state = "running"
        self.pct: float | None = None
        self.detail = ""
        self.lines: list[str] = []
        self.created = time.time()
        self.cancel = False
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 1200:
                del self.lines[:600]
        print(f"[{self.kind}] {msg}", flush=True)

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def view(self, since: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "title": self.title,
                    "meta": self.meta, "state": self.state,
                    "pct": None if self.pct is None else round(self.pct, 1),
                    "detail": self.detail, "created": self.created,
                    "cursor": len(self.lines), "lines": self.lines[since:]}


class Tasks:
    def __init__(self) -> None:
        self._items: dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, task: Task) -> Task:
        with self._lock:
            self._items[task.id] = task
            finished = sorted((t for t in self._items.values() if t.state != "running"),
                              key=lambda t: t.created)
            for old in finished[:-40]:
                self._items.pop(old.id, None)
        return task

    def get(self, task_id: str) -> Task | None:
        return self._items.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            return sorted(self._items.values(), key=lambda t: t.created, reverse=True)

    def running(self, kind: str = "") -> list[Task]:
        return [t for t in self.list() if t.state == "running" and (not kind or t.kind == kind)]


TASKS = Tasks()


def spawn(kind: str, title: str, fn, meta: dict | None = None) -> Task:
    task = TASKS.add(Task(kind, title, meta))

    def wrapper():
        try:
            fn(task)
            if task.state == "running":
                task.set(state="cancelled" if task.cancel else "done",
                         pct=None if task.cancel else 100)
        except Exception as exc:  # noqa: BLE001
            task.log(f"FAILED: {exc}")
            task.set(state="error", detail=str(exc))

    threading.Thread(target=wrapper, daemon=True).start()
    return task


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r")
PIP_PROGRESS = re.compile(r"^Progress (\d+) of (\d+)$")
PIP_DOWNLOADING = re.compile(r"^\s*Downloading (\S+?)(?:\.metadata)? \(([\d.]+ [kMG]B)\)")


def run_logged(task: Task, cmd: list[str], cwd: Path | None = None,
               env: dict | None = None, pip_progress: bool = False) -> int:
    """Run a command, streaming its output into the task. Returns the exit code.

    With pip_progress, pip's `--progress-bar raw` lines ("Progress N of M")
    drive the task's bar and detail instead of flooding the log — off a
    terminal pip otherwise shows nothing while a 500 MB wheel comes down,
    which looks exactly like a hang.
    """
    task.log("$ " + " ".join(cmd))
    full_env = dict(os.environ)
    full_env["PYTHONUNBUFFERED"] = "1"
    full_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    full_env["PYTHONIOENCODING"] = "utf-8"      # both ends UTF-8, see server.py
    full_env["PYTHONUTF8"] = "1"
    if env:
        full_env.update(env)
    proc = subprocess.Popen(cmd, cwd=str(cwd or APP_DIR), env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    current = ""
    last_logged = -1
    for line in proc.stdout:
        line = ANSI.sub("", line.rstrip())
        if not line:
            continue
        m = PIP_PROGRESS.match(line) if pip_progress else None
        if m:
            got, total = int(m.group(1)), int(m.group(2))
            pct = got / total * 100 if total else None
            task.set(pct=pct, detail=f"{current or 'download'}  {got / 1e6:.0f}/{total / 1e6:.0f} MB")
            if total and got >= total and last_logged != total:
                last_logged = total
                task.log(f"  {current}: {total / 1e6:.0f} MB done")
        else:
            d = PIP_DOWNLOADING.match(line) if pip_progress else None
            if d:
                current = d.group(1)
                last_logged = -1
                task.set(pct=None, detail=f"Downloading {current} ({d.group(2)})")
            elif pip_progress and line.startswith("Installing collected packages"):
                task.set(pct=None, detail="Installing the downloaded packages")
            task.log(line)
        if task.cancel:
            proc.kill()
            task.log("Cancelled")
            break
    return proc.wait()


# --------------------------------------------------------------------------- #
# machine facts
# --------------------------------------------------------------------------- #
def python_facts() -> dict:
    v = sys.version_info
    return {"version": f"{v.major}.{v.minor}.{v.micro}", "executable": sys.executable,
            "venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
            "melo_ok": (v.major, v.minor) in ((3, 10), (3, 11)),
            "platform": platform.system()}


def pkg_version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def gpu_info() -> dict:
    """What nvidia-smi says about the card: name, memory, driver.

    Read from the driver rather than torch so the Engine page can show the card
    before PyTorch is installed, and say whether the build it is about to
    install will work with the driver.
    """
    smi = shutil.which("nvidia-smi")
    if not smi and platform.system() == "Windows":
        # Where the driver puts it when PATH does not say (rule 5b of the TTS app).
        for cand in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                     Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                     / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"):
            if cand.is_file():
                smi = str(cand)
                break
    if not smi:
        return {"present": False}
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return {"present": False}
    if not out:
        return {"present": False}
    name, total, used, driver = [p.strip() for p in out[0].split(",")[:4]]
    try:
        total_b = int(float(total)) * 1024 * 1024
        used_b = int(float(used)) * 1024 * 1024
    except ValueError:
        total_b = used_b = 0
    return {"present": True, "name": name, "vram_total": total_b, "vram_used": used_b,
            "driver": driver}


def driver_major(driver: str) -> int:
    m = re.match(r"(\d+)", driver or "")
    return int(m.group(1)) if m else 0


def recommended_build(gpu: dict | None = None) -> str:
    gpu = gpu if gpu is not None else gpu_info()
    if not gpu.get("present"):
        return "cpu"
    major = driver_major(gpu.get("driver", ""))
    if major and major < 525:
        return "cu121"
    if major and major < 560:
        return "cu121"
    if major and major < 570:
        return "cu126"
    return "cu128"


def torch_facts() -> dict:
    """Import torch in a child so a broken install cannot take the server down."""
    code = ("import json,torch;print(json.dumps({'version':torch.__version__,"
            "'cuda':torch.cuda.is_available(),'cuda_version':torch.version.cuda,"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"installed": False, "error": str(exc)}
    if r.returncode != 0:
        err = (r.stderr or r.stdout).strip().splitlines()
        return {"installed": bool(pkg_version("torch")),
                "error": err[-1] if err else f"exit {r.returncode}"}
    try:
        return {"installed": True, **json.loads(r.stdout.strip().splitlines()[-1])}
    except (ValueError, IndexError):
        return {"installed": True, "error": "could not read torch's answer"}


_torch_cache: dict = {"at": 0.0, "value": None}
_torch_lock = threading.Lock()


def torch_facts_cached(force: bool = False) -> dict:
    # One probe at a time: at boot the engine starter and the page's first
    # status call both ask, and two concurrent `import torch` subprocesses
    # double the wait on a cold disk for the same answer.
    with _torch_lock:
        now = time.time()
        if force or _torch_cache["value"] is None or now - _torch_cache["at"] > 60:
            _torch_cache["value"] = torch_facts()
            _torch_cache["at"] = now
        return _torch_cache["value"]


def importable(mod: str) -> bool:
    code = f"import {mod}"
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True,
                              timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def unidic_ready() -> bool:
    try:
        import unidic  # type: ignore  # noqa: WPS433
        return Path(unidic.DICDIR).joinpath("dicrc").is_file()
    except Exception:  # noqa: BLE001
        return False


def model_state(key: str) -> dict:
    m = MODELS[key]
    d: Path = m["dir"]
    missing = [f for f in m["key_files"] if not (d / f).is_file()]
    nbytes = 0
    nfiles = 0
    if d.is_dir():
        for p in d.rglob("*"):
            if p.is_file() and not p.name.endswith(".part"):
                nfiles += 1
                nbytes += p.stat().st_size
    return {"key": key, "label": m["label"], "repo": m["repo"], "dir": str(d),
            "present": not missing, "missing": missing, "files": nfiles,
            "bytes": nbytes, "about": m["about"], "size": m["size"], "vram": m["vram"],
            "downloading": any(t.meta.get("model") == key for t in TASKS.running("download"))}


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
ENGINE_PKGS = ["librosa>=0.9.1", "soundfile", "numpy<2", "scipy", "eng_to_ipa",
               "inflect", "unidecode", "pypinyin", "cn2an", "jieba", "pydub",
               "silero-vad", "requests"]


def deps(cfg: dict, fast: bool = False) -> list[dict]:
    py = python_facts()
    gpu = gpu_info()
    out: list[dict] = []

    out.append({"id": "python", "label": "Python",
                "state": "ok" if py["melo_ok"] else "warn",
                "detail": f"{py['version']} at {py['executable']}" + (
                    "" if py["melo_ok"] else
                    " — MeloTTS (V2 voices) pins packages that only build on "
                    "3.10 or 3.11; V1 works here, V2 may not install"),
                "installable": False})

    ensure_tools_path()
    ff = shutil.which("ffmpeg")
    out.append({"id": "ffmpeg", "label": "ffmpeg", "state": "ok" if ff else "warn",
                "detail": ff or "Not on PATH. Reference clips in mp3/m4a may fail to "
                "decode; wav and flac are fine. Install puts a static build in the "
                "app's tools/ folder (about 40 MB); `winget install Gyan.FFmpeg` "
                "or ffmpeg.org work too, after a restart of the app.",
                "installable": True, "optional": True})

    tf = torch_facts_cached(force=not fast)
    if not tf.get("installed"):
        state, detail = "missing", "Not installed"
    elif tf.get("error"):
        state, detail = "error", f"Installed but broken: {tf['error']}"
    elif gpu.get("present") and not tf.get("cuda"):
        state = "warn"
        detail = (f"{tf['version']} is a CPU-only build on a machine with "
                  f"{gpu.get('name')}. Reinstall to use the card.")
    else:
        state = "ok"
        detail = f"{tf.get('version')}" + (
            f", CUDA {tf.get('cuda_version')} on {tf.get('gpu')}" if tf.get("cuda")
            else ", CPU only")
    out.append({"id": "torch", "label": "PyTorch", "state": state, "detail": detail,
                "installable": True, "build": cfg.get("torch_build", "auto"),
                "recommended": recommended_build(gpu)})

    have = {p.split("<")[0].split(">")[0].split("=")[0]: pkg_version(
        p.split("<")[0].split(">")[0].split("=")[0]) for p in ENGINE_PKGS}
    missing = [k for k, v in have.items() if not v]
    out.append({"id": "engine", "label": "OpenVoice packages",
                "state": "ok" if not missing else "missing",
                "detail": ("librosa " + have.get("librosa", "") + ", silero-vad " +
                           (have.get("silero-vad") or "—")) if not missing
                else "Missing: " + ", ".join(missing),
                "installable": True})

    melo = pkg_version("melotts") or pkg_version("melo")
    uni = unidic_ready()
    if melo and uni:
        m_state, m_detail = "ok", f"version {melo}"
    elif melo:
        m_state = "warn"
        m_detail = (f"version {melo} is installed, but it cannot import until the "
                    "unidic dictionary below is downloaded.")
    else:
        m_state = "missing"
        m_detail = ("Not installed. V2 reads every line with a MeloTTS voice; "
                    "V1 has its own base speakers and does not need this.")
    out.append({"id": "melo", "label": "MeloTTS (base voices for V2)",
                "state": m_state, "detail": m_detail,
                "installable": True, "optional": True})

    # Not a Japanese-only extra: MeloTTS's text front end loads the MeCab
    # tagger at import time, so without this dictionary `import melo` fails
    # for every language. install_melo fetches it; this row is for when
    # that part did not finish.
    out.append({"id": "unidic", "label": "unidic dictionary (MeloTTS needs it)",
                "state": "ok" if uni else ("missing" if melo else "missing"),
                "detail": "MeloTTS imports this at start-up, whatever the language. "
                          "About 500 MB." if not uni else "Downloaded",
                "installable": True, "optional": not melo})

    wm = pkg_version("wavmark")
    out.append({"id": "wavmark", "label": "wavmark (audio watermark)",
                "state": "ok" if wm else "missing",
                "detail": f"version {wm}" if wm else
                "Optional. Adds an inaudible watermark when the switch is on.",
                "installable": True, "optional": True})

    for key in ("v2", "v1"):
        ms = model_state(key)
        out.append({"id": f"model_{key}", "label": f"{ms['label']} checkpoints",
                    "state": "ok" if ms["present"] else "missing",
                    "detail": f"{ms['files']} files, {ms['bytes'] / 1e6:.0f} MB in "
                              f"{Path(ms['dir']).name}/" if ms["present"]
                    else f"Not downloaded ({ms['size']}). Models page.",
                    "installable": True, "model": key,
                    "optional": key == "v1"})
    return out


def pip_version() -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", pkg_version("pip") or "0.0")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def pip(task: Task, args: list[str]) -> int:
    # The progress bar needs pip 24.1+. A venv made by an older Python ships an
    # older pip; one small upgrade first, then every install shows progress.
    if pip_version() < PIP_RAW_MIN:
        task.log("Updating pip so downloads can show progress")
        run_logged(task, [sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    extra = ["--progress-bar", "raw"] if pip_version() >= PIP_RAW_MIN else []
    cache: list[str] = []
    for attempt in range(3):
        rc = run_logged(task, [sys.executable, "-m", "pip", "install", "--upgrade",
                               *extra, *cache, *args], pip_progress=bool(extra))
        if rc == 0:
            return 0
        why = pip_retry_reason(task.lines)
        if why == "raw" and extra:
            # A pip that does not know "raw" exits with "invalid choice" before
            # doing anything; losing the bar is fine, losing the install is not.
            task.log("This pip has no raw progress mode; installing without a bar")
            extra = []
        elif why == "cache" and not cache:
            # pip refused a wheel because the copy in its own HTTP cache no
            # longer matches the index's hash — a half-written cache entry, or
            # something on the way (an antivirus, a proxy) rewrote it. The
            # wheel itself is fine; fetching it fresh is the whole fix.
            task.log("pip's cache handed back a file that does not match its hash; "
                     "fetching fresh copies instead of the cache")
            cache = ["--no-cache-dir"]
        else:
            return rc
    return rc


def pip_retry_reason(lines: list[str]) -> str:
    """What the last pip run tripped on, when it is something a retry fixes."""
    tail = "\n".join(lines[-20:])
    if "invalid choice: 'raw'" in tail:
        return "raw"
    if "DO NOT MATCH THE HASHES" in tail or "HashMismatch" in tail:
        return "cache"
    return ""


def install_torch(task: Task, build: str) -> None:
    gpu = gpu_info()
    if build == "auto" or not build:
        build = recommended_build(gpu)
        task.log(f"Build picked for this machine: {build}"
                 + (f" ({gpu['name']}, driver {gpu['driver']})" if gpu.get("present")
                    else " (no NVIDIA driver found)"))
    args = ["torch", "torchaudio", "--index-url", PYTORCH_INDEX + build]
    if build != "cpu":
        # A CUDA build replacing a CPU one keeps the old wheel's name, so pip
        # thinks nothing changed; force it.
        args.insert(0, "--force-reinstall")
        args.insert(1, "--no-cache-dir")
    task.set(detail=f"Installing PyTorch ({build}) — a few GB, this takes a while")
    rc = pip(task, args)
    torch_facts_cached(force=True)
    if rc != 0:
        raise RuntimeError(f"pip exited with {rc}")
    tf = torch_facts_cached(force=True)
    if tf.get("error"):
        raise RuntimeError(f"PyTorch installed but does not import: {tf['error']}")
    if gpu.get("present") and not tf.get("cuda"):
        raise RuntimeError("PyTorch installed, but it cannot see the card. Pick a "
                           "CUDA build that matches your driver and try again.")
    task.log(f"PyTorch {tf.get('version')} ready"
             + (f" on {tf.get('gpu')}" if tf.get("cuda") else " (CPU)"))


def install_engine(task: Task) -> None:
    task.set(detail="Installing the OpenVoice packages")
    rc = pip(task, ENGINE_PKGS)
    if rc != 0:
        raise RuntimeError(f"pip exited with {rc}")


def install_melo(task: Task) -> None:
    py = python_facts()
    if not py["melo_ok"]:
        task.log(f"Python {py['version']}: MeloTTS pins transformers 4.27 and "
                 "tokenizers 0.13, which have wheels for 3.10 and 3.11 only. "
                 "Trying anyway; if this fails, run the app on Python 3.11.")
    if not shutil.which("git"):
        raise RuntimeError("git is not installed, and pip needs it to fetch MeloTTS. "
                           "Install Git from git-scm.com and try again.")
    task.set(detail="Installing MeloTTS from GitHub")
    rc = pip(task, [MELO_GIT])
    if rc != 0:
        raise RuntimeError(f"pip exited with {rc}")
    # MeloTTS's English front end wants two NLTK corpora it fetches on first
    # use; doing it here means the first take does not stall on a download.
    task.set(detail="Fetching the NLTK data MeloTTS English needs")
    code = ("import nltk\n"
            "for n in ('averaged_perceptron_tagger_eng','averaged_perceptron_tagger','cmudict'):\n"
            "    try: nltk.download(n, quiet=True)\n"
            "    except Exception as e: print('skip', n, e)\n")
    run_logged(task, [sys.executable, "-c", code])
    # MeloTTS's librosa pin drags numpy; make sure it stayed below 2.
    pip(task, ["numpy<2"])
    # Not optional: melo imports the MeCab tagger at start-up, and without
    # the dictionary `import melo` fails for every language, not only Japanese.
    if not unidic_ready():
        install_unidic(task)
    # First import: melo's text front ends fetch a few tokenizers from
    # HuggingFace (a Japanese BERT among them) the first time they load. Doing
    # it here, with the output in this log, beats a first take that sits on
    # "base voice" for minutes with nothing to show.
    task.set(detail="Importing MeloTTS once so it fetches what it needs")
    rc = run_logged(task, [sys.executable, "-c", "import melo.api; print('MeloTTS imports')"])
    if rc != 0:
        raise RuntimeError("MeloTTS installed but does not import; the log above says why "
                           "(a blocked download from huggingface.co is the usual cause)")


def install_unidic(task: Task) -> None:
    task.set(detail="Downloading the unidic dictionary (about 500 MB)")
    if not pkg_version("unidic"):
        pip(task, ["unidic"])
    rc = run_logged(task, [sys.executable, "-m", "unidic", "download"])
    if rc != 0:
        raise RuntimeError(f"unidic download exited with {rc}")


def install_wavmark(task: Task) -> None:
    task.set(detail="Installing wavmark")
    rc = pip(task, ["wavmark"])
    if rc != 0:
        raise RuntimeError(f"pip exited with {rc}")


def install_ffmpeg(task: Task) -> None:
    """A static ffmpeg through the imageio-ffmpeg wheel, copied into tools/.

    pip is the one installer every machine here already has working, the
    wheel carries a build for Windows, macOS and Linux, and tools/ goes on
    PATH for the server and every engine it starts — so no system install,
    no restart of the terminal, nothing outside the app folder.
    """
    task.set(detail="Fetching a static ffmpeg build (imageio-ffmpeg)")
    rc = pip(task, ["imageio-ffmpeg"])
    if rc != 0:
        raise RuntimeError(f"pip exited with {rc}")
    r = subprocess.run([sys.executable, "-c",
                        "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                       capture_output=True, text=True, timeout=120)
    src = Path((r.stdout or "").strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
    if not src or not src.is_file():
        raise RuntimeError("imageio-ffmpeg installed but did not hand over a binary: "
                           + (r.stderr or r.stdout).strip()[-300:])
    TOOLS_DIR.mkdir(exist_ok=True)
    dest = TOOLS_DIR / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
    shutil.copy2(src, dest)
    if platform.system() != "Windows":
        dest.chmod(dest.stat().st_mode | 0o111)
    ensure_tools_path()
    r = subprocess.run([str(dest), "-version"], capture_output=True, text=True, timeout=30)
    first = (r.stdout or r.stderr).strip().splitlines()[:1]
    if r.returncode != 0:
        raise RuntimeError("ffmpeg was copied but does not run: " + " ".join(first))
    task.log(f"ffmpeg ready at {dest}: {first[0] if first else ''}")
    task.log("The engine picks it up the next time it starts.")


INSTALLERS = {
    "torch": install_torch,
    "ffmpeg": install_ffmpeg,
    "engine": install_engine,
    "melo": install_melo,
    "unidic": install_unidic,
    "wavmark": install_wavmark,
}


# --------------------------------------------------------------------------- #
# model downloads
# --------------------------------------------------------------------------- #
def hf_headers(cfg: dict) -> dict:
    tok = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def hf_endpoint(cfg: dict) -> str:
    return (cfg.get("hf_endpoint") or HF_DEFAULT).rstrip("/")


def hf_files(cfg: dict, repo: str) -> list[dict]:
    """[{path, size}] for every file in the repo, via the tree API."""
    url = f"{hf_endpoint(cfg)}/api/models/{repo}/tree/main?recursive=true"
    r = requests.get(url, headers=hf_headers(cfg), timeout=30)
    if r.status_code == 401:
        raise RuntimeError("HuggingFace answered 401: the repo wants a token. "
                           "Add one on the Models page.")
    r.raise_for_status()
    out = []
    for item in r.json():
        if item.get("type") != "file":
            continue
        size = item.get("size") or (item.get("lfs") or {}).get("size") or 0
        out.append({"path": item["path"], "size": int(size)})
    return out


def local_path(model: dict, repo_path: str) -> Path | None:
    name = repo_path.rsplit("/", 1)[-1]
    if name in SKIP_FILES or name.startswith("."):
        return None
    rel = repo_path
    if model["strip"] and rel.startswith(model["strip"]):
        rel = rel[len(model["strip"]):]
    if not rel or ".." in rel.split("/"):
        return None
    return model["dir"] / rel


def download_file(task: Task, url: str, dest: Path, headers: dict, size: int,
                  done_before: int, total: int) -> int:
    """Stream url into dest with resume. Returns bytes written this call."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    hdr = dict(headers)
    if have:
        hdr["Range"] = f"bytes={have}-"
    with requests.get(url, headers=hdr, stream=True, timeout=60) as r:
        if r.status_code == 416:            # the .part is already complete
            pass
        else:
            r.raise_for_status()
            if r.status_code != 206:
                have = 0                    # server ignored the range
            mode = "ab" if have else "wb"
            written = have
            last = time.time()
            with open(part, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if task.cancel:
                        return written - have
                    f.write(chunk)
                    written += len(chunk)
                    if time.time() - last > 0.5:
                        last = time.time()
                        pct = (done_before + written) / total * 100 if total else None
                        task.set(pct=pct, detail=f"{dest.name}  "
                                 f"{written / 1e6:.0f}/{size / 1e6:.0f} MB")
            have = written
    os.replace(part, dest)
    return have


def download_model(task: Task, key: str, cfg: dict) -> None:
    model = MODELS[key]
    zip_url = (cfg.get("zip_urls") or {}).get(key, "").strip()
    if zip_url:
        task.log(f"Trying the zip first: {zip_url}")
        try:
            import_zip_url(task, key, zip_url)
            if model_state(key)["present"]:
                return
            task.log("The zip did not contain the expected files; "
                     "falling back to HuggingFace")
        except Exception as exc:  # noqa: BLE001
            task.log(f"Zip failed ({exc}); falling back to HuggingFace")
    repo = model["repo"]
    task.log(f"Listing {repo} on {hf_endpoint(cfg)}")
    files = hf_files(cfg, repo)
    plan = []
    for f in files:
        dest = local_path(model, f["path"])
        if dest is None:
            continue
        if dest.is_file() and (f["size"] == 0 or dest.stat().st_size == f["size"]):
            continue
        plan.append((f, dest))
    if not plan:
        task.log("Everything is already on disk")
        return
    total = sum(f["size"] for f, _ in plan)
    task.log(f"{len(plan)} file(s), {total / 1e6:.0f} MB to fetch")
    done = 0
    for f, dest in plan:
        if task.cancel:
            task.log("Cancelled — what was fetched so far resumes next time")
            return
        url = f"{hf_endpoint(cfg)}/{repo}/resolve/main/{f['path']}"
        task.log(f"{f['path']}  ({f['size'] / 1e6:.1f} MB)")
        download_file(task, url, dest, hf_headers(cfg), f["size"], done, total)
        done += f["size"]
    st = model_state(key)
    if not st["present"]:
        raise RuntimeError("Download finished but these are still missing: "
                           + ", ".join(st["missing"]))
    task.set(pct=100, detail=f"{model['label']} ready")
    task.log(f"{model['label']} is ready: {st['files']} files, {st['bytes'] / 1e6:.0f} MB")


def extract_zip(task: Task, key: str, zip_path: Path) -> int:
    """Unpack a checkpoint zip into the model folder, dropping any wrapper folder.

    Both of MyShell's zips (and most re-uploads of them) wrap everything in a
    top-level checkpoints/ or checkpoints_v2/ folder; some do not. Either way,
    the files land where engine.py looks.
    """
    import zipfile

    model = MODELS[key]
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        names = [i for i in z.infolist() if not i.is_dir()
                 and not i.filename.startswith("__MACOSX")
                 and ".." not in i.filename.split("/")]
        tops = {i.filename.split("/", 1)[0] for i in names if "/" in i.filename}
        wrapper = ""
        if len(tops) == 1 and all("/" in i.filename for i in names):
            top = next(iter(tops))
            if top not in ("converter", "base_speakers"):
                wrapper = top + "/"
        for info in names:
            rel = info.filename
            if wrapper and rel.startswith(wrapper):
                rel = rel[len(wrapper):]
            if not rel:
                continue
            dest = model["dir"] / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
            if task.cancel:
                break
    task.log(f"Unpacked {n} file(s) into {model['dir'].name}/")
    return n


def import_zip_url(task: Task, key: str, url: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / f"{key}_download.zip"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if task.cancel:
                    return
                f.write(chunk)
                got += len(chunk)
                if total:
                    task.set(pct=got / total * 100,
                             detail=f"zip {got / 1e6:.0f}/{total / 1e6:.0f} MB")
    try:
        extract_zip(task, key, tmp)
    finally:
        tmp.unlink(missing_ok=True)


def import_local(task: Task, key: str, path: str) -> None:
    """A zip or a folder someone already has on disk."""
    p = Path(path).expanduser()
    model = MODELS[key]
    if p.is_file() and p.suffix.lower() == ".zip":
        task.log(f"Unpacking {p}")
        extract_zip(task, key, p)
    elif p.is_dir():
        # Accept the folder itself or one that wraps it.
        src = p
        for cand in (p, p / model["dir"].name, p / "checkpoints", p / "checkpoints_v2"):
            if (cand / "converter" / "checkpoint.pth").is_file():
                src = cand
                break
        task.log(f"Copying {src} into {model['dir'].name}/")
        shutil.copytree(src, model["dir"], dirs_exist_ok=True)
    else:
        raise RuntimeError(f"{p} is neither a zip nor a folder")
    st = model_state(key)
    if not st["present"]:
        raise RuntimeError("Imported, but these are still missing: " + ", ".join(st["missing"]))
    task.log(f"{model['label']} is ready")


def delete_model(key: str) -> None:
    d = MODELS[key]["dir"]
    if d.is_dir():
        shutil.rmtree(d)
