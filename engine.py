"""
engine.py - the OpenVoice worker.

server.py starts this as a child process (`python engine.py`) and talks to it
over pipes: one JSON object per line on stdin in, one per line on stdout out.
Everything the models print goes to stderr, which the server shows as the
engine console. Running the model in its own process is what lets the Engine
page free the card with one click (stop the worker), lets a CUDA out-of-memory
kill nothing but the take that caused it, and lets the Engine page install
PyTorch into the environment without restarting the server that is doing the
installing.

Requests look like {"id": "..", "cmd": "speak", ...}. Replies look like
{"id": "..", "ok": true, "result": {...}} or {"id": "..", "ok": false,
"error": ".."}. A long command also sends {"id": "..", "event": "progress",
...} lines while it runs.

Commands:
  hello    torch/CUDA/card facts, nothing loaded
  status   what is resident and how much of the card it holds
  load     load the converter for a version ("v1" or "v2")
  embed    turn a reference clip into a tone-colour embedding (cached)
  speak    read a list of lines, each in a chosen voice, and join them
  unload   drop every model and free the card
  cancel   stop the running speak between lines
  quit
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# The protocol owns stdout. OpenVoice and MeloTTS both print as they work, so
# the real stdout is kept on a private handle and everything else is sent to
# stderr, where the server reads it as the engine console.
# sys.__stdout__ is the original stream object, kept by Python itself — no
# os.dup, no second handle on the pipe, which is where Windows gets picky.
_PROTO = sys.__stdout__
try:
    _PROTO.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
sys.stdout = sys.stderr

CKPT_V1 = APP_DIR / "checkpoints"
CKPT_V2 = APP_DIR / "checkpoints_v2"

# V2 base speakers, as MeloTTS names them. The embedding file for each is
# `base_speakers/ses/<key>.pth`, where key is the speaker name lower-cased
# with `_` turned into `-` — the rule demo_part3.ipynb applies.
V2_LANGUAGES = {
    "EN": ["EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"],
    "EN_NEWEST": ["EN-Newest"],
    "ES": ["ES"],
    "FR": ["FR"],
    "ZH": ["ZH"],
    "JP": ["JP"],
    "KR": ["KR"],
}
V1_STYLES = ["default", "whispering", "shouting", "excited", "cheerful",
             "terrified", "angry", "sad", "friendly"]


def log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def send(obj: dict) -> None:
    _PROTO.write(json.dumps(obj) + "\n")
    _PROTO.flush()


def ses_key(speaker: str) -> str:
    return speaker.lower().replace("_", "-")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def release_melo_bert() -> None:
    """Melo keeps BERT weights outside its TTS objects, in module globals.

    Only inspect modules already imported: importing a language here can
    download its tokenizer. Clear both cache shapes used by upstream Melo.
    """
    for name, module in tuple(sys.modules.items()):
        if name.startswith("melo.text.") and name.endswith("_bert") and module:
            if hasattr(module, "model"):
                module.model = None
            models = getattr(module, "models", None)
            if isinstance(models, dict):
                models.clear()


class Cancelled(Exception):
    pass


class Engine:
    def __init__(self) -> None:
        self.torch = None
        self.device = "cpu"
        self.converter = None
        self.converter_version = ""
        self.watermark = False
        self.v1_tts: dict[str, object] = {}       # "EN"/"ZH" -> BaseSpeakerTTS
        self.v2_tts: dict[str, object] = {}       # melo language -> TTS
        self.vad = None
        self.bert_language = ""
        self.cancel_flag = False
        self.busy = ""

    # ------------------------------------------------------------------ #
    # facts
    # ------------------------------------------------------------------ #
    def import_torch(self):
        if self.torch is None:
            log("Importing PyTorch (the first time on a machine can take a minute)")
            t0 = time.time()
            # While it runs, say so every 30 s, and after 90 s dump every
            # thread's Python stack to the console: an import that is merely
            # slow (antivirus reading 3 GB of DLLs) and one that is stuck look
            # the same from outside, and the stack says which module it is in.
            stop = threading.Event()

            def heartbeat():
                while not stop.wait(30):
                    log(f"still importing PyTorch after {time.time() - t0:.0f}s "
                        "(a first import on a slow disk or behind an antivirus "
                        "can take several minutes)")

            threading.Thread(target=heartbeat, daemon=True).start()
            faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
            try:
                import torch  # noqa: WPS433 - deliberately late
            finally:
                stop.set()
                faulthandler.cancel_dump_traceback_later()
            self.torch = torch
            log(f"PyTorch imported in {time.time() - t0:.1f}s")
            want = os.environ.get("OPENVOICE_DEVICE", "").strip().lower()
            if want in ("cpu", "cuda"):
                self.device = want if (want == "cpu" or torch.cuda.is_available()) else "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log(f"PyTorch {torch.__version__}, device {self.device}")
        return self.torch

    def hello(self) -> dict:
        torch = self.import_torch()
        info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                "device": self.device, "python": sys.version.split()[0],
                "gpu": "", "vram_total": 0}
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu"] = props.name
                info["vram_total"] = int(props.total_memory)
            except Exception as exc:  # noqa: BLE001 - a card torch sees but cannot query
                log(f"Could not read the card's properties: {exc}")
                info["gpu"] = "NVIDIA (CUDA)"
            info["cuda_version"] = torch.version.cuda
        try:
            import melo  # noqa: F401
            info["melo"] = True
        except Exception:  # noqa: BLE001
            info["melo"] = False
        return info

    def status(self) -> dict:
        torch = self.torch
        out = {"device": self.device, "converter": self.converter_version,
               "v1": sorted(self.v1_tts), "v2": sorted(self.v2_tts),
               "busy": self.busy, "vram_used": 0, "vram_reserved": 0}
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            out["vram_used"] = int(torch.cuda.memory_allocated())
            out["vram_reserved"] = int(torch.cuda.memory_reserved())
        return out

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load_converter(self, version: str, watermark: bool = False):
        if self.converter is not None and self.converter_version == version \
                and self.watermark == watermark:
            return self.converter
        self.import_torch()
        from openvoice.api import ToneColorConverter
        root = CKPT_V2 if version == "v2" else CKPT_V1
        cfg = root / "converter" / "config.json"
        ckpt = root / "converter" / "checkpoint.pth"
        if not cfg.is_file() or not ckpt.is_file():
            raise FileNotFoundError(
                f"OpenVoice {version.upper()} converter is not downloaded "
                f"(looked for {ckpt.relative_to(APP_DIR)}). Get it from the Models page.")
        self.converter = None
        t0 = time.time()
        conv = ToneColorConverter(str(cfg), device=self.device,
                                  enable_watermark=watermark)
        conv.load_ckpt(str(ckpt))
        self.converter = conv
        self.converter_version = version
        self.watermark = watermark
        log(f"Converter {version} ready in {time.time() - t0:.1f}s "
            f"(sampling rate {conv.hps.data.sampling_rate})")
        return conv

    def v1_base(self, lang: str):
        """lang is "EN" or "ZH"."""
        if lang in self.v1_tts:
            return self.v1_tts[lang]
        from openvoice.api import BaseSpeakerTTS
        root = CKPT_V1 / "base_speakers" / lang
        if not (root / "checkpoint.pth").is_file():
            raise FileNotFoundError(
                f"The V1 {lang} base speaker is not downloaded. Get OpenVoice V1 from the Models page.")
        t0 = time.time()
        tts = BaseSpeakerTTS(str(root / "config.json"), device=self.device)
        tts.load_ckpt(str(root / "checkpoint.pth"))
        self.v1_tts[lang] = tts
        log(f"V1 base speaker {lang} ready in {time.time() - t0:.1f}s")
        return tts

    def v2_base(self, language: str):
        # The language BERTs are not owned by the evicted TTS instances.
        # Keep only the current language's feature models on the card.
        if self.bert_language != language:
            release_melo_bert()
            self.bert_language = language
            self.empty_cache()
        if language in self.v2_tts:
            return self.v2_tts[language]
        try:
            from melo.api import TTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MeloTTS is not installed, and OpenVoice V2 needs it for the base "
                f"voice. Install it from the Engine page. ({exc})") from exc
        # Only two base models stay resident. They are small, but a script that
        # walks through six languages would otherwise keep six of them on an
        # 8 GB card that also has to hold the converter.
        while len(self.v2_tts) >= 2:
            old = next(iter(self.v2_tts))
            log(f"Releasing MeloTTS {old} to make room")
            del self.v2_tts[old]
            self.empty_cache()
        t0 = time.time()
        tts = TTS(language=language, device=self.device)
        self.v2_tts[language] = tts
        log(f"MeloTTS {language} ready in {time.time() - t0:.1f}s; "
            f"speakers {list(tts.hps.data.spk2id)}")
        return tts

    def source_se(self, version: str, base: dict):
        torch = self.torch
        if version == "v1":
            lang = base.get("language", "EN")
            style = base.get("style", "default")
            root = CKPT_V1 / "base_speakers" / lang
            if lang == "ZH":
                name = "zh_default_se.pth"
            else:
                name = "en_default_se.pth" if style == "default" else "en_style_se.pth"
            path = root / name
        else:
            key = ses_key(base.get("speaker", "EN-Default"))
            path = CKPT_V2 / "base_speakers" / "ses" / f"{key}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Base speaker embedding missing: {path.relative_to(APP_DIR)}")
        return torch.load(str(path), map_location=self.device)

    def empty_cache(self) -> None:
        import gc
        gc.collect()
        if self.torch is not None and self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def unload(self) -> dict:
        self.converter = None
        self.converter_version = ""
        self.v1_tts.clear()
        self.v2_tts.clear()
        self.vad = None
        release_melo_bert()
        self.bert_language = ""
        self.empty_cache()
        log("Everything unloaded; the card is free")
        return self.status()

    # ------------------------------------------------------------------ #
    # reference clips -> tone colour embeddings
    # ------------------------------------------------------------------ #
    def speech_segments(self, audio16k, sr=16000) -> list[tuple[int, int]]:
        """(start, end) sample pairs of speech at 16 kHz.

        Silero VAD when it is installed — the same detector upstream uses
        through whisper-timestamped, without the 1.5 GB Whisper that package
        drags in. Without it, an energy split from librosa, which is cruder but
        still keeps silence out of the embedding.
        """
        torch = self.torch
        try:
            if self.vad is None:
                from silero_vad import get_speech_timestamps, load_silero_vad
                self.vad = (load_silero_vad(), get_speech_timestamps)
            model, get_ts = self.vad
            wav = torch.as_tensor(audio16k, dtype=torch.float32)
            stamps = get_ts(wav, model, sampling_rate=sr,
                            min_speech_duration_ms=100,
                            min_silence_duration_ms=1000)
            return [(int(s["start"]), int(s["end"])) for s in stamps]
        except ImportError:
            log("silero-vad is not installed; using an energy-based split instead")
        except Exception as exc:  # noqa: BLE001
            log(f"Silero VAD failed ({exc}); using an energy-based split instead")
        import librosa
        ivals = librosa.effects.split(audio16k, top_db=35, frame_length=1024, hop_length=256)
        return [(int(s), int(e)) for s, e in ivals]

    def embed(self, clip: str, version: str, cache_dir: str, req_id: str = "") -> dict:
        import librosa
        import numpy as np
        import soundfile

        torch = self.import_torch()
        conv = self.load_converter(version, self.watermark)
        src = Path(clip)
        if not src.is_file():
            raise FileNotFoundError(f"Reference clip not found: {clip}")
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        key = f"{src.stem}_{version}_{file_hash(src)}"
        se_path = cache / f"{key}.pth"
        if se_path.is_file():
            return {"se": str(se_path), "cached": True}

        self.progress(req_id, stage="embed", msg=f"Listening to {src.name}")
        sr = int(conv.hps.data.sampling_rate)
        audio, _ = librosa.load(str(src), sr=sr, mono=True)
        audio16, _ = librosa.load(str(src), sr=16000, mono=True)
        segs = self.speech_segments(audio16)
        if segs:
            scale = sr / 16000.0
            parts = [audio[int(s * scale): int(e * scale)] for s, e in segs]
            active = np.concatenate(parts) if parts else audio
        else:
            active = audio
        dur = len(active) / sr
        log(f"{src.name}: {len(audio) / sr:.1f}s, {dur:.1f}s of speech after VAD")
        if dur < 1.0:
            raise ValueError(
                "Less than a second of speech was found in that clip. Use a clip "
                "with clear speech, ideally 5 to 30 seconds of it.")

        # ~10 s pieces, as upstream does: the embedding is the mean over pieces,
        # and one long piece would weight a single breath or pause too heavily.
        split = 10.0
        n = max(1, int(round(dur / split)))
        step = len(active) / n
        seg_dir = cache / key
        seg_dir.mkdir(exist_ok=True)
        pieces = []
        for i in range(n):
            a = int(i * step)
            b = len(active) if i == n - 1 else int((i + 1) * step)
            p = seg_dir / f"seg{i}.wav"
            soundfile.write(str(p), active[a:b], sr)
            pieces.append(str(p))
        self.progress(req_id, stage="embed", msg=f"Extracting tone colour from {n} piece(s)")
        with torch.no_grad():
            conv.extract_se(pieces, se_save_path=str(se_path))
        shutil.rmtree(seg_dir, ignore_errors=True)
        return {"se": str(se_path), "cached": False, "speech_seconds": round(dur, 1)}

    # ------------------------------------------------------------------ #
    # speaking
    # ------------------------------------------------------------------ #
    def progress(self, req_id: str, **kw) -> None:
        if req_id:
            send({"id": req_id, "event": "progress", **kw})

    def check_cancel(self) -> None:
        if self.cancel_flag:
            raise Cancelled()

    def base_tts(self, version: str, base: dict, text: str, out: Path, speed: float) -> None:
        if version == "v1":
            lang = base.get("language", "EN")
            tts = self.v1_base(lang)
            style = base.get("style", "default") if lang == "EN" else "default"
            tts.tts(text, str(out), speaker=style,
                    language="Chinese" if lang == "ZH" else "English", speed=speed)
        else:
            language = base.get("language", "EN")
            tts = self.v2_base(language)
            spk2id = tts.hps.data.spk2id
            speaker = base.get("speaker") or next(iter(spk2id))
            if speaker not in spk2id:
                raise ValueError(f"MeloTTS {language} has no speaker {speaker!r}; "
                                 f"it has {list(spk2id)}")
            tts.tts_to_file(text, spk2id[speaker], str(out), speed=speed, quiet=True)

    def speak(self, req: dict) -> dict:
        import librosa
        import numpy as np
        import soundfile

        torch = self.import_torch()
        version = req.get("version", "v2")
        opts = req.get("opts", {})
        lines = req.get("lines", [])
        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        req_id = req.get("id", "")
        pause = float(opts.get("pause", 0.35))
        tau = float(opts.get("tau", 0.3))
        watermark = bool(opts.get("watermark", False))
        message = str(opts.get("message", "@MyShell"))[:8] or "@MyShell"
        if not lines:
            raise ValueError("Nothing to read")

        self.cancel_flag = False
        self.busy = "speak"
        conv = src_se = tgt_se = None
        completed = False
        try:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            conv = self.load_converter(version, watermark)
            sr = int(conv.hps.data.sampling_rate)
            clips: list[np.ndarray] = []
            files: list[str] = []
            tmp = out_dir / "base.wav"
            t_start = time.time()
            for i, line in enumerate(lines):
                self.check_cancel()
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                base = line.get("base") or {}
                speed = float(line.get("speed", 1.0))
                self.progress(req_id, line=i, total=len(lines), stage="tts",
                              msg=f"Line {i + 1}/{len(lines)}: base voice")
                self.base_tts(version, base, text, tmp, speed)
                self.check_cancel()
                final = out_dir / f"line{i:03d}.wav"
                se_path = line.get("se")
                if se_path:
                    self.progress(req_id, line=i, total=len(lines), stage="convert",
                                  msg=f"Line {i + 1}/{len(lines)}: cloning tone colour")
                    src_se = self.source_se(version, base)
                    tgt_se = torch.load(str(se_path), map_location=self.device)
                    with torch.no_grad():
                        conv.convert(audio_src_path=str(tmp), src_se=src_se, tgt_se=tgt_se,
                                     output_path=str(final), tau=tau, message=message,
                                     check_cancel=self.check_cancel)
                    self.check_cancel()
                else:
                    # Base voice only — still resampled to the converter's rate so
                    # every line in the take shares one sample rate.
                    audio, _ = librosa.load(str(tmp), sr=sr, mono=True)
                    soundfile.write(str(final), audio, sr)
                audio, _ = librosa.load(str(final), sr=sr, mono=True)
                clips.append(audio.astype(np.float32))
                files.append(str(final))
                self.progress(req_id, line=i, total=len(lines), stage="done",
                              msg=f"Line {i + 1}/{len(lines)}: {len(audio) / sr:.1f}s",
                              file=str(final), seconds=round(len(audio) / sr, 2))
            if not clips:
                raise ValueError("Every line was empty")
            gap = np.zeros(int(sr * pause), dtype=np.float32)
            joined = []
            for k, c in enumerate(clips):
                if k:
                    joined.append(gap)
                joined.append(c)
            full = np.concatenate(joined)
            take = out_dir / "take.wav"
            soundfile.write(str(take), full, sr)
            tmp.unlink(missing_ok=True)
            elapsed = time.time() - t_start
            log(f"Take done: {len(clips)} line(s), {len(full) / sr:.1f}s of audio "
                f"in {elapsed:.1f}s")
            result = {"file": str(take), "lines": files, "seconds": round(len(full) / sr, 2),
                      "sample_rate": sr, "elapsed": round(elapsed, 1)}
            if self.device.startswith("cuda") and torch.cuda.is_available():
                result["vram_peak"] = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            completed = True
            return result
        finally:
            # Drop local references before empty_cache. Clearing only the
            # Engine attributes leaves the converter alive in this frame.
            conv = src_se = tgt_se = None
            if opts.get("free_after") or not completed:
                self.unload()
            self.busy = ""


def reader(q: "queue.Queue[dict | None]", eng: Engine) -> None:
    """stdin -> queue. `cancel` is handled here so it reaches a running speak."""
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            log(f"Ignored a line that is not JSON: {raw[:80]}")
            continue
        if req.get("cmd") == "cancel":
            eng.cancel_flag = True
            send({"id": req.get("id", ""), "ok": True, "result": {"cancelling": True}})
            continue
        q.put(req)
    q.put(None)


def main() -> None:
    eng = Engine()
    q: "queue.Queue[dict | None]" = queue.Queue()
    threading.Thread(target=reader, args=(q, eng), daemon=True).start()
    send({"id": "", "event": "ready", "pid": os.getpid()})
    log(f"Worker started (pid {os.getpid()}, Python {sys.version.split()[0]}); "
        "waiting for the server's first command")
    while True:
        req = q.get()
        if req is None:
            break
        rid = req.get("id", "")
        cmd = req.get("cmd", "")
        try:
            if cmd == "hello":
                result = eng.hello()
            elif cmd == "status":
                result = eng.status()
            elif cmd == "load":
                eng.load_converter(req.get("version", "v2"), bool(req.get("watermark", False)))
                result = eng.status()
            elif cmd == "embed":
                result = eng.embed(req["clip"], req.get("version", "v2"),
                                   req["cache_dir"], rid)
            elif cmd == "speak":
                req["id"] = rid
                result = eng.speak(req)
            elif cmd == "unload":
                result = eng.unload()
            elif cmd == "quit":
                send({"id": rid, "ok": True, "result": {}})
                break
            else:
                raise ValueError(f"Unknown command {cmd!r}")
            send({"id": rid, "ok": True, "result": result})
        except Cancelled:
            send({"id": rid, "ok": False, "error": "Cancelled", "cancelled": True})
        except Exception as exc:  # noqa: BLE001
            text = str(exc) or exc.__class__.__name__
            if "out of memory" in text.lower():
                # Clear unwound model frames as well as the owner's caches.
                # A traceback can otherwise retain the failed CUDA tensors.
                log("".join(traceback.format_exception(exc)).rstrip())
                traceback.clear_frames(exc.__traceback__)
                eng.unload()
                text = ("The card ran out of memory; the worker released its "
                        "loaded models. Close other GPU programs or shorten the "
                        "failing line, then retry. CPU mode is available through "
                        f"OPENVOICE_DEVICE=cpu. ({text.splitlines()[0]})")
            else:
                log("".join(traceback.format_exception(exc)).rstrip())
            send({"id": rid, "ok": False, "error": text})
    log("Worker stopped")


if __name__ == "__main__":
    main()
