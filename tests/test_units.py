"""Unit tests for OpenVoice Studio's server and manager.

Standard library plus Flask only — `python -m unittest discover tests` needs
nothing beyond requirements.txt. tests/mock_engine.py stands in for engine.py,
so no PyTorch, no checkpoints and no card are needed.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import wave
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# server.py reads DATA_DIR and the engine script at import, so point both
# somewhere disposable before importing it.
_SANDBOX = tempfile.mkdtemp(prefix="ov-tests-")
os.environ["OPENVOICE_STUDIO_DATA"] = _SANDBOX
os.environ["OPENVOICE_STUDIO_ENGINE"] = str(REPO / "tests" / "mock_engine.py")

import manager  # noqa: E402
import server  # noqa: E402


def tearDownModule():
    server.ENGINE.stop()
    shutil.rmtree(_SANDBOX, ignore_errors=True)


def make_wav(path: Path, seconds=1.0, rate=22050):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x10\x00" * int(rate * seconds))
    return path


def wav_frames(data: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes(), w.getframerate()


# --------------------------------------------------------------------------- #
class ScriptParsing(unittest.TestCase):
    def test_named_lines_and_continuations(self):
        lines = server.parse_script("Anna: Hello there.\nhow are you?\n\nBen: Fine.\n")
        self.assertEqual(lines, [{"speaker": "Anna", "text": "Hello there. how are you?"},
                                 {"speaker": "Ben", "text": "Fine."}])
        self.assertEqual(server.speakers_in(lines), ["Anna", "Ben"])

    def test_plain_text_has_no_speaker(self):
        lines = server.parse_script("Just one line.\n\nAnd a second paragraph.")
        self.assertEqual([l["speaker"] for l in lines], ["", ""])
        self.assertEqual(lines[1]["text"], "And a second paragraph.")

    def test_a_clock_time_is_not_a_speaker(self):
        lines = server.parse_script("12:30 is lunch.")
        self.assertEqual(lines, [{"speaker": "", "text": "12:30 is lunch."}])

    def test_fullwidth_colon(self):
        lines = server.parse_script("Mei：你好")
        self.assertEqual(lines[0]["speaker"], "Mei")


class Advice(unittest.TestCase):
    def test_known_failures_get_a_sentence(self):
        self.assertIn("Visual C++", server.advice_for("OSError: [WinError 126] ... c10.dll"))
        self.assertIn("Install", server.advice_for("ModuleNotFoundError: No module named 'torch'"))
        self.assertIn("driver", server.advice_for("CUDA driver version is insufficient"))
        self.assertEqual(server.advice_for("something else"), "")


class LineValidation(unittest.TestCase):
    def test_rejects_empty_and_long(self):
        with self.assertRaises(ValueError):
            server.validate_lines({"lines": []})
        with self.assertRaises(ValueError):
            server.validate_lines({"lines": [{"text": "x" * 700}]})
        with self.assertRaises(ValueError):
            server.validate_lines({"lines": [{"text": "  "}]})

    def test_v1_style_checked_and_defaults_filled(self):
        out = server.validate_lines({"version": "v1", "lines": [{"text": "hi", "speed": "9"}]})
        self.assertEqual(out[0]["base"], {"language": "EN", "style": "default"})
        self.assertEqual(out[0]["speed"], 2.0)       # clamped
        with self.assertRaises(ValueError):
            server.validate_lines({"version": "v1", "lines": [
                {"text": "hi", "base": {"language": "ZH", "style": "angry"}}]})

    def test_v2_speaker_checked(self):
        out = server.validate_lines({"version": "v2", "lines": [
            {"text": "hi", "base": {"language": "FR"}}]})
        self.assertEqual(out[0]["base"], {"language": "FR", "speaker": "FR"})
        with self.assertRaises(ValueError):
            server.validate_lines({"version": "v2", "lines": [
                {"text": "hi", "base": {"language": "EN", "speaker": "EN-Mars"}}]})

    def test_unknown_voice_rejected(self):
        with self.assertRaises(ValueError):
            server.validate_lines({"lines": [{"text": "hi", "voice": "nope"}]})


# --------------------------------------------------------------------------- #
class ManagerBits(unittest.TestCase):
    def test_config_merges_defaults(self):
        cfg = manager.load_config()
        for k in manager.DEFAULT_CONFIG:
            self.assertIn(k, cfg)
        cfg["pause"] = 0.5
        cfg["zip_urls"]["v1"] = "http://x/y.zip"
        manager.save_config(cfg)
        again = manager.load_config()
        self.assertEqual(again["pause"], 0.5)
        self.assertEqual(again["zip_urls"]["v1"], "http://x/y.zip")
        self.assertEqual(again["zip_urls"]["v2"], "")

    def test_local_path_strips_wrapper_and_skips_housekeeping(self):
        v1 = manager.MODELS["v1"]
        self.assertEqual(manager.local_path(v1, "checkpoints/converter/checkpoint.pth"),
                         v1["dir"] / "converter" / "checkpoint.pth")
        self.assertIsNone(manager.local_path(v1, ".gitattributes"))
        self.assertIsNone(manager.local_path(v1, "README.md"))
        self.assertIsNone(manager.local_path(v1, "checkpoints/../x.pth"))
        v2 = manager.MODELS["v2"]
        self.assertEqual(manager.local_path(v2, "converter/config.json"),
                         v2["dir"] / "converter" / "config.json")

    def test_recommended_build_follows_driver(self):
        self.assertEqual(manager.recommended_build({"present": False}), "cpu")
        self.assertEqual(manager.recommended_build({"present": True, "driver": "572.16"}), "cu128")
        self.assertEqual(manager.recommended_build({"present": True, "driver": "561.0"}), "cu126")
        self.assertEqual(manager.recommended_build({"present": True, "driver": "535.1"}), "cu121")
        self.assertEqual(manager.recommended_build({"present": True, "driver": ""}), "cu128")

    def test_extract_zip_drops_wrapper_folder(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        old_dir = manager.MODELS["v2"]["dir"]
        manager.MODELS["v2"]["dir"] = tmp / "checkpoints_v2"
        self.addCleanup(lambda: manager.MODELS["v2"].__setitem__("dir", old_dir))
        z = tmp / "c.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("checkpoints_v2/converter/config.json", "{}")
            zf.writestr("checkpoints_v2/base_speakers/ses/en-default.pth", "x")
            zf.writestr("__MACOSX/._junk", "x")
        task = manager.Task("t", "t")
        n = manager.extract_zip(task, "v2", z)
        self.assertEqual(n, 2)
        self.assertTrue((tmp / "checkpoints_v2" / "converter" / "config.json").is_file())
        self.assertTrue((tmp / "checkpoints_v2" / "base_speakers" / "ses" / "en-default.pth").is_file())

    def test_model_state_reports_missing(self):
        st = manager.model_state("v2")
        self.assertIn("converter/checkpoint.pth", st["missing"] or ["converter/checkpoint.pth"])
        self.assertIn("label", st)

    def test_pip_progress_lines_drive_the_bar_not_the_log(self):
        fake = ("import sys\n"
                "print('Collecting torch')\n"
                "print('  Downloading torch-9.whl.metadata (1 kB)')\n"
                "print('Downloading torch-9.whl (554.6 MB)')\n"
                "for n in (0, 200000000, 554600000): print(f'Progress {n} of 554600000')\n"
                "print('Installing collected packages: torch')\n")
        task = manager.Task("install", "t")
        rc = manager.run_logged(task, [sys.executable, "-c", fake], pip_progress=True)
        self.assertEqual(rc, 0)
        joined = "\n".join(task.lines)
        self.assertNotIn("Progress 200000000", joined)       # the bar took it
        self.assertIn("torch-9.whl: 555 MB done", joined)
        self.assertIsNone(task.pct)                           # cleared at "Installing"
        self.assertEqual(task.detail, "Installing the downloaded packages")
        # and without the flag, the lines go to the log untouched
        task2 = manager.Task("install", "t")
        manager.run_logged(task2, [sys.executable, "-c", fake])
        self.assertIn("Progress 200000000 of 554600000", "\n".join(task2.lines))

    def test_pip_retry_reasons(self):
        self.assertEqual(manager.pip_retry_reason(
            ["Downloading x.whl", "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE "
             "REQUIREMENTS FILE.", "    unknown package:", "FAILED: pip exited with 1"]), "cache")
        self.assertEqual(manager.pip_retry_reason(
            ["option --progress-bar: invalid choice: 'raw' (choose from 'on', 'off')"]), "raw")
        self.assertEqual(manager.pip_retry_reason(["ERROR: No matching distribution found"]), "")

    def test_tools_dir_goes_first_on_path(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        old_tools, old_path = manager.TOOLS_DIR, os.environ.get("PATH", "")
        manager.TOOLS_DIR = tmp
        try:
            manager.ensure_tools_path()
            self.assertTrue(os.environ["PATH"].startswith(str(tmp) + os.pathsep))
            manager.ensure_tools_path()                       # idempotent
            self.assertEqual(os.environ["PATH"].count(str(tmp)), 1)
        finally:
            manager.TOOLS_DIR = old_tools
            os.environ["PATH"] = old_path

    def test_task_log_and_view(self):
        t = manager.Task("k", "title")
        t.log("one")
        t.set(pct=12.34, detail="d")
        v = t.view()
        self.assertEqual(v["pct"], 12.3)
        self.assertEqual(len(v["lines"]), 1)
        self.assertEqual(t.view(since=1)["lines"], [])


# --------------------------------------------------------------------------- #
class EngineProtocol(unittest.TestCase):
    def setUp(self):
        server.ENGINE.stop()
        for k in ("MOCK_ENGINE_FAIL", "MOCK_ENGINE_CRASH", "MOCK_ENGINE_SLOW_HELLO"):
            os.environ.pop(k, None)

    def tearDown(self):
        server.ENGINE.stop()
        for k in ("MOCK_ENGINE_FAIL", "MOCK_ENGINE_CRASH", "MOCK_ENGINE_SLOW_HELLO"):
            os.environ.pop(k, None)

    def test_start_hello_speak_stop(self):
        e = server.ENGINE
        e.start()
        self.assertEqual(e.state, "ready")
        self.assertEqual(e.hello["torch"], "0.0-mock")
        out = Path(_SANDBOX) / "t1"
        seen = []
        res = e.call("speak", version="v2", out_dir=str(out), opts={"pause": 0.5},
                     lines=[{"text": "a", "base": {}}, {"text": "b", "base": {}}],
                     on_progress=seen.append)
        self.assertEqual(len(res["lines"]), 2)
        self.assertTrue(Path(res["file"]).is_file())
        self.assertGreaterEqual(len(seen), 2)
        # two 0.3 s tones and one 0.5 s gap
        with wave.open(res["file"], "rb") as w:
            self.assertAlmostEqual(w.getnframes() / w.getframerate(), 1.1, places=2)
        e.stop()
        self.assertEqual(e.state, "stopped")

    def test_engine_error_is_raised_not_swallowed(self):
        os.environ["MOCK_ENGINE_FAIL"] = "speak"
        e = server.ENGINE
        e.start()
        with self.assertRaises(server.EngineError) as cm:
            e.call("speak", version="v2", out_dir=_SANDBOX, lines=[{"text": "a"}])
        self.assertIn("mock failure", str(cm.exception))
        self.assertTrue(e.alive())          # one failed command does not kill it

    def test_stop_does_not_wait_behind_a_slow_start(self):
        os.environ["MOCK_ENGINE_SLOW_HELLO"] = "20"
        e = server.ENGINE
        threading.Thread(target=lambda: self._swallow(e.start), daemon=True).start()
        for _ in range(100):
            if e.state == "starting" and e.alive():
                break
            time.sleep(0.05)
        self.assertEqual(e.state, "starting")
        t0 = time.time()
        e.stop()
        self.assertLess(time.time() - t0, 5)         # not the 20 s hello
        self.assertEqual(e.state, "stopped")
        self.assertFalse(e.alive())
        os.environ.pop("MOCK_ENGINE_SLOW_HELLO", None)
        e.start()                                    # and it comes back clean
        self.assertEqual(e.state, "ready")

    @staticmethod
    def _swallow(fn):
        try:
            fn()
        except server.EngineError:
            pass

    def test_crash_mid_command_fails_the_call_and_marks_state(self):
        os.environ["MOCK_ENGINE_CRASH"] = "1"
        e = server.ENGINE
        e.start()
        with self.assertRaises(server.EngineError):
            e.call("speak", version="v2", out_dir=_SANDBOX,
                   lines=[{"text": "a"}, {"text": "b"}], timeout=20)
        time.sleep(0.2)
        self.assertEqual(e.state, "error")
        self.assertIn("exited", e.error)
        self.assertIn("crashing on purpose", e.error)   # the engine's own last words
        # and it comes back
        e.start()
        self.assertEqual(e.state, "ready")


# --------------------------------------------------------------------------- #
class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = server.app.test_client()
        if not getattr(server, "_test_worker", None):
            server._test_worker = threading.Thread(target=server.job_worker, daemon=True)
            server._test_worker.start()

    def setUp(self):
        os.environ.pop("MOCK_ENGINE_FAIL", None)

    def test_status_shape(self):
        r = self.client.get("/api/status?fast=1")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        for k in ("python", "gpu", "models", "engine", "config", "ready"):
            self.assertIn(k, j)
        self.assertNotIn("hf_token", j["config"])

    def test_index_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"OpenVoice Studio", r.data)

    def test_bases(self):
        j = self.client.get("/api/bases").get_json()
        self.assertIn("EN", j["v2"])
        self.assertEqual(j["v2"]["EN"]["speakers"][0]["key"], "EN-Default")
        self.assertIn("whispering", j["v1"]["EN"]["styles"])

    def test_config_round_trip_and_clamps(self):
        r = self.client.post("/api/config", json={"pause": 9, "tau": -1, "version": "v9",
                                                  "free_after": 1, "hf_token": "leak"})
        j = r.get_json()["config"]
        self.assertEqual(j["pause"], 5.0)
        self.assertEqual(j["tau"], 0.0)
        self.assertIn(j["version"], ("v1", "v2"))
        self.assertTrue(j["free_after"])
        self.assertNotIn("hf_token", j)
        self.client.post("/api/config", json={"pause": 0.35, "tau": 0.3, "free_after": False})

    def test_voice_upload_bad_type(self):
        r = self.client.post("/api/voices", data={"file": (io.BytesIO(b"x"), "notes.txt")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_parse_endpoint(self):
        j = self.client.post("/api/parse", json={"text": "A: one\nB: two"}).get_json()
        self.assertEqual(j["speakers"], ["A", "B"])

    def test_full_take_with_a_cloned_voice(self):
        clip = make_wav(Path(_SANDBOX) / "clip.wav")
        with open(clip, "rb") as f:
            r = self.client.post("/api/voices", data={"file": (f, "my clip.wav"), "name": "Me"},
                                 content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.data)
        vid = r.get_json()["voice"]["id"]
        self.assertEqual(self.client.get(f"/api/voices/{vid}/audio").status_code, 200)

        r = self.client.post("/api/speak", json={"version": "v2", "title": "Test take", "lines": [
            {"text": "Hello", "voice": vid, "speaker": "Me", "base": {"language": "EN", "speaker": "EN-US"}},
            {"text": "World", "voice": "", "base": {"language": "ES"}}],
            "opts": {"pause": 0.25}})
        self.assertEqual(r.status_code, 200, r.data)
        jid = r.get_json()["job"]["id"]
        job = None
        for _ in range(200):
            job = self.client.get(f"/api/jobs/{jid}").get_json()["job"]
            if job["state"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.05)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(job["pct"], 100)
        takes = self.client.get("/api/takes").get_json()["takes"]
        take = next(t for t in takes if t["id"] == job["take"])
        self.assertEqual(take["title"], "Test take")
        self.assertEqual(len(take["lines"]), 2)
        self.assertEqual(take["lines"][0]["voice_name"], "Me")
        self.assertEqual(take["lines"][1]["voice_name"], "")
        self.assertEqual(take["pause"], 0.25)       # the player needs it to find the current line
        audio = self.client.get(f"/api/takes/{take['id']}/audio")
        self.assertEqual(audio.status_code, 200)
        frames, rate = wav_frames(audio.data)
        self.assertAlmostEqual(frames / rate, 0.3 + 0.25 + 0.3, places=2)
        self.assertEqual(self.client.get(f"/api/takes/{take['id']}/line/1").status_code, 200)
        self.assertEqual(self.client.get(f"/api/takes/{take['id']}/line/2").status_code, 404)
        # the embedding was cached under the voice's file stem
        self.assertTrue(list((Path(_SANDBOX) / "se").glob("*.pth")))
        # rename, delete
        self.client.post(f"/api/takes/{take['id']}", json={"title": "Renamed"})
        self.assertEqual(self.client.get("/api/takes").get_json()["takes"][0]["title"], "Renamed")
        self.assertEqual(self.client.delete(f"/api/takes/{take['id']}").status_code, 200)
        self.assertFalse((Path(_SANDBOX) / "takes" / take["id"]).exists())
        self.assertEqual(self.client.delete(f"/api/voices/{vid}").status_code, 200)
        self.assertFalse(list((Path(_SANDBOX) / "se").glob("*.pth")))

    def test_failed_take_reports_the_error(self):
        os.environ["MOCK_ENGINE_FAIL"] = "speak"
        server.ENGINE.stop()           # so the next start picks up the env
        r = self.client.post("/api/speak", json={"lines": [{"text": "x"}]})
        jid = r.get_json()["job"]["id"]
        job = None
        for _ in range(200):
            job = self.client.get(f"/api/jobs/{jid}").get_json()["job"]
            if job["state"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.05)
        self.assertEqual(job["state"], "error")
        self.assertIn("mock failure", job["error"])
        os.environ.pop("MOCK_ENGINE_FAIL", None)
        server.ENGINE.stop()

    def test_speak_validation_errors_are_400(self):
        r = self.client.post("/api/speak", json={"lines": [{"text": ""}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_engine_log_and_free(self):
        server.ENGINE.start()
        j = self.client.get("/api/engine/log").get_json()
        self.assertEqual(j["state"], "ready")
        self.assertTrue(any("mock" in l or "Engine" in l for l in j["lines"]))
        self.assertEqual(self.client.post("/api/engine/free").status_code, 200)
        self.assertEqual(self.client.post("/api/engine/stop").get_json()["state"], "stopped")

    def test_models_listing_and_hf_settings(self):
        j = self.client.get("/api/models").get_json()
        self.assertEqual([m["key"] for m in j["models"]], ["v2", "v1"])
        r = self.client.post("/api/hf/settings", json={"hf_endpoint": "https://hf-mirror.com/",
                                                       "hf_token": "abc"})
        self.assertEqual(r.get_json()["hf_endpoint"], "https://hf-mirror.com")
        self.assertTrue(r.get_json()["has_token"])
        self.assertEqual(self.client.post("/api/hf/settings", json={"hf_endpoint": "ftp"}).status_code, 400)
        self.client.post("/api/hf/settings", json={"hf_endpoint": "", "hf_token": ""})

    def test_unknown_dep_install_is_404(self):
        self.assertEqual(self.client.post("/api/deps/nothing/install").status_code, 404)


if __name__ == "__main__":
    unittest.main()
