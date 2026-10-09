"""Exercise worker ownership, rather than the protocol-only mock engine."""
import contextlib
import importlib.util
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

with contextlib.redirect_stdout(sys.stdout):
    spec = importlib.util.spec_from_file_location("worker", Path(__file__).resolve().parents[1] / "engine.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)


class Model:
    pass


class WorkerMemory(unittest.TestCase):
    def test_success_releases_local_converter_before_collecting_memory(self):
        class Audio(list):
            def astype(self, dtype):
                return self
        eng = worker.Engine()
        eng.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        refs, collected = [], []
        def load(*args):
            eng.converter = Model()
            eng.converter.hps = types.SimpleNamespace(data=types.SimpleNamespace(sampling_rate=24))
            refs.append(weakref.ref(eng.converter))
            return eng.converter
        eng.load_converter = load
        eng.base_tts = mock.Mock()
        eng.empty_cache = lambda: collected.append(refs[0]() is None)
        np = types.SimpleNamespace(float32=float, zeros=lambda n, **kw: Audio([0] * n),
                                   concatenate=lambda clips: Audio(sum(clips, [])))
        librosa = types.SimpleNamespace(load=lambda *a, **kw: (Audio([0.1] * 24), 24))
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.dict(sys.modules, {"numpy": np, "librosa": librosa,
                                           "soundfile": types.SimpleNamespace(write=mock.Mock())}):
            result = eng.speak({"out_dir": root, "lines": [{"text": "Hello"}],
                                "opts": {"free_after": True}})
        self.assertEqual(result["seconds"], 1.0)
        self.assertEqual(collected, [True])

    def test_unload_clears_both_melo_cache_shapes_without_importing_languages(self):
        singular = types.ModuleType("melo.text.english_bert")
        plural = types.ModuleType("melo.text.japanese_bert")
        singular.model = Model()
        plural.model = Model()
        plural.models = {"japanese": plural.model}
        refs = [weakref.ref(singular.model), weakref.ref(plural.model)]
        with mock.patch.dict(sys.modules, {singular.__name__: singular, plural.__name__: plural}):
            eng = worker.Engine()
            eng.unload()
            self.assertEqual(plural.models, {})
            self.assertTrue(all(ref() is None for ref in refs))

    def test_language_switch_releases_feature_cache_even_with_cached_tts(self):
        eng = worker.Engine()
        eng.bert_language = "EN"
        eng.v2_tts["FR"] = Model()
        with mock.patch.object(worker, "release_melo_bert") as release:
            eng.v2_base("FR")
            eng.v2_base("FR")
        release.assert_called_once()

    def test_failed_and_cancelled_takes_release_converter_and_allow_next_command(self):
        for error in (RuntimeError("CUDA out of memory"), worker.Cancelled()):
            eng = worker.Engine()
            eng.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            def load(*args):
                eng.converter = Model()
                eng.converter.hps = types.SimpleNamespace(data=types.SimpleNamespace(sampling_rate=24000))
                eng.converter_version = "v2"
                return eng.converter
            eng.load_converter = load
            eng.base_tts = mock.Mock(side_effect=error)
            with tempfile.TemporaryDirectory() as root, \
                 mock.patch.dict(sys.modules, {n: types.ModuleType(n) for n in ("librosa", "numpy", "soundfile")}):
                with self.assertRaises(type(error)):
                    eng.speak({"out_dir": root, "lines": [{"text": "Hello"}], "opts": {"free_after": False}})
            self.assertIsNone(eng.converter)
            self.assertEqual(eng.busy, "")
            self.assertEqual(eng.status()["converter"], "")


if __name__ == "__main__":
    unittest.main()
