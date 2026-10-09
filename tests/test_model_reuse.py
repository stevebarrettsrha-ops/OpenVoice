"""Real local cache fixtures; every unexpected network call fails the test."""
import copy
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Do not initialize the server's manager module before test_units selects its
# data directory. This isolated copy is only exercised against temporary files.
spec = importlib.util.spec_from_file_location("reuse_manager", Path(__file__).resolve().parents[1] / "manager.py")
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ExistingModels(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = copy.deepcopy(manager.MODELS)
        for key, model in self.models.items():
            model["dir"] = self.root / key
        patch = mock.patch.object(manager, "MODELS", self.models)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "hub")})
        patch.start()
        self.addCleanup(patch.stop)
        self.task = manager.Task("download", "test")

    def populate(self, key, root, content=b"existing model fixture"):
        for rel in self.models[key]["key_files"]:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

    def snapshot(self, key):
        repo = self.root / "hub" / ("models--" + self.models[key]["repo"].replace("/", "--"))
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text("a" * 40)
        return repo / "snapshots" / ("a" * 40)

    def test_complete_local_files_skip_cache_and_network_even_with_saved_zip(self):
        for key, model in self.models.items():
            with self.subTest(key=key):
                self.populate(key, model["dir"])
                with mock.patch.object(manager, "reuse_hf_cache") as cache, \
                     mock.patch.object(manager.requests, "get", side_effect=AssertionError("network")):
                    manager.download_model(self.task, key, {"zip_urls": {key: "https://example.org/model.zip"}})
                cache.assert_not_called()
                self.assertTrue(manager.model_state(key)["present"])

    def test_hf_snapshots_reused_offline_in_both_repository_layouts(self):
        for key, model in self.models.items():
            with self.subTest(key=key):
                snapshot = self.snapshot(key)
                source = snapshot / model["strip"] if key == "v1" else snapshot
                self.populate(key, source)
                with mock.patch.object(manager.requests, "get", side_effect=AssertionError("network")):
                    manager.download_model(self.task, key, {})
                for rel in model["key_files"]:
                    self.assertEqual((model["dir"] / rel).read_bytes(), (source / rel).read_bytes())

    def test_reuse_preserves_existing_local_model_and_shared_cache(self):
        key = "v2"
        snapshot = self.snapshot(key)
        self.populate(key, snapshot)
        converter = self.models[key]["dir"] / "converter/checkpoint.pth"
        converter.parent.mkdir(parents=True)
        converter.write_bytes(b"my existing converter")
        count = manager.reuse_hf_cache(self.task, key)
        self.assertEqual(count, len(self.models[key]["key_files"]) - 1)
        self.assertEqual(converter.read_bytes(), b"my existing converter")
        self.assertEqual((snapshot / "converter/checkpoint.pth").read_bytes(), b"existing model fixture")

    def test_partial_or_pointer_files_are_not_ready(self):
        model = self.models["v2"]
        self.populate("v2", model["dir"])
        (model["dir"] / "converter/checkpoint.pth").write_bytes(b"")
        (model["dir"] / "base_speakers/ses/fr.pth").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:123")
        self.assertEqual(manager.model_state("v2")["missing"],
                         ["converter/checkpoint.pth", "base_speakers/ses/fr.pth"])

    def test_importing_its_own_folder_is_a_noop(self):
        model = self.models["v1"]
        self.populate("v1", model["dir"])
        with mock.patch.object(manager.shutil, "copytree", side_effect=AssertionError("self copy")):
            manager.import_local(self.task, "v1", str(model["dir"]))

    def test_status_does_not_search_cache(self):
        with mock.patch.object(manager, "reuse_hf_cache", side_effect=AssertionError("discovery")):
            for _ in range(3):
                manager.model_state("v2")

    def test_complete_partial_is_promoted_without_a_network_request(self):
        dest = self.root / "checkpoint.pth"
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(b"already downloaded")
        with mock.patch.object(manager.requests, "get", side_effect=AssertionError("network")):
            manager.download_file(self.task, "url", dest, {}, part.stat().st_size, 0, 18)
        self.assertEqual(dest.read_bytes(), b"already downloaded")
        self.assertFalse(part.exists())

    def test_truncated_download_preserves_previous_file_and_resumable_partial(self):
        dest = self.root / "checkpoint.pth"
        dest.write_bytes(b"previous complete checkpoint")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.iter_content.return_value = [b"short"]
        with mock.patch.object(manager.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Incomplete download"):
                manager.download_file(self.task, "url", dest, {}, 20, 0, 20)
        self.assertEqual(dest.read_bytes(), b"previous complete checkpoint")
        self.assertEqual(dest.with_name(dest.name + ".part").read_bytes(), b"short")

    def test_rejected_resume_never_promotes_an_incomplete_file(self):
        dest = self.root / "checkpoint.pth"
        dest.with_name(dest.name + ".part").write_bytes(b"short")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = 416
        with mock.patch.object(manager.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "resume range"):
                manager.download_file(self.task, "url", dest, {}, 20, 0, 20)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
