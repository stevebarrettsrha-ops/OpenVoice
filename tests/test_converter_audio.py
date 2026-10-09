"""Signal assembly tests require NumPy, but no PyTorch or checkpoints."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openvoice.audio_processing import convert_in_chunks


def converter_api():
    """Import the API with model dependencies replaced at their boundaries."""
    fake = {
        "torch": types.SimpleNamespace(load=mock.Mock()),
        "librosa": types.SimpleNamespace(load=mock.Mock()),
        "soundfile": types.SimpleNamespace(write=mock.Mock()),
        "openvoice.models": types.SimpleNamespace(SynthesizerTrn=mock.Mock()),
        "openvoice.text": types.SimpleNamespace(text_to_sequence=mock.Mock()),
        "openvoice.commons": types.ModuleType("openvoice.commons"),
        "openvoice.mel_processing": types.SimpleNamespace(spectrogram_torch=mock.Mock()),
    }
    path = Path(__file__).resolve().parents[1] / "openvoice" / "api.py"
    spec = importlib.util.spec_from_file_location("converter_test_api", path)
    api = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake):
        spec.loader.exec_module(api)
    return api


class ConversionWindows(unittest.TestCase):
    def test_short_audio_retains_exact_single_call_output(self):
        source = np.linspace(-1, 1, 243, dtype=np.float32)
        expected = source[:-3]
        render = mock.Mock(return_value=expected)
        result = convert_in_chunks(source, render, sample_rate=100, hop_length=4)
        self.assertIs(result, expected)
        self.assertIs(render.call_args.args[0], source)
        render.assert_called_once()

    def test_long_identity_conversion_preserves_complete_hops_and_bounds_calls(self):
        for count in (1001, 1983, 2000, 6001):
            with self.subTest(count=count):
                source = np.linspace(-1, 1, count, dtype=np.float32)
                lengths = []
                def render(part):
                    lengths.append(len(part))
                    return part.copy()
                result = convert_in_chunks(source, render, sample_rate=100, hop_length=4)
                self.assertEqual(len(result), count // 4 * 4)
                self.assertTrue(all(n <= 1000 and n % 4 == 0 for n in lengths))
                np.testing.assert_allclose(result, source[:len(result)], atol=1e-7)

    def test_overlap_crossfades_without_a_gap_or_amplitude_doubling(self):
        calls = []
        def render(part):
            calls.append(len(part))
            return np.full(len(part), len(calls), dtype=np.float32)
        result = convert_in_chunks(np.zeros(1500), render, sample_rate=100, hop_length=4)
        np.testing.assert_array_equal(result[:976], np.ones(976))
        np.testing.assert_allclose(result[976:1000], np.linspace(1, 2, 24), atol=1e-7)
        np.testing.assert_array_equal(result[1000:], np.full(500, 2))

    def test_failed_window_stops_conversion(self):
        render = mock.Mock(side_effect=[np.zeros(1000), RuntimeError("model failure")])
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            convert_in_chunks(np.zeros(3000), render, sample_rate=100, hop_length=4)
        self.assertEqual(render.call_count, 2)

    def test_unexpected_model_length_does_not_leave_uninitialized_audio(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected window length"):
            convert_in_chunks(np.zeros(1500), lambda part: part[:-1], 100, 4)


class ConverterApi(unittest.TestCase):
    def converter(self, count):
        api = converter_api()
        converter = api.ToneColorConverter.__new__(api.ToneColorConverter)
        converter.hps = types.SimpleNamespace(data=types.SimpleNamespace(sampling_rate=100, hop_length=4))
        source = np.linspace(-1, 1, count, dtype=np.float32)
        api.librosa.load.return_value = (source, 100)
        converter._convert_audio = mock.Mock(side_effect=lambda part, *args: part.copy())
        converter.add_watermark = mock.Mock(side_effect=lambda audio, message: audio + 0.125)
        return api, converter, source

    def test_resampling_watermark_and_return_path_for_short_and_long_audio(self):
        for count in (243, 1500):
            with self.subTest(count=count):
                api, converter, source = self.converter(count)
                actual = converter.convert("source.wav", "src", "target", tau=0.5, message="mark")
                api.librosa.load.assert_called_once_with("source.wav", sr=100)
                converter.add_watermark.assert_called_once()
                self.assertEqual(converter.add_watermark.call_args.args[1], "mark")
                np.testing.assert_allclose(actual, source[:len(actual)] + 0.125, atol=1e-7)
                api.soundfile.write.assert_not_called()
                for call in converter._convert_audio.call_args_list:
                    self.assertEqual(call.args[1:], ("src", "target", 0.5))

    def test_file_output_is_assembled_once(self):
        api, converter, source = self.converter(1500)
        result = converter.convert("source.wav", "src", "target", output_path="final.wav")
        self.assertIsNone(result)
        api.soundfile.write.assert_called_once()
        path, audio, rate = api.soundfile.write.call_args.args
        self.assertEqual((path, rate), ("final.wav", 100))
        np.testing.assert_allclose(audio, source + 0.125, atol=1e-7)

    def test_cancellation_between_windows_prevents_next_inference_and_save(self):
        api, converter, source = self.converter(3000)
        check = mock.Mock(side_effect=[None, RuntimeError("cancelled")])
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            converter.convert("source.wav", "src", "target", output_path="final.wav", check_cancel=check)
        converter._convert_audio.assert_called_once()
        converter.add_watermark.assert_not_called()
        api.soundfile.write.assert_not_called()

    def test_checkpoint_is_staged_on_cpu_before_loading_gpu_model(self):
        api = converter_api()
        model = api.OpenVoiceBaseClass.__new__(api.OpenVoiceBaseClass)
        model.device = "cuda"
        model.model = mock.Mock()
        model.model.load_state_dict.return_value = ([], [])
        weights = {"weight": object()}
        api.torch.load.return_value = {"model": weights}
        model.load_ckpt("checkpoint.pth")
        api.torch.load.assert_called_once_with("checkpoint.pth", map_location="cpu")
        model.model.load_state_dict.assert_called_once_with(weights, strict=False)


if __name__ == "__main__":
    unittest.main()
