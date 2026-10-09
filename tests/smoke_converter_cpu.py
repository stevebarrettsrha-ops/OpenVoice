"""Optional real-tensor smoke: python tests/smoke_converter_cpu.py.

Needs CPU PyTorch, NumPy, librosa and soundfile; downloads no weights. The
small random SynthesizerTrn exercises the real spectrogram, conversion and
decoder paths. Its noise output cannot establish trained-model audio quality.
"""
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Only the unused base-TTS text frontend is omitted. Model/tensor/audio code
# is real, and must import successfully for this smoke to pass.
frontend = types.ModuleType("openvoice.text")
frontend.text_to_sequence = mock.Mock(side_effect=AssertionError("TTS frontend is not part of this smoke"))
with mock.patch.dict(sys.modules, {"openvoice.text": frontend}):
    from openvoice.api import ToneColorConverter
from openvoice.models import SynthesizerTrn


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    converter = ToneColorConverter.__new__(ToneColorConverter)
    converter.hps = types.SimpleNamespace(data=types.SimpleNamespace(
        sampling_rate=16000, filter_length=512, hop_length=128, win_length=512))
    converter.device = "cpu"
    converter.watermark_model = None
    converter.model = SynthesizerTrn(
        n_vocab=0, spec_channels=257, inter_channels=8, hidden_channels=8,
        filter_channels=16, n_heads=2, n_layers=1, kernel_size=3, p_dropout=0,
        resblock="1", resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]],
        upsample_rates=[8, 4, 4], upsample_initial_channel=32,
        upsample_kernel_sizes=[16, 8, 8], n_speakers=0, gin_channels=4).eval()
    embedding = torch.zeros(1, 4, 1)
    lengths = []
    render = converter._convert_audio
    def observed(part, *args):
        output = render(part, *args)
        lengths.append((len(part), len(output)))
        assert len(output) == len(part) // 128 * 128, lengths[-1]
        return output
    converter._convert_audio = observed
    with tempfile.TemporaryDirectory() as root:
        source = Path(root) / "input.wav"
        final = Path(root) / "output.wav"
        for count in (16031, 160000, 160001, 336333):
            lengths.clear()
            wave = np.sin(np.arange(count, dtype=np.float32) * (2 * np.pi * 220 / 16000)) * 0.1
            soundfile.write(source, wave, 16000)
            converter.convert(str(source), embedding, embedding, output_path=str(final))
            audio, rate = soundfile.read(final)
            assert rate == 16000
            assert len(audio) == count // 128 * 128
            assert np.isfinite(audio).all()
            assert all(given <= 160000 for given, _ in lengths)
            print(f"PASS {count} input samples -> {len(audio)} output; windows {lengths}")


if __name__ == "__main__":
    main()
