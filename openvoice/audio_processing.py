"""Bound voice conversion windows without retaining tensors between calls."""
import numpy as np


def convert_in_chunks(audio, convert, sample_rate, hop_length):
    """Convert at most 10 seconds at once; overlap 250 ms and join on CPU.

    Boundaries align to the model's hop length. Like a single model call, the
    result contains only complete output hops. Short inputs use the original
    single-call path, including its exact output length and samples.
    """
    window = max(hop_length, int(sample_rate * 10) // hop_length * hop_length)
    if len(audio) <= window:
        return convert(audio)
    overlap = max(hop_length, int(sample_rate * 0.25) // hop_length * hop_length)
    overlap = min(overlap, window // 2)
    stride = window - overlap
    length = len(audio) // hop_length * hop_length
    output = np.empty(length, dtype=np.float32)
    previous_end = 0
    for start in range(0, length, stride):
        end = min(start + window, length)
        converted = convert(audio[start:end])
        if len(converted) != end - start:
            raise RuntimeError("Tone converter returned an unexpected window length")
        shared = max(0, previous_end - start)
        if shared:
            fade = np.linspace(0, 1, shared, dtype=np.float32)
            output[start:previous_end] *= 1 - fade
            output[start:previous_end] += converted[:shared] * fade
        output[start + shared:end] = converted[shared:]
        previous_end = end
        if end == length:
            break
    return output
