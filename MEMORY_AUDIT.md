# 8 GB memory audit — 9 October 2026

## Fixed

- `Engine.unload()` dropped the TTS objects but left Melo's BERT models in
  module-global caches. It now clears both upstream cache formats, without
  importing or downloading unused languages. Language switches also release
  those feature models.
- Successful cleanup kept a local reference to the converter alive while
  calling `empty_cache()`. Local model references are now dropped first.
- Failed and cancelled takes now unload models. The worker also clears
  unwound traceback frames after an out-of-memory error, before final cleanup.
- New installations default to freeing memory after each take. Existing
  saved choices remain intact. Peak-memory statistics reset before a take.
- The server no longer discards `OPENVOICE_DEVICE=cpu` when its saved device
  is `auto`. An explicit saved device still takes precedence. CPU takes no
  longer reset or query CUDA allocator statistics just because a card exists.
- Converter and V1 checkpoints load into CPU memory before copying into the
  existing model, avoiding a second full set of weights on the GPU.
- Tone conversion previously inferred over a whole line after Melo had joined
  its synthesized sentences. It now uses at most 10 seconds per inference,
  with hop-aligned 250 ms overlaps joined on CPU. Short inputs keep the single
  call path. Watermarking runs once after assembly, and cancellation is checked
  between windows. This bounds conversion activations, not base TTS synthesis.
- The Models and Engine pages no longer promise that every V2 take fits in
  1–2 GB; language feature models and temporary tensors also consume memory.

## Validation

`python tests/check.py`: all six gate checks passed.
`python -m unittest discover -s tests -p 'test_*.py'`: **49 tests passed**,
including model ownership at cleanup, both Melo cache shapes, language
switching, failed/cancelled takes, CPU device propagation, bounded conversion,
signal alignment and overlap, watermarking, output writing and cancellation.
These tests use stand-ins for synthesis; signal tests use NumPy.

`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python tests/smoke_converter_cpu.py`:
**four real CPU tensor cases passed** with a small randomly initialized
`SynthesizerTrn` and the real spectrogram/decoder path. A 336,333-sample input
produced 336,256 output samples (complete 128-sample hops) in three windows,
none exceeding 160,000 samples at 16 kHz. Short and exact-window inputs also
passed. This verifies tensor shapes and bounded calls without trained weights.

No CUDA device or real synthesis checkpoint was available in this audit.
These tests establish cleanup and conversion shape behavior, not an 8 GB
peak-memory figure. Audio quality at chunk boundaries still needs listening
tests with trained weights; the smoke model emits meaningless audio.

## Apply

Update the source and close/relaunch OpenVoice to replace its worker.
Enable **Free GPU memory after each take** on an existing installation.
Test a short English line first with other AI engines closed. If it still
fails, keep the Engine console's final error, the GPU model, system RAM,
and the failing line/reference length. `OPENVOICE_DEVICE=cpu` provides a
slower alternative when GPU memory is unavailable.
