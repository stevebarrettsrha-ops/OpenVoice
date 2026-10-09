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

## Validation

`python tests/check.py`: all six gate checks passed.
`python -m unittest discover -s tests -p 'test_*.py'`: **38 tests passed**,
including model ownership at cleanup, both Melo cache shapes, language
switching, and failed/cancelled takes. Tests use stand-ins for synthesis.

No CUDA device or real synthesis checkpoint was available in this audit.
These tests establish the cleanup behavior, not an 8 GB peak-memory figure.

## Apply

Update the source and close/relaunch OpenVoice to replace its worker.
Enable **Free GPU memory after each take** on an existing installation.
Test a short English line first with other AI engines closed. If it still
fails, keep the Engine console's final error, the GPU model, system RAM,
and the failing line/reference length. `OPENVOICE_DEVICE=cpu` provides a
slower alternative when GPU memory is unavailable.
