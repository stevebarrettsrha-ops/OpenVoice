# OpenVoice Studio — working notes

A local Flask app around MyShell's OpenVoice. `openvoice/` is upstream code;
leave it alone beyond bug fixes. The studio is `server.py`, `engine.py`,
`manager.py` and `web/index.html`.

## Rules

1. **One page, no build step.** `web/index.html` is the whole interface, with
   its CSS and script inline. No external scripts.
2. **The models never run in the server process.** `engine.py` is a child
   process spoken to over JSON lines (see `EngineProcess` in server.py).
   Stdout of the worker is protocol; everything else goes to stderr, which the
   Engine page shows as the console. Keep it that way — it is what lets an
   install replace PyTorch under a running server and what lets an
   out-of-memory kill only a take.
3. **Installs go into `sys.executable`'s environment**, which the launchers
   make `.venv`. Stop the engine before installing.
4. **Checkpoints come from HuggingFace**, not the S3 links in `docs/USAGE.md`
   (that bucket is gone). Repo names and the local layout are in
   `manager.MODELS`; the file list is read from the HF tree API.
5. **8 GB is the target card.** Keep at most two MeloTTS languages resident,
   offer `free_after`, report `vram_peak` per take.
6. **Tests need no GPU.** `tests/mock_engine.py` answers for the worker; the
   suite points `OPENVOICE_STUDIO_DATA` at a temp folder. Add a test when a
   fault ships.

Borrowed from the sibling apps (Script Builder, Llama Studio, Ideogram
Studio), where each was a fault that shipped:

7. **Every Python child talks UTF-8 on its pipe, both ends.** The server sets
   `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` on the engine and on every
   pip run, and opens its pipes `encoding="utf-8", errors="replace"`; the
   worker reconfigures `sys.__stdout__` the same way. Windows hands a piped
   child the ANSI code page with strict errors, and OpenVoice prints IPA.
8. **The console is plain text.** `plain()` strips ANSI escapes and `\r` from
   engine stderr and from task logs; loguru and tqdm colour even into a pipe.
9. **Long work reports a number.** pip gets `--progress-bar raw` and its
   `Progress N of M` lines drive the task bar; a pip that rejects `raw` is
   retried without it, never left failed. Model downloads carry byte
   progress. `pct` is None until there is a real number.
10. **Detection is by execution, not PATH lookup.** The launchers run each
    Python candidate; `gpu_info` runs nvidia-smi from PATH and from the
    places the driver puts it on Windows. A missing nvidia-smi is not a
    missing card.
11. **A dead engine is a sentence, not a stack trace.** Every way `start()`
    can fail writes its reason to the console; the error carries the
    worker's last complaint and, for the usual Windows causes, one line of
    advice (`advice_for`). Home and Create show it with a Restart button.
12. **The gate checks element ids.** `tests/check.py` lists every id the
    inline script asks for and fails on one the markup lacks — the mismatch
    that killed every button silently in two sibling apps.

## Gate before pushing

```
python tests/check.py
python -m unittest discover -s tests -p 'test_*.py'
```

A missing declaration in the inline script kills every button silently; the
gate's `node --check` catches it.

## Engine protocol

Request `{"id","cmd",...}`; reply `{"id","ok":true,"result"}` or
`{"id","ok":false,"error"}`; long commands send `{"id","event":"progress",...}`
lines first. Commands: hello, status, load, embed, speak, unload, cancel, quit.
`cancel` is handled on the reader thread so it reaches a running `speak`.
