"""The cheap gate: everything compiles and the page's inline script parses.

    python tests/check.py

A missing declaration in the inline script kills every button silently, and
py_compile catches the Python equivalent. Neither needs the app running. Node
is used for the script check when it is on PATH and skipped with a note when
it is not.
"""
from __future__ import annotations

import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
failed = 0


def step(name, problem):
    global failed
    if problem:
        failed += 1
        print(f"FAIL {name}\n" + "\n".join("     " + l for l in str(problem).rstrip().splitlines()))
    else:
        print(f"ok   {name}")


def compiles(paths):
    for p in paths:
        try:
            py_compile.compile(str(REPO / p), doraise=True)
        except py_compile.PyCompileError as exc:
            return str(exc)
    return None


step("python modules compile", compiles(["server.py", "manager.py", "engine.py"]))
step("test helpers compile", compiles(["tests/mock_engine.py", "tests/test_units.py"]))
step("openvoice package compiles", compiles([str(p.relative_to(REPO))
                                             for p in (REPO / "openvoice").rglob("*.py")]))

html = (REPO / "web" / "index.html").read_text(encoding="utf-8")
blocks = re.findall(r"<script>([\s\S]*?)</script>", html)
if not blocks:
    step("the inline script parses", "no inline <script> in web/index.html")
elif shutil.which("node"):
    tmp = Path(tempfile.mkdtemp()) / "inline.js"
    tmp.write_text("\n".join(blocks), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
    step("the inline script parses", None if r.returncode == 0 else (r.stderr or f"exit {r.returncode}"))
else:
    print("skip the inline script parses (node is not on PATH)")

step("the page has one script and no build step",
     "web/index.html pulls in an external script" if re.search(r"<script[^>]+src=", html) else None)

sys.exit(1 if failed else 0)
