"""run.sh on a folder first set up on Windows.

The app folder can sit on a drive both systems share, so Linux finds the
Windows `.venv` (Scripts/python.exe, a pyvenv.cfg whose home is C:\\...) and
`.uvenv`. The launcher read them as "incomplete" and deleted them, which broke
the app for Windows. These run the real run.sh against a stand-in Python, so
nothing is installed and nothing touches the network.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# A Python 3.11 that only knows the four things run.sh asks of it.
FAKE_PY = r"""#!/bin/sh
case "$1" in
  -c) echo 3.11; exit 0 ;;
  -m)
    case "$2" in
      venv) mkdir -p "$3/bin"; cp "$FAKE_PY" "$3/bin/python"
            printf 'home = /usr/bin\n' > "$3/pyvenv.cfg"; exit 0 ;;
      pip) for a in "$@"; do last="$a"; done
           [ "$last" = uv ] && cp "$FAKE_UV" "$(dirname "$0")/uv"
           exit 0 ;;
    esac ;;
esac
echo "SERVER $0 $*" >> "$LAUNCH_LOG"
"""

# uv: fetch a Python (nothing to do) or build a venv.
FAKE_UV = r"""#!/bin/sh
if [ "$1" = venv ]; then
  for a in "$@"; do last="$a"; done
  mkdir -p "$last/bin"; cp "$FAKE_PY" "$last/bin/python"
  printf 'home = /usr/bin\n' > "$last/pyvenv.cfg"
fi
exit 0
"""

WIN_CFG = "home = C:\\Users\\steven\\AppData\\Local\\Programs\\Python\\Python311\n"


@unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "run.sh needs bash")
class WindowsVenvOnLinux(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ov-launch-"))
        self.app = self.tmp / "app"
        self.app.mkdir()
        for name in ("run.sh", "requirements.txt"):
            shutil.copy(REPO / name, self.app / name)
        fakebin = self.tmp / "bin"
        fakebin.mkdir()
        self.fake_py = fakebin / "python3.11"
        self.fake_py.write_text(FAKE_PY)
        fake_uv = self.tmp / "uv"
        fake_uv.write_text(FAKE_UV)
        for f in (self.fake_py, fake_uv):
            f.chmod(0o755)
        (fakebin / "python3").symlink_to(self.fake_py)
        self.log = self.tmp / "launch.log"
        self.env = {
            "PATH": f"{fakebin}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(self.tmp),
            "FAKE_PY": str(self.fake_py), "FAKE_UV": str(fake_uv),
            "LAUNCH_LOG": str(self.log),
            "OPENVOICE_STUDIO_NO_TERMINAL": "1",
        }
        self.native = ".venv-" + platform.system().lower()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def windows_venv(self, name: str, *exes: str) -> Path:
        venv = self.app / name
        (venv / "Scripts").mkdir(parents=True)
        for exe in ("python.exe",) + exes:
            (venv / "Scripts" / exe).write_bytes(b"MZ\x90\x00" + b"\x00" * 60)
        (venv / "pyvenv.cfg").write_text(WIN_CFG)
        return venv

    def launch(self, **extra) -> str:
        r = subprocess.run(["bash", str(self.app / "run.sh")], cwd=self.app,
                           env={**self.env, **extra}, capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return self.log.read_text() if self.log.exists() else ""

    def test_a_windows_venv_is_left_alone_and_linux_gets_its_own(self):
        venv = self.windows_venv(".venv")
        ran = self.launch()
        self.assertTrue((venv / "Scripts" / "python.exe").is_file(),
                        "the Windows .venv was deleted")
        self.assertEqual((venv / "pyvenv.cfg").read_text(), WIN_CFG)
        self.assertFalse((venv / "bin").exists(), "built over the Windows .venv")
        self.assertTrue((self.app / self.native / "bin" / "python").is_file())
        self.assertIn(f"{self.native}/bin/python server.py", ran)

    def test_a_venv_whose_pyvenv_cfg_names_windows_is_foreign_too(self):
        # What the old launchers left behind: Linux rebuilt .venv, then
        # Windows built over it. Both bin/ and Scripts/, a Windows home.
        venv = self.windows_venv(".venv")
        (venv / "bin").mkdir()
        shutil.copy(self.fake_py, venv / "bin" / "python")
        ran = self.launch()
        self.assertEqual((venv / "pyvenv.cfg").read_text(), WIN_CFG)
        self.assertIn(f"{self.native}/bin/python server.py", ran)

    def test_a_windows_uvenv_is_left_alone(self):
        uvenv = self.windows_venv(".uvenv", "uv.exe")
        ran = self.launch(OPENVOICE_STUDIO_FORCE_UV="1")
        self.assertTrue((uvenv / "Scripts" / "uv.exe").is_file(),
                        "the Windows .uvenv was deleted")
        self.assertTrue((self.app / (".uvenv-" + platform.system().lower())
                         / "bin" / "uv").is_file())
        self.assertIn(".venv/bin/python server.py", ran)

    def test_a_native_venv_is_still_built_in_place(self):
        ran = self.launch()
        self.assertTrue((self.app / ".venv" / "bin" / "python").is_file())
        self.assertFalse((self.app / self.native).exists())
        self.assertIn(".venv/bin/python server.py", ran)


if __name__ == "__main__":
    unittest.main()
