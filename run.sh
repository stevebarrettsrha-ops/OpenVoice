#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 3.11 and 3.10 first: MeloTTS (the V2 base voices) pins packages that only
# have wheels for those two. 3.12+ still runs the app and OpenVoice V1.
find_python() {
  PY=""
  for cand in python3.11 python3.10 python3.12 python3.13 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$cand"; return 0
    fi
  done
  return 1
}

if ! find_python; then
  echo "  Python 3.10 or newer was not found." >&2
  if command -v brew >/dev/null 2>&1; then
    INSTALL=(brew install python@3.11)
  elif command -v apt-get >/dev/null 2>&1; then
    INSTALL=(sudo apt-get install -y python3 python3-venv python3-pip)
  elif command -v dnf >/dev/null 2>&1; then
    INSTALL=(sudo dnf install -y python3 python3-pip)
  elif command -v pacman >/dev/null 2>&1; then
    INSTALL=(sudo pacman -S --noconfirm python python-pip)
  else
    echo "  No package manager was found. Install Python 3.11, then run this again." >&2
    exit 1
  fi
  echo "  This can be done for you with: ${INSTALL[*]}"
  printf "  Install Python now? [y/N] "
  read -r reply </dev/tty || reply=""
  case "$reply" in
    [yY]*) ;;
    *) echo "  Nothing was installed. Install Python 3.11, then run this again." >&2
       exit 1 ;;
  esac
  if ! "${INSTALL[@]}"; then
    echo "  That did not finish. Install Python 3.11 by hand, then run this again." >&2
    exit 1
  fi
  if ! find_python; then
    echo "  Python is installed but this shell cannot see it yet." >&2
    echo "  Open a new terminal and run ./run.sh again." >&2
    exit 1
  fi
fi

echo "  Using: $("$PY" -c 'import sys;print(sys.executable)')"

# Everything OpenVoice Studio installs — Flask now, PyTorch, OpenVoice's
# packages and MeloTTS later from the Engine page — goes into .venv beside this
# script, never into the Python that was found. Healthy means pip runs: a venv
# cut off at ensurepip has a python and no pip, so it is checked and rebuilt.
VENV=".venv"
if [ -e "$VENV" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "  The existing environment is incomplete. Building it again."
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "  Setting up OpenVoice Studio's environment (first run only)..."
  if ! "$PY" -m venv "$VENV"; then
    rm -rf "$VENV"
    echo "  Could not create the environment. On Debian and Ubuntu the venv" >&2
    echo "  module ships separately:  sudo apt-get install -y python3-venv" >&2
    exit 1
  fi
fi

"$VENV/bin/python" -m pip install --disable-pip-version-check --quiet \
    -r requirements.txt
exec "$VENV/bin/python" server.py
