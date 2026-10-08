#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# OpenVoice Studio runs on Python 3.10 or 3.11, in its own environment.
#
# MeloTTS (OpenVoice V2's base voices) pins packages that have wheels for
# those two versions only, so a 3.12+ venv can run the app and V1 but never
# V2. Rather than hope the machine has the right Python, this launcher:
#   1. uses a 3.11 or 3.10 already installed, if there is one;
#   2. otherwise fetches a managed CPython 3.11 through uv, into this app's
#      own folders — nothing touches the system Python;
#   3. builds .venv from it, and rebuilds a .venv that was made on 3.12+
#      (the old one is kept beside it, renamed).
# Set OPENVOICE_STUDIO_FORCE_UV=1 to take route 2 even when route 1 would do.

WANT_MIN="3.10"
WANT_MAX="3.11"
VENV=".venv"
UVENV=".uvenv"          # a tiny venv whose only job is to hold uv

version_of() {           # prints "3.11" for an interpreter, or nothing
  "$1" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true
}
is_wanted() {            # 3.10 or 3.11
  case "$1" in 3.10|3.11) return 0;; *) return 1;; esac
}
find_wanted() {          # a 3.11 or 3.10 on this machine, by execution
  PY=""
  for cand in python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && is_wanted "$(version_of "$cand")"; then
      PY="$cand"; return 0
    fi
  done
  return 1
}
find_any() {             # any Python 3.8+, enough to bootstrap uv
  BOOT=""
  for cand in python3 python python3.13 python3.12 python3.11 python3.10 python3.9 python3.8; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,8) else 1)' 2>/dev/null; then
      BOOT="$cand"; return 0
    fi
  done
  return 1
}

USE_UV=""
if [ -z "${OPENVOICE_STUDIO_FORCE_UV:-}" ] && find_wanted; then
  echo "  Using: $("$PY" -c 'import sys;print(sys.executable)') (Python $(version_of "$PY"))"
else
  if ! find_any; then
    echo "  No Python was found at all. Install Python 3.11 from python.org (or" >&2
    echo "  'sudo apt-get install -y python3 python3-venv' on Debian/Ubuntu)," >&2
    echo "  then run this again." >&2
    exit 1
  fi
  if [ -n "${OPENVOICE_STUDIO_FORCE_UV:-}" ]; then
    echo "  OPENVOICE_STUDIO_FORCE_UV is set: fetching Python 3.11 through uv."
  else
    echo "  No Python 3.10 or 3.11 on this machine (found $(version_of "$BOOT"))."
    echo "  OpenVoice V2's voices need one, so Python 3.11 will be fetched into"
    echo "  this app's own environment. Your system Python is not touched."
  fi
  if [ ! -x "$UVENV/bin/uv" ]; then
    echo "  Setting up uv (the fetcher) ..."
    rm -rf "$UVENV"
    if ! "$BOOT" -m venv "$UVENV"; then
      echo "  Could not create a virtual environment. On Debian and Ubuntu the" >&2
      echo "  venv module ships separately:  sudo apt-get install -y python3-venv" >&2
      exit 1
    fi
    "$UVENV/bin/python" -m pip install --disable-pip-version-check --quiet uv
  fi
  UV="$UVENV/bin/uv"
  echo "  Fetching Python 3.11 (about 30 MB, kept under uv's data folder) ..."
  "$UV" python install "$WANT_MAX"
  USE_UV=1
fi

# A .venv made on 3.12+ (an earlier launch, before this rule) cannot run V2.
# Keep it aside rather than delete it; the Engine page installs into the new
# one again. Healthy means pip runs: a venv cut off at ensurepip has a python
# and no pip, so that is checked too.
if [ -e "$VENV" ]; then
  HAVE="$(version_of "$VENV/bin/python")"
  if [ -n "$HAVE" ] && ! is_wanted "$HAVE"; then
    echo "  The existing environment is Python $HAVE. Setting it aside as $VENV-py$HAVE"
    echo "  and building a fresh one on 3.11/3.10 (PyTorch and the models'"
    echo "  packages will need installing again from the Engine page)."
    rm -rf "$VENV-py$HAVE"
    mv "$VENV" "$VENV-py$HAVE"
  elif ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "  The existing environment is incomplete. Building it again."
    rm -rf "$VENV"
  fi
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "  Setting up OpenVoice Studio's environment (first run only)..."
  if [ -n "$USE_UV" ]; then
    "$UV" venv --seed --python "$WANT_MAX" "$VENV"
  elif ! "$PY" -m venv "$VENV"; then
    rm -rf "$VENV"
    echo "  Could not create the environment. On Debian and Ubuntu the venv" >&2
    echo "  module ships separately:  sudo apt-get install -y python3-venv" >&2
    exit 1
  fi
fi

echo "  Environment: Python $(version_of "$VENV/bin/python") in $VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --quiet \
    -r requirements.txt
exec "$VENV/bin/python" server.py
