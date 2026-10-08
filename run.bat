@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title OpenVoice Studio

rem OpenVoice Studio runs on Python 3.10 or 3.11, in its own environment.
rem
rem MeloTTS (OpenVoice V2's base voices) pins packages that have wheels for
rem those two versions only, so a 3.12+ venv can run the app and V1 but never
rem V2. Rather than hope the machine has the right Python, this launcher:
rem   1. uses a 3.11 or 3.10 already installed, if there is one;
rem   2. otherwise fetches a managed CPython 3.11 through uv, into this app's
rem      own folders - nothing touches the system Python;
rem   3. builds .venv from it, and rebuilds a .venv that was made on 3.12+
rem      (the old one is kept beside it, renamed).
rem Set OPENVOICE_STUDIO_FORCE_UV=1 to take route 2 even when route 1 would do.
rem
rem Flat control flow on purpose: nested parenthesised blocks with delayed
rem expansion are where batch fails silently, so every branch is a label.

set "VENV=.venv"
set "UVENV=.uvenv"
set "USE_UV="
set "PY="

if defined OPENVOICE_STUDIO_FORCE_UV goto needuv
call :findwanted
if defined PY goto haspy

:needuv
call :findany
if defined BOOT goto haveboot
echo.
echo   No Python was found at all. Install Python 3.11 from
echo   https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo   then run this file again.
echo.
pause
exit /b 1

:haveboot
echo   No Python 3.10 or 3.11 on this machine. OpenVoice V2's voices need one,
echo   so Python 3.11 will be fetched into this app's own environment.
echo   Your system Python is not touched.
if exist "%UVENV%\Scripts\uv.exe" goto haveuv
echo   Setting up uv (the fetcher) ...
if exist "%UVENV%" rmdir /s /q "%UVENV%"
%BOOT% -m venv "%UVENV%"
if errorlevel 1 goto uvfail
"%UVENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet uv
if errorlevel 1 goto uvfail
:haveuv
set "UV=%UVENV%\Scripts\uv.exe"
echo   Fetching Python 3.11 (about 30 MB, kept under uv's data folder) ...
"%UV%" python install 3.11
if errorlevel 1 goto uvfail
set "USE_UV=1"
goto venv

:haspy
echo   Using: %PY%

:venv
rem A .venv made on 3.12+ cannot run V2: set it aside and build afresh.
if not exist "%VENV%\Scripts\python.exe" goto makevenv
set "HAVE="
for /f "delims=" %%V in ('"%VENV%\Scripts\python.exe" -c "import sys;print('%%d.%%d'%%sys.version_info[:2])" 2^>nul') do set "HAVE=%%V"
if "!HAVE!"=="3.10" goto checkpip
if "!HAVE!"=="3.11" goto checkpip
echo   The existing environment is Python !HAVE!. Setting it aside as %VENV%-py!HAVE!
echo   and building a fresh one on 3.11/3.10 (PyTorch and the models' packages
echo   will need installing again from the Engine page).
if exist "%VENV%-py!HAVE!" rmdir /s /q "%VENV%-py!HAVE!"
move "%VENV%" "%VENV%-py!HAVE!" >nul
goto makevenv

:checkpip
"%VENV%\Scripts\python.exe" -m pip --version >nul 2>nul
if not errorlevel 1 goto hasvenv
echo   The existing environment is incomplete. Building it again.
rmdir /s /q "%VENV%"

:makevenv
echo   Setting up OpenVoice Studio's environment (first run only)...
if defined USE_UV goto uvvenv
%PY% -m venv "%VENV%"
if errorlevel 1 goto badvenv
"%VENV%\Scripts\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 goto badvenv
goto hasvenv

:uvvenv
"%UV%" venv --seed --python 3.11 "%VENV%"
if errorlevel 1 goto uvfail

:hasvenv
for /f "delims=" %%V in ('"%VENV%\Scripts\python.exe" -c "import sys;print('%%d.%%d'%%sys.version_info[:2])" 2^>nul') do set "HAVE=%%V"
echo   Environment: Python !HAVE! in %VENV%
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto pipfail
"%VENV%\Scripts\python.exe" server.py
pause
exit /b 0

:badvenv
if exist "%VENV%" rmdir /s /q "%VENV%"
echo   Could not build the environment with %PY%. Trying the fetched Python instead.
goto needuv

:uvfail
echo.
echo   Could not fetch Python 3.11. Install it by hand from
echo   https://www.python.org/downloads/release/python-3119/ (tick "Add
echo   python.exe to PATH"), then run this file again.
echo.
pause
exit /b 1

:pipfail
echo   Could not install the Python packages. Check your internet connection.
pause
exit /b 1

rem --------------------------------------------------------------------- rem
rem Sets PY to a 3.11 or 3.10 interpreter, tested by running it: Microsoft
rem Store stubs answer `where` and then fail on execution.
:findwanted
set "PY="
for %%C in ("py -3.11" "py -3.10" "python3.11" "python3.10" "python") do (
  if not defined PY (
    %%~C -c "import sys;raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,11) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "PY=%%~C"
  )
)
exit /b 0

rem Sets BOOT to any Python 3.8+, enough to bootstrap uv.
:findany
set "BOOT="
for %%C in ("py -3" "python" "py -3.13" "py -3.12" "py -3.11" "py -3.10" "py -3.9" "py -3.8") do (
  if not defined BOOT (
    %%~C -c "import sys;raise SystemExit(0 if sys.version_info>=(3,8) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "BOOT=%%~C"
  )
)
exit /b 0
