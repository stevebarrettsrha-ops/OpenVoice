@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title OpenVoice Studio

rem Flat control flow on purpose. Nested parenthesised blocks with delayed
rem expansion are where batch fails silently, so every branch is a label.

call :findpy
if defined PY goto haspy

echo.
echo   Python 3.10 or newer was not found.
echo.

where winget >nul 2>nul
if errorlevel 1 goto manualpy

echo   WinGet is here, so this can be done for you: the Python install manager
echo   from the Microsoft Store, then Python 3.11 through it.
echo   (3.11 rather than the newest: MeloTTS, which gives OpenVoice V2 its
echo   voices, only installs cleanly on 3.10 and 3.11.)
echo.
choice /c YN /n /m "  Install Python now? [Y/N] "
if errorlevel 2 goto declined

echo.
echo   Installing the Python install manager...
winget install 9NQ7512CXL7T --accept-package-agreements --accept-source-agreements
echo.
echo   Installing Python 3.11...
py install 3.11

call :findpy
if defined PY goto haspy
goto restartneeded

:manualpy
echo   Install Python 3.11 from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
exit /b 1

:declined
echo.
echo   Nothing was installed. Get Python 3.11 from
echo   https://www.python.org/downloads/ and run this file again.
echo.
pause
exit /b 1

:restartneeded
echo.
echo   Python is installed, but this window cannot see it yet.
echo   Close it and run run.bat again.
echo.
pause
exit /b 1

:haspy
echo   Using: %PY%

rem First run only, and only when the Python found is 3.12 or newer: MeloTTS
rem (OpenVoice V2's voices) pins tokenizers 0.13, which has no wheels past
rem 3.11, so offer 3.11 now rather than after a failed install. Declining
rem keeps the Python found; V1 and the app itself run fine on it.
if exist ".venv\Scripts\python.exe" goto skipoffer
%PY% -c "import sys;raise SystemExit(0 if sys.version_info<(3,12) else 1)" >nul 2>nul
if not errorlevel 1 goto skipoffer
where winget >nul 2>nul
if errorlevel 1 goto skipoffer
echo.
echo   This Python is 3.12 or newer. OpenVoice V2's voices (MeloTTS) only
echo   install on 3.10 or 3.11. Python 3.11 can be added now through the
echo   Python install manager; it sits beside your current Python.
echo.
choice /c YN /n /m "  Install Python 3.11 for this app? [Y/N] "
if errorlevel 2 goto skipoffer
echo.
winget install 9NQ7512CXL7T --accept-package-agreements --accept-source-agreements
py install 3.11
call :findpy
echo   Using: %PY%
%PY% -c "import sys;raise SystemExit(0 if sys.version_info<(3,12) else 1)" >nul 2>nul
if not errorlevel 1 goto skipoffer
echo.
echo   Python 3.11 is installed, but this window cannot see it yet.
echo   Close it and run run.bat again.
echo.
pause
exit /b 1

:skipoffer
rem Everything the app installs goes into .venv beside this file - Flask now,
rem PyTorch and the models' packages later from the Engine page - so the Python
rem that was found is left as it was found.
if not exist ".venv\Scripts\python.exe" goto makevenv
".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if not errorlevel 1 goto hasvenv
echo   The existing environment is incomplete. Building it again.
rmdir /s /q ".venv"

:makevenv
echo   Setting up OpenVoice Studio's environment (first run only)...
%PY% -m venv .venv
if errorlevel 1 goto badvenv
".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 goto badvenv

:hasvenv
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto pipfail
".venv\Scripts\python.exe" server.py
pause
exit /b 0

:badvenv
if exist ".venv" rmdir /s /q ".venv"
echo   Could not build a separate environment; using %PY% as it is.
%PY% -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto pipfail
%PY% server.py
pause
exit /b 0

:pipfail
echo   Could not install the Python packages. Check your internet connection.
pause
exit /b 1

rem --------------------------------------------------------------------- rem
rem Sets PY to the first interpreter that is really 3.10+, preferring 3.11 and
rem 3.10 (see above). Tested by running each one: Microsoft Store stubs answer
rem `where` and then fail on execution.
:findpy
set "PY="
for %%C in ("py -3.11" "py -3.10" "py -3.12" "py -3.13" "py -3" "python") do (
  if not defined PY (
    %%~C -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "PY=%%~C"
  )
)
exit /b 0
