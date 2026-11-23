@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create a virtual environment and install requirements first; see README.md
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m src.crawler --dry-run
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m src.crawler
if errorlevel 1 exit /b 1
echo P1 COMPLETE. Inspect data\audit\p1_integrity.json
