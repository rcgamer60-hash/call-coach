@echo off
cd /d "%~dp0"
if not exist ".venv" (
  python -m venv .venv
  .venv\Scripts\pip install -r backend\requirements.txt
)
.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
