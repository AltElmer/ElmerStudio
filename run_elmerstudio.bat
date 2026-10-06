@echo off
rem Launch Elmer Studio (Windows). Creates the virtual environment on first run.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :err
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
)
start "" ".venv\Scripts\pythonw.exe" -m elmerstudio %*
exit /b 0
:err
echo Setup failed. Install Python 3.10+ and try again.
pause
exit /b 1
