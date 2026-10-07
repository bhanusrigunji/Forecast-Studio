@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON=py"
) else (
  set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the project environment...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto failed
)

echo Installing or updating project packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Starting Forecast Studio. Close this window to stop it.
".venv\Scripts\python.exe" -m streamlit run app.py
goto end

:failed
echo.
echo Setup failed. Confirm Python 3.10 or newer is installed and try again.
pause

:end
endlocal
