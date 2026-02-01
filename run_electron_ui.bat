@echo off
echo ============================================
echo   Starting Quiz Assistant (Electron UI)
echo ============================================
echo.

:: Set the current directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

:: Start Python backend in background
echo [1/2] Starting Python backend...
start "" /B "%~dp0venv\Scripts\python.exe" "%~dp0terminal_app.py" --ui-mode
timeout /t 5 /nobreak >nul

:: Start Electron UI
echo [2/2] Starting Electron UI...
call npm start

echo.
echo Application closed.
pause
