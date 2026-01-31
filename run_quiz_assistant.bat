@echo off
:: Check for administrator privileges
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    :: Not running as administrator, elevate the script
    echo Requesting administrator privileges...
    PowerShell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

:: If we get here, we're running with admin rights
echo Starting HPMA Quiz Assistant with administrator privileges...
echo.

:: Set the current directory to where the batch file is located
cd /d "%~dp0"
echo Working directory: %CD%

:: Check for Virtual Environment
if not exist "%~dp0venv\Scripts\python.exe" (
    echo.
    echo [FIRST RUN DETECTED]
    echo Virtual environment not found. Running setup...
    echo.
    call "%~dp0setup_env.bat"
)

:: Make sure Python is in the path (in venv)
if not exist "%~dp0venv\Scripts\python.exe" (
    echo Setup failed or canceled. Python not found in venv.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)


:: Start the terminal app with full path using Windows Terminal
start cmd /k "cd /d "%~dp0" && "%~dp0venv\Scripts\python.exe" "%~dp0terminal_app.py""

echo.
echo HPMA Quiz Assistant started successfully!
echo F2 - Capture and process quiz
echo.
:: Auto-close this launcher window after 2 seconds
timeout /t 2 /nobreak >nul
exit 