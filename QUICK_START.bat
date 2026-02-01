
@echo off
setlocal
title HPMA Quiz Assistant Setup & Run

echo ==================================================
echo      HPMA Quiz Assistant - One-Click Setup
echo ==================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: Check for Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js (LTS version) and try again.
    pause
    exit /b 1
)

:: Create Virtual Environment if missing
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

:: Install/Update Python Dependencies
echo [INFO] Installing/Updating Python dependencies...
venv\Scripts\pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
:: Explicitly ensure specific versions for OCR stability
venv\Scripts\pip install paddlepaddle-gpu==2.6.2 paddleocr websockets rich

:: Install Node Dependencies
if not exist "node_modules" (
    echo [INFO] Installing Node.js dependencies...
    call npm install
)

echo ==================================================
echo             Setup Complete!
echo ==================================================
echo.
echo Starting application...
echo.

:: Launch the main runner
call run_electron_ui.bat
