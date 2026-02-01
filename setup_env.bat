@echo off
echo ============================================
echo   HPMA Quiz Assistant - First Run Setup
echo ============================================
echo.
echo This script will:
echo - Create a Python virtual environment
echo - Install PaddleOCR and dependencies
echo - Configure the application
echo.

:: Set the current directory
cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing dependencies in correct order...
echo    - Installing Torch (CPU version to avoid DLL conflicts)...
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo    - Installing PaddlePaddle GPU...
pip install paddlepaddle-gpu==2.6.2

echo    - Installing PaddleOCR...
pip install paddleocr==2.8.1

echo    - Installing OpenCV...
pip install opencv-python

echo    - Installing remaining dependencies...
pip install -r requirements.txt

echo [5/6] Installing Node.js dependencies...
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Node.js is not installed. UI features may not work.
    echo Please install Node.js from https://nodejs.org/
) else (
    call npm install
    if %errorlevel% neq 0 (
        echo [WARNING] npm install failed.
    ) else (
        echo Node modules installed successfully.
    )
)

echo [6/6] Verifying installation...
python -c "from paddleocr import PaddleOCR; import cv2; print('✓ PaddleOCR installed successfully')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] PaddleOCR verification failed, but continuing...
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo You can now run the application using:
echo    run_quiz_assistant.bat
echo.
pause
