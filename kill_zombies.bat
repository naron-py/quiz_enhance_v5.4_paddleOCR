@echo off
echo ===============================================
echo   Zombie Protocol Initiated: Purging Processes
echo ===============================================
echo.

echo [1/3] Terminating Electron...
taskkill /F /IM electron.exe /T 2>nul
if %errorlevel% equ 0 ( echo    - Electron processes terminated. ) else ( echo    - No Electron processes found. )

echo [2/3] Terminating Node.js...
taskkill /F /IM node.exe /T 2>nul
if %errorlevel% equ 0 ( echo    - Node processes terminated. ) else ( echo    - No Node processes found. )

echo [3/3] Terminating Python Backend...
taskkill /F /IM python.exe /T 2>nul
if %errorlevel% equ 0 ( echo    - Python processes terminated. ) else ( echo    - No Python processes found. )

echo.
echo ===============================================
echo   Purge Complete. You may now run the App.
echo ===============================================
pause
