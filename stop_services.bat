@echo off
title UniHelp - Stop Services
cd /d "%~dp0"

echo =============================================
echo        UniHelp - Stopping Services
echo =============================================
echo.

echo Stopping UniHelp processes by window title...
taskkill /FI "WINDOWTITLE eq UniHelp-*" /F >nul 2>&1

echo Cleaning up ports 8000, 8002, 4200...
for %%p in (8000 8002 4200) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p "') do (
        for /f %%b in ("%%a") do (
            if not "%%b"=="0" (
                taskkill /F /PID %%b >nul 2>&1
                echo   [KILL] Port %%p (PID %%b)
            )
        )
    )
)

echo.
echo [OK] All UniHelp services stopped.
echo.
pause
