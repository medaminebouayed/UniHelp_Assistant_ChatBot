@echo off
title UniHelp - Services Launcher
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

set "BACKEND_PORT=8000"
set "RAG_PORT=8002"
set "FRONTEND_PORT=4200"

set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "RAG_DIR=%PROJECT_ROOT%\RAG Core"
set "FRONTEND_DIR=%PROJECT_ROOT%\frontend"

set "LOGS_DIR=%PROJECT_ROOT%\logs"

if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

echo =============================================
echo        UniHelp - Services Launcher
echo =============================================
echo   Project : %PROJECT_ROOT%
echo   Date    : %DATE% %TIME%
echo =============================================
echo.

:: ── Prerequisites ─────────────────────────────────────────────────────
echo [1/5] Checking prerequisites...
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found in PATH. Install Python 3.10+.
    pause
    exit /b 1
)
echo   [OK] Python found

where node >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Node.js not found in PATH.
    pause
    exit /b 1
)
echo   [OK] Node.js found

where npm >nul 2>&1
if errorlevel 1 (
    echo [FAIL] npm not found.
    pause
    exit /b 1
)
echo   [OK] npm found

:: ── Check databases ───────────────────────────────────────────────────
echo.

echo [2/5] Checking database connectivity...
echo.

:: PostgreSQL — try pg_isready, fallback to port check
pg_isready -h localhost -p 5433 >nul 2>&1
if errorlevel 1 (
    netstat -an | findstr "127.0.0.1:5433" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] PostgreSQL not detected on localhost:5433
        echo        Start PostgreSQL first, then re-run this script.
        echo        Docker: docker run -d --name unihelp-pg ^
             -e POSTGRES_PASSWORD=password1234 ^
             -e POSTGRES_DB=graphrag ^
             -p 5433:5432 postgres:15
    ) else (
        echo   [OK] PostgreSQL detected on port 5433
    )
) else (
    echo   [OK] PostgreSQL is ready
)

:: Neo4j
netstat -an | findstr "127.0.0.1:7687" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Neo4j not detected on localhost:7687
    echo        Start Neo4j first, then re-run this script.
    echo        Docker: docker run -d --name unihelp-neo4j ^
         -e NEO4J_AUTH=neo4j/password123 ^
         -p 7687:7687 -p 7474:7474 ^
         neo4j:5
) else (
    echo   [OK] Neo4j detected on port 7687
)

echo.

:: ── Kill stale processes ──────────────────────────────────────────────
echo [3/5] Cleaning up stale processes...

for %%p in (%BACKEND_PORT% %RAG_PORT% %FRONTEND_PORT%) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p "') do (
        for /f %%b in ("%%a") do (
            if not "%%b"=="0" (
                taskkill /F /PID %%b >nul 2>&1
                echo   [KILL] Port %%p (PID %%b)
            )
        )
    )
)
timeout /t 2 /nobreak >nul
echo   [OK] Ports cleaned
echo.

:: ── Start Services ────────────────────────────────────────────────────
echo [4/5] Starting services...
echo.

:: ── 4a. RAG API ───────────────────────────────────────────────────────
echo   [RAG API] Starting on port %RAG_PORT%...
cd /d "%RAG_DIR%"
start "UniHelp-RAG-API" /MIN cmd /c "python -m uvicorn rag_api.main:app --host 127.0.0.1 --port %RAG_PORT% --log-level warning 2>>"%LOGS_DIR%\rag-api.log""
cd /d "%PROJECT_ROOT%"
timeout /t 8 /nobreak >nul
echo            http://127.0.0.1:%RAG_PORT%/docs
echo.

:: ── 4b. Backend ───────────────────────────────────────────────────────
echo   [Backend] Starting on port %BACKEND_PORT%...
cd /d "%BACKEND_DIR%"
start "UniHelp-Backend" /MIN cmd /c "python -m uvicorn main:app --host 127.0.0.1 --port %BACKEND_PORT% --reload --log-level warning 2>>"%LOGS_DIR%\backend.log""
cd /d "%PROJECT_ROOT%"
timeout /t 8 /nobreak >nul
echo            http://127.0.0.1:%BACKEND_PORT%/docs
echo.

:: ── 4c. Frontend ──────────────────────────────────────────────────────
echo   [Frontend] Starting on port %FRONTEND_PORT%...
cd /d "%FRONTEND_DIR%"
start "UniHelp-Frontend" /MIN cmd /c "npm start -- --port %FRONTEND_PORT% 2>>"%LOGS_DIR%\frontend.log""
cd /d "%PROJECT_ROOT%"
echo            http://127.0.0.1:%FRONTEND_PORT%
echo.

:: ── Summary ───────────────────────────────────────────────────────────
echo [5/5] Done!
echo.
echo =============================================
echo       UniHelp - All Services Running
echo =============================================
echo.
echo   Frontend   : http://127.0.0.1:%FRONTEND_PORT%
echo   Backend    : http://127.0.0.1:%BACKEND_PORT%/docs
echo   RAG API    : http://127.0.0.1:%RAG_PORT%/docs
echo.
echo   Logs       : %LOGS_DIR%
echo.
echo   To stop:  Close each terminal window
echo   or run:   taskkill /FI "WINDOWTITLE eq UniHelp-*"
echo.
echo =============================================
echo.
pause
