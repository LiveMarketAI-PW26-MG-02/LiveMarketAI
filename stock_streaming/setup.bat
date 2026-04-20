@echo off
REM ============================================================
REM  setup.bat — Stock Streaming Classification System
REM  Sets up a Python virtual environment and installs deps
REM ============================================================

SETLOCAL ENABLEDELAYEDEXPANSION

echo.
echo ============================================================
echo   Stock Streaming Window-Based Classification System
echo   Setup Script
echo ============================================================
echo.

REM --- Check Python availability ---
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

FOR /F "tokens=2 delims= " %%V IN ('python --version 2^>^&1') DO SET PY_VER=%%V
echo [INFO] Python version: %PY_VER%

REM --- Create virtual environment ---
IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment in .venv ...
    python -m venv .venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]   Virtual environment created.
) ELSE (
    echo [INFO] Virtual environment already exists — skipping creation.
)

REM --- Activate virtual environment ---
CALL .venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo [OK]   Virtual environment activated.

REM --- Upgrade pip ---
echo [INFO] Upgrading pip ...
python -m pip install --upgrade pip --quiet
echo [OK]   pip upgraded.

REM --- Install dependencies ---
echo [INFO] Installing required packages ...

pip install scikit-learn --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] scikit-learn install failed — online classifier will use heuristic fallback.
)

pip install numpy --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] numpy install failed — some features may be unavailable.
)

echo [OK]   Packages installed.

REM --- Create output directories ---
IF NOT EXIST "reports" mkdir reports
echo [OK]   Output directories ready.

REM --- Verify import ---
echo [INFO] Verifying installation ...
python -c "import sklearn; import numpy; print('[OK]   sklearn', sklearn.__version__, '/ numpy', numpy.__version__)"
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Verification failed — the system will run in heuristic-only mode.
)

echo.
echo ============================================================
echo   Setup complete!
echo   Run  run.bat  to start the system.
echo ============================================================
echo.
pause
ENDLOCAL
