@echo off
REM ============================================================
REM  run.bat — Stock Streaming Classification System
REM  Launcher with interactive mode selection
REM ============================================================

SETLOCAL ENABLEDELAYEDEXPANSION

echo.
echo ============================================================
echo   Stock Streaming Window-Based Classification System
echo ============================================================
echo.

REM --- Activate virtual environment if it exists ---
IF EXIST ".venv\Scripts\activate.bat" (
    CALL .venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) ELSE (
    echo [WARN] No .venv found — using system Python. Run setup.bat first.
)

echo.
echo  Select a run mode:
echo.
echo    1  Stream   — Live streaming classification demo  (default, 15 s)
echo    2  Batch    — Batch vs streaming comparison       (600 ticks)
echo    3  Respond  — Window responsiveness analysis
echo    4  Full     — All modes + generate evaluation report
echo    5  Config   — Print active configuration and exit
echo    6  Custom   — Enter custom arguments manually
echo.
SET /P CHOICE="  Enter choice [1-6, default=1]: "

IF "%CHOICE%"=="" SET CHOICE=1

IF "%CHOICE%"=="1" (
    echo.
    echo [RUN] Mode: Stream (15 seconds)
    python main.py --mode stream --duration 15
    GOTO END
)

IF "%CHOICE%"=="2" (
    echo.
    echo [RUN] Mode: Batch vs Streaming comparison
    python main.py --mode batch --ticks 600
    GOTO END
)

IF "%CHOICE%"=="3" (
    echo.
    echo [RUN] Mode: Window Responsiveness Analysis
    python main.py --mode responsiveness
    GOTO END
)

IF "%CHOICE%"=="4" (
    echo.
    echo [RUN] Mode: Full evaluation (this may take 1-2 minutes)
    python main.py --mode full
    GOTO END
)

IF "%CHOICE%"=="5" (
    echo.
    echo [RUN] Printing configuration
    python main.py --config
    GOTO END
)

IF "%CHOICE%"=="6" (
    echo.
    SET /P ARGS="  Enter arguments (e.g. --mode stream --duration 30 --seed 99): "
    echo [RUN] python main.py !ARGS!
    python main.py !ARGS!
    GOTO END
)

echo [ERROR] Invalid choice: %CHOICE%
echo         Please enter a number between 1 and 6.

:END
echo.
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] The program exited with an error (code %ERRORLEVEL%).
    echo         Make sure setup.bat has been run successfully.
) ELSE (
    echo [INFO] Run complete.
)
echo.
pause
ENDLOCAL
