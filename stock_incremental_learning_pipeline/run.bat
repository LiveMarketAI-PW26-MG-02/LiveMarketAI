@echo off
REM ============================================================
REM  run.bat — Stock Incremental Learning Pipeline
REM  Executes the full pipeline and opens results
REM ============================================================

echo.
echo ============================================================
echo   Stock Incremental Learning Pipeline — Run
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Run setup.bat first.
    pause
    exit /b 1
)

REM Check pipeline.py exists
if not exist pipeline.py (
    echo [ERROR] pipeline.py not found.
    echo         Please run this script from the stock_pipeline directory.
    pause
    exit /b 1
)

REM Create directories if missing
if not exist models  mkdir models
if not exist results mkdir results

echo Starting pipeline at %DATE% %TIME%
echo (This may take several minutes on first run)
echo.

REM ── Run the pipeline ─────────────────────────────────────────
python pipeline.py
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo ============================================================
    echo   Pipeline finished successfully!
    echo ============================================================
    echo.
    echo   Results saved in the 'results\' folder:
    echo     • incremental_metrics.csv    — batch-by-batch incremental stats
    echo     • retrain_metrics.csv        — full-retraining comparison stats
    echo     • comparison_summary.json    — high-level efficiency comparison
    echo     • pipeline.log               — full execution log
    echo.
    echo   Model checkpoints in 'models\':
    echo     • base_model.pt              — trained base LSTM model
    echo.

    REM Open results directory
    start "" results

    REM Optional: run the visualiser if matplotlib is available
    if exist visualise.py (
        echo Running visualiser ...
        python visualise.py
    )
) else (
    echo ============================================================
    echo   Pipeline exited with error code %EXIT_CODE%
    echo   Check results\pipeline.log for details.
    echo ============================================================
)

echo.
pause
