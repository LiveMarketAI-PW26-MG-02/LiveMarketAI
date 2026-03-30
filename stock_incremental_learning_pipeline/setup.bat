@echo off
REM ============================================================
REM  setup.bat — Stock Incremental Learning Pipeline
REM  Installs all Python dependencies
REM ============================================================

echo.
echo ============================================================
echo   Stock Incremental Learning Pipeline — Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM Upgrade pip silently
echo [1/6] Upgrading pip ...
python -m pip install --upgrade pip --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] pip upgrade failed — continuing with current version.
)

REM Core scientific stack
echo [2/6] Installing NumPy, SciPy, Pandas ...
python -m pip install numpy scipy pandas --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install NumPy / SciPy / Pandas.
    pause
    exit /b 1
)

REM PyTorch (CPU-only for portability)
echo [3/6] Installing PyTorch (CPU) ...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch installation failed.
    echo        Try manually: pip install torch --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)

REM Visualisation
echo [4/6] Installing Matplotlib ...
python -m pip install matplotlib --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Matplotlib not installed — plots will be skipped.
)

REM Optional extras
echo [5/6] Installing tqdm (progress bars) ...
python -m pip install tqdm --quiet

REM Create output directories
echo [6/6] Creating output directories ...
if not exist models  mkdir models
if not exist results mkdir results

echo.
echo ============================================================
echo   Setup complete!  Run the pipeline with:  run.bat
echo ============================================================
echo.
pause
