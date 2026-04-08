@echo off
SETLOCAL

echo ============================================================
echo  AAAI Multimodal Equity Discovery Endpoint — SETUP
echo ============================================================

:: Check Docker
where docker >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

:: Check Node.js (for local dev mode)
where node >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Node.js not found. Frontend will run via Docker only.
) ELSE (
    echo [INFO] Node.js found.
)

:: Check Python
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Python not found. Backend will run via Docker only.
) ELSE (
    echo [INFO] Python found.
)

:: Copy .env if not present
IF NOT EXIST backend\.env (
    echo [INFO] Creating backend\.env from .env.example ...
    copy backend\.env.example backend\.env
    echo [ACTION REQUIRED] Edit backend\.env and fill in your API keys before running.
) ELSE (
    echo [INFO] backend\.env already exists. Skipping copy.
)

:: Build Docker images
echo.
echo [INFO] Building Docker images (this may take a few minutes)...
docker-compose build --no-cache

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker build failed. Check Docker Desktop is running.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  SETUP COMPLETE
echo  Run run.bat to start the system.
echo ============================================================
pause
