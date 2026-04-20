@echo off
SETLOCAL

echo ============================================================
echo  AAAI Multimodal Equity Discovery Endpoint — START
echo ============================================================

:: Verify .env exists
IF NOT EXIST backend\.env (
    echo [ERROR] backend\.env not found. Run setup.bat first.
    pause
    exit /b 1
)

:: Start all services
echo [INFO] Starting PostgreSQL, Backend, and Frontend via Docker Compose...
docker-compose up -d

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start services. Ensure Docker Desktop is running.
    pause
    exit /b 1
)

:: Wait for backend to be healthy
echo [INFO] Waiting for backend to become available...
:WAIT_BACKEND
timeout /t 3 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Backend not ready yet, retrying...
    goto WAIT_BACKEND
)

echo.
echo ============================================================
echo  SYSTEM RUNNING
echo  Backend  API : http://localhost:8000/api/v1
echo  API Docs     : http://localhost:8000/docs
echo  Frontend UI  : http://localhost:3000
echo ============================================================
echo.
echo  FIRST TIME? Open the UI and click "Seed Instruments"
echo  to populate all multimodal streams.
echo ============================================================

:: Auto-open browser
start http://localhost:3000

pause
