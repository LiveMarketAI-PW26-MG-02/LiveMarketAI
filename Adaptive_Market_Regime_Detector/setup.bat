@echo off
echo MODULE 03 — Market Regime Detection Setup
python --version >nul 2>&1 || (echo Python not found. & pause & exit /b 1)
pip install -r requirements.txt
java -version >nul 2>&1 && (javac RegimeDetector.java && echo Java compiled OK.) || echo Java not found, skipping.
echo Setup complete! Run: run.bat
pause
