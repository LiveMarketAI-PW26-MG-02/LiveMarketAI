@echo off
echo Setting up environment...
python -m venv venv
call venv\Scripts\activate
pip install numpy pandas
echo Setup complete!
pause
