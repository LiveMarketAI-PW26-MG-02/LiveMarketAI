@echo off
echo Setting up environment...
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

echo Compiling C++ modules...
for /d %%d in (novelty*) do (
    if exist %%d\logic.cpp (
        g++ %%d\logic.cpp -o %%d\logic.exe
    )
)

echo Setup complete!
pause
