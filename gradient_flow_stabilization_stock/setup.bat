@echo off
echo Setting up environment...
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

echo Compiling Java...
javac main_runner\MainRunner.java

echo Compiling C++...
g++ main_runner\main.cpp -o main_runner\main.exe

echo Setup complete!
pause
