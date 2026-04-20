@echo off
call venv\Scripts\activate

echo Running Python modules...
python main_runner\run_all.py

echo Running Java...
java -cp main_runner MainRunner

echo Running C++...
main_runner\main.exe

echo Running R scripts...
Rscript main_runner/run_all.R

pause
