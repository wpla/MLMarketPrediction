@echo off
call ..\venv\Scripts\activate.bat
cd ..\scripts
python ..\prediction\fit_models.py %1 %2 %3 %4 %5 %6 %7 %8 %9
call ..\venv\Scripts\deactivate.bat
