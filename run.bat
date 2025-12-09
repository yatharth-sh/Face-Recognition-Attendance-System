@echo off
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python.
    pause
    exit /b
)

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Face Recognition System...
python DemoVersion.py
pause
