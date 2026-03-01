@echo off
REM Batch script to run the Hospital Readmission Prediction project

echo Hospital Readmission Prediction Project
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher from https://www.python.org/downloads/
    echo Then run this script again.
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not available. Please ensure Python was installed with pip.
    pause
    exit /b 1
)

echo Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install required packages.
    pause
    exit /b 1
)

echo Running the main project script...
python main.py

echo.
echo Process completed. Press any key to exit.
pause >nul