@echo off
REM Environment setup script for rvc_stream (Windows)

echo === RVC Stream Environment Setup ===

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda not found. Please install Miniconda or Anaconda.
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Create conda environment
echo Creating conda environment 'rvcstream'...
conda env create -f environment.yml

REM Activate message
echo.
echo === Setup Complete ===
echo.
echo To activate the environment, run:
echo   conda activate rvcstream
echo.
echo Then to test:
echo   python -m tests.test_simple
echo.
echo To run the client:
echo   python -m src.rvc_client --help
echo.
echo To run the server (requires RVC dependencies):
echo   python -m src.rvc_server --help
