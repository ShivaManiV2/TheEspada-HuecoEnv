@echo off
title HuecoEnv Dashboard
echo.
echo  ========================================
echo   HuecoEnv Dashboard Starting...
echo   Open: http://127.0.0.1:7860
echo  ========================================
echo.
cd /d "%~dp0"
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
pause
