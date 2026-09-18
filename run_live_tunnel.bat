@echo off
title PolyDubAI Live Public Tunnel
echo =======================================================
echo   PolyDubAI -- AI-Powered Multilingual Video Dubbing
echo   Starting Web Server and Live Public Tunnel...
echo =======================================================
echo.
echo 1. Starting FastAPI Web Server on port 8000...
start "PolyDubAI Server" python run_web.py --no-browser --port 8000
timeout /t 3 /nobreak >nul
echo.
echo 2. Launching Secure Cloudflare Tunnel...
echo.
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
