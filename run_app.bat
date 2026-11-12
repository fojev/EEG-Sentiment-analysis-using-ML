@echo off
title EEG-Based Sentiment Analysis - Full Stack Launcher
echo ======================================================================
echo Starting EEG-Based Sentiment Analysis Application
echo ======================================================================
echo.

:: Start Flask Backend in background
echo [1/2] Launching Flask Backend API on http://127.0.0.1:5000 ...
start "Flask Backend (EEG ML)" cmd /k "cd /d %~dp0 && venv\Scripts\python backend\app.py"

:: Wait 3 seconds for model loading
timeout /t 3 /nobreak >nul

:: Start React Vite Frontend in background
echo [2/2] Launching React Vite Frontend on http://localhost:5173 ...
start "React Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

:: Wait 2 seconds for Vite to initialize
timeout /t 2 /nobreak >nul

:: Open default web browser
echo.
echo Opening http://localhost:5173 in your default browser...
start http://localhost:5173

echo.
echo ======================================================================
echo Both servers are running!
echo - Frontend: http://localhost:5173
echo - Backend : http://127.0.0.1:5000
echo ======================================================================
pause

