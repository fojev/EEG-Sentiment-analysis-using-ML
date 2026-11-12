@echo off
title EEG-Based Sentiment Analysis - Automated QA Suite
echo ======================================================================
echo Running Automated QA Verification Suite
echo ======================================================================
echo.

echo [1/3] Running Backend API Unit Tests...
venv\Scripts\python backend\test_backend.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Backend API tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Running End-to-End System & Workflow Verification...
venv\Scripts\python backend\test_e2e_workflow.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] End-to-end workflow verification failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Running Chrome Automated Browser QA Suite...
cd /d %~dp0frontend
node test_qa_suite.mjs
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Browser QA tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo ALL TESTS PASSED SUCCESSFULLY!
echo ======================================================================
pause
