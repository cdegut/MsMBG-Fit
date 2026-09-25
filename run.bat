@echo off
rem Double-clickable launcher for Windows (bypasses the PowerShell script execution policy)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 pause
