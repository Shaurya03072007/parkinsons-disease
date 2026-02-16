@echo off
echo Starting Backend Server...
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
pause
