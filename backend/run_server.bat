@echo off
cd ..
echo Starting Backend Server from Project Root...
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
pause
