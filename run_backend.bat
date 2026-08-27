@echo off
echo Starting ENDO-X FastAPI Backend...
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
