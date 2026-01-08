@echo off
setlocal
chcp 65001 >nul

cd /d C:\Users\Astra\Desktop\spotify_pipeline

if not exist logs (
    mkdir logs
)

call .venv\Scripts\activate

python spotify_extract.py >> logs\extract.log 2>&1

endlocal
