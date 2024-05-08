cd AI-Model
echo server starting
cd /d %~dp0 & call Scripts/activate & uvicorn main:app --reload & Scripts/deactivate