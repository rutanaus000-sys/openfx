@echo off
REM Starts the Comfy API proxy and the OpenFX web app.
REM Start ComfyUI yourself first (it must be listening on 127.0.0.1:8188).
setlocal
cd /d "%~dp0\.."

if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

if "%COMFYUI_DIR%"=="" set COMFYUI_DIR=%USERPROFILE%\ComfyUI

start "Comfy API proxy" .venv\Scripts\comfy-api-proxy.exe --comfyui-base-dir "%COMFYUI_DIR%"
set COMFY_BASE_URL=http://127.0.0.1:8189
echo OpenFX: http://127.0.0.1:7860
.venv\Scripts\python.exe -m openfx
