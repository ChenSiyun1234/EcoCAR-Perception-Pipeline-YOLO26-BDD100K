@echo off
setlocal
if "%~1"=="" (
  echo Usage: apply_zip_patch.bat PROJECT_ROOT ZIP_OR_ZIP_DIR
  exit /b 1
)
if "%~2"=="" (
  echo Usage: apply_zip_patch.bat PROJECT_ROOT ZIP_OR_ZIP_DIR
  exit /b 1
)
set PROJECT=%~1
set INPUT=%~2
if exist "%INPUT%\" (
  python tools\apply_zip_patch.py --project "%PROJECT%" --zip-dir "%INPUT%"
) else (
  python tools\apply_zip_patch.py --project "%PROJECT%" --zip "%INPUT%"
)
endlocal
