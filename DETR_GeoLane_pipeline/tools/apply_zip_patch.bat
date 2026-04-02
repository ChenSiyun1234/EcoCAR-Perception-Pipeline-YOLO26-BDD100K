@echo off
setlocal
if "%~1"=="" goto usage
if "%~2"=="" goto usage
python "%~dp0apply_zip_patch.py" --project "%~1" --zip "%~2"
goto end
:usage
echo Usage:
echo   apply_zip_patch.bat "C:\path\to\project" "C:\path\to\patch.zip"
:end
endlocal
