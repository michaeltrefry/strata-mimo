@echo off
rem Starts the MiMo server: start-mimo.bat [config.json]
rem The config is the first argument, else %STRATA_MIMO_CONFIG%, else the strata-mimo-mxfp4.json setup writes in this
rem folder, else strata-mimo.json there.  Its "port" is used (8096 if it has none); its "host" and "api_key" decide who can connect.
rem Ready after about 90 seconds.  Ctrl+C or closing the window stops it; stop-mimo.bat stops it from anywhere.
setlocal
cd /d "%~dp0"
set "CFG=%~1"
if "%CFG%"=="" set "CFG=%STRATA_MIMO_CONFIG%"
if "%CFG%"=="" if exist "%~dp0strata-mimo-mxfp4.json" set "CFG=%~dp0strata-mimo-mxfp4.json"
if "%CFG%"=="" set "CFG=%~dp0strata-mimo.json"
if not exist "%CFG%" (
  echo No MiMo config at "%CFG%": pass one, set STRATA_MIMO_CONFIG, or run START-HERE.bat --family mimo.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo No .venv in %~dp0: run START-HERE.bat --family mimo first.
  pause
  exit /b 1
)
for /f %%p in ('".venv\Scripts\python.exe" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('port', 8096))" "%CFG%"') do set "PORT=%%p"
title Strata-MiMo (port %PORT%)
".venv\Scripts\python.exe" serve\server.py --engine strata --config "%CFG%" --port %PORT%
pause
