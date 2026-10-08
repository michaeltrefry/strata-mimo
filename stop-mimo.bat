@echo off
rem Stops the MiMo server and its engine started from this folder, wherever they were started (a window, a
rem terminal, another program).  A hard stop: the engine's learned expert profile keeps its last periodic save.
rem Ctrl+C in the server's window is the gentle way.
setlocal
set "HERE=%~dp0"
powershell -NoProfile -Command "$h = $env:HERE.TrimEnd('\'); Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*server.py*' -and ($_.ExecutablePath -like \"$h\*\" -or $_.CommandLine -like \"*$h*\")) -or ($_.Name -eq 'strata-mimo.exe' -and $_.ExecutablePath -like \"$h\*\") } | ForEach-Object { Write-Host ('stopping ' + $_.Name + ' (' + $_.ProcessId + ')'); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo Strata-MiMo stopped.
