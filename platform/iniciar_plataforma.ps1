# Inicia OCEAN GUARD AI en http://127.0.0.1:8000 (Ctrl+C para detener).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here "..\unet_oilspill\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No se encuentra el entorno virtual: $py" }
Set-Location $here
Write-Host "Abra http://127.0.0.1:8000 en el navegador (la primera carga del modelo tarda unos segundos)."
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
