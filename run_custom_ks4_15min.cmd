@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Activate the analysis environment first:
    echo     conda activate npixel_analysis
    exit /b 1
)

if not "%~1"=="" (
    set "CFG=%~1"
) else (
    for /f "delims=" %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = 'G:\SpikeGLX_data_saving_dir_Marcin_Szymon\full_preprocessed_data_files'; $configs = Get-ChildItem -Path $root -Filter pipeline_config.json -Recurse -ErrorAction SilentlyContinue; $cfg = $configs | Where-Object { $_.FullName -match '\\processing_work\\logs\\pipeline_config\.json$' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName; if (-not $cfg) { $cfg = $configs | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName }; if (-not $cfg) { throw ('No pipeline_config.json found under ' + $root) }; Write-Output $cfg"') do set "CFG=%%I"
)

if not defined CFG (
    echo Could not resolve a pipeline_config.json under full_preprocessed_data_files.
    exit /b 1
)

if not exist "%CFG%" (
    echo Config does not exist:
    echo     "%CFG%"
    exit /b 1
)

set "TMP_CFG=%TEMP%\somatic_custom_ks4_15min_pipeline_config.json"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfgPath = '%CFG%'; $outPath = '%TMP_CFG%'; $cfg = Get-Content -Raw -Path $cfgPath | ConvertFrom-Json; $cfg.custom_kilosort_repository = '%~dp0Kilosort_state_enhanced'; $cfg | Add-Member -NotePropertyName custom_ks4_reference_duration_s -NotePropertyValue 900 -Force; $cfg | Add-Member -NotePropertyName custom_ks4_run_quality_metrics -NotePropertyValue $true -Force; $cfg | ConvertTo-Json -Depth 100 | Set-Content -Path $outPath -Encoding UTF8"
if errorlevel 1 (
    echo Failed to write temporary patched config.
    exit /b 1
)

echo Using config:
echo     "%CFG%"
echo Temporary patched config:
echo     "%TMP_CFG%"
echo Running custom somatic fragment merge with custom_ks4_reference_duration_s=900

python pipeline_cli.py --config "%TMP_CFG%" --stages custom_ks4 --validated-output
set "PIPELINE_EXIT=%ERRORLEVEL%"
if not "%PIPELINE_EXIT%"=="0" exit /b %PIPELINE_EXIT%

echo Copying custom somatic outputs into preprocessed_data\custom_ks4_somatic
python package_custom_ks4_results.py --config "%TMP_CFG%"
exit /b %errorlevel%
