$ErrorActionPreference = "Stop"

$repoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $repoDir "Data_processing_GUI_launcher.pyw"
$icon = Join-Path $repoDir "processing.ico"
$desktop = [Environment]::GetFolderPath("Desktop")

$pythonwCandidates = @()
$pythonwCandidates += @(
    "E:\SZYMON\Documents\Miniconda\envs\data_proc_gui\pythonw.exe",
    "G:\Miniconda3\envs\NeuropixelsGUI\pythonw.exe",
    "C:\Users\slic\miniconda3\envs\NeuropixelsGUI\pythonw.exe",
    "G:\Miniconda3\envs\npixel_analysis\pythonw.exe",
    "C:\Users\slic\miniconda3\envs\npixel_analysis\pythonw.exe",
    "C:\Miniconda3\envs\npixel_analysis\pythonw.exe",
    "C:\ProgramData\Miniconda3\envs\npixel_analysis\pythonw.exe",
    "C:\ProgramData\Anaconda3\envs\npixel_analysis\pythonw.exe"
)
if ($env:CONDA_PREFIX) {
    $pythonwCandidates += Join-Path $env:CONDA_PREFIX "pythonw.exe"
}
$pythonwCandidates = $pythonwCandidates | Where-Object { $_ -and (Test-Path $_) }

$pythonwCommand = Get-Command pythonw.exe -ErrorAction SilentlyContinue
if ($pythonwCommand) {
    $pythonwCandidates += $pythonwCommand.Source
}

$pythonw = $pythonwCandidates | Select-Object -First 1
if (-not (Test-Path $pythonw)) {
    throw "Could not find pythonw.exe. Activate the npixel_analysis conda environment, then run this script again."
}
if (-not (Test-Path $launcher)) {
    throw "Could not find launcher at $launcher"
}
if (-not (Test-Path $icon)) {
    throw "Could not find icon at $icon"
}

$shell = New-Object -ComObject WScript.Shell

$shortcutPaths = @(
    (Join-Path $desktop "Data Processing GUI.lnk"),
    (Join-Path $repoDir "Data Processing GUI.lnk")
)

foreach ($shortcutPath in $shortcutPaths) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $pythonw
    $shortcut.Arguments = "`"$launcher`""
    $shortcut.WorkingDirectory = $repoDir
    $shortcut.IconLocation = $icon
    $shortcut.Description = "Launch Neuropixels data processing GUI in the data_proc_gui conda environment."
    $shortcut.Save()
    Write-Host "Created shortcut:" $shortcutPath
}
