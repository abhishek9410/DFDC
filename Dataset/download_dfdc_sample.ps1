param(
    [string]$OutputDir = "$PSScriptRoot\deepfake-detection-challenge",
    [string]$KaggleConfigDir = "$PSScriptRoot\.kaggle"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI not found. Install it with: ..\.venv\Scripts\python.exe -m pip install kaggle"
}

$repoToken = Join-Path $KaggleConfigDir "kaggle.json"
$homeToken = "$env:USERPROFILE\.kaggle\kaggle.json"

if (Test-Path $repoToken) {
    $env:KAGGLE_CONFIG_DIR = $KaggleConfigDir
} elseif (Test-Path $homeToken) {
    $env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"
} else {
    throw "Kaggle token not found. Save kaggle.json to either $repoToken or $homeToken."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
kaggle datasets download -d krishna191919/dfdc-train-sample-dataset -p $OutputDir

$zip = Join-Path $OutputDir "dfdc-train-sample-dataset.zip"
if (Test-Path $zip) {
    Expand-Archive -Force -Path $zip -DestinationPath $OutputDir
}

Write-Host "Downloaded DFDC files to $OutputDir"
Write-Host "For this project, train against: $OutputDir\train_sample_videos"
