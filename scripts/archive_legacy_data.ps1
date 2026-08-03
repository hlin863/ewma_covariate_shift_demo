param(
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archiveRoot = Join-Path $projectRoot "archive\data"

$legacyPaths = @(
    "data\processed\bci_2b",
    "outputs\figures\bci_distributions",
    "outputs\metrics\bci_2b_table3_results.csv",
    "outputs\metrics\bci_2b_table1_comparison.csv"
)

New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

foreach ($relativePath in $legacyPaths) {
    $source = Join-Path $projectRoot $relativePath
    if (-not (Test-Path $source)) {
        Write-Host "Skip (not found): $relativePath"
        continue
    }

    $destination = Join-Path $archiveRoot $relativePath
    $destinationParent = Split-Path $destination -Parent
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null

    if (Test-Path $destination) {
        throw "Archive destination already exists: $destination"
    }

    if ($WhatIf) {
        Write-Host "Would move: $source -> $destination"
    }
    else {
        Move-Item -Path $source -Destination $destination
        Write-Host "Moved: $relativePath -> archive\data\$relativePath"
    }
}

Write-Host "Legacy-data archival pass complete."
Write-Host "Review with: git status --short"
