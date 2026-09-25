# Launch MsMBG-Fit on Windows. Installs uv first if it is missing.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Reload PATH in case uv was installed after this terminal was opened
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found, installing it..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;" + $env:Path
}

# Update to the latest version if this is a git clone (skipped if offline or with local changes)
if ((Test-Path .git) -and (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Checking for updates..."
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not update, starting the current version." }
}

# uv run installs Python 3.13 and syncs dependencies if needed before starting
uv run python main.py
exit $LASTEXITCODE
