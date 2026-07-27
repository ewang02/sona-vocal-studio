<#
.SYNOPSIS
  One-command setup for Windows.

.DESCRIPTION
  Checks for Node.js 22.13+, Python 3.13, FFmpeg, and FFprobe, offers to
  install whatever is missing with winget, then runs `npm install` and
  `npm run pipeline:setup`. Re-running it is safe: anything already present
  is skipped.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
  powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Yes
  powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Check
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$RequiredNodeMajor = 22
$RequiredNodeMinor = 13

function Write-Ok    { param($m) Write-Host "  [ok] $m"      -ForegroundColor Green }
function Write-Miss  { param($m) Write-Host "  [--] $m"      -ForegroundColor Red }
function Write-Note  { param($m) Write-Host "  ->   $m"      -ForegroundColor Yellow }
function Write-Step  { param($m) Write-Host "`n$m"           -ForegroundColor White }

function Test-Have { param($name) [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Test-NodeVersion {
    if (-not (Test-Have node)) { return $false }
    $raw = (& node -v) -replace '^v', ''
    $parts = $raw.Split('.')
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -gt $RequiredNodeMajor) { return $true }
    return ($major -eq $RequiredNodeMajor -and $minor -ge $RequiredNodeMinor)
}

# Sona's setup script accepts the first working `python` on PATH, which may be
# an older interpreter. Resolve an explicit 3.13 and hand it over via $env:PYTHON.
$script:PythonBin = $null
function Find-Python {
    foreach ($candidate in @("python3.13", "python", "python3")) {
        if (-not (Test-Have $candidate)) { continue }
        & $candidate -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 13) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $script:PythonBin = $candidate; return $true }
    }
    # The Windows launcher can reach a 3.13 that is not first on PATH.
    if (Test-Have py) {
        & py -3.13 -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $script:PythonBin = "py -3.13"; return $true }
    }
    return $false
}

$missing = New-Object System.Collections.Generic.List[string]

Write-Step "Checking prerequisites"

if (Test-NodeVersion) {
    Write-Ok "Node.js $(& node -v)"
} else {
    if (Test-Have node) {
        Write-Miss "Node.js $(& node -v) is older than $RequiredNodeMajor.$RequiredNodeMinor"
    } else {
        Write-Miss "Node.js $RequiredNodeMajor.$RequiredNodeMinor+ is not installed"
    }
    $missing.Add("node") | Out-Null
}

if (Find-Python) {
    Write-Ok "Python 3.13+ ($script:PythonBin)"
} else {
    Write-Miss "Python 3.13+ is not installed"
    $missing.Add("python") | Out-Null
}

$mediaMissing = $false
foreach ($tool in @("ffmpeg", "ffprobe")) {
    if (Test-Have $tool) { Write-Ok "$tool" } else { Write-Miss "$tool is not installed"; $mediaMissing = $true }
}
if ($mediaMissing) { $missing.Add("ffmpeg") | Out-Null }

if ($Check) {
    if ($missing.Count -eq 0) { Write-Step "Everything is installed."; exit 0 }
    Write-Step "Missing: $($missing -join ', ')"
    exit 1
}

if ($missing.Count -gt 0) {
    Write-Step "Missing: $($missing -join ', ')"

    if (-not (Test-Have winget)) {
        Write-Host @"

winget is required to install the missing tools automatically. It ships with
Windows 10 1809+ and Windows 11 as "App Installer". Install it from the
Microsoft Store, or install these manually and re-run:
  Node.js 22.13+   https://nodejs.org/
  Python 3.13      https://www.python.org/downloads/
  FFmpeg           https://ffmpeg.org/download.html
"@ -ForegroundColor Red
        exit 1
    }

    if (-not $Yes) {
        $reply = Read-Host "Install them now? [y/N]"
        if ($reply -notmatch '^[yY]') { Write-Host "Aborted. Nothing was installed."; exit 1 }
    }

    $packages = @()
    foreach ($item in $missing) {
        switch ($item) {
            "node"   { $packages += "OpenJS.NodeJS.LTS" }
            "python" { $packages += "Python.Python.3.13" }
            "ffmpeg" { $packages += "Gyan.FFmpeg" }
        }
    }

    foreach ($package in $packages) {
        Write-Note "winget install $package"
        & winget install --id $package --exact --silent --accept-package-agreements --accept-source-agreements
        # winget returns a non-zero code when a package is already installed.
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
            Write-Host "winget failed for $package (exit $LASTEXITCODE)." -ForegroundColor Red
            exit 1
        }
    }

    # winget updates the machine PATH but not this already-running shell.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")

    Write-Step "Re-checking"
    if (-not (Test-NodeVersion)) {
        Write-Miss "Node.js is still missing. Close this window, open a new terminal, and re-run."
        exit 1
    }
    Write-Ok "Node.js $(& node -v)"
    if (-not (Find-Python)) {
        Write-Miss "Python 3.13 is still missing. Close this window, open a new terminal, and re-run."
        exit 1
    }
    Write-Ok "Python 3.13+ ($script:PythonBin)"
    if (-not ((Test-Have ffmpeg) -and (Test-Have ffprobe))) {
        Write-Miss "FFmpeg is still missing. Close this window, open a new terminal, and re-run."
        exit 1
    }
    Write-Ok "FFmpeg and FFprobe"
}

Write-Step "Installing JavaScript dependencies"
& npm install
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "Setting up the Python pipeline"
Write-Note "this downloads PyTorch and ~249 MiB of separation models; it can take several minutes"
if ($script:PythonBin) { $env:PYTHON = $script:PythonBin }
& npm run pipeline:setup
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "Done"
Write-Host @"
Start Sona with:

  npm start

That runs the web interface and the local processing companion together.
"@
