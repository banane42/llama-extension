param(
    [string]$Target = "template_debug",
    [string]$Platform = "windows",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "Building llama-extension ($Platform / $Target)..." -ForegroundColor Cyan

# ── Optional clean ─────────────────────────────────────────────────────────────
if ($Clean) {
    $ObjDir = Join-Path $Root "build\obj"
    if (Test-Path $ObjDir) {
        Write-Host "Cleaning build artifacts: $ObjDir" -ForegroundColor Yellow
        Remove-Item $ObjDir -Recurse -Force
    }
}

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    Write-Error "vswhere not found. Is Visual Studio Build Tools installed?"
}

$VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsPath) {
    Write-Error "No Visual Studio installation with C++ tools found."
}

$VsDevCmd = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $VsDevCmd)) {
    Write-Error "vcvars64.bat not found at: $VsDevCmd"
}

$SconsCmd = "scons platform=$Platform target=$Target arch=x86_64 --implicit-cache"
$Command = "`"$VsDevCmd`" && cd /d `"$Root`" && $SconsCmd"

Write-Host "Invoking: $Command" -ForegroundColor DarkGray
cmd /c $Command

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed with exit code $LASTEXITCODE"
} else {
    Write-Host "Build succeeded." -ForegroundColor Green
}