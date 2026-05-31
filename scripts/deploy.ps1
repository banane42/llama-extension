param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectFile,

    [string]$Target = "template_debug",
    [string]$Platform = "windows",
    [string]$BinDir = "bin",

    # Pass -Build to compile before deploying (calls build.ps1 with matching args).
    [switch]$Build,
    [switch]$Clean   # forwarded to build.ps1 when -Build is set
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

# ── Validate project.godot ────────────────────────────────────────────────────
$ProjectFile = Resolve-Path $ProjectFile -ErrorAction SilentlyContinue
if (-not $ProjectFile) {
    Write-Error "project.godot not found at the path provided."
    return
}
if ((Split-Path $ProjectFile -Leaf) -ne "project.godot") {
    Write-Error "Provided path does not point to a project.godot file."
    return
}

# ── Optional build step ────────────────────────────────────────────────────────
if ($Build) {
    $BuildScript = Join-Path $PSScriptRoot "build.ps1"
    $BuildArgs = @{ Target = $Target; Platform = $Platform }
    if ($Clean) { $BuildArgs["Clean"] = $true }
    Write-Host "Running build.ps1 first..." -ForegroundColor Cyan
    & $BuildScript @BuildArgs
}

$ProjectDir = Split-Path $ProjectFile -Parent
$GameBin    = Join-Path $ProjectDir $BinDir

New-Item -ItemType Directory -Force -Path $GameBin | Out-Null

Write-Host "Deploying to: $GameBin" -ForegroundColor Cyan

# ── Extension DLL ─────────────────────────────────────────────────────────────
$ExtDll = "$Root\demo\bin\libllama_ext.windows.$Target.x86_64.dll"
if (-not (Test-Path $ExtDll)) {
    Write-Error "Extension DLL not found: $ExtDll`nRun build.ps1 first."
}
Copy-Item $ExtDll $GameBin -Force
Write-Host "  Copied: $(Split-Path $ExtDll -Leaf)" -ForegroundColor DarkGray

# ── llama.cpp DLLs ────────────────────────────────────────────────────────────
$LlamaBinDir = "$Root\llama.cpp\build\bin\Release"
$DllsToCopy  = @("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll")

foreach ($dll in $DllsToCopy) {
    $src = Join-Path $LlamaBinDir $dll
    if (Test-Path $src) {
        Copy-Item $src $GameBin -Force
        Write-Host "  Copied: $dll" -ForegroundColor DarkGray
    } else {
        Write-Warning "  Not found (skipping): $dll"
    }
}

# ── .gdextension descriptor ───────────────────────────────────────────────────
$GdExt = "$Root\demo\bin\llama_ext.gdextension"
if (Test-Path $GdExt) {
    Copy-Item $GdExt $GameBin -Force
    Write-Host "  Copied: llama_ext.gdextension" -ForegroundColor DarkGray
} else {
    Write-Warning "  .gdextension not found at $GdExt - skipping"
}

Write-Host "Deploy complete." -ForegroundColor Green