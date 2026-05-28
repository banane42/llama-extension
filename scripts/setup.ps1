$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "Initializing submodules..." -ForegroundColor Cyan
git -C $Root submodule update --init --recursive

Write-Host "Building godot-cpp..." -ForegroundColor Cyan
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VsPath  = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$VsDevCmd = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"

$GodotCppDir = Join-Path $Root "godot-cpp"
$Command = "`"$VsDevCmd`" && cd /d `"$GodotCppDir`" && scons platform=windows target=template_debug arch=x86_64"
cmd /c $Command
if ($LASTEXITCODE -ne 0) { Write-Error "godot-cpp build failed." }

Write-Host "Building llama.cpp..." -ForegroundColor Cyan
$LlamaDir = Join-Path $Root "llama.cpp"
cmake -S $LlamaDir -B "$LlamaDir\build" -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$LlamaDir\build" --config Release -j $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE -ne 0) { Write-Error "llama.cpp build failed." }

Write-Host "Setup complete. You can now run build.ps1." -ForegroundColor Green