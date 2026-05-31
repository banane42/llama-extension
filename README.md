# llama-extension

A Godot 4 GDExtension that exposes llama.cpp inference to GDScript via a `LlamaNode` class.

## Requirements

- Windows 10/11
- [Visual Studio Build Tools 2022](https://aka.ms/vs/17/release/vs_BuildTools.exe) with the **C++ build tools** workload
- [CMake](https://cmake.org/download/) — enable "Add to PATH" during install
- [Python 3](https://www.python.org/downloads/) with SCons: `pip install scons`
- Git

## First time setup

Clone the repo then run the setup script. This initializes submodules and builds
both godot-cpp and llama.cpp:

```powershell
git clone <your-repo-url>
cd llama-extension
.\scripts\setup.ps1
```

## Building the extension

Run this after making any changes to the C++ source:

```powershell
.\scripts\build.ps1
```

To build a release version instead of debug:

```powershell
.\scripts\build.ps1 -Target template_release
```

## Deploying to a Godot project

Pass the path to your game's `project.godot` file. The script will copy the
extension DLL and all llama.cpp dependencies into a `bin/` folder inside your
project:

```powershell
.\scripts\deploy.ps1 -ProjectFile "D:\Godot\my-game\project.godot"
```

To deploy a release build:

```powershell
.\scripts\deploy.ps1 -ProjectFile "D:\Godot\my-game\project.godot" -Target template_release
```

To deploy into a different folder than the default `bin/`:

```powershell
.\scripts\deploy.ps1 -ProjectFile "D:\Godot\my-game\project.godot" -BinDir "extensions\llama"
```

## Typical workflow

```powershell
# Edit source files in src/, then:
.\scripts\build.ps1
.\scripts\deploy.ps1 -ProjectFile "D:\Godot\my-game\project.godot"
# Restart the Godot editor to reload the extension
```