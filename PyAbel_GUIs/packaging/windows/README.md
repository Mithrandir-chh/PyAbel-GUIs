# PyAbel Studio for Windows x64

This directory builds a standalone 64-bit Windows application containing both
PyAbel Studio GUIs, all transform methods, and the compiled Cython direct
backend.

## Native build

Install 64-bit Python 3.11 and Microsoft C++ Build Tools, open PowerShell in the
repository root, and run:

```powershell
powershell -ExecutionPolicy Bypass -File PyAbel_GUIs\packaging\windows\build_windows_app.ps1
```

The script creates an isolated environment, installs pinned dependencies
(the same library versions as the macOS build), compiles the Cython extension
with MSVC, and packages the program. It then runs the packaged backend,
transform, and GUI smoke tests — the last also checks that both control panels
scroll — and reruns all three from a fresh extraction of the ZIP archive. Its
outputs are:

- `dist\PyAbel Studio-Windows-x64\`
- `dist\PyAbel-Studio-Windows-x64.zip`

PyInstaller is not a cross-compiler. This recipe must run on Windows. The
`.github/workflows/studio-packages.yml` workflow runs it on a GitHub-hosted
Windows machine: start it from the repository's **Actions** tab ("Build
PyAbel Studio packages" → "Run workflow"), or with
`gh workflow run studio-packages.yml`. The ZIP is attached to the finished run
as the `PyAbel-Studio-Windows-x64` artifact.
