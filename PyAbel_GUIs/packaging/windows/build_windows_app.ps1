param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($env:OS -ne "Windows_NT") {
    throw "This build script must be run on native 64-bit Windows."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
$BuildEnv = Join-Path $RepoRoot ".build\windows-venv"
$OutputDir = Join-Path $RepoRoot "dist"
$OutputApp = Join-Path $OutputDir "PyAbel Studio-Windows-x64"
$OutputZip = Join-Path $OutputDir "PyAbel-Studio-Windows-x64.zip"
$StageDir = Join-Path ([System.IO.Path]::GetTempPath()) (
    "pyabel-studio-windows-" + [System.Guid]::NewGuid().ToString("N")
)
$Checks = "--self-test", "--transform-smoke-test", "--gui-smoke-test"

function Invoke-PackagedCheck {
    param([string]$Executable, [string]$Check)

    # The application is a windowed program, and PowerShell does not wait for
    # those to exit. Start it explicitly and relay its output from files.
    $Log = Join-Path $StageDir ("check" + $Check)
    $Process = Start-Process -FilePath $Executable -ArgumentList $Check `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError "$Log.err"
    foreach ($Stream in "$Log.out", "$Log.err") {
        if (Test-Path $Stream) {
            Get-Content $Stream | Write-Host
        }
    }
    if ($Process.ExitCode -ne 0) {
        throw "The packaged application failed $Check (exit code $($Process.ExitCode))."
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot ".build") | Out-Null
if (-not (Test-Path (Join-Path $BuildEnv "Scripts\python.exe"))) {
    if ($Python -eq "py") {
        & py -3.11-64 -m venv $BuildEnv
    } else {
        & $Python -m venv $BuildEnv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Windows build environment."
    }
}

$BuildPython = Join-Path $BuildEnv "Scripts\python.exe"
& $BuildPython -c (
    "import struct, sys; " +
    "assert sys.version_info[:2] == (3, 11), sys.version; " +
    "assert struct.calcsize('P') == 8, 'A 64-bit Python is required'"
)
if ($LASTEXITCODE -ne 0) {
    throw "The build environment must use 64-bit Python 3.11."
}

& $BuildPython -m pip install --disable-pip-version-check `
    --requirement (Join-Path $ScriptDir "requirements-build.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Push-Location $RepoRoot
try {
    Get-ChildItem (Join-Path $RepoRoot "abel\lib") -Filter "direct*.pyd" `
        -ErrorAction SilentlyContinue | Remove-Item -Force
    & $BuildPython setup.py build_ext --inplace --force
    if ($LASTEXITCODE -ne 0) {
        throw "The Cython direct backend failed to compile."
    }

    & $BuildPython -c (
        "from PyAbel_GUIs.PyAbel_Application import verify_cython_backend; " +
        "ok, detail = verify_cython_backend(); print(detail); " +
        "assert ok, detail"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "The compiled Cython backend did not pass validation."
    }

    $env:PYINSTALLER_CONFIG_DIR = Join-Path $RepoRoot ".build\pyinstaller-windows"
    $env:MPLCONFIGDIR = Join-Path $RepoRoot ".build\matplotlib-windows"
    $env:XDG_CACHE_HOME = Join-Path $RepoRoot ".build\xdg-windows"
    $env:PYABEL_BUILD_SITE_PACKAGES = (& $BuildPython -c (
        "import sysconfig; print(sysconfig.get_path('purelib'))"
    )).Trim()

    New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
    & $BuildPython -m PyInstaller --noconfirm --clean `
        --distpath $StageDir `
        (Join-Path $ScriptDir "PyAbelStudio.spec")
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }

    $StageApp = Join-Path $StageDir "PyAbel Studio"
    $Executable = Join-Path $StageApp "PyAbel Studio.exe"
    if (-not (Test-Path $Executable)) {
        throw "PyInstaller did not produce the expected Windows executable."
    }

    foreach ($Check in $Checks) {
        Invoke-PackagedCheck $Executable $Check
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    if (Test-Path $OutputApp) {
        Remove-Item -Recurse -Force $OutputApp
    }
    if (Test-Path $OutputZip) {
        Remove-Item -Force $OutputZip
    }
    Copy-Item -Recurse $StageApp $OutputApp
    Compress-Archive -Path $StageApp -DestinationPath $OutputZip -CompressionLevel Optimal

    # Rerun every check from a fresh extraction of the distributable.
    $VerifyDir = Join-Path $StageDir "verify"
    Expand-Archive -Path $OutputZip -DestinationPath $VerifyDir
    $VerifiedExecutable = Join-Path $VerifyDir "PyAbel Studio\PyAbel Studio.exe"
    foreach ($Check in $Checks) {
        Invoke-PackagedCheck $VerifiedExecutable $Check
    }

    Write-Host "Built application: $OutputApp"
    Write-Host "Verified distributable: $OutputZip"
} finally {
    Pop-Location
    if (Test-Path $StageDir) {
        Remove-Item -Recurse -Force $StageDir
    }
}
