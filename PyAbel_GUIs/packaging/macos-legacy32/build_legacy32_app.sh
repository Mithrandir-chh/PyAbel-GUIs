#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE_PYTHON="${PYABEL_LEGACY_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.7/bin/python3}"
BUILD_ENV="$REPO_ROOT/.build/macos-legacy32-venv"
SOURCE_ROOT="$REPO_ROOT/.build/macos-legacy32-source"
PYINSTALLER_SOURCE="$SOURCE_ROOT/pyinstaller-5.13.2"
OUTPUT_DIR="$REPO_ROOT/dist"
OUTPUT_APP="$OUTPUT_DIR/PyAbel Studio-macOS-i386.app"
OUTPUT_ZIP="$OUTPUT_DIR/PyAbel-Studio-macOS-i386.zip"
STAGE_DIR="$(mktemp -d /private/tmp/pyabel-studio-i386.XXXXXX)"
APP_PATH="$STAGE_DIR/PyAbel Studio.app"
trap 'rm -rf -- "$STAGE_DIR"' EXIT

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This build must run on a native Intel Mac." >&2
    exit 1
fi

DARWIN_MAJOR="$(uname -r | cut -d. -f1)"
if [[ "$DARWIN_MAJOR" -gt 18 ]]; then
    echo "macOS 10.15 and newer cannot execute 32-bit applications." >&2
    echo "Build on macOS 10.14 Mojave or an older Intel macOS release." >&2
    exit 1
fi
if [[ ! -x "$BASE_PYTHON" ]]; then
    echo "A universal 32/64-bit Python 3.7 installation was not found." >&2
    echo "Set PYABEL_LEGACY_PYTHON to its python3 executable." >&2
    exit 1
fi

arch -i386 "$BASE_PYTHON" -c \
    "import struct, sys; assert sys.version_info[:2] == (3, 7), sys.version; assert struct.calcsize('P') == 4"

mkdir -p "$REPO_ROOT/.build" "$SOURCE_ROOT"
if [[ ! -x "$BUILD_ENV/bin/python" ]]; then
    arch -i386 "$BASE_PYTHON" -m venv "$BUILD_ENV"
fi
PYTHON=(arch -i386 "$BUILD_ENV/bin/python")
"${PYTHON[@]}" -c \
    "import struct, sys; assert sys.version_info[:2] == (3, 7), sys.version; assert struct.calcsize('P') == 4"

"${PYTHON[@]}" -m pip install --disable-pip-version-check \
    --requirement "$SCRIPT_DIR/requirements-build.txt"

if [[ ! -d "$PYINSTALLER_SOURCE/PyInstaller" ]]; then
    DOWNLOAD_DIR="$SOURCE_ROOT/download"
    mkdir -p "$DOWNLOAD_DIR"
    "${PYTHON[@]}" -m pip download --disable-pip-version-check \
        --no-deps --no-binary=:all: --dest "$DOWNLOAD_DIR" \
        PyInstaller==5.13.2
    SOURCE_ARCHIVE="$(find "$DOWNLOAD_DIR" -maxdepth 1 -type f -name 'pyinstaller-5.13.2.tar.gz' -print -quit)"
    if [[ -z "$SOURCE_ARCHIVE" ]]; then
        echo "Could not download the pinned PyInstaller source archive." >&2
        exit 1
    fi
    tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE_ROOT"
fi

pushd "$PYINSTALLER_SOURCE/bootloader" >/dev/null
"${PYTHON[@]}" ./waf distclean || true
"${PYTHON[@]}" ./waf all --target-arch=32bit
popd >/dev/null

BOOTLOADER=""
while IFS= read -r CANDIDATE; do
    if file "$CANDIDATE" | grep -q 'i386'; then
        BOOTLOADER="$CANDIDATE"
        break
    fi
done < <(find "$PYINSTALLER_SOURCE/PyInstaller/bootloader" -type f -name runw -print)
if [[ -z "$BOOTLOADER" ]]; then
    echo "PyInstaller did not produce an i386 windowed bootloader." >&2
    echo "Use an Xcode release whose compiler and macOS SDK still support i386." >&2
    exit 1
fi

cd "$REPO_ROOT"
export ARCHFLAGS="-arch i386"
export CFLAGS="${CFLAGS:-} -arch i386"
export CXXFLAGS="${CXXFLAGS:-} -arch i386"
export LDFLAGS="${LDFLAGS:-} -arch i386"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.9}"
export PYINSTALLER_CONFIG_DIR="$REPO_ROOT/.build/pyinstaller-legacy32"
export MPLCONFIGDIR="$REPO_ROOT/.build/matplotlib-legacy32"
export XDG_CACHE_HOME="$REPO_ROOT/.build/xdg-legacy32"

find "$REPO_ROOT/abel/lib" -maxdepth 1 -type f -name 'direct*.so' -delete
"${PYTHON[@]}" setup.py build_ext --inplace --force
DIRECT_EXTENSION="$(find "$REPO_ROOT/abel/lib" -maxdepth 1 -type f -name 'direct*.so' -print -quit)"
if [[ -z "$DIRECT_EXTENSION" ]] || ! file "$DIRECT_EXTENSION" | grep -q 'i386'; then
    echo "The Cython direct backend is not an i386 Mach-O binary." >&2
    exit 1
fi

"${PYTHON[@]}" - <<'PY'
import struct
import tkinter

import matplotlib
import numpy
import scipy
from PyAbel_GUIs.PyAbel_Application import verify_cython_backend

assert struct.calcsize("P") == 4
ok, detail = verify_cython_backend()
print(detail)
if not ok:
    raise SystemExit(1)
PY

PYABEL_BUILD_SITE_PACKAGES="$("${PYTHON[@]}" -c 'import site; print(site.getsitepackages()[0])')" \
PYTHONPATH="$PYINSTALLER_SOURCE" \
"${PYTHON[@]}" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$STAGE_DIR" \
    "$SCRIPT_DIR/PyAbelStudioLegacy32.spec"

EXECUTABLE="$APP_PATH/Contents/MacOS/PyAbel Studio"
if [[ ! -x "$EXECUTABLE" ]] || ! file "$EXECUTABLE" | grep -q 'i386'; then
    echo "PyInstaller did not produce the expected i386 application." >&2
    exit 1
fi

find "$APP_PATH" -type f -print0 | while IFS= read -r -d '' CANDIDATE; do
    DESCRIPTION="$(file "$CANDIDATE")"
    case "$DESCRIPTION" in
        *Mach-O*)
            if ! printf '%s\n' "$DESCRIPTION" | grep -q 'i386'; then
                echo "Non-i386 Mach-O file in bundle: $CANDIDATE" >&2
                exit 1
            fi
            ;;
    esac
done

arch -i386 "$EXECUTABLE" --self-test
arch -i386 "$EXECUTABLE" --transform-smoke-test
arch -i386 "$EXECUTABLE" --gui-smoke-test

xattr -cr "$APP_PATH"
codesign --force --deep --sign - "$APP_PATH"
codesign --verify --deep --strict "$APP_PATH"

mkdir -p "$OUTPUT_DIR"
ditto --norsrc -c -k --keepParent "$APP_PATH" "$STAGE_DIR/PyAbel-Studio-macOS-i386.zip"
mv -f "$STAGE_DIR/PyAbel-Studio-macOS-i386.zip" "$OUTPUT_ZIP"
if [[ -e "$OUTPUT_APP" ]]; then
    rm -rf -- "$OUTPUT_APP"
fi
ditto --norsrc "$APP_PATH" "$OUTPUT_APP"

echo "Built application: $OUTPUT_APP"
echo "Verified distributable: $OUTPUT_ZIP"
