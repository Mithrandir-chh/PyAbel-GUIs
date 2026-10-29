#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_ENV="$REPO_ROOT/.build/macos-venv"
BASE_PYTHON="${PYABEL_BUILD_PYTHON:-/opt/anaconda3/bin/python}"
OUTPUT_DIR="$REPO_ROOT/dist"
OUTPUT_APP="$OUTPUT_DIR/PyAbel Studio.app"
OUTPUT_ZIP="$OUTPUT_DIR/PyAbel-Studio-macOS-arm64.zip"
STAGE_DIR="$(mktemp -d /private/tmp/pyabel-studio-build.XXXXXX)"
APP_PATH="$STAGE_DIR/PyAbel Studio.app"
ENTITLEMENTS="$SCRIPT_DIR/PyAbelStudio.entitlements"
# Set PYABEL_SIGN_IDENTITY to a "Developer ID Application" identity to produce a
# build that opens without a Gatekeeper warning on other Macs. The default
# ad-hoc signature is enough to run locally but is rejected by Gatekeeper when
# the app has been downloaded. Set PYABEL_NOTARY_PROFILE to a notarytool
# keychain profile to also notarize and staple the result.
SIGN_IDENTITY="${PYABEL_SIGN_IDENTITY:--}"
NOTARY_PROFILE="${PYABEL_NOTARY_PROFILE:-}"
trap 'rm -rf -- "$STAGE_DIR"' EXIT
export PYINSTALLER_CONFIG_DIR="$REPO_ROOT/.build/pyinstaller-cache"
export MPLCONFIGDIR="$REPO_ROOT/.build/matplotlib-cache"
export XDG_CACHE_HOME="$REPO_ROOT/.build/xdg-cache"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This build script must be run on macOS." >&2
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "This first macOS build targets Apple silicon (arm64)." >&2
    exit 1
fi
if [[ ! -x "$BASE_PYTHON" ]]; then
    echo "Python was not found at $BASE_PYTHON" >&2
    echo "Set PYABEL_BUILD_PYTHON to a Python 3.11 executable." >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/.build"
if [[ ! -x "$BUILD_ENV/bin/python" ]]; then
    "$BASE_PYTHON" -m venv "$BUILD_ENV"
fi

if grep -q '^include-system-site-packages = true' "$BUILD_ENV/pyvenv.cfg"; then
    echo "The packaging environment must not expose global site-packages." >&2
    echo "Recreate $BUILD_ENV without --system-site-packages." >&2
    exit 1
fi

"$BUILD_ENV/bin/python" -m pip install \
    --requirement "$SCRIPT_DIR/requirements-build.txt"

cd "$REPO_ROOT"
BUILD_TOOLS_SITE="$($BUILD_ENV/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
PYTHONPATH="$BUILD_TOOLS_SITE" "$BASE_PYTHON" setup.py build_ext --inplace --force
PYTHONPATH="$BUILD_TOOLS_SITE" "$BASE_PYTHON" - <<'PY'
from PyAbel_GUIs.PyAbel_Application import verify_cython_backend

ok, detail = verify_cython_backend()
print(detail)
if not ok:
    raise SystemExit(1)
PY

PYABEL_BUILD_SITE_PACKAGES="$($BASE_PYTHON -c 'import site; print(site.getsitepackages()[0])')" \
"$BUILD_ENV/bin/python" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$STAGE_DIR" \
    "$SCRIPT_DIR/PyAbelStudio.spec"

EXECUTABLE="$APP_PATH/Contents/MacOS/PyAbel Studio"
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "PyInstaller did not produce the expected application executable." >&2
    exit 1
fi

"$EXECUTABLE" --self-test
"$EXECUTABLE" --transform-smoke-test
"$EXECUTABLE" --gui-smoke-test

# Sign nested code from the inside out. Apple discourages "--deep", and
# notarization is more reliable when every Mach-O file carries its own
# signature. The hardened runtime is mandatory for notarization, so it is
# applied even to ad-hoc builds to keep the two paths identical.
nested_flags=(--force --options runtime --sign "$SIGN_IDENTITY")
bundle_flags=(--force --options runtime
              --entitlements "$ENTITLEMENTS" --sign "$SIGN_IDENTITY")
if [[ "$SIGN_IDENTITY" != "-" ]]; then
    nested_flags=(--timestamp "${nested_flags[@]}")
    bundle_flags=(--timestamp "${bundle_flags[@]}")
fi

xattr -cr "$APP_PATH"
while IFS= read -r -d '' binary; do
    codesign "${nested_flags[@]}" "$binary"
done < <(find "$APP_PATH" -type f \( -name "*.dylib" -o -name "*.so" \) -print0)
codesign "${bundle_flags[@]}" "$APP_PATH"
codesign --verify --deep --strict "$APP_PATH"

mkdir -p "$OUTPUT_DIR"
package_app() {
    ditto --norsrc -c -k --keepParent "$APP_PATH" "$STAGE_DIR/package.zip"
    mv -f "$STAGE_DIR/package.zip" "$OUTPUT_ZIP"
}
package_app

if [[ -n "$NOTARY_PROFILE" ]]; then
    # Notarization runs against the archive, but the resulting ticket is
    # stapled to the application, so the archive is rebuilt afterwards.
    xcrun notarytool submit "$OUTPUT_ZIP" \
        --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$APP_PATH"
    xcrun stapler validate "$APP_PATH"
    package_app
fi

# Keep an unpacked copy for convenient local launching. The repository lives
# under Desktop's file-provider domain, which may attach Finder metadata after
# the verified app is copied; the ZIP above remains the canonical signed build.
if [[ -e "$OUTPUT_APP" ]]; then
    rm -rf -- "$OUTPUT_APP"
fi
ditto --norsrc "$APP_PATH" "$OUTPUT_APP"
xattr -cr "$OUTPUT_APP" || true
"$OUTPUT_APP/Contents/MacOS/PyAbel Studio" --self-test

VERIFY_DIR="$STAGE_DIR/verify"
mkdir -p "$VERIFY_DIR"
ditto -x -k "$OUTPUT_ZIP" "$VERIFY_DIR"
codesign --verify --deep --strict "$VERIFY_DIR/PyAbel Studio.app"
"$VERIFY_DIR/PyAbel Studio.app/Contents/MacOS/PyAbel Studio" --self-test
"$VERIFY_DIR/PyAbel Studio.app/Contents/MacOS/PyAbel Studio" --transform-smoke-test
"$VERIFY_DIR/PyAbel Studio.app/Contents/MacOS/PyAbel Studio" --gui-smoke-test

echo
echo "Built application: $OUTPUT_APP"
echo "Verified distributable: $OUTPUT_ZIP"
if spctl --assess --type exec -vv "$VERIFY_DIR/PyAbel Studio.app" 2>&1; then
    echo "Gatekeeper: accepted. Recipients can open this build normally."
else
    echo
    echo "Gatekeeper: REJECTED. This build is ad-hoc signed, so a Mac that"
    echo "downloads it will refuse to open it until the recipient approves it"
    echo "once under System Settings > Privacy & Security, or removes the"
    echo "quarantine flag with:"
    echo "  xattr -dr com.apple.quarantine '/Applications/PyAbel Studio.app'"
    echo "Rebuild with PYABEL_SIGN_IDENTITY and PYABEL_NOTARY_PROFILE set to"
    echo "produce a build that opens with no warning at all."
fi
