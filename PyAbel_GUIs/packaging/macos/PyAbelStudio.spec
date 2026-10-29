# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the standalone macOS PyAbel Studio app."""

import os
from pathlib import Path


PACKAGING_DIR = Path(SPEC).resolve().parents[1]
REPO_ROOT = PACKAGING_DIR.parents[1]
APP_VERSION = "0.9.1"
SYSTEM_SITE_PACKAGES = Path(
    os.environ.get(
        "PYABEL_BUILD_SITE_PACKAGES",
        "/opt/anaconda3/lib/python3.11/site-packages",
    )
)

datas = [
    (
        str(REPO_ROOT / "PyAbel_GUIs" / "Master_GUI_2.0.py"),
        "PyAbel_GUIs",
    ),
]
datas += [
    (str(path), "abel/tests/data")
    for path in (REPO_ROOT / "abel" / "tests" / "data").glob("*")
    if path.is_file()
]

direct_extensions = list((REPO_ROOT / "abel" / "lib").glob("direct*.so"))
if len(direct_extensions) != 1:
    raise RuntimeError(
        "Expected exactly one compiled abel.lib.direct extension; found "
        f"{direct_extensions}"
    )
binaries = [(str(direct_extensions[0]), "abel/lib")]

# Master_GUI_2.0.py is bundled as data and loaded by path, so PyInstaller
# cannot see its imports.
hiddenimports = [
    "PyAbel_GUIs.called_functions.anni",
    "PyAbel_GUIs.gui_core",
    "PyAbel_GUIs.scrollable_panel",
    "scipy.ndimage",
    "scipy.optimize",
    "scipy.signal",
]

a = Analysis(
    [str(PACKAGING_DIR / "PyAbel_Studio.py")],
    pathex=[str(REPO_ROOT), str(SYSTEM_SITE_PACKAGES)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PACKAGING_DIR / "macos" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "bcrypt",
        "chardet",
        "charset_normalizer",
        "cryptography",
        "docutils",
        "jupyter",
        "pkg_resources",
        "pytest",
        "requests",
        "setuptools",
        "sphinx",
        "urllib3",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PyAbel Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PyAbel Studio",
)

app = BUNDLE(
    coll,
    name="PyAbel Studio.app",
    icon=None,
    bundle_identifier="org.pyabel.studio",
    version=APP_VERSION,
    info_plist={
        "CFBundleDisplayName": "PyAbel Studio",
        "CFBundleShortVersionString": APP_VERSION,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
    },
)
