# PyAbel Studio for legacy 32-bit macOS

This is a native **Intel i386** builder for macOS releases that can still run
32-bit applications. It is intentionally separate from the modern Apple
silicon build.

## Required build machine

- Intel Mac or Intel macOS virtual machine
- macOS 10.14 Mojave or older (10.15 removed 32-bit execution)
- Xcode/Command Line Tools old enough to compile `-arch i386`
- Python 3.7's legacy macOS 64-bit/32-bit installer
- A working i386 compiler stack for NumPy, SciPy, and Matplotlib, including
  Fortran/BLAS requirements needed when those packages build from source

Python 3.7.3 was one of the official releases that supplied a universal
64-bit/32-bit installer for macOS 10.6 and later. Modern Python and scientific
Python wheels are 64-bit-only, so simply changing a PyInstaller flag on a new
Mac does not create a valid legacy application.

## Build

From the repository root on the legacy machine:

```bash
chmod +x PyAbel_GUIs/packaging/macos-legacy32/build_legacy32_app.sh
./PyAbel_GUIs/packaging/macos-legacy32/build_legacy32_app.sh
```

If Python is installed elsewhere:

```bash
PYABEL_LEGACY_PYTHON=/path/to/universal/python3 \
  ./PyAbel_GUIs/packaging/macos-legacy32/build_legacy32_app.sh
```

The script forces every Python command through `arch -i386`, compiles a custom
32-bit PyInstaller bootloader, builds the Cython backend as i386, scans every
Mach-O file in the finished bundle, and runs backend, transform, and GUI smoke
tests. It refuses to package if any native binary lacks an i386 slice.

Successful outputs are:

- `dist/PyAbel Studio-macOS-i386.app`
- `dist/PyAbel-Studio-macOS-i386.zip`

This target cannot be built or executed on Apple silicon or macOS 10.15 and
newer. It therefore cannot be validated by the current modern macOS GitHub
hosted runners; use the native legacy machine described above.
