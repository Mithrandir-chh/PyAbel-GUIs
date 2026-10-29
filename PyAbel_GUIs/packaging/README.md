# PyAbel Studio standalone packages

PyAbel Studio combines the Image Generator and the analysis GUI in one window.
To run it from a source checkout, use `python -m PyAbel_GUIs.PyAbel_Application`
from the repository root.

Run each builder from the repository root; outputs are written to `dist/`.

PyInstaller bundles are specific to the operating system and CPU architecture
of the Python interpreter that creates them. Each supported target therefore
has its own native, self-testing builder:

| Target | Builder | Output |
| --- | --- | --- |
| Apple-silicon macOS | `macos/build_macos_app.sh` | `PyAbel-Studio-macOS-arm64.zip` |
| Windows 10/11 x64 | `windows/build_windows_app.ps1` | `PyAbel-Studio-Windows-x64.zip` |
| Legacy macOS i386 (10.9–10.14) | `macos-legacy32/build_legacy32_app.sh` | `PyAbel-Studio-macOS-i386.zip` |

Every builder compiles and tests the Cython direct backend and runs all exposed
Abel transform methods. The GUI smoke test also checks the generator/analysis
bridge, that both control panels scroll, the coordinated result layout, radius
selection, and the 3D model export.

The Windows build is also available as the manually triggered
`.github/workflows/studio-packages.yml` GitHub Actions workflow. No current
hosted macOS runner can execute or validate i386 applications, so the legacy
package must be produced on a real or virtual pre-Catalina Intel Mac.
