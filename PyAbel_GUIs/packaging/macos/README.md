# PyAbel Studio for macOS

`build_macos_app.sh` builds an Apple-silicon standalone application containing
the Image Generator, the Master analysis GUI, PyAbel, the compiled Cython
direct-transform backend, NumPy, SciPy, Matplotlib, Pillow, Tcl/Tk, and the
Python runtime.

Build from the repository root:

```bash
./PyAbel_GUIs/packaging/macos/build_macos_app.sh
```

The finished application is written to `dist/PyAbel Studio.app`, and the
canonical signed distributable is `dist/PyAbel-Studio-macOS-arm64.zip`. The
build performs a numerical Cython-versus-Python backend comparison, executes
all 11 bundled inversion methods, constructs both GUI workspaces and exercises
their bridge, radius selection, and 3D dot-cloud export as glTF at the default
sphere size,
and then reruns every
of those checks through a fresh extraction of the ZIP archive.

This build targets Apple silicon (`arm64`). A universal or Intel build requires
matching x86_64 Python and numerical-library binaries.

## Sharing the application

macOS refuses to open a downloaded application unless it carries a Developer ID
signature and an Apple notarization ticket. The default build is ad-hoc signed,
which runs on the machine that built it but is rejected by Gatekeeper anywhere
else. The build prints its own `spctl --assess` verdict at the end, so you can
always tell which of the two cases you have.

### Transfer methods that avoid the problem entirely

The quarantine flag is attached by whichever application receives the file. A
copy made with `scp`, `rsync`, a USB drive, or a local file share arrives
without it and opens normally even when ad-hoc signed. Browsers, Mail,
Messages, Slack, AirDrop, and iCloud Drive all attach it.

### What a recipient sees without notarization

The first launch is blocked. On macOS 15 and newer the Control-click shortcut no
longer bypasses this, so the recipient opens System Settings, goes to Privacy &
Security, and presses "Open Anyway" beside the blocked application. One command
does the same thing:

```bash
xattr -dr com.apple.quarantine "/Applications/PyAbel Studio.app"
```

### Producing a build that opens with no warning

This requires an Apple Developer Program membership, which is a paid annual
subscription. There is no alternative: Apple issues notarization tickets only to
members. With a membership:

1. Create a **Developer ID Application** certificate in the Apple Developer
   account and install it in the login keychain. Confirm it is visible with
   `security find-identity -v -p codesigning`.
2. Store notarization credentials once, using an app-specific password:

   ```bash
   xcrun notarytool store-credentials pyabel-notary \
       --apple-id you@example.com --team-id TEAMID \
       --password APP_SPECIFIC_PASSWORD
   ```

3. Build with both variables set:

   ```bash
   PYABEL_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
   PYABEL_NOTARY_PROFILE=pyabel-notary \
       ./PyAbel_GUIs/packaging/macos/build_macos_app.sh
   ```

The script signs every bundled binary from the inside out under the hardened
runtime, applies `PyAbelStudio.entitlements`, submits the archive to Apple,
staples the returned ticket onto the application, and rebuilds the archive
around the stapled bundle. Nothing else changes, so an ad-hoc build and a
notarized build are byte-for-byte the same application with different
signatures.

The hardened runtime is enabled for ad-hoc builds too, so the local build
exercises the same restrictions that notarization will impose. The two
entitlements it grants are the ones a frozen Python application needs: CPython
and NumPy generate executable code at run time, and PyInstaller loads its
collected libraries from inside the bundle.
