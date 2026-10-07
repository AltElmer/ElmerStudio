"""Build portable bundles and installers for the current platform.

    python packaging/build.py [--skip-pyinstaller] [--no-smoke]

Outputs in dist/:
  Windows  ElmerStudio-<v>-windows-x64-portable.zip, ElmerStudio-<v>-windows-x64-setup.exe (needs Inno Setup)
  macOS    ElmerStudio-<v>-macos-<arch>.dmg, ElmerStudio-<v>-macos-<arch>.zip
  Linux    ElmerStudio-<v>-linux-x86_64.tar.gz, ElmerStudio-<v>-linux-x86_64.AppImage (needs appimagetool)
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "packaging")
DIST = os.path.join(ROOT, "dist")
sys.path.insert(0, ROOT)
from elmerstudio import __version__ as VERSION  # noqa: E402


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def arch():
    m = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x86_64" if sys.platform.startswith("linux") else "x64",
            "arm64": "arm64", "aarch64": "aarch64"}.get(m, m)


# --------------------------------------------------------------------------- icons
def make_icons():
    """Render the app icon (from the SVG icon set) to PNG/ICO/ICNS with Qt; iconutil on macOS."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtCore import QByteArray, Qt
    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    from elmerstudio.ui.icons import SVG
    out = os.path.join(PKG, "icons")
    os.makedirs(out, exist_ok=True)
    r = QSvgRenderer(QByteArray(SVG["app"].encode()))

    def render(size):
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        r.render(p)
        p.end()
        return img
    render(512).save(os.path.join(out, "elmerstudio.png"))
    render(256).save(os.path.join(out, "elmerstudio.ico"), "ICO")
    if sys.platform == "darwin":
        iconset = os.path.join(out, "elmerstudio.iconset")
        os.makedirs(iconset, exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            render(s).save(os.path.join(iconset, f"icon_{s}x{s}.png"))
            render(2 * s).save(os.path.join(iconset, f"icon_{s}x{s}@2x.png"))
        run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(out, "elmerstudio.icns")])
        shutil.rmtree(iconset, ignore_errors=True)


# --------------------------------------------------------------------------- packaging
def bundle_dir():
    return os.path.join(DIST, "ElmerStudio.app" if sys.platform == "darwin" else "ElmerStudio")


def executable():
    if sys.platform == "win32":
        return os.path.join(DIST, "ElmerStudio", "ElmerStudio.exe")
    if sys.platform == "darwin":
        return os.path.join(DIST, "ElmerStudio.app", "Contents", "MacOS", "ElmerStudio")
    return os.path.join(DIST, "ElmerStudio", "ElmerStudio")


def zip_dir(src, dest, root_name):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _dirs, files in os.walk(src):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.join(root_name, os.path.relpath(full, src)))


def package_windows():
    out = []
    z = os.path.join(DIST, f"ElmerStudio-{VERSION}-windows-x64-portable.zip")
    zip_dir(bundle_dir(), z, "ElmerStudio")
    out.append(z)
    iscc = shutil.which("iscc") or next((p for p in (r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
                                                       r"C:\Program Files\Inno Setup 6\ISCC.exe") if os.path.exists(p)), None)
    if iscc:
        run([iscc, f"/DAppVersion={VERSION}", f"/DSourceDir={bundle_dir()}", f"/DOutputDir={DIST}", f"/DRepoDir={ROOT}",
             os.path.join(PKG, "windows", "installer.iss")])
        out.append(os.path.join(DIST, f"ElmerStudio-{VERSION}-windows-x64-setup.exe"))
    else:
        print("Inno Setup (iscc) not found: skipping the installer", flush=True)
    return out


def package_macos():
    a = arch()
    dmg = os.path.join(DIST, f"ElmerStudio-{VERSION}-macos-{a}.dmg")
    stage = os.path.join(DIST, "dmg")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    run(["ditto", bundle_dir(), os.path.join(stage, "ElmerStudio.app")])
    os.symlink("/Applications", os.path.join(stage, "Applications"))
    run(["hdiutil", "create", "-volname", "Elmer Studio", "-srcfolder", stage, "-ov", "-format", "UDZO", dmg])
    z = os.path.join(DIST, f"ElmerStudio-{VERSION}-macos-{a}.zip")
    run(["ditto", "-c", "-k", "--keepParent", bundle_dir(), z])
    return [dmg, z]


DESKTOP = """[Desktop Entry]
Type=Application
Name=Elmer Studio
Comment=Model builder for the Elmer FEM multiphysics solver
Exec=ElmerStudio %f
Icon=elmerstudio
Categories=Science;Engineering;Physics;
Terminal=false
"""


def package_linux():
    a = arch()
    out = []
    tgz = os.path.join(DIST, f"ElmerStudio-{VERSION}-linux-{a}.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(bundle_dir(), arcname="ElmerStudio")
    out.append(tgz)
    tool = shutil.which("appimagetool") or os.environ.get("APPIMAGETOOL")
    if tool:
        appdir = os.path.join(DIST, "ElmerStudio.AppDir")
        shutil.rmtree(appdir, ignore_errors=True)
        shutil.copytree(bundle_dir(), os.path.join(appdir, "usr", "lib", "elmerstudio"), symlinks=True)
        with open(os.path.join(appdir, "elmerstudio.desktop"), "w") as f:
            f.write(DESKTOP)
        shutil.copy(os.path.join(PKG, "icons", "elmerstudio.png"), os.path.join(appdir, "elmerstudio.png"))
        apprun = os.path.join(appdir, "AppRun")
        with open(apprun, "w") as f:
            f.write('#!/bin/sh\nHERE="$(dirname "$(readlink -f "$0")")"\n'
                    'exec "$HERE/usr/lib/elmerstudio/ElmerStudio" "$@"\n')
        os.chmod(apprun, 0o755)
        img = os.path.join(DIST, f"ElmerStudio-{VERSION}-linux-{a}.AppImage")
        # APPIMAGE_EXTRACT_AND_RUN=1 (set in CI) lets appimagetool run without FUSE
        env = dict(os.environ, ARCH=platform.machine())
        run([tool, appdir, img], env=env)
        out.append(img)
    else:
        print("appimagetool not found: skipping the AppImage", flush=True)
    return out


def smoke():
    """Start the frozen app on an example, build its geometry, quit; fail unless it reports OK."""
    res = os.path.join(DIST, "selftest.txt")
    if os.path.exists(res):
        os.remove(res)
    cmd = [executable(), "--example", "heat_sink", "--quit-after", "5", "--selftest-out", res, "--selftest-compute"]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, timeout=180)
    except subprocess.TimeoutExpired:
        print("frozen application did not exit (modal error dialog?)", flush=True)
    txt = open(res).read() if os.path.exists(res) else ""
    print("selftest:", txt.strip() or "(no output)", flush=True)
    if not txt.startswith("OK") or "geometry=built" not in txt:
        raise SystemExit("frozen application self-test failed")
    if "elmer=None" not in txt and "frames=1" not in txt:
        raise SystemExit("frozen application found Elmer but the solve/plot self-test failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pyinstaller", action="store_true")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    make_icons()
    if not args.skip_pyinstaller:
        run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", DIST,
             "--workpath", os.path.join(ROOT, "build"), os.path.join(PKG, "elmerstudio.spec")])
    if not args.no_smoke:
        smoke()
    if sys.platform == "win32":
        files = package_windows()
    elif sys.platform == "darwin":
        files = package_macos()
    else:
        files = package_linux()
    print("ARTIFACTS:")
    for f in files:
        print(f"  {f}  ({os.path.getsize(f) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
