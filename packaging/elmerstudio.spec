# PyInstaller spec: one-folder bundle (portable), plus a .app bundle on macOS.
# Build with:  python packaging/build.py
import os
import sys

import gmsh
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICONS = os.path.join(SPECPATH, "icons")
sys.path.insert(0, ROOT)
from elmerstudio import __version__  # noqa: E402

# gmsh.py looks for its shared library next to itself; in the bundle that is the bundle root
binaries = [(gmsh.libpath, ".")]
datas = collect_data_files("pyvista")
for pkg in ("pyvista", "pyvistaqt", "vtk", "gmsh", "matplotlib", "numpy", "PySide6"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=["vtkmodules.all", "vtkmodules.util.numpy_support", "vtkmodules.qt.QVTKRenderWindowInteractor",
                   "pyvistaqt", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg", "psutil"],
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "pytest", "jupyter", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = os.path.join(ICONS, "elmerstudio.ico" if sys.platform == "win32" else "elmerstudio.icns")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ElmerStudio", console=False,
          icon=icon if os.path.exists(icon) else None, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="ElmerStudio", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="ElmerStudio.app", icon=icon if os.path.exists(icon) else None,
                 bundle_identifier="org.altelmer.elmerstudio", version=__version__,
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": __version__,
                             "CFBundleDocumentTypes": [{"CFBundleTypeName": "Elmer Studio model",
                                                        "CFBundleTypeExtensions": ["esm"],
                                                        "CFBundleTypeRole": "Editor"}]})
