"""CHANGED variant tests for Save/reopen/recovery (ladder row 9).

CORRECT: test_recovery.py::test_a_crashed_session_is_recovered
WRONG: test_recovery.py::test_a_normal_close_leaves_nothing
CHANGED: Different model state, different job types, different checkpoint content
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from elmerstudio.core import recovery

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(code: str, env):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)


@pytest.fixture
def rec_root(monkeypatch):
    d = tempfile.mkdtemp(prefix="es_rec_changed_")
    monkeypatch.setenv("ELMERSTUDIO_RECOVERY_DIR", d)
    return d


# CHANGED: Different model (structural) with different edits
def test_recovery_changed_structural_model(rec_root):
    """CHANGED: Structural model with different physics and edits."""
    pytest.importorskip("PySide6")

    CRASH_STRUCTURAL = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {root!r})
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from elmerstudio.core import builder as B, project
        from elmerstudio.ui.main_window import MainWindow
        w = MainWindow(None, show_new=False)
        w.no_modal = True
        m = B.new_model("3D", ["solid"], "stationary")
        m.create("geom.block", m.component.child("geometry"), index=0, w="0.1", d="0.05", h="0.02")
        B.add_material(m, "Structural steel", all_domains=True)
        w._set_model(m, {path!r}, project.new_workdir())
        w.rename(m.component.child("physics.solid"), "Structural load changed")
        w.checkpoint()
        w.recovery.note_job("Computing Structural Study")
        print("CHECKPOINT", w.recovery.dir, flush=True)
        os._exit({code})
    """)

    path = os.path.join(tempfile.mkdtemp(prefix="es_rec_"), "structural.esm")
    code = CRASH_STRUCTURAL.format(root=ROOT, path=path, code=3)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ELMERSTUDIO_RECOVERY_DIR=rec_root)
    r = _run(code, env)
    assert r.returncode == 3 and "CHECKPOINT" in r.stdout, r.stdout + r.stderr

    found = recovery.orphans()
    assert len(found) == 1 and found[0]["file_path"] == path and found[0]["job"]["title"] == "Computing Structural Study"

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    try:
        assert w.recover(found[0])
        assert "Structural load changed" in w.model.component.child("physics.solid").label
        assert w.dirty and w.file_path == path
        text = w.messages.view.toPlainText()
        assert "Recovered the unsaved model" in text and "'Computing Structural Study' was running" in text
        assert recovery.orphans() == []
        assert w.recovery.has_checkpoint()
        assert w.file_save_as(path) and not w.recovery.has_checkpoint()
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()
    assert recovery.orphans() == []


# CHANGED: Time-dependent model with running time-dependent job
def test_recovery_changed_time_dependent_job(rec_root):
    """CHANGED: Time-dependent model with a time-dependent job."""
    pytest.importorskip("PySide6")

    CRASH_TD = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {root!r})
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from elmerstudio.core import builder as B, project
        from elmerstudio.ui.main_window import MainWindow
        w = MainWindow(None, show_new=False)
        w.no_modal = True
        m = B.new_model("2D", ["tds"], "time")
        m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="0.1")
        w._set_model(m, {path!r}, project.new_workdir())
        w.rename(m.component.child("physics.tds"), "Transient diffusion edited")
        w.checkpoint()
        w.recovery.note_job("Solving Time Dependent")
        print("CHECKPOINT", w.recovery.dir, flush=True)
        os._exit({code})
    """)

    path = os.path.join(tempfile.mkdtemp(prefix="es_rec_"), "transient.esm")
    code = CRASH_TD.format(root=ROOT, path=path, code=3)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ELMERSTUDIO_RECOVERY_DIR=rec_root)
    r = _run(code, env)
    assert r.returncode == 3 and "CHECKPOINT" in r.stdout, r.stdout + r.stderr

    found = recovery.orphans()
    assert len(found) == 1 and found[0]["job"]["title"] == "Solving Time Dependent"

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    try:
        assert w.recover(found[0])
        assert "Transient diffusion edited" in w.model.component.child("physics.tds").label
        assert w.dirty and w.file_path == path
        text = w.messages.view.toPlainText()
        assert "'Solving Time Dependent' was running" in text
        assert recovery.orphans() == []
        assert w.file_save_as(path) and not w.recovery.has_checkpoint()
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()
    assert recovery.orphans() == []


# CHANGED: Model with assembly pair and multiphysics
def test_recovery_changed_assembly_multiphysics(rec_root):
    """CHANGED: Model with assembly and multiphysics."""
    pytest.importorskip("PySide6")

    CRASH_ASSEMBLY = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {root!r})
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from elmerstudio.core import builder as B, project
        from elmerstudio.ui.main_window import MainWindow
        w = MainWindow(None, show_new=False)
        w.no_modal = True
        m = B.new_model("2D", ["ht", "spf"], "stationary")
        g = m.component.child("geometry")
        m.create("geom.rectangle", g, index=0, w="0.5", h="0.1")
        m.create("geom.rectangle", g, index=1, w="0.5", h="0.1", x="0.5")
        g.child("geom.finalize").props["action"] = "none"
        B.add_material(m, "Copper", all_domains=True)
        w._set_model(m, {path!r}, project.new_workdir())
        w.rename(m.component.child("physics.ht"), "Assembly thermal edited")
        w.checkpoint()
        w.recovery.note_job("Computing Multiphysics Assembly")
        print("CHECKPOINT", w.recovery.dir, flush=True)
        os._exit({code})
    """)

    path = os.path.join(tempfile.mkdtemp(prefix="es_rec_"), "assembly.mp.mph")
    code = CRASH_ASSEMBLY.format(root=ROOT, path=path, code=3)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ELMERSTUDIO_RECOVERY_DIR=rec_root)
    r = _run(code, env)
    assert r.returncode == 3 and "CHECKPOINT" in r.stdout, r.stdout + r.stderr

    found = recovery.orphans()
    assert len(found) == 1 and found[0]["job"]["title"] == "Computing Multiphysics Assembly"

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    try:
        assert w.recover(found[0])
        assert "Assembly thermal edited" in w.model.component.child("physics.ht").label
        assert w.dirty and w.file_path == path
        text = w.messages.view.toPlainText()
        assert "'Computing Multiphysics Assembly' was running" in text
        assert recovery.orphans() == []
        assert w.file_save_as(path) and not w.recovery.has_checkpoint()
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()
    assert recovery.orphans() == []


# CHANGED: Model with multiple geometry objects and custom parameters
def test_recovery_changed_complex_geometry(rec_root):
    """CHANGED: Complex geometry with parameters and multiple objects."""
    pytest.importorskip("PySide6")

    CRASH_COMPLEX = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {root!r})
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from elmerstudio.core import builder as B, project
        from elmerstudio.ui.main_window import MainWindow
        w = MainWindow(None, show_new=False)
        w.no_modal = True
        m = B.new_model("3D", ["ht"], "stationary")
        m.global_defs.child("params").props["table"] = [
            {{\"name\": \"L\", \"expr\": \"50[mm]\", \"descr\": \"Length\"}},
            {{\"name\": \"W\", \"expr\": \"20[mm]\", \"descr\": \"Width\"}},
            {{\"name\": \"H\", \"expr\": \"10[mm]\", \"descr\": \"Height\"}},
        ]
        m.create("geom.block", m.component.child("geometry"), index=0, w="L", d="W", h="H")
        m.create("geom.cylinder", m.component.child("geometry"), index=1, r="W/4", h="H", x="L/2", y="W/2", z="0")
        B.add_material(m, "Copper", all_domains=True)
        w._set_model(m, {path!r}, project.new_workdir())
        w.rename(m.component.child("physics.ht"), "Complex busbar edited")
        m.global_defs.child("params").props["table"][0][\"expr\"] = \"60[mm]\"
        w.checkpoint()
        w.recovery.note_job("Computing Complex Model")
        print("CHECKPOINT", w.recovery.dir, flush=True)
        os._exit({code})
    """)

    path = os.path.join(tempfile.mkdtemp(prefix="es_rec_"), "complex.esm")
    code = CRASH_COMPLEX.format(root=ROOT, path=path, code=3)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ELMERSTUDIO_RECOVERY_DIR=rec_root)
    r = _run(code, env)
    assert r.returncode == 3 and "CHECKPOINT" in r.stdout, r.stdout + r.stderr

    found = recovery.orphans()
    assert len(found) == 1

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    try:
        assert w.recover(found[0])
        assert "Complex busbar edited" in w.model.component.child("physics.ht").label
        assert w.dirty and w.file_path == path
        text = w.messages.view.toPlainText()
        assert "'Computing Complex Model' was running" in text
        assert recovery.orphans() == []
        assert w.file_save_as(path) and not w.recovery.has_checkpoint()
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()
    assert recovery.orphans() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])