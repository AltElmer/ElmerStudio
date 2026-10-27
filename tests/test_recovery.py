"""Crash recovery: a session that dies with unsaved edits (a hard kill, not a close) leaves a checkpoint that the next
session finds and opens as an unsaved model, with the job that was running named; a session that closes normally
leaves nothing behind (the control)."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from elmerstudio.core import recovery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CRASH = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {root!r})
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from elmerstudio.core import builder as B, project
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    m = B.new_model("2D", ["ht"], "stationary")
    w._set_model(m, {path!r}, project.new_workdir())
    w.rename(m.component.child("physics.ht"), "Edited before the crash")
    w.checkpoint()
    w.recovery.note_job("Computing Study 1")
    print("CHECKPOINT", w.recovery.dir, flush=True)
    os._exit({code})                                 # no close, no clean-up: as a crash
""")


def _run(code: str, env):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)


@pytest.fixture
def rec_root(monkeypatch):
    d = tempfile.mkdtemp(prefix="es_recovery_")
    monkeypatch.setenv("ELMERSTUDIO_RECOVERY_DIR", d)
    return d


def test_a_live_session_is_not_an_orphan(rec_root):
    s = recovery.Session()
    s.checkpoint({"format": "elmerstudio-model", "root": {"kind": "root"}}, None, None, "x")
    assert s.has_checkpoint() and recovery.orphans() == []          # this process is alive
    info = json.load(open(os.path.join(s.dir, "info.json")))
    info["pid"], info["process_started"] = 999999, 1.0                 # a process that is gone
    with open(os.path.join(s.dir, "info.json"), "w") as f:
        json.dump(info, f)
    assert [o["dir"] for o in recovery.orphans()] == [s.dir]
    s.discard()
    assert recovery.orphans() == []


def test_a_crashed_session_is_recovered(rec_root):
    pytest.importorskip("PySide6")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ELMERSTUDIO_RECOVERY_DIR=rec_root)
    path = os.path.join(tempfile.mkdtemp(prefix="es_rec_"), "model.esm")
    r = _run(CRASH.format(root=ROOT, path=path, code=3), env)
    assert r.returncode == 3 and "CHECKPOINT" in r.stdout, r.stdout + r.stderr
    found = recovery.orphans()
    assert len(found) == 1 and found[0]["file_path"] == path and found[0]["job"]["title"] == "Computing Study 1"

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    try:
        assert w.recover(found[0])
        assert w.model.component.child("physics.ht").label == "Edited before the crash"
        assert w.dirty and w.file_path == path                        # unsaved, under its file name
        text = w.messages.view.toPlainText()
        assert "Recovered the unsaved model" in text and "'Computing Study 1' was running" in text
        assert recovery.orphans() == []                               # the old checkpoint is consumed ...
        assert w.recovery.has_checkpoint()                            # ... and this session holds the model now
        assert w.file_save_as(path) and not w.recovery.has_checkpoint()   # saving makes it unnecessary
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()
    assert recovery.orphans() == []


def test_a_normal_close_leaves_nothing(rec_root):
    """Control: the same edits, then a normal close (Discard), leave no recovery file."""
    pytest.importorskip("PySide6")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ELMERSTUDIO_RECOVERY_DIR=rec_root)
    code = CRASH.format(root=ROOT, path="", code=0).replace(
        "os._exit(0)", "w.dirty = False; w.close(); app.processEvents(); print('CLOSED', flush=True)")
    r = _run(code, env)
    assert "CLOSED" in r.stdout, r.stdout + r.stderr
    assert recovery.orphans() == []
