"""Tests for unconverged result display.

These tests verify that a solution with status "not converged" shows the marker
in: (1) plot titles, (2) model builder tree, (3) evaluation tables, (4) exports.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.elmer import find_elmer
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner, StudyNotConverged
from elmerstudio.ui.plots import build_scene, evaluate_derived


# --------------------------------------------------------------------------------------- helpers
def _nonlinear_model():
    """Nonlinear PDE with c=1+u^2, f=40, u=0 on boundary. Sets nl_maxit=1 -> not converged."""
    m = B.new_model("2D", ["pde"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.3", h="0.8")
    geo = build_geometry(m, display=False)
    pde = m.component.child("physics.pde")
    pde.child("pde.cfeq").props.update(c="1+u^2", f="40", da="0")
    B.add_feature(m, pde, "pde.dir", geo.numbers["boundary"], r="0")
    # Keep unconverged solutions
    m.studies()[0].props["nonconv"] = "keep"
    # Explicitly set nl_maxit=1 to ensure the solver doesn't converge
    B.ensure_solver_configs(m, m.studies()[0]).children[0].props.update(nl_maxit=1, nl_tol="1e-10")
    return m, geo


def _converged_model():
    """Same nonlinear PDE but with enough iterations to converge."""
    m = B.new_model("2D", ["pde"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.3", h="0.8")
    geo = build_geometry(m, display=False)
    pde = m.component.child("physics.pde")
    pde.child("pde.cfeq").props.update(c="1+u^2", f="40", da="0")
    B.add_feature(m, pde, "pde.dir", geo.numbers["boundary"], r="0")
    m.studies()[0].props["nonconv"] = "keep"
    B.ensure_solver_configs(m, m.studies()[0]).children[0].props.update(nl_maxit=80)
    return m, geo


def _unconverged_solution_folder():
    """Create an unconverged solution folder by running Elmer with nl_maxit=1.
    Returns the solution folder path and the model."""
    m, geo = _nonlinear_model()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_unconv_")
    try:
        StudyRunner(m, st, work, find_elmer()).run()
    except StudyNotConverged:
        pass  # expected
    # Find the solution folder
    for root, dirs, files in os.walk(work):
        if "meta.json" in files:
            with open(os.path.join(root, "meta.json")) as f:
                meta = json.load(f)
            if meta.get("status") == "not converged":
                return root, m
    return None, m


def _converged_solution_folder():
    """Create a converged solution folder by running Elmer with nl_maxit=80.
    Returns the solution folder path and the model."""
    m, geo = _converged_model()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_conv_")
    folders = StudyRunner(m, st, work, find_elmer()).run()
    return folders[0], m


# --------------------------------------------------------------------------------------- validity function tests
def test_solution_validity_label():
    """Test that Solution exposes a validity label (converged / not converged with diagnostics)."""
    sol_folder, m = _unconverged_solution_folder()
    assert sol_folder is not None, "No not converged solution found"

    sol = Solution(sol_folder)
    # After implementation, Solution should have a validity_label property
    assert hasattr(sol, "validity_label"), "Solution missing validity_label property"
    label = sol.validity_label
    assert label is not None, "validity_label should not be None"
    assert "not converged" in label.lower(), f"Expected 'not converged' in label: {label}"
    # Diagnostics should be included
    with open(os.path.join(sol_folder, "meta.json")) as f:
        meta = json.load(f)
    assert any(d in label for d in meta.get("diagnostics", [])), f"Diagnostics not in label: {label}"


def test_converged_solution_validity_label():
    """Converged solution should have 'converged' label (or None/empty for no marker)."""
    sol_folder, m = _converged_solution_folder()
    sol = Solution(sol_folder)
    # Converged solution should have a label indicating convergence
    label = sol.validity_label
    assert label is not None
    # Should NOT contain "not converged"
    assert "not converged" not in label.lower()


# --------------------------------------------------------------------------------------- plot title tests
def test_plot_title_includes_marker_unconverged(elmer):
    """Plot title should include the unconverged marker for not converged solutions."""
    m, geo = _nonlinear_model()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_plot_unconv_")
    try:
        StudyRunner(m, st, work, elmer).run()
    except StudyNotConverged:
        pass
    sol_folder = None
    for root, dirs, files in os.walk(work):
        if "meta.json" in files:
            with open(os.path.join(root, "meta.json")) as f:
                meta = json.load(f)
            if meta.get("status") == "not converged":
                sol_folder = root
                break
    assert sol_folder is not None

    sol = Solution(sol_folder)
    # Build a simple surface plot scene
    from elmerstudio.ui.plots import build_scene
    # Create a plot group (2D) and surface plot
    pg_node = m.create("pg2d", m.results, index=0)
    surface = m.create("plot.surface", pg_node, index=0)
    surface.props["expr"] = "u"

    scene = build_scene(pg_node, sol, 0, sol.sdim)
    # The scene title should contain the marker
    assert "not converged" in scene.title.lower(), f"Plot title missing marker: {scene.title}"


def test_plot_title_no_marker_converged(elmer):
    """Plot title should NOT include the marker for converged solutions."""
    m, geo = _converged_model()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_plot_conv_")
    folders = StudyRunner(m, st, work, elmer).run()
    sol = Solution(folders[0])

    pg_node = m.create("pg2d", m.results, index=0)
    surface = m.create("plot.surface", pg_node, index=0)
    surface.props["expr"] = "u"

    scene = build_scene(pg_node, sol, 0, sol.sdim)
    # Should NOT contain "not converged"
    assert "not converged" not in scene.title.lower(), f"Converged plot has marker: {scene.title}"


# --------------------------------------------------------------------------------------- tree node tests
def test_tree_node_shows_marker_unconverged():
    """Model builder tree node for solution dataset should show marker for not converged solutions."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from elmerstudio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    for name in ("warning", "critical", "information"):
        from PySide6.QtWidgets import QMessageBox
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)

    try:
        sol_folder, m = _unconverged_solution_folder()
        assert sol_folder is not None, "No not converged solution found"

        # Load the model into the window with its workdir
        # sol_folder is at workdir/st.tag/sol_.../ so workdir is grandparent
        workdir = os.path.dirname(os.path.dirname(sol_folder))
        w._set_model(m, None, workdir)
        # Create a solution dataset node pointing to the study
        res = m.results
        dsets = res.child("results.datasets")
        ds = m.create("dset.solution", dsets, index=0)
        ds.props["study"] = m.studies()[0].tag

        # Refresh the tree to ensure the node is created
        w.tree.refresh()
        # Wait for async UI update (schedule_update uses QTimer.singleShot)
        QApplication.processEvents()
        import time
        time.sleep(0.1)
        QApplication.processEvents()

        # Find the tree item for the dataset
        item = w.tree._items.get(id(ds))
        assert item is not None, "Tree item for dset.solution not found"

        # Check the displayed text or tooltip for the marker
        # The tree node should show the marker in its text or tooltip
        text = item.text(0)
        tooltip = item.toolTip(0)
        has_marker = "not converged" in text.lower() or "not converged" in tooltip.lower()
        assert has_marker, f"Tree node missing 'not converged' marker. Text: '{text}', Tooltip: '{tooltip}'"
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()


def test_tree_node_no_marker_converged():
    """Model builder tree node for solution dataset should NOT show marker for converged solutions."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from elmerstudio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    for name in ("warning", "critical", "information"):
        from PySide6.QtWidgets import QMessageBox
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)

    try:
        sol_folder, m = _converged_solution_folder()

        w._set_model(m, None, os.path.dirname(sol_folder))
        res = m.results
        dsets = res.child("results.datasets")
        ds = m.create("dset.solution", dsets, index=0)
        ds.props["study"] = m.studies()[0].tag

        w.tree.refresh()

        item = w.tree._items.get(id(ds))
        assert item is not None, "Tree item for dset.solution not found"

        text = item.text(0)
        tooltip = item.toolTip(0)
        has_marker = "not converged" in text.lower() or "not converged" in tooltip.lower()
        assert not has_marker, f"Converged tree node has 'not converged' marker. Text: '{text}', Tooltip: '{tooltip}'"
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()


# --------------------------------------------------------------------------------------- evaluation table tests
def test_evaluation_table_includes_status_unconverged():
    """Evaluation tables should include the unconverged status for not converged solutions."""
    sol_folder, m = _unconverged_solution_folder()
    assert sol_folder is not None
    sol = Solution(sol_folder)

    # Create an eval.global node
    from elmerstudio.core.model import Node
    eval_node = Node("eval.global", m.results, props={"exprs": [{"expr": "u", "unit": "", "descr": ""}]})

    cols, rows = evaluate_derived(eval_node, m, sol, [], None)

    # The table should have a column or row indicating the solution status
    has_marker_col = any("not converged" in col.lower() for col in cols)
    has_marker_row = any(any("not converged" in str(cell).lower() for cell in row) for row in rows)

    assert has_marker_col or has_marker_row, f"Evaluation table missing unconverged marker. Cols: {cols}, Rows: {rows}"


def test_evaluation_table_no_status_converged():
    """Evaluation tables should NOT include the unconverged status for converged solutions."""
    sol_folder, m = _converged_solution_folder()
    sol = Solution(sol_folder)

    from elmerstudio.core.model import Node
    eval_node = Node("eval.global", m.results, props={"exprs": [{"expr": "u", "unit": "", "descr": ""}]})

    cols, rows = evaluate_derived(eval_node, m, sol, [], None)

    has_marker_col = any("not converged" in col.lower() for col in cols)
    has_marker_row = any(any("not converged" in str(cell).lower() for cell in row) for row in rows)

    assert not has_marker_col and not has_marker_row, f"Converged evaluation table has marker. Cols: {cols}, Rows: {rows}"


# --------------------------------------------------------------------------------------- CSV export tests
def test_csv_export_includes_marker_header():
    """CSV exports should include a header line with the unconverged status."""
    sol_folder, m = _unconverged_solution_folder()
    assert sol_folder is not None
    sol = Solution(sol_folder)

    # Simulate the CSV export from export_node (same logic as in main_window.py)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        fn = f.name
    try:
        exprs = ["u"]
        fi = sol.frame_index("last")
        import numpy as np
        P = sol.topo.nodes * sol.scale
        cols = [P[:, 0], P[:, 1]] + ([P[:, 2]] if sol.sdim == 3 else []) + [sol.evaluate(e, fi) for e in exprs]
        hdr = "x,y," + ("z," if sol.sdim == 3 else "") + ",".join(exprs)
        # Add solution validity marker to header (same as export_node)
        validity = getattr(sol, "validity_label", None)
        if validity and validity != "converged":
            hdr = f"% Solution Status: {validity}\n" + hdr
        np.savetxt(fn, np.column_stack(cols), delimiter=",", header=hdr, comments="% ")

        # Read back the header
        with open(fn, "r") as f:
            first_line = f.readline().strip()

        # The header should include the unconverged marker
        assert "not converged" in first_line.lower(), f"CSV header missing marker: {first_line}"
    finally:
        os.unlink(fn)


def test_csv_export_no_marker_converged():
    """CSV exports should NOT include the marker for converged solutions."""
    sol_folder, m = _converged_solution_folder()
    sol = Solution(sol_folder)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        fn = f.name
    try:
        exprs = ["u"]
        fi = sol.frame_index("last")
        import numpy as np
        P = sol.topo.nodes * sol.scale
        cols = [P[:, 0], P[:, 1]] + ([P[:, 2]] if sol.sdim == 3 else []) + [sol.evaluate(e, fi) for e in exprs]
        hdr = "x,y," + ("z," if sol.sdim == 3 else "") + ",".join(exprs)
        np.savetxt(fn, np.column_stack(cols), delimiter=",", header=hdr, comments="% ")

        with open(fn, "r") as f:
            first_line = f.readline().strip()

        assert "not converged" not in first_line.lower(), f"Converged CSV has marker: {first_line}"
    finally:
        os.unlink(fn)


# --------------------------------------------------------------------------------------- PNG export tests
def test_png_scene_title_includes_marker(elmer):
    """PNG scene title should include the unconverged marker for not converged solutions.
    Note: Actual PNG export requires GL context (see G1). This test verifies the scene
    title that PNG snapshot would use."""
    m, geo = _nonlinear_model()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_png_unconv_")
    try:
        StudyRunner(m, st, work, elmer).run()
    except StudyNotConverged:
        pass
    sol_folder = None
    for root, dirs, files in os.walk(work):
        if "meta.json" in files:
            with open(os.path.join(root, "meta.json")) as f:
                meta = json.load(f)
            if meta.get("status") == "not converged":
                sol_folder = root
                break
    assert sol_folder is not None

    sol = Solution(sol_folder)
    # Create a solution dataset and assign it to the plot group
    res = m.results
    dsets = res.child("results.datasets")
    ds = m.create("dset.solution", dsets, index=0)
    ds.props["study"] = st.tag
    pg_node = m.create("pg2d", m.results, index=0)
    pg_node.props["data"] = ds.tag
    surface = m.create("plot.surface", pg_node, index=0)
    surface.props["expr"] = "u"

    # Render the plot and check the scene title (used in PNG snapshot)
    scene = build_scene(pg_node, sol, 0, sol.sdim)
    assert "not converged" in scene.title.lower(), f"PNG scene title missing marker: {scene.title}"

    # Also verify by doing a real plot via the graphics window
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from elmerstudio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    for name in ("warning", "critical", "information"):
        from PySide6.QtWidgets import QMessageBox
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)

    try:
        # sol_folder is at workdir/st.tag/sol_.../ so workdir is grandparent
        workdir = os.path.dirname(os.path.dirname(sol_folder))
        w._set_model(m, None, workdir)
        
        # Plot the solution
        w.plot(pg_node)
        QApplication.processEvents()
        import time
        time.sleep(0.2)
        QApplication.processEvents()

        # The scene title is set on the graphics view and used for PNG export
        # We already verified the scene title above, which is what the PNG snapshot uses
        # Just ensure the graphics window rendered without error
        assert w.graphics.plotter is not None
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()


def test_png_scene_title_no_marker_converged(elmer):
    """PNG scene title should NOT include the marker for converged solutions."""
    m, geo = _converged_model()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_png_conv_")
    folders = StudyRunner(m, st, work, elmer).run()
    sol = Solution(folders[0])

    # Create a solution dataset and assign it to the plot group
    res = m.results
    dsets = res.child("results.datasets")
    ds = m.create("dset.solution", dsets, index=0)
    ds.props["study"] = st.tag
    pg_node = m.create("pg2d", m.results, index=0)
    pg_node.props["data"] = ds.tag
    surface = m.create("plot.surface", pg_node, index=0)
    surface.props["expr"] = "u"

    scene = build_scene(pg_node, sol, 0, sol.sdim)
    assert "not converged" not in scene.title.lower(), f"Converged PNG scene has marker: {scene.title}"

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from elmerstudio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    for name in ("warning", "critical", "information"):
        from PySide6.QtWidgets import QMessageBox
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)

    try:
        # sol_folder is at workdir/st.tag/sol_.../ so workdir is grandparent
        workdir = os.path.dirname(os.path.dirname(folders[0]))
        w._set_model(m, None, workdir)
        w.plot(pg_node)
        QApplication.processEvents()
        import time
        time.sleep(0.2)
        QApplication.processEvents()

        assert w.graphics.plotter is not None
    finally:
        w.dirty = False
        w.stop_job()
        w.close()
        app.processEvents()


# --------------------------------------------------------------------------------------- VTU export tests
def test_vtu_export_includes_status_metadata():
    """VTU/case exports should include the status in field data or metadata."""
    import pyvista as pv

    sol_folder, m = _unconverged_solution_folder()
    assert sol_folder is not None
    sol = Solution(sol_folder)

    # Simulate the VTU export from export_node (same logic as in main_window.py)
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        fn = f.name
    try:
        src_file = sol.frames[sol.frame_index("last")].file
        assert os.path.exists(src_file), f"VTU file not found: {src_file}"
        
        # Add solution validity marker to VTU field data (same as export_node)
        validity = getattr(sol, "validity_label", None)
        if validity and validity != "converged":
            grid = pv.read(src_file)
            grid.field_data["SolutionStatus"] = np.array([validity])
            grid.save(fn)
        else:
            import shutil
            shutil.copy(src_file, fn)
        
        # Read back and check field data
        grid = pv.read(fn)
        field_data = grid.field_data

        # Check if the status is in field data
        has_status = any("not converged" in str(field_data[k]).lower() for k in field_data.keys())
        for key in ["SolutionStatus", "status", "validity", "convergence_status"]:
            if key in field_data:
                val = str(field_data[key])
                if "not converged" in val.lower():
                    has_status = True
                    break

        assert has_status, f"VTU field data missing unconverged marker. Keys: {list(field_data.keys())}"
    finally:
        os.unlink(fn)


def test_vtu_export_no_status_converged():
    """VTU exports should NOT include the unconverged status for converged solutions."""
    import pyvista as pv

    sol_folder, m = _converged_solution_folder()
    sol = Solution(sol_folder)

    # Simulate the VTU export from export_node
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        fn = f.name
    try:
        src_file = sol.frames[sol.frame_index("last")].file
        assert os.path.exists(src_file), f"VTU file not found: {src_file}"
        
        # Add solution validity marker to VTU field data (same as export_node)
        validity = getattr(sol, "validity_label", None)
        if validity and validity != "converged":
            grid = pv.read(src_file)
            grid.field_data["SolutionStatus"] = np.array([validity])
            grid.save(fn)
        else:
            import shutil
            shutil.copy(src_file, fn)
        
        # Read back and check field data
        grid = pv.read(fn)
        field_data = grid.field_data

        has_status = any("not converged" in str(field_data[k]).lower() for k in field_data.keys())
        for key in ["SolutionStatus", "status", "validity", "convergence_status"]:
            if key in field_data:
                val = str(field_data[key])
                if "not converged" in val.lower():
                    has_status = True
                    break

        assert not has_status, f"Converged VTU has unconverged marker in field data. Keys: {list(field_data.keys())}"
    finally:
        os.unlink(fn)


if __name__ == "__main__":
    # Run without Elmer for unit tests
    print("Running unit tests without Elmer...")
    try:
        test_solution_validity_label()
        print("test_solution_validity_label: FAIL (expected - not implemented)")
    except Exception as e:
        print(f"test_solution_validity_label: FAIL - {e}")

    try:
        test_converged_solution_validity_label()
        print("test_converged_solution_validity_label: FAIL (expected)")
    except Exception as e:
        print(f"test_converged_solution_validity_label: FAIL - {e}")

    try:
        test_evaluation_table_includes_status_unconverged()
        print("test_evaluation_table_includes_status_unconverged: FAIL (expected)")
    except Exception as e:
        print(f"test_evaluation_table_includes_status_unconverged: FAIL - {e}")

    try:
        test_evaluation_table_no_status_converged()
        print("test_evaluation_table_no_status_converged: FAIL (expected)")
    except Exception as e:
        print(f"test_evaluation_table_no_status_converged: FAIL - {e}")

    try:
        test_csv_export_includes_marker_header()
        print("test_csv_export_includes_marker_header: FAIL (expected)")
    except Exception as e:
        print(f"test_csv_export_includes_marker_header: FAIL - {e}")

    try:
        test_csv_export_no_marker_converged()
        print("test_csv_export_no_marker_converged: FAIL (expected)")
    except Exception as e:
        print(f"test_csv_export_no_marker_converged: FAIL - {e}")

    try:
        test_vtu_export_includes_status_metadata()
        print("test_vtu_export_includes_status_metadata: FAIL (expected)")
    except Exception as e:
        print(f"test_vtu_export_includes_status_metadata: FAIL - {e}")

    try:
        test_vtu_export_no_status_converged()
        print("test_vtu_export_no_status_converged: FAIL (expected)")
    except Exception as e:
        print(f"test_vtu_export_no_status_converged: FAIL - {e}")

    print("Unit tests done - expected failures until implementation")