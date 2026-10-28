"""Main window: ribbon, Model Builder, Settings, Graphics, Messages/Progress/Log/Table."""
from __future__ import annotations

import copy
import glob
import json
import os
import re
import time
import traceback

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
                               QStackedWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, DOCS_URL, FILE_EXT, __version__
from ..core import builder as B
from ..core import project, recovery, units
from ..core.helpmap import help_url
from ..core.elmer import elmer_version, find_elmer, write_case
from ..core.examples import EXAMPLES
from ..core.geometry import (WP_KINDS, GeometryError, GeometryResult, apply_named_selections, build_geometry,
                             object_names_before)
from ..core.materials import MATPROPS
from ..core.meshing import SIZE_CHOICES, build_mesh
from ..core.model import REGISTRY, Model, Node, is_axisymmetric, level_plural, space_dim, type_title
from ..core.physics import COUPLINGS, PHYSICS, STUDY_TYPES, physics_def
from ..core.results import Solution, compatible_units
from ..core.sif import SifError, build_sif
from ..core.study_runner import StudyCancelled, StudyRunner
from . import icons, theme
from .graphics import GraphicsView
from .material_browser import MaterialBrowser
from .model_builder import ModelBuilder
from .panes import ConvergencePlot, LogPane, MessagesPane, MplCanvas, ProgressPane, TablePane
from ..core.datasets import resolve as resolve_dataset
from ..core.datasets import root_dataset
from .plots import build_scene, default_expr, draw_1d, evaluate_derived
from .ribbon import Ribbon
from .settings import SettingsWindow, pretty_unit
from .wizard import ModelWizard, NewDialog

GEOM_LABELS = {k: REGISTRY[k].title for k in REGISTRY if k.startswith("geom.")}


class Job(QThread):
    event = Signal(str, object)
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.runner = None
        self.cancelled = False          # Stop: the result of a job that cannot be killed is discarded

    def run(self):
        try:
            r = self.fn(lambda k, p=None: self.event.emit(k, p))
            self.done.emit(r)
        except StudyCancelled:
            self.failed.emit("Computation stopped by the user.", "")
        except Exception as exc:  # report everything to the GUI thread
            self.failed.emit(str(exc), traceback.format_exc())


def _caps_text(inst) -> str:
    """One line on the optional Elmer features this installation has."""
    from ..core.elmer import capabilities
    c = capabilities(inst)
    parts = [("diffusion current (multi-ion electrolytes)", c.get("diffusion_current")),
             ("user solvers (Fortran compiler found)", c.get("user_solvers"))]
    return "Features: " + ", ".join(f"{n}: {'yes' if v else 'no'}" for n, v in parts) + (
        "  [from-source build tree]" if getattr(inst, "build_tree", False) else "")


def _inputs(model: Model, solvers=True) -> str:
    """What a solution depends on: the model without Results, labels and bookkeeping (to tell an out-of-date
    solution from a current one). *solvers* False leaves out Solver Configurations: the run adds the default ones to
    its copy when the study has none, which is not an edit."""
    def strip(d):
        out = {k: v for k, v in d.items() if k not in ("label", "meta", "children")}
        if isinstance(out.get("selection"), dict):                # entity signatures are bookkeeping
            out["selection"] = {k: v for k, v in out["selection"].items() if k != "sig"}
        kids = [strip(c) for c in d.get("children", []) if c.get("kind") != "results"
                and (solvers or c.get("kind") != "study.solvers")]
        if kids:
            out["children"] = kids
        return out
    return json.dumps(strip(model.root.to_dict()), sort_keys=True, default=str)


class MainWindow(QMainWindow):
    def __init__(self, open_path: str | None = None, show_new=True):
        super().__init__()
        self.qs = QSettings("ElmerStudio", "ElmerStudio")
        self.inst = find_elmer(self.qs.value("elmer_home") or None)
        self.model: Model = B.new_model(None)
        self.workdir = project.new_workdir()
        self.file_path: str | None = None
        # model revisions: every edit makes a new one, undo/redo return to the revision of the state they restore;
        # the file is modified when the revision differs from the saved one, and a geometry/mesh job's result is
        # current only if the model is still at the revision the job started from
        self._revs = 0
        self.rev = self._saved_rev = 0
        self._slots: dict = {}          # component tag -> its geometry and mesh (geo, mesh, ... are the active one's)
        self.undo_stack: list[tuple[str, int]] = []      # (snapshot, revision)
        self.redo_stack: list[tuple[str, int]] = []
        self.sol_cache: dict[str, Solution] = {}
        self.job: Job | None = None
        self.active_sel: Node | None = None
        self.current_plot: Node | None = None
        self._pending_after_geo = None
        self.no_modal = False          # tests: never block on modal dialogs/menus
        self._job_started = 0.0
        self._job_stage = ""
        self._stopping = False
        self._actions_pending = False
        self._maximized_state = None
        self.recovery = recovery.Session()
        self._checkpoint_rev = None
        self.setWindowIcon(icons.icon("app"))
        self.resize(1500, 920)
        self._build_actions()
        self._build_ribbon()
        self._build_panes()
        self._build_statusbar()
        geom = self.qs.value("window_geometry")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)
        mode = self.qs.value("layout_mode") or self._auto_layout()
        self.apply_layout(mode if mode in ("widescreen", "regular") else "widescreen", restore=True)
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._flush_log)
        self.log_timer.start(250)
        self.mem_timer = QTimer(self)
        self.mem_timer.timeout.connect(self._update_mem)
        self.mem_timer.start(2000)
        self.job_timer = QTimer(self)                  # elapsed time of the running job
        self.job_timer.timeout.connect(self._update_job_status)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.checkpoint)
        if int(self.qs.value("autosave_s", 60)) > 0:
            self.autosave_timer.start(int(self.qs.value("autosave_s", 60)) * 1000)
        self.graphics.export_theme = self.qs.value("export_theme", "white")
        self.graphics.set_theme(self.qs.value("graphics_theme", "light"), redraw=False)
        self._set_model(self.model, None)
        self.messages.add(f"{APP_NAME} {__version__}")
        if self.inst:
            v = elmer_version(self.inst)
            self.messages.add(f"Elmer found: {self.inst.solver}" + (f"  [{v}]" if v else ""))
        else:
            self.messages.add("Elmer was not found. Set the installation folder in File > Preferences.", "warning")
        self._startup = (open_path, show_new)
        QTimer.singleShot(0, self._after_start)

    def _after_start(self):
        """Offer the recovery of an interrupted session first, then open the file or the New window."""
        open_path, show_new = self._startup
        if self.offer_recovery():
            return
        if open_path:
            self.open_file(open_path)
        elif show_new:
            QTimer.singleShot(150, self.file_new)

    # ================================================================== construction
    def _a(self, name, text, icon_name=None, cb=None, shortcut=None, tip=None, checkable=False, menu=None):
        a = QAction(icons.icon(icon_name) if icon_name else QIcon(), text, self)
        keys = [shortcut] if isinstance(shortcut, str) else list(shortcut or [])
        if keys:
            a.setShortcuts([QKeySequence(k) for k in keys])
            a.setShortcutContext(Qt.WindowShortcut)
        a.setToolTip(f"{tip or text.replace(chr(10), ' ')}" + (f" ({keys[0]})" if keys else ""))
        a.setProperty("base_tip", a.toolTip())
        if cb:
            a.triggered.connect(lambda _=False: cb())
        a.setCheckable(checkable)
        if menu is not None:
            a.setMenu(menu)
        self.act[name] = a
        self.addAction(a)
        return a

    def _menu(self, items):
        m = QMenu(self)
        m.setToolTipsVisible(True)
        for it in items:
            if it is None:
                m.addSeparator()
            elif isinstance(it, QAction):
                m.addAction(it)
            else:
                text, icon_name, cb = it
                a = m.addAction(icons.icon(icon_name), text)
                a.triggered.connect(lambda _=False, c=cb: c())
        return m

    def _build_actions(self):
        self.act: dict[str, QAction] = {}
        a = self._a
        a("new", "New", "new", self.file_new, "Ctrl+N")
        a("open", "Open...", "open", self.file_open, "Ctrl+O")
        a("save", "Save", "save", self.file_save, "Ctrl+S")
        a("save_as", "Save As...", "save_as", self.file_save_as, "Ctrl+Shift+S")
        a("undo", "Undo", "undo", self.undo, "Ctrl+Z")
        a("redo", "Redo", "redo", self.redo, ["Ctrl+Y", "Ctrl+Shift+Z"])
        a("delete", "Delete", "delete", lambda: self.delete_node(self.tree.current()), "Delete")
        a("rename", "Rename", "rename", lambda: self.tree.edit_current(), "F2")
        a("duplicate", "Duplicate", "duplicate", lambda: self.duplicate_node(self.tree.current()), "Ctrl+Shift+D")
        a("disable", "Disable", "disable", lambda: self.set_enabled(self.tree.current(), False), "F3")
        a("enable", "Enable", "enable", lambda: self.set_enabled(self.tree.current(), True), "F4")
        a("move_up", "Move Up", "up", lambda: self.move_node(self.tree.current(), -1), "Ctrl+Up")
        a("move_down", "Move Down", "down", lambda: self.move_node(self.tree.current(), 1), "Ctrl+Down")
        a("build_selected", "Build\nSelected", "build_selected", self.build_selected, "F7")
        a("build_all", "Build\nAll", "build_all", self.build_all_context, "F8")
        a("geom_build", "Build\nAll", "build_all", self.build_geometry)
        a("mesh_build", "Build\nMesh", "mesh", self.build_mesh)
        a("compute", "Compute", "compute", self.compute, "Ctrl+=")
        a("stop", "Stop", "stop", self.stop_job)
        a("prefs", "Preferences...", "preferences", self.preferences)
        a("help", "Help", "help", self.context_help, "F1", tip="Help on the selected node")
        a("docs", "Documentation", "help", self.open_docs, "Ctrl+F1", tip="The documentation's front page")
        a("about", f"About {APP_NAME}", "app", self.about)
        a("reset_desktop", "Reset\nDesktop", "reset_desktop", self.reset_desktop,
          tip="Restore the current layout's windows")
        a("recovery_files", "Recovery Files...", "refresh", self.show_recovery_files,
          tip="Models of interrupted sessions that were never saved")
        lay = QActionGroup(self)
        for key, text, tip in (("widescreen", "Widescreen Layout", "Model Builder, Settings and Graphics side by side"),
                               ("regular", "Regular Screen Layout", "Model Builder above Settings, Graphics beside "
                                                                     "them (narrow or 4:3 screens)")):
            act = a(f"layout_{key}", text, "layout", (lambda k=key: self.apply_layout(k)), tip=tip, checkable=True)
            lay.addAction(act)
        a("maximize", "Maximize\nGraphics", "graphics_window", self.toggle_maximize, checkable=True,
          tip="Hide the other windows (double-click the Graphics tab does the same)")
        a("minimize_ribbon", "Minimize the Ribbon", None, lambda: self.ribbon.set_minimized(
            self.act["minimize_ribbon"].isChecked()), checkable=True,
          tip="Show only the tab names; click a tab to show its commands (double-click a tab does the same)")
        # moving between windows, as in COMSOL
        a("next_window", "Next Window", None, lambda: self.cycle_focus(1), "Ctrl+Tab")
        a("prev_window", "Previous Window", None, lambda: self.cycle_focus(-1), "Ctrl+Shift+Tab")
        a("focus_tree", "Model Builder", "model_builder", lambda: self.focus_pane("tree"), "Ctrl+Alt+Left")
        a("focus_settings", "Settings", "settings_window", lambda: self.focus_pane("settings"), "Ctrl+Alt+Right")
        a("next_section", "Next Settings Section", None, lambda: self.settings.focus_section(1), "Ctrl+Alt+Down")
        a("prev_section", "Previous Settings Section", None, lambda: self.settings.focus_section(-1), "Ctrl+Alt+Up")
        a("filter_tree", "Filter the Model Builder", None, lambda: self.tree.focus_filter(), "Ctrl+L")
        a("go_back", "Go Back", "back", lambda: self.tree.go_back(), "Alt+Left")
        a("go_forward", "Go Forward", "forward", lambda: self.tree.go_forward(), "Alt+Right")
        a("sel_all", "Select All", "select_all", lambda: self.graphics.select_all(), "Ctrl+A",
          tip="Select all entities of the active selection")
        a("sel_clear", "Clear Selection", "clear_selection", lambda: self.graphics.clear_selection(), "Ctrl+D",
          tip="Clear the active selection")
        a("show_sif", "Show Elmer\nInput File", "sif", self.show_sif)
        a("export_case", "Export Elmer\nCase", "export", self.export_case)
        a("run_script", "Run Python\nScript", "python", self.run_script)
        a("open_case", "Open Case\nFolder", "folder", self.open_case_folder)
        a("zoom_extents", "Zoom Extents", "zoom_extents", lambda: self.graphics.zoom_extents(), "F5")

    def _build_ribbon(self):
        r = Ribbon(self)
        self.ribbon = r
        for n in ("new", "open", "save", None, "undo", "redo", None, "compute"):
            if n is None:
                r.add_quick_sep()
            else:
                r.add_quick(self.act[n])
        r.add_right(self.act["help"])
        self.recent_menu = QMenu("Recent", self)
        fm = QMenu(self)
        fm.setToolTipsVisible(True)
        for n in ("new", "open"):
            fm.addAction(self.act[n])
        fm.addMenu(self.recent_menu)
        fm.addAction(self.act["recovery_files"])
        fm.addSeparator()
        for n in ("save", "save_as"):
            fm.addAction(self.act[n])
        fm.addSeparator()
        ex = fm.addMenu(icons.icon("app"), "Application Libraries")
        for key, (title, _fn) in EXAMPLES.items():
            act = ex.addAction(icons.icon("app"), title)
            act.triggered.connect(lambda _=False, k=key: self.open_example(k))
        fm.addAction(self.act["export_case"])
        fm.addSeparator()
        fm.addAction(self.act["prefs"])
        fm.addSeparator()
        fm.addAction(self.act["help"])
        fm.addAction(self.act["docs"])
        fm.addAction(self.act["about"])
        fm.addSeparator()
        q = fm.addAction(icons.icon("close"), "Exit")
        q.triggered.connect(self.close)
        r.set_file_menu(fm)
        self._refresh_recent()

        # --- Home
        t = r.add_tab("Home")
        g = t.add_group("Model")
        self.m_comp = self._menu([("3D", "component", lambda: self.add_component("3D")),
                                  ("2D Axisymmetric", "revolve", lambda: self.add_component("2Daxi")),
                                  ("2D", "rectangle", lambda: self.add_component("2D"))])
        g.add_large(self._a("add_comp", "Add\nComponent", "component", menu=self.m_comp))
        g = t.add_group("Definitions")
        g.add_large(self._a("params", "Parameters", "parameters", lambda: self.goto_kind("params")))
        g.add_small([self._a("vars", "Variables", "variables", lambda: self.add_child_kind("definitions", "variables")),
                     self._a("funcs", "Functions", "functions", menu=self._menu([
                         ("Analytic", "functions", lambda: self.add_child_kind("global", "func.analytic")),
                         ("Interpolation", "functions", lambda: self.add_child_kind("global", "func.interp"))]))])
        g = t.add_group("Geometry")
        g.add_large(self.act["geom_build"])
        g.add_large(self._a("import", "Import", "import", lambda: self.add_geom("geom.import")))
        g = t.add_group("Materials")
        g.add_large(self._a("add_mat", "Add\nMaterial", "materials", self.show_material_browser))
        g = t.add_group("Physics")
        self.m_add_phys = QMenu(self)
        self.m_add_phys.aboutToShow.connect(lambda: self._fill_add_physics(self.m_add_phys))
        g.add_large(self._a("add_phys", "Add\nPhysics", "physics_list", menu=self.m_add_phys))
        self.m_mp = QMenu(self)
        self.m_mp.aboutToShow.connect(lambda: self._fill_multiphysics(self.m_mp))
        g.add_large(self._a("mp", "Multiphysics", "multiphysics", menu=self.m_mp))
        g = t.add_group("Mesh")
        g.add_large(self.act["mesh_build"])
        self.m_size = QMenu(self)
        for val, text in SIZE_CHOICES:
            act = self.m_size.addAction(text)
            act.triggered.connect(lambda _=False, v=val: self.set_mesh_size(v))
        g.add_small([self._a("mesh_size", "Element Size", "mesh_size", menu=self.m_size)])
        g = t.add_group("Study")
        g.add_large(self.act["compute"])
        self.m_add_study = self._menu([(STUDY_TYPES[k][0], STUDY_TYPES[k][1], (lambda kk=k: self.add_study(kk)))
                                       for k in STUDY_TYPES])
        g.add_large(self._a("add_study", "Add\nStudy", "study", menu=self.m_add_study))
        g = t.add_group("Results")
        g.add_large(self._a("add_pg", "Add Plot\nGroup", "plot_group_3d", menu=self._menu([
            ("3D Plot Group", "plot_group_3d", lambda: self.add_plot_group("pg3d")),
            ("2D Plot Group", "plot_group_2d", lambda: self.add_plot_group("pg2d")),
            ("1D Plot Group", "plot_group_1d", lambda: self.add_plot_group("pg1d")),
            ("Polar Plot Group", "plot_group_1d", lambda: self.add_plot_group("pgpolar"))])))
        g = t.add_group("Layout")
        self.m_layout = self._menu([self.act["layout_widescreen"], self.act["layout_regular"], None,
                                    self.act["maximize"], self.act["minimize_ribbon"], None,
                                    self.act["reset_desktop"]])
        g.add_large(self._a("layout", "Layout", "layout", menu=self.m_layout, tip="Desktop layout"))
        g.add_large(self.act["reset_desktop"])

        # --- Definitions
        t = r.add_tab("Definitions")
        g = t.add_group("Definitions")
        g.add_large(self._a("params2", "Parameters", "parameters", lambda: self.add_child_kind("global", "params")))
        g.add_large(self._a("vars2", "Variables", "variables", lambda: self.add_child_kind("definitions", "variables")))
        g = t.add_group("Functions")
        g.add_large(self._a("an", "Analytic", "functions", lambda: self.add_child_kind("global", "func.analytic")))
        g.add_large(self._a("interp", "Interpolation", "functions", lambda: self.add_child_kind("global", "func.interp")))
        g = t.add_group("Component Couplings")
        g.add_small([self._a(f"cpl_{k}", REGISTRY[f"cpl.{k}"].title, REGISTRY[f"cpl.{k}"].icon,
                             (lambda kk=k: self.add_child_kind("definitions", f"cpl.{kk}")))
                     for k in ("average", "integration")])
        g.add_small([self._a(f"cpl_{k}", REGISTRY[f"cpl.{k}"].title, REGISTRY[f"cpl.{k}"].icon,
                             (lambda kk=k: self.add_child_kind("definitions", f"cpl.{kk}")))
                     for k in ("maximum", "minimum")])
        g = t.add_group("Deformed Geometry")
        g.add_large(self._a("dg_ddom", "Deforming\nDomain", "feature_domain",
                            lambda: self.add_child_kind("definitions", "dg.ddom")))
        g.add_large(self._a("dg_pnmv", "Prescribed Normal\nMesh Velocity", "feature_boundary",
                            lambda: self.add_child_kind("definitions", "dg.pnmv")))

        # --- Geometry
        t = r.add_tab("Geometry")
        g = t.add_group("Build")
        g.add_large(self._a("geom_build2", "Build\nAll", "build_all", self.build_geometry))
        g.add_large(self._a("import2", "Import", "import", lambda: self.add_geom("geom.import")))
        g = t.add_group("Primitives")
        self.geo3d_actions = []
        self.geo2d_actions = []
        for kinds, store in ((["geom.block", "geom.cone", "geom.cylinder", "geom.sphere", "geom.torus", "geom.ellipsoid"],
                              self.geo3d_actions),
                             (["geom.rectangle", "geom.circle", "geom.ellipse", "geom.polygon", "geom.paramcurve"],
                              self.geo2d_actions)):
            for k in kinds:
                nt = REGISTRY[k]
                act = self._a(k, nt.title, nt.icon, (lambda kk=k: self.add_geom(kk)))
                act.setData(k)
                store.append(act)
        for act in self.geo3d_actions[:3]:
            g.add_large(act)
        g.add_small(self.geo3d_actions[3:])
        for act in self.geo2d_actions[:2]:
            g.add_large(act)
        g.add_small(self.geo2d_actions[2:])
        g = t.add_group("Work Plane")
        self.wp_actions = [self._a("geom.workplane", "Work\nPlane", "work_plane", lambda: self.add_geom("geom.workplane")),
                           self._a("geom.extrude", "Extrude", "extrude", lambda: self.add_geom("geom.extrude")),
                           self._a("geom.revolve", "Revolve", "revolve", lambda: self.add_geom("geom.revolve"))]
        g.add_large(self.wp_actions[0])
        g.add_small(self.wp_actions[1:])
        g = t.add_group("Booleans and Partitions")
        for k in ("geom.union", "geom.intersection", "geom.difference"):
            g.add_large(self._a(k, REGISTRY[k].title, REGISTRY[k].icon, (lambda kk=k: self.add_geom(kk))))
        g = t.add_group("Transforms")
        tr = [self._a(k, REGISTRY[k].title, REGISTRY[k].icon, (lambda kk=k: self.add_geom(kk)))
              for k in ("geom.array", "geom.copy", "geom.mirror", "geom.move", "geom.rotate", "geom.scale")]
        g.add_small(tr[:3])
        g.add_small(tr[3:])
        g = t.add_group("Conversions")
        self.fc_actions = [self._a(k, REGISTRY[k].title, REGISTRY[k].icon, (lambda kk=k: self.add_geom(kk)))
                           for k in ("geom.fillet3d", "geom.chamfer3d")]
        g.add_small(self.fc_actions)
        tos = self._a("geom.tosolid", "Convert\nto Solid", REGISTRY["geom.tosolid"].icon, lambda: self.add_geom("geom.tosolid"))
        tos.setData("geom.tosolid")
        self.geo2d_actions.append(tos)
        g.add_large(tos)
        g = t.add_group("Selections")
        g.add_large(self._a("geom.boxsel", "Box\nSelection", REGISTRY["geom.boxsel"].icon,
                            lambda: self.add_geom("geom.boxsel")))
        g = t.add_group("Views")
        g.add_large(self.act["zoom_extents"])

        # --- Materials
        t = r.add_tab("Materials")
        g = t.add_group("Materials")
        g.add_large(self._a("add_mat2", "Add\nMaterial", "materials", self.show_material_browser))
        g.add_large(self._a("blank_mat", "Blank\nMaterial", "material", self.add_blank_material))

        # --- Physics (contextual)
        t = r.add_tab("Physics")
        g = t.add_group("Physics")
        self.m_dom = QMenu(self)
        self.m_dom.aboutToShow.connect(lambda: self._fill_features(self.m_dom, "domain"))
        self.m_bnd = QMenu(self)
        self.m_bnd.aboutToShow.connect(lambda: self._fill_features(self.m_bnd, "boundary"))
        g.add_large(self._a("phys_dom", "Domains", "feature_domain", menu=self.m_dom))
        g.add_large(self._a("phys_bnd", "Boundaries", "feature_boundary", menu=self.m_bnd))
        g = t.add_group("Physics Interfaces")
        self.m_add_phys2 = QMenu(self)
        self.m_add_phys2.aboutToShow.connect(lambda: self._fill_add_physics(self.m_add_phys2))
        g.add_large(self._a("add_phys2", "Add\nPhysics", "physics_list", menu=self.m_add_phys2))
        self.m_mp2 = QMenu(self)
        self.m_mp2.aboutToShow.connect(lambda: self._fill_multiphysics(self.m_mp2))
        g.add_large(self._a("mp2", "Multiphysics\nCouplings", "multiphysics", menu=self.m_mp2))

        # --- Mesh
        t = r.add_tab("Mesh")
        g = t.add_group("Mesh")
        g.add_large(self._a("mesh_build2", "Build\nMesh", "mesh", self.build_mesh))
        g.add_large(self._a("mesh_size2", "Element\nSize", "mesh_size", menu=self.m_size))
        g = t.add_group("Generators")
        g.add_large(self._a("mesh_ftet", "Free\nTetrahedral", "free_tet", lambda: self.add_mesh_node("mesh.ftet")))
        g.add_small([self._a("mesh_ftri", "Free Triangular", "free_tri", lambda: self.add_mesh_node("mesh.ftri")),
                     self._a("mesh_fquad", "Free Quad", "free_tri", lambda: self.add_mesh_node("mesh.fquad")),
                     self._a("mesh_mapped", "Mapped", "free_tri", lambda: self.add_mesh_node("mesh.mapped"))])
        g = t.add_group("Attributes")
        g.add_small([self._a("mesh_sizen", "Size", "mesh_size", lambda: self.add_mesh_node("mesh.size")),
                     self._a("mesh_dist", "Distribution", "distribution", lambda: self.add_mesh_node("mesh.distribution"))])
        g = t.add_group("Statistics")
        g.add_large(self._a("mesh_stats", "Statistics", "mesh_stats", self.mesh_statistics))
        g.add_large(self._a("mesh_qual", "Plot\nQuality", "mesh_plot", lambda: self.show_mesh(quality=True)))
        g.add_large(self._a("mesh_clear", "Clear\nMesh", "clear", self.clear_mesh))

        # --- Study
        t = r.add_tab("Study")
        g = t.add_group("Study")
        g.add_large(self._a("compute2", "Compute", "compute", self.compute))
        g.add_large(self.act["stop"])
        g = t.add_group("Study Steps")
        g.add_large(self._a("steps", "Study\nSteps", "stationary", menu=self._menu(
            [(STUDY_TYPES[k][0], STUDY_TYPES[k][1], (lambda kk=k: self.add_step(kk))) for k in STUDY_TYPES])))
        g.add_large(self._a("sweep", "Parametric\nSweep", "parametric_sweep", self.add_sweep))
        g = t.add_group("Study")
        g.add_large(self._a("add_study2", "Add\nStudy", "study", menu=self.m_add_study))
        g.add_large(self._a("show_solver", "Show Default\nSolver", "solver_config", self.show_default_solver))
        g.add_large(self._a("clear_sol", "Clear All\nSolutions", "clear", self.clear_solutions))
        g = t.add_group("Elmer")
        g.add_large(self.act["show_sif"])
        g.add_large(self.act["open_case"])

        # --- Results
        t = r.add_tab("Results")
        g = t.add_group("Plot Groups")
        g.add_large(self._a("pg3", "3D Plot\nGroup", "plot_group_3d", lambda: self.add_plot_group("pg3d")))
        g.add_large(self._a("pg2", "2D Plot\nGroup", "plot_group_2d", lambda: self.add_plot_group("pg2d")))
        g.add_large(self._a("pg1", "1D Plot\nGroup", "plot_group_1d", lambda: self.add_plot_group("pg1d")))
        g.add_large(self._a("pgp", "Polar Plot\nGroup", "plot_group_1d", lambda: self.add_plot_group("pgpolar")))
        g = t.add_group("Plots")
        plots = [self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon, (lambda kk=k: self.add_plot(kk)))
                 for k in ("plot.surface", "plot.volume", "plot.slice", "plot.isosurface", "plot.contour", "plot.arrow",
                           "plot.streamline", "plot.mesh", "plot.line", "plot.linegraph", "plot.globalgraph",
                           "plot.pointgraph", "plot.tablegraph", "plot.annotation", "plot.multislice",
                           "plot.radiation", "plot.directivity")]
        g.add_large(plots[0])
        g.add_small(plots[1:4])
        g.add_small(plots[4:7])
        g.add_small(plots[7:9] + [plots[14]])
        g.add_small(plots[9:12])
        g.add_small(plots[12:14])
        g.add_small(plots[15:17])
        g = t.add_group("Attributes")
        g.add_large(self._a("r_def", "Deformation", "deformation", lambda: self.add_plot("plot.deform")))
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon, (lambda kk=k: self.add_plot(kk)))
                     for k in ("plot.selection", "plot.colorexpr")])
        g = t.add_group("Datasets")
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.datasets", kk)))
                     for k in ("dset.cutline", "dset.cutpoint", "dset.cutplane")])
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.datasets", kk)))
                     for k in ("dset.parext", "dset.revolution", "dset.mirror")])
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.datasets", kk)))
                     for k in ("dset.grid3d", "dset.grid2d", "dset.paramcurve")])
        g = t.add_group("Evaluation")
        g.add_large(self._a("r_glob", "Global\nEvaluation", "global_evaluation",
                            lambda: self.add_results_child("results.derived", "eval.global")))
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.derived", kk)))
                     for k in ("eval.volint", "eval.surfint", "eval.avg")])
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.derived", kk)))
                     for k in ("eval.max", "eval.min", "eval.point")])
        g = t.add_group("Export")
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.export", kk)))
                     for k in ("export.image", "export.data", "export.vtu")])
        g.add_large(self._a("r_report", "Report", "report", lambda: self.add_results_child("results.export", "export.report")))

        # --- Developer
        t = r.add_tab("Developer")
        g = t.add_group("Scripting")
        g.add_large(self.act["run_script"])
        g = t.add_group("Elmer Solver")
        g.add_large(self._a("show_sif2", "Show Elmer\nInput File", "sif", self.show_sif))
        g.add_large(self._a("export_case2", "Export Elmer\nCase", "export", self.export_case))
        g.add_large(self._a("prefs2", "Elmer\nInstallation", "preferences", self.preferences))

        # --- contextual tab of the selected plot group (named after it, as in COMSOL)
        t = r.add_tab("Plot Group")
        g = t.add_group("Plot")
        g.add_large(self._a("pg_plot", "Plot", "plot", lambda: self.plot(self._current_pg()),
                            tip="Plot the selected plot group (F7 and F8 on a plot group do the same)"))
        g.add_large(self._a("pg_snap", "Image\nSnapshot", "snapshot", lambda: self.graphics.snapshot_dialog()))
        g = t.add_group("Add Plot")
        self.pg3_plots = plots
        g.add_large(plots[0])
        g.add_small(plots[1:4])
        g.add_small(plots[4:7])
        g.add_small(plots[7:9] + [plots[14]])
        g = t.add_group("Add 1D Plot")
        g.add_small(plots[9:12])
        g.add_small(plots[12:14] + plots[16:17])
        g = t.add_group("Attributes")
        g.add_large(self.act["r_def"])
        g.add_small([self.act["r_plot.selection"], self.act["r_plot.colorexpr"]])
        r.set_tab_visible("Plot Group", False)
        for a in self.act.values():                    # a command chosen in a minimized ribbon folds it again
            a.triggered.connect(r.command_triggered)
        r.minimizedChanged.connect(self.act["minimize_ribbon"].setChecked)
        r.setContextMenuPolicy(Qt.CustomContextMenu)
        r.customContextMenuRequested.connect(
            lambda pos: None if self.no_modal else self._menu([self.act["minimize_ribbon"]]).exec(r.mapToGlobal(pos)))
        self.setMenuWidget(r)

    def _build_panes(self):
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.tree = ModelBuilder(self)
        self.tree.nodeSelected.connect(self.on_node_selected)
        self.tree.contextRequested.connect(self.context_menu)
        self.tree.renameRequested.connect(self.rename)
        self.tree.activated.connect(lambda n: self.build_all_context())
        self.tree.move_cb = self.move_node
        self.tree.issues = self.node_issues
        self.dock_tree = self._dock("Model Builder", "model_builder", self.tree, Qt.LeftDockWidgetArea)
        self.settings = SettingsWindow(self, self)
        self.settings.actionRequested.connect(self.on_settings_action)
        self.dock_settings = self._dock("Settings", "settings_window", self.settings, Qt.LeftDockWidgetArea)
        # central graphics
        self.central = QTabWidget(self)
        self.central.setDocumentMode(False)
        self.central.tabBar().tabBarDoubleClicked.connect(lambda _i: self.act["maximize"].trigger())
        self.gstack = QStackedWidget()
        self.graphics = GraphicsView(self)
        self.graphics.entityClicked.connect(self.on_entity_clicked)
        self.graphics.entitiesSet.connect(self.on_entities_set)
        self.graphics.status.connect(lambda s: self.statusBar().showMessage(s, 3000))
        self.plot1d = MplCanvas(self)
        self.gstack.addWidget(self.graphics)
        self.gstack.addWidget(self.plot1d)
        self.central.addTab(self.gstack, icons.icon("graphics_window"), "Graphics")
        self.conv = ConvergencePlot(self)
        self.conv.hide()           # becomes a tab of the central widget at the first solve; until then it would sit
                                   # over the Quick Access Toolbar at the window's top-left corner
        self.setCentralWidget(self.central)
        # bottom panes
        self.messages = MessagesPane(self)
        self.progress = ProgressPane(self)
        self.logpane = LogPane(self)
        self.table = TablePane(self)
        self.messages.nodeLinkClicked.connect(self.goto_tag)
        self.dock_msg = self._dock("Messages", "messages", self.messages, Qt.BottomDockWidgetArea)
        self.dock_prog = self._dock("Progress", "progress", self.progress, Qt.BottomDockWidgetArea)
        self.dock_log = self._dock("Log", "log", self.logpane, Qt.BottomDockWidgetArea)
        self.dock_table = self._dock("Table", "table", self.table, Qt.BottomDockWidgetArea)
        self.matbrowser = MaterialBrowser(self)
        self.matbrowser.addRequested.connect(self.add_library_material)
        self.dock_mat = self._dock("Add Material", "materials", self.matbrowser, Qt.RightDockWidgetArea)
        self.dock_mat.hide()

    # ================================================================== desktop layouts
    def _auto_layout(self) -> str:
        """COMSOL picks Widescreen or Regular Screen by the monitor: side by side needs a wide screen."""
        scr = self.screen() or QApplication.primaryScreen()
        g = scr.availableGeometry() if scr is not None else None
        return "widescreen" if g is None or (g.width() >= 1500 and g.width() / max(g.height(), 1) >= 1.5) else "regular"

    def _arrange(self, mode: str):
        """The default arrangement of a layout: Widescreen puts Model Builder, Settings and Graphics side by side;
        Regular Screen stacks Model Builder above Settings. Messages, Progress, Log and Table are tabbed below."""
        for d in self.findChildren(QDockWidget):     # start from nothing: re-tabifying docks in place duplicates tabs
            d.setFloating(False)
            self.removeDockWidget(d)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_tree)
        self.splitDockWidget(self.dock_tree, self.dock_settings, Qt.Horizontal if mode == "widescreen" else Qt.Vertical)
        for d in (self.dock_msg, self.dock_prog, self.dock_log, self.dock_table):
            self.addDockWidget(Qt.BottomDockWidgetArea, d)
        self.tabifyDockWidget(self.dock_msg, self.dock_prog)
        self.tabifyDockWidget(self.dock_prog, self.dock_log)
        self.tabifyDockWidget(self.dock_log, self.dock_table)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_mat)
        for d in (self.dock_tree, self.dock_settings, self.dock_msg, self.dock_prog, self.dock_log, self.dock_table):
            d.show()
        self.dock_mat.hide()
        self.dock_msg.raise_()
        if mode == "widescreen":
            self.resizeDocks([self.dock_tree, self.dock_settings], [300, 380], Qt.Horizontal)
        else:
            self.resizeDocks([self.dock_tree], [380], Qt.Horizontal)
            self.resizeDocks([self.dock_tree, self.dock_settings], [300, 520], Qt.Vertical)
        self.resizeDocks([self.dock_msg], [190], Qt.Vertical)
        self.resizeDocks([self.dock_mat], [300], Qt.Horizontal)

    def apply_layout(self, mode: str, restore=True):
        """Switch to a layout. Each layout keeps its own window arrangement (saved when you leave it); *restore*
        False gives its default arrangement (Reset Desktop)."""
        old = getattr(self, "layout_mode", None)
        if self._maximized_state is not None:
            self.act["maximize"].setChecked(False)
            self._maximized_state = None
        if old and old != mode:
            self.qs.setValue(f"window_state/{old}", self.saveState())
        self.layout_mode = mode
        self.act[f"layout_{mode}"].setChecked(True)
        self._arrange(mode)
        st = self.qs.value(f"window_state/{mode}") if restore else None
        if st is None and restore and mode == "widescreen":
            st = self.qs.value("window_state")        # saved by an earlier version: its only layout
        if isinstance(st, QByteArray):
            self.restoreState(st)
        self.qs.setValue("layout_mode", mode)

    def reset_desktop(self):
        self.apply_layout(self.layout_mode, restore=False)

    def toggle_maximize(self):
        """Hide every window but Graphics (again: bring them back as they were)."""
        if self._maximized_state is None:
            self._maximized_state = self.saveState()
            for d in self.findChildren(QDockWidget):
                d.hide()
            self.act["maximize"].setChecked(True)
        else:
            st, self._maximized_state = self._maximized_state, None
            self.restoreState(st)
            self.act["maximize"].setChecked(False)

    def _dock(self, title, icon_name, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(f"dock_{title.replace(' ', '_')}")
        d.setAccessibleName(title)
        widget.setAccessibleName(title)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        d.setWindowIcon(icons.icon(icon_name))
        self.addDockWidget(area, d)
        return d

    def _build_statusbar(self):
        sb = self.statusBar()
        self.pbar = QProgressBar()
        self.pbar.setMaximumWidth(220)
        self.pbar.setMaximumHeight(14)
        self.pbar.hide()
        self.mem = QLabel("")
        self.mem.setToolTip("Memory used by Elmer Studio: resident | virtual")
        self.st_job = QLabel("")
        self.st_job.setAccessibleName("Running job")
        self.st_job.hide()
        self.st_stop = QToolButton()
        self.st_stop.setDefaultAction(self.act["stop"])
        self.st_stop.setAutoRaise(True)
        self.st_stop.setAccessibleName("Stop")
        self.st_stop.hide()
        sb.addPermanentWidget(self.st_job)
        sb.addPermanentWidget(self.pbar)
        sb.addPermanentWidget(self.st_stop)
        self.st_geo, self.st_mesh, self.st_sol = (QToolButton(), QToolButton(), QToolButton())
        for b, which in ((self.st_geo, "geo"), (self.st_mesh, "mesh"), (self.st_sol, "sol")):
            b.setObjectName("StatusState")
            b.setAutoRaise(True)
            b.clicked.connect(lambda _=False, w=which: self._status_click(w))
            b.hide()
            sb.addPermanentWidget(b)
        sb.addPermanentWidget(self.mem)

    def _update_mem(self):
        try:
            import psutil
            mi = psutil.Process().memory_info()
            self.mem.setText(f"{mi.rss / 2**30:.2f} GB | {getattr(mi, 'vms', mi.rss) / 2**30:.2f} GB")
        except Exception:
            self.mem.setText("")

    # ================================================================== model lifecycle
    def _set_model(self, model: Model, path: str | None, workdir: str | None = None):
        self.model = model
        self.file_path = path
        if workdir:
            if getattr(self, "workdir", None) and os.path.abspath(self.workdir) != os.path.abspath(workdir):
                project.discard_workdir(self.workdir)      # the previous model's solutions are no longer shown
            self.workdir = workdir
        self._slots = {}
        model.set_active(None)
        self.sol_cache.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.rev = self._new_rev()
        self.dirty = False
        self.current_plot = None
        self.active_sel = None
        self.tree.set_model(model, self._file_name())
        self.graphics.clear()
        self.table.show_table("", [], [])               # no results table from the previous model
        self._update_title()
        self._update_geom_actions()
        self.tree.select(model.root)
        self.recovery.discard()                          # a new, opened or recovered model: nothing unsaved yet
        self._checkpoint_rev = None
        self.schedule_update()

    # -- per-component state: self.geo, self.mesh, self.geo_stale and self.mesh_stale are the active component's
    def _slot(self, comp: Node | None = None) -> dict:
        comp = comp if comp is not None else (self.model.component if getattr(self, "model", None) else None)
        return self._slots.setdefault(comp.tag if comp is not None else "",
                                      {"geo": None, "geo_stale": True, "mesh": None, "mesh_stale": True})

    geo = property(lambda self: self._slot()["geo"], lambda self, v: self._slot().__setitem__("geo", v))
    geo_stale = property(lambda self: self._slot()["geo_stale"],
                         lambda self, v: self._slot().__setitem__("geo_stale", v))
    mesh = property(lambda self: self._slot()["mesh"], lambda self, v: self._slot().__setitem__("mesh", v))
    mesh_stale = property(lambda self: self._slot()["mesh_stale"],
                          lambda self, v: self._slot().__setitem__("mesh_stale", v))

    def _all_stale(self):
        """Something every component reads changed (a parameter, a function): all geometries and meshes."""
        for sl in self._slots.values():
            sl["geo_stale"] = sl["mesh_stale"] = True

    def _snapshot(self) -> Model:
        """A copy of the model for a worker thread, with the same active component."""
        m = Model.from_dict(json.loads(json.dumps(self.model.to_dict())))
        m._active = getattr(self.model, "_active", None)
        return m

    def _file_name(self):
        if self.file_path:
            return os.path.basename(self.file_path)
        return f"{self.model.root.label or 'Untitled'}{FILE_EXT}"

    def _update_title(self):
        self.setWindowTitle(f"{self._file_name()}{' *' if self.dirty else ''} - {APP_NAME}")

    @property
    def dirty(self) -> bool:
        """Modified since the last save: the model is at another revision than the saved one."""
        return self.rev != self._saved_rev

    @dirty.setter
    def dirty(self, value: bool):
        if value:                   # a change outside the undo history (new solutions)
            self.rev = self._new_rev()
        else:
            self._saved_rev = self.rev

    def _new_rev(self) -> int:
        self._revs += 1
        return self._revs

    def _update_geom_actions(self, node: Node | None = None):
        sd = self.sdim() if self.model.component else 3
        tree = getattr(self, "tree", None)
        cur = node if node is not None else (tree.current() if tree is not None else None)
        # inside a work plane a 3D geometry takes the planar sketch primitives (add_geom puts them into the plane)
        in_wp = sd == 3 and cur is not None and (cur.kind == "geom.workplane" or (
            cur.parent is not None and cur.parent.kind == "geom.workplane"))
        nocomp = "Add a component first (Home > Add Component)" if self.model.component is None else None
        for a in self.geo3d_actions + self.wp_actions:
            self._set_avail(a, nocomp or (None if sd == 3 else "Needs a 3D component"))
        for a in self.fc_actions:                  # 3D: edges of a solid; 2D or inside a work plane: sketch vertices
            self._set_avail(a, nocomp or (None if sd in (2, 3) else "Needs a 2D or 3D component"))
        for a in self.geo2d_actions:
            self._set_avail(a, nocomp or (None if sd == 2 or (in_wp and a.data() in WP_KINDS) else
                                          "A planar primitive: select a Work Plane (or a node inside one) to add it "
                                          "to the plane's sketch"))
        self._set_avail(self.act["add_comp"], None)

    @staticmethod
    def _set_avail(a: QAction, reason: str | None):
        """Enable *a*, or disable it and say why in its tooltip and status tip."""
        base = a.property("base_tip") or a.toolTip()
        a.setEnabled(reason is None)
        tip = base if reason is None else f"{base}\nUnavailable: {reason}"
        if a.toolTip() != tip:
            a.setToolTip(tip)
            a.setStatusTip(tip.replace("\n", " - "))

    def schedule_update(self):
        """Command availability, the status bar and the tree's markers follow the model: refresh them once the
        current edit is done."""
        if not self._actions_pending:
            self._actions_pending = True
            QTimer.singleShot(0, self._update_ui_state)

    def _update_ui_state(self):
        self._actions_pending = False
        try:
            self.update_actions()
            self.update_status()
            self._update_tabs()
            self.tree.update_marks()
        except RuntimeError:            # the window is being destroyed
            pass

    def _units(self) -> dict:
        """Unit messages of the whole model (COMSOL's orange), recomputed once per model revision."""
        key = (self.rev, id(self.model))
        if getattr(self, "_units_key", None) != key:
            from ..core.diagnostics import unit_report
            try:
                self._units_report = unit_report(self.model)
            except Exception:  # noqa: BLE001 - a check must never break the window
                self._units_report = {}
            self._units_key = key
        return self._units_report

    def node_issues(self, node: Node) -> list[tuple[str, str]]:
        """What is wrong with or out of date in a node, for the Model Builder's marks and tooltips."""
        out = [("warning", f"Unit: {w}") for w in self._units().get(node.tag, [])]
        k = node.kind
        par = node.parent
        nt = REGISTRY.get(k)
        sel = node.selection
        if sel is not None and node.enabled and not node.meta.get("default") and not sel.all and not sel.named \
                and not sel.entities and (k == "material" or k.startswith(("mp.", "dg.", "cpl."))
                                          or (par is not None and par.kind.startswith("physics."))):
            try:
                shown = nt is None or nt.selection_visible is None or nt.selection_visible(
                    {**node.props, "__sdim": self.sdim(), "__axi": self.is_axi()})
            except Exception:  # noqa: BLE001
                shown = True
            if shown:
                out.append(("warning", f"No {level_plural(sel.level)} selected: it applies nowhere."))
        if node.meta.get("lost"):
            out.append(("error", f"{sel.level.capitalize() if sel else 'Entity'} {', '.join(map(str, node.meta['lost']))} "
                                 "no longer exist after the geometry changed: check the selection."))
        if k == "material" and node.enabled and self.model.component is not None:
            vals = node.props.get("values") or {}
            missing = [MATPROPS[p][0] for p in self.required_matprops(node) if not str(vals.get(p, "")).strip()]
            if missing:
                out.append(("error", "Missing material properties: " + ", ".join(missing) + "."))
        if k == "geometry" and self.geo is not None and self.geo_stale:
            out.append(("stale", "Edited since it was built: build it again (F8)."))
        if k == "mesh" and self.mesh is not None and self.mesh_stale:
            out.append(("stale", "The model changed since the mesh was built."))
        if k == "study" and self.model.component is not None:
            text, state, tip = self._solution_state(node)
            if state == "stale":
                out.append(("stale", tip))
            elif state == "error":
                out.append(("error", f"{text}. {tip}"))
            src = self.model.by_tag(str(next((s.get("initsol") for s in node.children if s.get("initsol")), "")))
            if src is not None and self._computed_at(src) > self._computed_at(node) > 0:
                out.append(("stale", f"Starts from {src.label}, which was computed again since: compute "
                                     f"{node.label} again"))
        if k.startswith(("pg", "plot.", "eval.", "dset.", "export.")):
            for st in self._results_studies().get(node.tag, []):
                if self._solution_state(st)[1] == "stale":
                    out.append(("stale", f"Shows {st.label}, which is out of date: compute it again"))
        return out

    def _computed_at(self, st: Node) -> float:
        metas = glob.glob(os.path.join(self.workdir, st.tag, "**", "meta.json"), recursive=True)
        return max((os.path.getmtime(f) for f in metas), default=0.0)

    def _results_studies(self) -> dict:
        """Results node tag -> the studies it shows (dependency graph), once per model revision."""
        key = (self.rev, id(self.model))
        if getattr(self, "_rs_key", None) != key:
            from ..core import deps
            res = self.model.results
            self._rs = {n.tag: deps.studies_of(self.model, n) for n in (res.walk() if res else [])}
            self._rs_key = key
        return self._rs

    def update_actions(self):
        """One availability rule per command; the ribbon, the shortcut, the menus and the context menu share the
        QAction, so they agree. A disabled command says why."""
        cur = self.tree.current()
        comp = self.model.component
        nt = REGISTRY.get(cur.kind) if cur is not None else None
        sa = self._set_avail
        a = self.act
        nocomp = "Add a component first (Home > Add Component)" if comp is None else None
        sa(a["undo"], None if self.undo_stack else "Nothing to undo")
        sa(a["redo"], None if self.redo_stack else "Nothing to redo")
        pick = "Select a node in the Model Builder" if cur is None else None
        fixed = None
        if cur is not None and cur.meta.get("default"):
            fixed = f"'{cur.label}' is a default feature: add a feature after it to override it"
        elif cur is not None and (cur.parent is None or (nt is not None and not nt.deletable)):
            fixed = f"'{cur.label}' is a fixed part of the model"
        sa(a["delete"], pick or fixed)
        sa(a["rename"], pick or (None if cur.kind != "root" and (nt is None or nt.renamable) else
                                 "The root's name is the file name: use Save As"))
        sa(a["duplicate"], pick or fixed or (None if cur.kind not in ("global", "results", "geometry", "mesh",
                                                                      "materials", "geom.finalize")
                                             else f"'{cur.label}' cannot be duplicated"))
        cant_dis = pick or (None if nt is not None and nt.can_disable and not cur.meta.get("default") else
                            f"'{cur.label}' cannot be disabled" if cur is not None else None)
        sa(a["disable"], cant_dis or (None if cur.enabled else "Already disabled"))
        sa(a["enable"], cant_dis or (None if not cur.enabled else "Not disabled"))
        for name, d in (("move_up", -1), ("move_down", 1)):
            reason = pick
            if reason is None:
                sib = cur.parent.children if cur.parent is not None else [cur]
                j = sib.index(cur) + d
                if cur.parent is None or not (0 <= j < len(sib)):
                    reason = "Already the " + ("first" if d < 0 else "last") + " node of its branch"
                elif sib[j].kind == "geom.finalize" or sib[j].meta.get("default") or cur.meta.get("default"):
                    reason = "Default and finalizing nodes keep their place"
            sa(a[name], reason)
        for name in ("build_selected", "build_all"):
            sa(a[name], None if comp is not None or (cur is not None and cur.kind.startswith(("study", "pg", "plot.",
                                                                                              "eval."))) else nocomp)
        for name in ("geom_build", "geom_build2", "import", "import2", "mesh_build", "mesh_build2", "mesh_size",
                     "mesh_size2", "mesh_ftet", "mesh_ftri", "mesh_fquad", "mesh_mapped", "mesh_sizen", "mesh_dist",
                     "mesh_stats", "mesh_qual", "add_mat", "add_mat2", "blank_mat", "add_phys", "add_phys2", "vars",
                     "vars2", "cpl_average", "cpl_integration", "cpl_maximum", "cpl_minimum", "dg_ddom", "dg_pnmv",
                     "geom.boxsel", "add_study", "add_study2", "steps", "geom.union", "geom.intersection",
                     "geom.difference", "geom.array", "geom.copy", "geom.mirror", "geom.move", "geom.rotate",
                     "geom.scale", "geom_build2"):
            sa(a[name], nocomp)
        self._update_geom_actions(cur)          # the per-dimension rules of the primitives (they say no component too)
        if comp is not None:
            sd = self.sdim()
            for name, kind in (("mesh_ftet", "mesh.ftet"), ("mesh_ftri", "mesh.ftri"), ("mesh_fquad", "mesh.fquad"),
                               ("mesh_mapped", "mesh.mapped")):
                ok = (kind == "mesh.ftet") == (sd == 3)        # the mesh sequence's generators per dimension
                sa(a[name], None if ok else f"{REGISTRY[kind].title} meshes {'3D' if kind == 'mesh.ftet' else '2D'} "
                                            "components")
        phys = B.physics_nodes(self.model)
        nophys = nocomp or (None if phys else "Add a physics interface first (Home > Add Physics)")
        for name in ("phys_dom", "phys_bnd"):
            sa(a[name], nophys)
        ids = {physics_def(p.kind).id for p in phys}
        any_mp = any(set(cp["requires"]) <= ids for cp in COUPLINGS.values())
        for name in ("mp", "mp2"):
            sa(a[name], nophys or (None if any_mp else "No multiphysics coupling applies to these physics interfaces"))
        has_study = bool(self.model.studies())
        nostudy = None if has_study else "Add a study first (Home > Add Study)"
        noelmer = None if self.inst is not None else "Elmer was not found: set its folder in File > Preferences"
        for name in ("compute", "compute2"):
            sa(a[name], nostudy or noelmer)
        for name in ("show_sif", "show_sif2", "export_case", "export_case2", "sweep", "show_solver", "clear_sol"):
            sa(a[name], nocomp or nostudy)
        sa(a["stop"], None if self.busy() else "Nothing is running")
        sa(a["mesh_clear"], nocomp or (None if self.mesh is not None else "There is no mesh"))
        pl = cur if cur is not None and cur.kind.startswith("plot.") else None
        for kind in ("plot.deform", "plot.selection", "plot.colorexpr"):
            ok = pl is not None and kind in REGISTRY[pl.kind].child_kinds(pl)
            sa(a["r_def" if kind == "plot.deform" else f"r_{kind}"],
               None if ok else f"Select a plot that takes a {REGISTRY[kind].title} attribute")
        nosol = None if any(d.kind == "dset.solution" for d in self.datasets()) else "Compute a study first"
        for name in ("pg3", "pg2", "pgp", "add_pg"):
            sa(a[name], nocomp)
        if comp is not None:
            sa(a["pg3"], None if self.sdim() == 3 else "A 3D plot group needs a 3D component")
        sa(a["pg_plot"], None if self._current_pg() is not None else "Select a plot group")
        for name in ("r_report",):
            sa(a[name], nosol)
        sa(a["sel_all"], None if self.active_sel is not None else "No selection list is active")
        sa(a["sel_clear"], None if self.active_sel is not None else "No selection list is active")
        sa(a["go_back"], None if self.tree.can_go(-1) else "No earlier node")
        sa(a["go_forward"], None if self.tree.can_go(1) else "No later node")
        sa(a["recovery_files"], None if recovery.orphans() else "No interrupted sessions to recover")

    def _current_pg(self) -> Node | None:
        cur = self.tree.current() if hasattr(self, "tree") else None
        if cur is None:
            return None
        return cur if cur.kind.startswith("pg") else cur.ancestor("pg")

    def _update_tabs(self):
        """Tabs of stages that need a component appear with it; the selected plot group gets its contextual tab."""
        has = self.model.component is not None
        for name in ("Geometry", "Materials", "Physics", "Mesh"):
            self.ribbon.set_tab_visible(name, has)
        pg = self._current_pg()
        self.ribbon.set_tab_visible("Plot Group", pg is not None, pg.label if pg is not None else None)

    # ================================================================== status bar
    def update_status(self):
        """Persistent state of the model in the status bar: geometry, mesh and solution of the current study, each
        with what it means and a click that goes to its node."""
        comp = self.model.component
        st = self._study_of(self.tree.current()) if comp is not None else None
        if comp is None:
            items = [(self.st_geo, "", "", ""), (self.st_mesh, "", "", ""), (self.st_sol, "", "", "")]
        else:
            if self.geo is None:
                geo = ("Geometry: not built", "stale", "Build it: Geometry > Build All (F8)")
            elif self.geo_stale:
                geo = ("Geometry: needs update", "stale", "The geometry or a parameter changed since it was built")
            else:
                nd = self.geo.count("domain")
                geo = (f"Geometry: {nd} domain{'s' if nd != 1 else ''}", "ok", "Built and up to date")
            if self.mesh is None:
                mesh = ("Mesh: not built", "", "Built when you compute, or with Mesh > Build Mesh")
            elif self.mesh_stale:
                mesh = ("Mesh: needs update", "stale", "The model changed since the mesh was built")
            else:
                mesh = (f"Mesh: {self.mesh.stats['elements']} elements", "ok", "Built and up to date")
            sol = self._solution_state(st)
            if len(self.model.components()) > 1:           # whose geometry and mesh: the active component's
                geo = (f"{comp.tag} {geo[0][0].lower()}{geo[0][1:]}",) + geo[1:]
                mesh = (f"{comp.tag} {mesh[0][0].lower()}{mesh[0][1:]}",) + mesh[1:]
            items = [(self.st_geo, *geo), (self.st_mesh, *mesh), (self.st_sol, *sol)]
        narrow = self.width() < 1100              # short forms; the tooltip keeps the whole sentence
        self.mem.setVisible(not narrow)
        for lab, text, state, tip in items:
            lab.setVisible(bool(text))
            short = text.split(":")[0] + {"ok": " ok", "stale": " *", "error": " !"}.get(state, " -")
            lab.setText(short if narrow and text else text)
            lab.setToolTip((text + "\n" if narrow else "") + tip + ("\nClick to go to the node." if text else ""))
            lab.setAccessibleName(text)
            if lab.property("state") != state:
                lab.setProperty("state", state)
                lab.style().unpolish(lab)
                lab.style().polish(lab)

    def _solution_state(self, st: Node | None):
        if st is None:
            return ("Study: none", "", "Add a study: Home > Add Study")
        folders = sorted(glob.glob(os.path.join(self.workdir, st.tag, "**", "sol_*_*", "meta.json"), recursive=True),
                         key=os.path.getmtime)
        if not folders:
            return (f"{st.label}: not computed", "", "Compute it: Ctrl+= or Study > Compute")
        try:
            with open(folders[-1], encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
        status = meta.get("status", "converged")
        if status == "failed":
            return (f"{st.label}: failed", "error", "The last computation failed: see Messages and Log")
        if status == "not converged":
            return (f"{st.label}: NOT CONVERGED", "error", "; ".join(meta.get("diagnostics", [])[:3]))
        mj = os.path.join(os.path.dirname(folders[-1]), "model.json")
        try:
            keep = st.child("study.solvers") is not None
            key = (mj, os.path.getmtime(mj), keep)
            cache = self.__dict__.setdefault("_solved_inputs", {})
            if key not in cache:
                with open(mj, encoding="utf-8") as f:
                    cache.clear()
                    cache[key] = _inputs(Model.from_dict(json.load(f)), keep)
            same = cache[key] == _inputs(self.model, keep)
        except (OSError, ValueError, KeyError):
            same = True                                   # an older solution without its model: unknown, say nothing
        if not same:
            return (f"{st.label}: out of date", "stale",
                    "The model's inputs changed since this solution was computed; results still show the computed "
                    "solution. Compute again to update it.")
        return (f"{st.label}: solved", "ok", "The solution matches the model's current inputs")

    def _status_click(self, which):
        comp = self.model.component
        if comp is None:
            return
        target = {"geo": comp.child("geometry"), "mesh": comp.child("mesh"),
                  "sol": self._study_of(self.tree.current())}.get(which)
        if target is not None:
            self.tree.select(target)

    def _update_job_status(self):
        if self.job is None:
            self.job_timer.stop()
            self.st_job.hide()
            self.st_stop.hide()
            return
        el = int(time.time() - self._job_started)
        txt = f"{self._job_title}  {el // 60:02d}:{el % 60:02d}"
        if self._job_stage:
            txt += f"  {self._job_stage}"
        if self._stopping:
            txt += "  - stopping..."
        self.st_job.setText(txt)
        self.st_job.show()
        self.st_stop.setVisible(not self._stopping)

    # ================================================================== focus and help
    def _panes(self):
        return [("tree", self.tree.tree), ("settings", self.settings),
                ("graphics", self.graphics.plotter.interactor if self.graphics.plotter is not None else self.graphics),
                ("messages", self.messages.view)]

    def focus_pane(self, name: str):
        """Move the keyboard focus to a window (and show it if it was hidden or tabbed away)."""
        dock = {"tree": self.dock_tree, "settings": self.dock_settings, "messages": self.dock_msg}.get(name)
        if dock is not None:
            dock.show()
            dock.raise_()
        if name == "settings":
            self.settings.focus_first()
            return
        w = dict(self._panes()).get(name)
        if w is not None:
            w.setFocus(Qt.ShortcutFocusReason)

    def cycle_focus(self, d: int):
        """Ctrl+Tab / Ctrl+Shift+Tab: the next or previous window (Model Builder, Settings, Graphics, Messages)."""
        panes = [(n, w) for n, w in self._panes() if w.isVisible()]
        if not panes:
            return
        fw = QApplication.focusWidget()
        cur = next((i for i, (_n, w) in enumerate(panes) if fw is not None and (fw is w or w.isAncestorOf(fw))), -1)
        self.focus_pane(panes[(cur + d) % len(panes)][0])

    def context_help(self):
        """F1: the documentation of the selected node (its section of the Reference)."""
        cur = self.tree.current()
        self.open_url(help_url(cur.kind if cur is not None and cur.kind != "root" else "root", DOCS_URL))

    def open_url(self, url: str):
        self.last_url = url
        self.statusBar().showMessage(f"Help: {url}", 5000)
        if self.no_modal:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        if not QDesktopServices.openUrl(QUrl(url)):
            self.messages.add(f"Documentation: {url}")

    def goto_tag(self, tag: str):
        """A link in Messages: select the node it names."""
        n = self.model.by_tag(tag)
        if n is not None:
            self.tree.select(n)
            self.focus_pane("tree")

    # ================================================================== recovery
    def checkpoint(self, force=False):
        """Write the unsaved model to this session's recovery folder (when it changed since the last checkpoint)."""
        if not self.dirty and not force:
            return
        if self._checkpoint_rev == self.rev and not force:
            return
        try:
            self.recovery.checkpoint(self.model.to_dict(), self.file_path, self.workdir, self._file_name())
            self._checkpoint_rev = self.rev
        except OSError as exc:
            self.statusBar().showMessage(f"Recovery checkpoint failed: {exc}", 5000)

    def offer_recovery(self) -> bool:
        """At start: models of sessions that ended without saving. Returns True if one was recovered."""
        if self.no_modal or not self.qs.value("offer_recovery", True, bool):
            return False
        found = recovery.orphans()
        if not found:
            return False
        return self.show_recovery_files(found)

    def show_recovery_files(self, found=None) -> bool:
        found = recovery.orphans() if found is None else found
        if not found:
            self.statusBar().showMessage("No recovery files.", 4000)
            return False
        dlg = QDialog(self)
        dlg.setWindowTitle("Recovery Files")
        dlg.resize(640, 300)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"{APP_NAME} closed without saving these models (a crash, a power cut or a killed "
                           "process). Open one to continue; it opens as an unsaved model."))
        lst = QListWidget()
        lst.setAccessibleName("Recovery files")
        for e in found:
            desc = f"{e.get('label') or 'Untitled'}  -  {e.get('saved', '').replace('T', ' ')}"
            if e.get("file_path"):
                desc += f"  -  {e['file_path']}"
            if e.get("job"):
                desc += f"  -  '{e['job']['title']}' was running"
            it = QListWidgetItem(icons.icon("app"), desc)
            it.setData(Qt.UserRole, e)
            lst.addItem(it)
        lst.setCurrentRow(0)
        v.addWidget(lst, 1)
        bb = QDialogButtonBox()
        b_open = bb.addButton("Open", QDialogButtonBox.AcceptRole)
        b_del = bb.addButton("Delete", QDialogButtonBox.DestructiveRole)
        bb.addButton("Later", QDialogButtonBox.RejectRole)
        chosen = {}

        def delete():
            it = lst.currentItem()
            if it is not None:
                recovery.discard(it.data(Qt.UserRole))
                lst.takeItem(lst.row(it))
            if lst.count() == 0:
                dlg.reject()
        b_open.clicked.connect(lambda: (chosen.__setitem__("e", lst.currentItem().data(Qt.UserRole))
                                        if lst.currentItem() else None, dlg.accept()))
        b_del.clicked.connect(delete)
        bb.rejected.connect(dlg.reject)
        lst.itemDoubleClicked.connect(lambda it: (chosen.__setitem__("e", it.data(Qt.UserRole)), dlg.accept()))
        v.addWidget(bb)
        if not dlg.exec() or "e" not in chosen:
            self.schedule_update()
            return False
        return self.recover(chosen["e"])

    def recover(self, entry: dict) -> bool:
        """Open a recovery checkpoint as an unsaved model (its file name kept, its solutions if they survived)."""
        if not self._confirm_discard():
            return False
        try:
            m, info = recovery.load(entry)
        except Exception as exc:  # noqa: BLE001
            self.messages.add(f"Could not recover {entry.get('dir')}: {exc}", "error")
            return False
        work = info.get("workdir") or ""
        work = work if work and os.path.isdir(work) else project.new_workdir()
        path = info.get("file_path") or None
        self._set_model(m, path, work)
        self.dirty = True                                 # never saved: Save writes it
        self._update_title()
        recovery.discard(entry)
        self.checkpoint(force=True)                       # it is this session's unsaved model now
        msg = f"Recovered the unsaved model of {info.get('saved', '').replace('T', ' ')}"
        if path:
            msg += f" (file {path}; the file itself still has the last saved version)"
        self.messages.add(msg + ". Save it to keep it.", "warning")
        if info.get("job"):
            self.messages.add(f"'{info['job']['title']}' was running when that session ended (started "
                              f"{info['job']['started'].replace('T', ' ')}); its solution may be missing or "
                              "incomplete. Compute the study again.", "warning")
        if m.component is not None:
            self.build_geometry()
        return True

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, f"Save changes to {self._file_name()}?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.file_save()
        return r == QMessageBox.Discard

    def file_new(self):
        if not self._confirm_discard():
            return
        dlg = NewDialog([(k, t) for k, (t, _f) in EXAMPLES.items()], self)
        if not dlg.exec():
            return
        if dlg.choice == "blank":
            self._set_model(B.new_model(None), None, project.new_workdir())
        elif dlg.choice == "example":
            self.open_example(dlg.example)
        else:
            w = ModelWizard(self)
            if not w.exec() or not w.sdim:
                return
            m = B.new_model(w.sdim, w.added, w.study)
            self._set_model(m, None, project.new_workdir())
            self.messages.add(f"New model: {w.sdim}, physics: {', '.join(w.added) or 'none'}, study: {w.study or 'none'}")
            g = m.component.child("geometry") if m.component else None
            if g is not None:
                self.tree.select(g)

    def open_example(self, key):
        title, fn = EXAMPLES[key]
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            m = fn()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, APP_NAME, f"Could not create example: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self._set_model(m, None, project.new_workdir())
        self.messages.add(f"Opened example: {title}")
        self.build_geometry()

    def file_open(self):
        if not self._confirm_discard():
            return
        fn, _ = QFileDialog.getOpenFileName(self, "Open", self.qs.value("last_dir", ""), f"{APP_NAME} model (*{FILE_EXT})")
        if fn:
            self.open_file(fn)

    def open_file(self, fn):
        try:
            m, work = project.load(fn)
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not open {fn}:\n{exc}")
            return
        self.qs.setValue("last_dir", os.path.dirname(fn))
        self._set_model(m, fn, work)
        self._add_recent(fn)
        self.messages.add(f"Opened file: {fn}")
        problems = getattr(m, "load_problems", [])
        if problems:                                   # a damaged archive, not one saved without solutions
            text = "\n".join(problems[:12]) + (f"\n... and {len(problems) - 12} more" if len(problems) > 12 else "")
            self.messages.add(f"{fn}: {len(problems)} stored file(s) missing or damaged; affected solutions "
                              "cannot be shown. Recompute them.", "error")
            QMessageBox.warning(self, APP_NAME, f"Some stored solution files are missing or damaged:\n\n{text}")
        elif getattr(m, "solutions_saved", "included") == "excluded":
            self.messages.add("This file was saved without solutions; compute the studies to see results.")
        if m.component is not None:
            self.build_geometry()

    def file_save(self) -> bool:
        if not self.file_path:
            return self.file_save_as()
        return self._save_to(self.file_path)

    def _save_to(self, fn: str, relabel=False) -> bool:
        """Write the model to *fn*; the window's file name, model label and saved revision change only once the
        file is written (a failed Save As leaves the previous file current)."""
        label = self.model.root.label
        if relabel:
            self.model.root.label = os.path.splitext(os.path.basename(fn))[0]
        try:
            project.save(self.model, fn, self.workdir, include_solutions=self.qs.value("save_solutions", True, bool))
        except Exception as exc:
            self.model.root.label = label
            self.messages.add(f"Save failed: {exc}", "error")
            if not self.no_modal:
                QMessageBox.critical(self, APP_NAME, f"Save failed: {exc}")
            return False
        self.file_path = fn
        self.dirty = False
        self.tree.file_name = self._file_name()
        self.tree.refresh()
        self._update_title()
        self.messages.add(f"Saved file: {fn}")
        self._add_recent(fn)
        self.recovery.discard()
        self._checkpoint_rev = None
        self.schedule_update()
        return True

    def file_save_as(self, fn: str | None = None) -> bool:
        if fn is None:
            fn, _ = QFileDialog.getSaveFileName(self, "Save As", os.path.join(self.qs.value("last_dir", ""),
                                                                               self._file_name()),
                                                f"{APP_NAME} model (*{FILE_EXT})")
        if not fn:
            return False
        if not fn.endswith(FILE_EXT):
            fn += FILE_EXT
        self.qs.setValue("last_dir", os.path.dirname(fn))
        return self._save_to(fn, relabel=True)

    def _add_recent(self, fn):
        rec = [r for r in (self.qs.value("recent", []) or []) if r != fn]
        rec.insert(0, fn)
        self.qs.setValue("recent", rec[:10])
        self._refresh_recent()

    def _refresh_recent(self):
        self.recent_menu.clear()
        for fn in self.qs.value("recent", []) or []:
            a = self.recent_menu.addAction(fn)
            a.triggered.connect(lambda _=False, f=fn: self._confirm_discard() and self.open_file(f))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if (ev.size().width() < 1100) != (ev.oldSize().width() < 1100):
            self.schedule_update()

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.stop_job()
        if self._maximized_state is not None:
            self.toggle_maximize()
        self.qs.setValue(f"window_state/{self.layout_mode}", self.saveState())
        self.qs.setValue("layout_mode", self.layout_mode)
        self.qs.setValue("window_geometry", self.saveGeometry())
        try:
            if self.graphics.plotter is not None:
                self.graphics.plotter.close()
        except Exception:
            pass
        project.discard_workdir(getattr(self, "workdir", None))
        self.recovery.discard()                       # closed on purpose: nothing to recover
        ev.accept()

    # ================================================================== undo / mutation
    def _push_undo(self, snap: str | None = None):
        """Record the state before an edit (*snap*, taken before it, or now) and move to a new revision."""
        self.undo_stack.append((snap if snap is not None else self.model.snapshot(), self.rev))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.rev = self._new_rev()
        self._update_title()
        self.schedule_update()

    def _restore(self, snap):
        cur = self.tree.current()
        tag = cur.tag if cur else None
        active = getattr(self.model, "_active", None)
        self.model.restore(snap)
        self.model._active = active
        self._all_stale()
        self.tree.set_model(self.model, self._file_name())
        n = self.model.by_tag(tag) if tag else None
        self.tree.refresh(select=n or self.model.root)
        self._update_geom_actions()
        self.schedule_update()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append((self.model.snapshot(), self.rev))
            snap, self.rev = self.undo_stack.pop()
            self._restore(snap)
            self._update_title()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.model.snapshot(), self.rev))
            snap, self.rev = self.redo_stack.pop()
            self._restore(snap)
            self._update_title()

    def _mark(self, node: Node):
        if node.kind == "params" or node.kind.startswith("func.") or node.kind == "variables":
            self._all_stale()
            return
        comp = self.model.component_of(node)
        if comp is not None and comp is not self.model.component:
            with self.model.scoped(comp):
                self._mark_in(node)
            return
        self._mark_in(node)

    def _mark_in(self, node: Node):
        if node.kind == "geometry" or node.kind.startswith("geom.") or node.ancestor("geom") is not None:
            self.geo_stale = True
            self.mesh_stale = True
        if node.kind == "mesh" or node.kind.startswith("mesh."):
            self.mesh_stale = True

    def set_prop(self, node: Node, key, value, rebuild=False):
        if node.props.get(key) == value:
            return
        self._push_undo()
        node.props[key] = value
        if (node.kind.startswith("cpl.") or node.kind == "plot.selection") and key == "level" and node.selection is not None:
            node.selection.level = value
            node.selection.entities = []
        if node.kind == "mesh.size" and key == "scope" and node.selection is not None and value != "entire":
            node.selection.level = value
            node.selection.entities = []
            node.selection.all = False
        self._mark(node)
        if key == "color" and node.kind == "material":
            self._show_geometry_for(node)
        if rebuild:
            self.settings.refresh()
        if node.kind.startswith(("plot.", "pg")) and self.current_plot is not None:
            pg = node if node.kind.startswith("pg") else node.ancestor("pg")
            if pg is self.current_plot or node.ancestor("pg") is self.current_plot:
                self.plot(self.current_plot)

    def rename(self, node: Node, text: str):
        text = text.strip()
        if not text or text == node.label:
            self.tree.update_node(node)
            return
        self._push_undo()
        node.label = text
        self.tree.update_node(node)

    def set_selection(self, node: Node, all_=None, entities=None, named=None):
        if node.selection is None:
            return
        self._push_undo()
        sel = node.selection
        if all_ is not None:
            sel.all = all_
            sel.named = ""
        if entities is not None:
            sel.entities = list(entities)
            sel.all = False
            sel.named = ""
        if named:
            sel.named, sel.all = named, False
            hit = self.geo.named.get(named) if self.geo is not None else None
            if hit is not None and hit[0] == sel.level:
                sel.entities = list(hit[1])
        self._sign(node)
        self._mark(node)
        self.settings.refresh()

    def _sign(self, node: Node):
        """A selection edited on the current geometry: remember what its entities are (so they are found again after
        the geometry changes), and forget entities it had lost."""
        sel = node.selection
        node.meta.pop("lost", None)
        geo = self.geo if not self.geo_stale else None
        if sel is None or geo is None or sel.all or sel.named:
            return
        sel.sig = {str(e): geo.signature(sel.level, e) for e in sel.entities if geo.signature(sel.level, e)}

    def remove_entities(self, node, ents):
        if node.selection is None or not ents:
            return
        self.set_selection(node, entities=[e for e in node.selection.entities if e not in ents])

    def highlight_entities(self, level, ents):
        if self.graphics.mode == "geometry" and self.active_sel is not None:
            self.graphics.set_selection_state(self.graphics.level, self.graphics.selected, ents)

    def overridden_entities(self, node: Node) -> set:
        """Entities of a physics feature overridden by a later feature of the same interface."""
        if node.selection is None or self.geo is None or node.parent is None or not node.parent.kind.startswith("physics."):
            return set()
        level = node.selection.level
        mine = set(node.selection.resolve(self.geo.numbers.get(level, [])))
        out = set()
        after = False
        for sib in node.parent.children:
            if sib is node:
                after = True
                continue
            if after and sib.enabled and sib.selection is not None and sib.selection.level == level:
                # a later boundary feature replaces the condition (build_sif); domain features merge key by key, so
                # only a later feature of the same type overrides (Reactions adds to Transport Properties)
                if level != "boundary" and sib.kind != node.kind:
                    continue
                out |= mine & set(sib.selection.resolve(self.geo.numbers.get(level, [])))
        return out

    def on_entity_clicked(self, level, ent, ctrl):
        node = self.active_sel
        if node is None or node.selection is None or node.selection.level != level:
            return
        self._push_undo()
        ents = list(node.selection.entities)
        if REGISTRY[node.kind].single_select:
            ents = [ent]
        elif ent in ents:
            ents.remove(ent)
        else:
            ents.append(ent)
            ents.sort()
        node.selection.entities = ents
        node.selection.all = False
        node.selection.named = ""          # editing the list makes it a manual selection, as in COMSOL
        self._sign(node)
        self._mark(node)
        self.graphics.set_selection_state(level, ents)
        self.settings.refresh()

    def on_entities_set(self, level, ents):
        """Select All / Clear Selection in the graphics: the active selection becomes *ents* in one edit."""
        node = self.active_sel
        if node is None or node.selection is None or node.selection.level != level:
            return
        if REGISTRY[node.kind].single_select and len(ents) > 1:
            return
        ents = sorted(int(e) for e in ents)
        if ents == sorted(node.selection.entities) and not node.selection.all and not node.selection.named:
            return
        self._push_undo()
        node.selection.entities = ents
        node.selection.all = False
        node.selection.named = ""
        self._sign(node)
        self._mark(node)
        self.graphics.set_selection_state(level, ents)
        self.settings.refresh()

    def activate_selection(self, node: Node | None, show: Node | None = None):
        self.active_sel = node
        self.schedule_update()
        target = node or show
        if target is None or target.selection is None:
            return
        geo = self.geometry_for(target)
        if geo is None:
            return
        level = target.selection.level
        if geo.sdim == 2 and level == "edge":
            level = "boundary"
        ents = self._display_entities(target, geo, level)
        if self.graphics.mode != "geometry" or self.graphics.geo is not geo:
            self.graphics.show_geometry(geo, level if node else None, ents,
                                        labels=level if self._needs_context(target) else None)
        else:
            self.graphics.set_selection_state(level if node else None, ents)
            self.graphics.level = level if node else None

    def _display_entities(self, node: Node, geo, level) -> list:
        """Entities to highlight: the picked selection, else numbers typed in the feature's fields."""
        ents = node.selection.resolve(geo.numbers.get(level, []))
        if ents:
            return ents
        if node.kind == "geom.workplane" and node.get("ptype") == "face":
            return [int(node.get("face", 0) or 0)]
        if node.kind in ("geom.fillet2d", "geom.chamfer2d"):
            return [int(float(x)) for x in str(node.get("points", "")).replace(",", " ").split() if x.strip()]
        return ents

    # -- context geometry (what a geometry feature "sees": the sequence before it, or its work-plane sketch)
    def _needs_context(self, node: Node | None) -> bool:
        if node is None or self.model.component is None:
            return False
        if node.kind == "geom.workplane":
            return node.get("ptype") == "face"
        return node.parent is not None and node.parent.kind == "geom.workplane"

    def geometry_for(self, node: Node | None):
        if not self._needs_context(node):
            return self.geo
        key = (id(node), hash(self.model.snapshot()))
        cache = getattr(self, "_ctx_geo", {})
        if key not in cache:
            from ..core.geometry import build_context_geometry
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                cache = {key: build_context_geometry(self.model, node)}   # keep only the latest
            except Exception as exc:
                self.messages.add(f"{node.label}: {exc}", "error")
                return self.geo
            finally:
                QApplication.restoreOverrideCursor()
            self._ctx_geo = cache
        return cache[key]

    # -- tree operations
    def create_node(self, kind: str, parent: Node, index=None, select=True, **props) -> Node:
        self._push_undo()
        nt = REGISTRY[kind]
        if kind.startswith("geom.") and parent.kind == "geometry" and index is None:
            index = next((i for i, c in enumerate(parent.children) if c.kind == "geom.finalize"), len(parent.children))
        n = self.model.create(kind, parent, index=index, **props)
        if n.selection is not None:
            if kind == "mesh.distribution":
                n.selection.level = "edge" if self.sdim() == 3 else "boundary"
            if kind.startswith("cpl."):
                n.props["opname"] = n.tag          # aveop1, intop1, ... as in COMSOL
            if kind.startswith("mp.") or kind == "mesh.ftet" or kind == "mesh.ftri" or kind == "mesh.fquad":
                n.selection.all = True
            if kind == "material":
                mats = [c for c in parent.children if c.kind == "material" and c is not n]
                n.selection.all = not mats
        if kind.startswith("plot.") and "data" in n.props:
            n.props["data"] = "fromparent"
        if (kind in ("pg3d", "pg2d", "pg1d", "pgpolar", "plot.polar") or kind.startswith(("eval.", "export.", "dset."))) \
                and kind != "dset.solution":
            ds = [d for d in self.datasets() if d.kind == "dset.solution" and d is not n]
            if ds and "data" in n.props:
                n.props["data"] = ds[-1].tag
        if kind == "plot.directivity":
            ds = [d for d in self.datasets() if d.kind == "dset.solution"]
            if ds:
                n.props["data"] = ds[-1].tag
        if kind == "plot.pointgraph":
            cp = [d for d in self.datasets() if d.kind == "dset.cutpoint"]
            if cp:
                n.props["data"] = cp[-1].tag
        if kind == "plot.tablegraph":
            tbls = self.tables()
            if tbls:
                n.props["table"] = tbls[-1].tag
        if kind == "plot.selection" and n.selection is not None:
            lvl = "domain" if parent.kind in ("plot.volume", "plot.mesh", "plot.isosurface", "plot.slice") or \
                (parent.kind in ("plot.surface", "plot.contour", "plot.arrow") and self.sdim() == 2) else "boundary"
            n.props["level"] = lvl
            n.selection.level = lvl
            n.selection.all = False
        if self.geo is not None and kind in ("plot.annotation", "dset.cutplane", "plot.polar"):
            b = self.geo.bbox
            c = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2]
            if kind == "plot.annotation":
                n.props.update({"x": f"{c[0]:.6g}", "y": f"{c[1]:.6g}", "z": f"{c[2]:.6g}"})
            elif kind == "dset.cutplane":
                n.props.update({"px": f"{c[0]:.6g}", "py": f"{c[1]:.6g}", "pz": f"{c[2]:.6g}"})
            else:
                n.props["radius"] = f"{0.45 * max(b[3] - b[0], b[4] - b[1]):.6g}"
        if kind in ("dset.cutline",) and self.geo is not None:
            b = self.geo.bbox
            n.props.update({"x0": f"{b[0]:.6g}", "y0": f"{(b[1] + b[4]) / 2:.6g}", "z0": f"{(b[2] + b[5]) / 2:.6g}",
                            "x1": f"{b[3]:.6g}", "y1": f"{(b[1] + b[4]) / 2:.6g}", "z1": f"{(b[2] + b[5]) / 2:.6g}"})
        if kind.startswith("study.") and "physics" in n.props:
            n.props["physics"] = {ph.tag: True for ph in B.all_physics(self.model)}
        if kind in ("study.stationary", "study.time", "study.eigen", "study.freq"):
            steps = [c for c in parent.children if c.kind.startswith("study.") and c.kind not in ("study.sweep", "study.solvers")]
            n.label = f"Step {len(steps)}: {nt.title}"
        self._mark(n)
        self.tree.refresh(select=n if select else None)
        self._update_geom_actions()
        return n

    def delete_node(self, node: Node | None):
        if node is None or node.parent is None:
            return
        nt = REGISTRY.get(node.kind)
        if nt is not None and not nt.deletable or node.meta.get("default"):
            self.statusBar().showMessage(f"'{node.label}' cannot be deleted", 4000)
            return
        from ..core import deps
        users = [u for u in deps.used_by(self.model, node) if node not in u.path()]
        self._push_undo()
        parent = node.parent
        idx = parent.children.index(node)
        parent.remove(node)
        if users:
            self.messages.add(f"Deleted '{node.label}', which " + ", ".join(f"'{u.label}'" for u in users[:6])
                              + (" and more" if len(users) > 6 else "") + " used: check them (Undo restores it)",
                              "warning", node=users[0])
        if node.kind.startswith("physics."):
            for st in self.model.studies():
                for s in st.children:
                    if isinstance(s.props.get("physics"), dict):
                        s.props["physics"].pop(node.tag, None)
        if node.kind == "study":
            ds = B.solution_dataset(self.model, node, create=False)
            if ds is not None:
                ds.parent.remove(ds)
        self._mark(node)
        nxt = parent.children[min(idx, len(parent.children) - 1)] if parent.children else parent
        self.tree.refresh(select=nxt)
        self._update_geom_actions()

    def duplicate_node(self, node: Node | None):
        if node is None or node.parent is None or node.kind in ("root", "global", "results", "geometry", "mesh",
                                                                 "materials", "geom.finalize") or node.meta.get("default"):
            return
        self._push_undo()
        d = Node.from_dict(json.loads(json.dumps(node.to_dict())))
        for n in d.walk():
            n.tag = self.model.new_tag(REGISTRY[n.kind].prefix if n.kind in REGISTRY else "n")
            n.meta.pop("default", None)
            n.meta.pop("default_plot", None)
        d.label = self.model.new_label(re.sub(r"\s+\d+$", "", node.label)) if REGISTRY[node.kind].numbered else node.label
        node.parent.add(d, node.parent.children.index(node) + 1)
        self._mark(d)
        self.tree.refresh(select=d)

    def set_enabled(self, node: Node | None, on: bool):
        if node is None:
            return
        nt = REGISTRY.get(node.kind)
        if nt is not None and not nt.can_disable or node.meta.get("default"):
            return
        self._push_undo()
        node.enabled = on
        self._mark(node)
        self.tree.refresh(select=node)

    def move_node(self, node: Node | None, d: int):
        if node is None or node.parent is None:
            return
        sib = node.parent.children
        i = sib.index(node)
        j = i + d
        if not (0 <= j < len(sib)) or sib[j].kind == "geom.finalize" or sib[j].meta.get("default") or node.meta.get("default"):
            return
        self._push_undo()
        sib[i], sib[j] = sib[j], sib[i]
        self._mark(node)
        self.tree.refresh(select=node)

    # ================================================================== ctx API for SettingsWindow
    @property
    def geometry(self):
        return self.geo

    def sdim(self):
        return space_dim(self.model.component)

    def is_axi(self):
        return is_axisymmetric(self.model.component)

    def length_unit(self):
        g = self.model.component.child("geometry") if self.model.component else None
        return g.get("unit", "m") if g else "m"

    def parameters(self):
        try:
            return self.model.parameters()
        except Exception:
            return {}

    def variable_names(self):
        out = []
        comp = self.model.component
        defs = comp.child("definitions") if comp else None
        for n in (defs.children if defs else []):
            for r in n.props.get("table", []) or []:
                if r.get("name"):
                    out.append(r["name"])
        return out

    def function_names(self):
        g = self.model.global_defs
        return [n.get("fname", n.tag) for n in (g.children if g else []) if n.kind.startswith("func.")]

    def object_names_before(self, node):
        g = node.ancestor("geometry") if node.parent is not None else None
        if g is None:
            return []
        return object_names_before(g, node)

    def physics_nodes(self):
        """Every component's physics interfaces (the study steps' Physics and Variables Selection)."""
        return B.all_physics(self.model)

    def step_kind_name(self, step):
        return {"study.stationary": "Stationary", "study.time": "Time dependent", "study.eigen": "Eigenfrequency",
                "study.freq": "Frequency domain"}.get(step.kind, "")

    def required_matprops(self, mat: Node) -> list[str]:
        req = []
        doms = set(mat.selection.resolve(self.geo.numbers["domain"])) if (self.geo and mat.selection) else None
        for ph in B.physics_nodes(self.model):
            if not ph.enabled:
                continue
            pd = physics_def(ph.kind)
            for f in ph.children:
                if not f.enabled or f.selection is None or f.selection.level != "domain":
                    continue
                for k in MATPROPS:
                    if f.props.get(f"{k}_src") == "mat" and k not in req:
                        req.append(k)
        mp = self.model.component.child("multiphysics") if self.model.component else None
        for c in (mp.children if mp else []):
            for k in MATPROPS:
                if c.props.get(f"{k}_src") == "mat" and k not in req:
                    if c.kind == "mp.nitf" and not c.get("buoy"):
                        continue
                    req.append(k)
        return req

    def header_actions(self, node: Node):
        k = node.kind
        if k == "geometry" or k == "geom.finalize":
            return [("Build All Objects", "build_all", "Build the whole geometry sequence (F8)")]
        if k.startswith("geom."):
            return [("Build Selected", "build_selected", "Build up to this node (F7)"),
                    ("Build All Objects", "build_all", "Build the whole geometry sequence (F8)")]
        if k == "mesh" or k.startswith("mesh."):
            return [("Build All", "mesh", "Build the mesh (F8)")]
        if k == "study" or k.startswith("study."):
            return [("Compute", "compute", "Compute the study")]
        if k.startswith("pg") or k.startswith("plot."):
            return [("Plot", "plot", "Plot")]
        if k.startswith("eval."):
            return [("Evaluate", "compute", "Evaluate into a table")]
        if k.startswith("export."):
            return [("Export", "export", "Write the file")]
        if k == "materials":
            return [("Add Material from Library", "materials", "Open the Add Material window")]
        return []

    def on_settings_action(self, name, node):
        if name in ("Build Selected",):
            self.build_selected()
        elif name in ("Build All Objects",):
            self.build_geometry()
        elif name == "Build All":
            self.build_mesh()
        elif name == "Compute":
            self.compute(self._study_of(node))
        elif name == "Plot":
            pg = node if node.kind.startswith("pg") else node.ancestor("pg")
            if pg:
                self.plot(pg)
        elif name == "Evaluate":
            self.evaluate_node(node)
        elif name == "Export":
            self.export_node(node)
        elif name == "Add Material from Library":
            self.show_material_browser()

    def datasets(self):
        res = self.model.results
        d = res.child("results.datasets") if res else None
        return [c for c in (d.children if d else [])]

    def plot_groups(self):
        res = self.model.results
        return [c for c in (res.children if res else []) if c.kind.startswith("pg")]

    def _dataset_for(self, node: Node) -> Node | None:
        tag = node.props.get("data", "")
        if tag == "fromparent" or not tag:
            pg = node.ancestor("pg") if not node.kind.startswith("pg") else node
            tag = pg.props.get("data", "") if pg is not None else ""
        d = self.model.by_tag(tag) if tag else None
        if d is None:
            ds = [x for x in self.datasets() if x.kind == "dset.solution"]
            d = ds[0] if ds else None
        return d

    def tables(self):
        res = self.model.results
        t = res.child("results.tables") if res else None
        return list(t.children) if t else []

    def solution_of(self, dset: Node | None):
        """Solution (or a Solution-like derived view: cut plane, mirror, extrusion) for a dataset node."""
        view, _rev = self.view_of(dset)
        return view

    def view_of(self, dset: Node | None):
        """(Solution-like view, revolution settings or None) for a dataset node, cached per dataset settings."""
        if dset is None:
            return None, None
        if dset.kind == "dset.solution":
            return self._solution_for(dset), None
        key = (dset.tag, json.dumps([(d.tag, d.props) for d in self.datasets()], default=str, sort_keys=True),
               tuple(id(s) for s in self.sol_cache.values()), json.dumps(self.parameters(), default=str))
        cache = self.__dict__.setdefault("_view_cache", {})
        if key not in cache:
            try:
                cache.clear()
                cache[key] = resolve_dataset(self.model, dset, self._solution_for,
                                             lambda d: self.sweep_solutions(d))
            except Exception as exc:
                self.messages.add(f"{dset.label}: {exc}", "error")
                return None, None
        return cache[key]

    def _solution_for(self, dset: Node | None) -> Solution | None:
        if dset is None:
            return None
        st = self.model.by_tag(dset.get("study", ""))
        if st is None:
            return None
        comp = B.dataset_component(self.model, dset)
        folders = sorted((f for f in glob.glob(os.path.join(self._comp_dir(st, comp), "sol_*_*"))
                          if os.path.isfile(os.path.join(f, "meta.json"))),      # skip solutions still being computed
                         key=lambda p: tuple(int(x) for x in re.findall(r"\d+", os.path.basename(p))))
        if not folders:
            return None
        sweep = int(dset.get("sweepidx", -1) if dset.get("sweepidx", -1) not in ("", None) else -1)
        if sweep < 0:   # COMSOL default: the last parameter value of a sweep
            sweep = max(int(re.findall(r"\d+", os.path.basename(f))[-2]) for f in folders)
        chosen = [f for f in folders if os.path.basename(f).startswith(f"sol_{sweep}_")] or folders
        folder = chosen[-1]
        with self.model.scoped(comp):
            return self._load_solution(folder)

    def _comp_dir(self, st: Node, comp: Node | None) -> str:
        from ..core.study_runner import component_dir
        sd = os.path.join(self.workdir, st.tag)
        return component_dir(self.model, sd, comp.tag) if comp is not None else sd

    def _load_solution(self, folder):
        if folder not in self.sol_cache:
            try:
                sol = Solution(folder)
            except Exception as exc:
                self.messages.add(f"Could not load solution {folder}: {exc}", "error")
                return None
            status = sol.meta.get("status", "converged")
            if status == "failed":                     # kept on disk for its log, never shown as a result
                return None
            if status == "not converged":
                self.messages.add("Solution NOT CONVERGED (kept by the study settings): "
                                  + "; ".join(sol.meta.get("diagnostics", [])), "warning",
                                  node=os.path.basename(os.path.dirname(folder)))
            self.sol_cache[folder] = sol
        sol = self.sol_cache[folder]
        from ..core.nodes import bind_results
        sol = bind_results(self.model, sol, self.geo if not self.geo_stale else None)
        changed = getattr(sol, "params_changed", [])
        if changed and folder not in self.__dict__.setdefault("_stale_noted", set()):
            self._stale_noted.add(folder)
            self.messages.add(f"The solution was computed with earlier values of {', '.join(changed)}; results use "
                              "those values. Compute the study again to see the current model.", "warning")
        return sol

    def sweep_info(self, dset: Node | None):
        """(parameter name, unit) of the dataset study's parametric sweep."""
        if dset is not None and dset.kind != "dset.solution":
            dset = root_dataset(self.model, dset) or dset
        st = self.model.by_tag(dset.get("study", "")) if dset is not None else None
        sw = B.study_sweep(st) if st is not None else None
        if sw is None:
            return None, ""
        return sw.get("pname", ""), str(sw.get("punit", "") or "")

    def sweep_solutions(self, dset: Node | None) -> list:
        """[(parameter value, Solution)] over a parametric sweep (last study step), or [] without a sweep."""
        if dset is None:
            return []
        if dset.kind != "dset.solution":
            dset = root_dataset(self.model, dset) or dset
        st = self.model.by_tag(dset.get("study", ""))
        if st is None:
            return []
        comp = B.dataset_component(self.model, dset)
        out = []
        for f in glob.glob(os.path.join(self._comp_dir(st, comp), "sol_*_*")):
            if not os.path.isfile(os.path.join(f, "meta.json")):
                continue
            vi, si = (int(x) for x in re.findall(r"\d+", os.path.basename(f))[-2:])
            out.append((vi, si, f))
        if not out:
            return []
        last = max(si for _vi, si, _f in out)
        res = []
        for vi, si, f in sorted(out):
            if si != last:
                continue
            with self.model.scoped(comp):
                sol = self._load_solution(f)
            if sol is not None and sol.meta.get("sweep_param"):
                res.append((sol.meta.get("sweep_value"), sol))
        return res

    def frame_labels(self, node):
        sol = self.solution_of(self._dataset_for(node))
        return sol.frame_labels() if sol else []

    def frame_kind_label(self, node):
        sol = self.solution_of(self._dataset_for(node))
        if sol is None:
            return ""
        return {"study.time": "Time (" + sol.meta.get("tunit", "s") + "):", "study.eigen": "Eigenfrequency (Hz):",
                "study.freq": "Parameter value (freq (Hz)):"}.get(sol.kind, "")

    def result_variables(self, node, vector=False):
        sol = self.solution_of(self._dataset_for(node))
        if sol is None:
            return []
        av = sol.available()
        if not vector:
            from ..core.acoustics import EFC_LABELS, INCIDENT_LABELS
            root = sol
            while not isinstance(root, Solution) and hasattr(root, "base"):
                root = root.base
            for tag in sorted(getattr(root, "efc", {}) or {}):          # exterior field (acpr.efc1.Lp_pext, ...)
                av = av + [(f"acpr.{tag}.{nm}", lbl, u, "scalar") for nm, lbl, u in EFC_LABELS]
            if any(nm in (getattr(root, "variables", {}) or {}) for nm, _l, _u in INCIDENT_LABELS):
                av = av + [(a, lbl, u, "scalar") for a, lbl, u in INCIDENT_LABELS]
        return [v for v in av if (v[3] == "vector") == vector]

    def result_units(self, node):
        sol = self.solution_of(self._dataset_for(node))
        expr = node.get("expr", "") or (default_expr(sol) if sol else "")
        if sol is None:
            return [node.get("unit", "")]
        _l, si = sol.describe(expr)
        return compatible_units(si) if si else [""]

    def extra_sections(self, node: Node):
        out = []
        if node.kind == "mesh" and self.mesh is not None and not self.mesh_stale:
            st = self.mesh.stats
            txt = (f"Number of elements: {st['elements']}\nBoundary elements: {st['boundary_elements']}\n"
                   f"Number of nodes: {st['nodes']}\nElement order: {st['order']}\n"
                   f"Minimum element quality: {st['min_quality']:.4g}\nAverage element quality: {st['avg_quality']:.4g}\n"
                   + "\n".join(f"{k}: {v}" for k, v in st["types"].items()))
            out.append(("Statistics", QLabel(txt)))
        if node.kind == "geometry" and self.geo is not None:
            g = self.geo
            txt = (f"Domains: {g.count('domain')}\nBoundaries: {g.count('boundary')}\n"
                   + (f"Edges: {g.count('edge')}\n" if g.sdim == 3 else "") + f"Points: {g.count('point')}\n"
                   f"Bounding box: [{g.bbox[0]:.4g}, {g.bbox[3]:.4g}] x [{g.bbox[1]:.4g}, {g.bbox[4]:.4g}]"
                   + (f" x [{g.bbox[2]:.4g}, {g.bbox[5]:.4g}]" if g.sdim == 3 else "") + f" {g.unit}")
            out.append(("Geometry Statistics", QLabel(txt)))
        from ..core import deps
        users = deps.used_by(self.model, node) if node.parent is not None else []
        if users and node.kind not in ("component", "geometry", "materials", "definitions", "mesh"):
            lab = QLabel(", ".join(f'<a href="{u.tag}">{u.label}</a>' for u in users[:30])
                         + (f" and {len(users) - 30} more" if len(users) > 30 else ""))
            lab.setWordWrap(True)
            lab.linkActivated.connect(self.goto_tag)
            lab.setToolTip("The nodes that use this one (its parameters, functions, operators, objects, solution...)")
            out.append(("Used By", lab))
        return out

    # ================================================================== selection / graphics updates
    def on_node_selected(self, node: Node):
        comp = self.model.component_of(node)            # the active component: the selected node's (COMSOL)
        if comp is not None and comp is not self.model.component:
            self.model.set_active(comp)
        self.active_sel = None
        self.settings.show_node(node)
        self._update_geom_actions(node)
        self.schedule_update()
        k = node.kind
        if k.startswith("pg"):
            self.plot(node)
        elif k == "plot.selection":           # pick the entities on the geometry, then Plot
            self._show_geometry_for(node)
        elif k.startswith("plot."):
            pg = node.ancestor("pg")
            if pg is not None and pg is not self.current_plot:
                self.plot(pg)
        elif k == "mesh" or k.startswith("mesh."):
            if self.mesh is not None and not self.mesh_stale:
                self.show_mesh()
            else:
                self._show_geometry_for(node)
        elif k in ("study", "results") or k.startswith(("study.", "results.", "eval.", "export.", "dset.", "table")):
            pass
        elif self.model.component is not None:
            self._show_geometry_for(node)

    def _show_geometry_for(self, node: Node):
        if self.model.component is None:
            return
        if self._needs_context(node):
            geo = self.geometry_for(node)
            if geo is not None:
                self.gstack.setCurrentIndex(0)
                level = node.selection.level if node.selection is not None else None
                ents = self._display_entities(node, geo, level) if level else []
                self.graphics.show_geometry(geo, None, ents, labels=level)
                self.current_plot = None
            return
        if self.geo is None or (self.geo_stale and not node.kind.startswith("geom.") and node.kind != "geometry"):
            self.build_geometry(silent=True, then=lambda: self._show_geometry_for(node))
            return
        if self.geo is None:
            return
        colors = None
        if node.kind in ("materials", "material"):
            colors = {}
            mats = self.model.component.child("materials")
            for m in (mats.children if mats else []):
                if m.enabled and m.selection is not None:
                    for d in m.selection.resolve(self.geo.numbers["domain"]):
                        colors[d] = m.props.get("color", [190, 190, 190])
        self.gstack.setCurrentIndex(0)
        if node.kind == "geom.boxsel" and node.tag in self.geo.named:      # what the Box Selection picks
            level, ents = self.geo.named[node.tag]
            self.graphics.show_geometry(self.geo, None, [], colors)
            self.graphics.set_selection_state(level, ents)
        elif node.selection is not None:
            level = node.selection.level
            ents = node.selection.resolve(self.geo.numbers.get(level, []))
            self.graphics.show_geometry(self.geo, None, ents, colors)
        else:
            self.graphics.show_geometry(self.geo, None, [], colors)
        self.current_plot = None

    def show_mesh(self, quality=False):
        if self.mesh is None or self.mesh_stale:
            self.build_mesh(then=lambda: self.show_mesh(quality))
            return
        self.gstack.setCurrentIndex(0)
        self.graphics.show_mesh(self.mesh, quality=quality)
        self.current_plot = None

    # ================================================================== context menu
    def context_menu(self, node: Node, pos):
        m = self.build_context_menu(node)
        if not self.no_modal:
            m.exec(pos)

    def build_context_menu(self, node: Node) -> QMenu:
        m = QMenu(self)
        m.setToolTipsVisible(True)
        k = node.kind
        nt = REGISTRY.get(k)

        def add_kind(menu, kind, parent=node):
            t = REGISTRY[kind]
            sd = self.sdim() if self.model.component else 3
            a = menu.addAction(icons.icon(t.icon), type_title(kind, sd))
            a.triggered.connect(lambda _=False: self.create_node(kind, parent))
            if sd not in t.dims and kind.startswith("geom.") and not node.kind == "geom.workplane":
                a.setEnabled(False)

        if k == "root":
            sub = m.addMenu(icons.icon("component"), "Add Component")
            for s, lbl in (("3D", "3D"), ("2Daxi", "2D Axisymmetric"), ("2D", "2D")):
                a = sub.addAction(lbl)
                a.triggered.connect(lambda _=False, ss=s: self.add_component(ss))
                a.setEnabled(True)
            sub = m.addMenu(icons.icon("study"), "Add Study")
            for key, (t, ic) in STUDY_TYPES.items():
                a = sub.addAction(icons.icon(ic), t)
                a.triggered.connect(lambda _=False, kk=key: self.add_study(kk))
        elif k == "component":
            sub = m.addMenu(icons.icon("physics_list"), "Add Physics")
            self._fill_add_physics(sub)
            sub = m.addMenu(icons.icon("multiphysics"), "Multiphysics Couplings")
            self._fill_multiphysics(sub)
        elif k.startswith("physics."):
            dm = m.addMenu(icons.icon("feature_domain"), "Domains")
            bm = m.addMenu(icons.icon("feature_boundary"), "Boundaries")
            for kind in nt.child_kinds(node):
                if REGISTRY[kind].selection == "domain":
                    add_kind(dm, kind)
                else:
                    add_kind(bm, kind)
        elif k == "geometry" or k == "geom.workplane":
            kinds = nt.child_kinds(node)
            groups = [("Primitives", [x for x in kinds if x in ("geom.block", "geom.sphere", "geom.cylinder", "geom.cone",
                                                                 "geom.torus", "geom.ellipsoid", "geom.rectangle", "geom.circle",
                                                                 "geom.ellipse", "geom.polygon", "geom.paramcurve")]),
                      ("Booleans and Partitions", [x for x in kinds if x in ("geom.union", "geom.difference", "geom.intersection")]),
                      ("Transforms", [x for x in kinds if x in ("geom.move", "geom.rotate", "geom.scale", "geom.mirror",
                                                                 "geom.array", "geom.copy")]),
                      ("Conversions", [x for x in kinds if x in ("geom.fillet3d", "geom.chamfer3d", "geom.fillet2d",
                                                                  "geom.chamfer2d", "geom.tosolid")])]
            listed = {x for _t, ks in groups for x in ks}
            groups.append(("Other", [x for x in kinds if x not in listed]))   # work plane, extrude, import, selections, ...
            for title, ks in groups:
                if not ks:
                    continue
                sub = m.addMenu(title)
                for kind in ks:
                    add_kind(sub, kind)
            m.addSeparator()
            m.addAction(icons.icon("build_all"), "Build All Objects", self.build_geometry)
        elif nt is not None and nt.child_kinds(node):
            for kind in nt.child_kinds(node):
                add_kind(m, kind)
        if k.startswith("geom.") and k not in ("geom.workplane",):
            m.addAction(self.act["build_selected"])
            m.addAction(icons.icon("build_all"), "Build All Objects", self.build_geometry)
        if k == "materials":
            m.addSeparator()
            m.addAction(icons.icon("materials"), "Add Material from Library", self.show_material_browser)
            m.addAction(icons.icon("material"), "Blank Material", self.add_blank_material)
        if k == "mesh":
            m.addSeparator()
            m.addAction(icons.icon("mesh"), "Build All", self.build_mesh)
            m.addAction(icons.icon("mesh_stats"), "Statistics", self.mesh_statistics)
            m.addAction(icons.icon("clear"), "Clear Mesh", self.clear_mesh)
        if k == "study" or k.startswith("study."):
            m.addSeparator()
            m.addAction(icons.icon("compute"), "Compute", lambda: self.compute(self._study_of(node)))
            m.addAction(icons.icon("sif"), "Show Elmer Input File", lambda: self.show_sif(self._study_of(node)))
            if k == "study":
                m.addAction(icons.icon("solver_config"), "Show Default Solver", self.show_default_solver)
        if k.startswith("pg"):
            m.addSeparator()
            m.addAction(icons.icon("plot"), "Plot", lambda: self.plot(node))
            m.addAction(icons.icon("image_export"), "Image Snapshot...", self.graphics.snapshot_dialog)
        if k.startswith("eval."):
            m.addAction(icons.icon("compute"), "Evaluate", lambda: self.evaluate_node(node))
        if k == "results":
            m.addSeparator()
        if node.parent is not None and k not in ("global", "results"):
            # the ribbon's own commands (same shortcut, availability and reason when unavailable); they act on the
            # current node, which the right-click made this node
            self.update_actions()
            m.addSeparator()
            if nt is None or nt.renamable:
                m.addAction(self.act["rename"])
            if nt is not None and nt.deletable and not node.meta.get("default"):
                m.addAction(self.act["duplicate"])
                m.addAction(self.act["delete"])
            if nt is not None and nt.can_disable and not node.meta.get("default"):
                m.addAction(self.act["disable" if node.enabled else "enable"])
            m.addAction(self.act["move_up"])
            m.addAction(self.act["move_down"])
        m.addSeparator()
        m.addAction(self.act["help"])
        return m

    # ================================================================== ribbon helpers
    def _current_component(self):
        return self.model.component

    def _current_physics(self) -> Node | None:
        n = self.tree.current()
        while n is not None:
            if n.kind.startswith("physics."):
                return n
            n = n.parent
        ph = B.physics_nodes(self.model)
        return ph[0] if ph else None

    def _fill_features(self, menu: QMenu, level):
        menu.clear()
        ph = self._current_physics()
        if ph is None:
            menu.addAction("(no physics interface)").setEnabled(False)
            return
        for kind in REGISTRY[ph.kind].child_kinds(ph):
            t = REGISTRY[kind]
            if t.selection != level:
                continue
            a = menu.addAction(icons.icon(t.icon), t.title)
            a.triggered.connect(lambda _=False, kk=kind, p=ph: self.create_node(kk, p))

    def _fill_add_physics(self, menu: QMenu):
        menu.clear()
        comp = self.model.component
        if comp is None:
            menu.addAction("(add a component first)").setEnabled(False)
            return
        sd = comp.get("sdim")
        from ..core.physics import WIZARD_TREE
        for cat, subs in WIZARD_TREE:
            sub = None
            for _s, pids in subs:
                for pid in pids:
                    if pid.startswith("@") or sd not in PHYSICS[pid].dims:
                        continue
                    if sub is None:
                        sub = menu.addMenu(cat)
                    pd = PHYSICS[pid]
                    a = sub.addAction(icons.icon(pd.icon), f"{pd.name} ({pd.id})")
                    a.triggered.connect(lambda _=False, p=pid: self.add_physics(p))

    def _fill_multiphysics(self, menu: QMenu):
        menu.clear()
        ids = {physics_def(p.kind).id for p in B.physics_nodes(self.model)}
        for kind, cp in COUPLINGS.items():
            a = menu.addAction(icons.icon("multiphysics"), cp["title"])
            a.setEnabled(set(cp["requires"]) <= ids)
            a.setToolTip("Requires " + " + ".join(PHYSICS[r].name for r in cp["requires"]))
            a.triggered.connect(lambda _=False, k=kind: self.add_coupling(k))

    def goto_kind(self, kind):
        n = self.model.root.find(kind)
        if n is not None:
            self.tree.select(n)

    def add_child_kind(self, parent_kind, kind):
        p = self.model.root.find(parent_kind)
        if p is not None:
            self.create_node(kind, p)

    def add_component(self, sdim):
        """Add a component (a model may have several, each with its own geometry, physics and mesh); it becomes
        the active one."""
        self._push_undo()
        comp = B.add_component(self.model, sdim)
        self.model.set_active(comp)
        self.tree.refresh(select=comp.child("geometry"))
        self._update_geom_actions()

    def add_physics(self, pid):
        if self.model.component is None:
            self.add_component("3D")
        pd = PHYSICS[pid]
        if self.model.component.get("sdim") not in pd.dims:
            QMessageBox.information(self, APP_NAME, f"{pd.name} is not available in this space dimension.")
            return
        snap = self.model.snapshot()
        try:
            ph = B.add_physics(self.model, pid)
        except ValueError as e:                   # a second instance of an interface with a fixed variable
            self.messages.add(str(e), "warning")
            if not self.no_modal:
                QMessageBox.information(self, APP_NAME, str(e))
            return
        self._push_undo(snap)
        self.tree.refresh(select=ph)

    def add_coupling(self, kind):
        self._push_undo()
        c = B.add_coupling(self.model, kind)
        self.tree.refresh(select=c)

    def add_geom(self, kind):
        comp = self.model.component
        if comp is None:
            return
        cur = self.tree.current()
        parent = comp.child("geometry")
        in_wp = cur is not None and (cur.kind == "geom.workplane" or (cur.parent is not None and cur.parent.kind == "geom.workplane"))
        if kind in ("geom.fillet3d", "geom.chamfer3d") and (in_wp or self.sdim() == 2):
            kind = kind.replace("3d", "2d")       # a planar sketch: fillet/chamfer its vertices (points), as in COMSOL
        if in_wp:
            wp = cur if cur.kind == "geom.workplane" else cur.parent
            if kind in REGISTRY["geom.workplane"].child_kinds(wp):
                parent = wp
        n = self.create_node(kind, parent)
        if kind in ("geom.union", "geom.intersection", "geom.difference", "geom.move", "geom.rotate", "geom.scale",
                    "geom.mirror", "geom.array", "geom.copy", "geom.extrude", "geom.revolve", "geom.fillet3d", "geom.chamfer3d"):
            names = self.object_names_before(n)
            if kind in ("geom.extrude", "geom.revolve"):
                wps = [x for x in names if x.startswith("wp")]
                n.props["input"] = wps[-1:] if wps else []
            elif kind == "geom.difference" and len(names) >= 2:
                n.props["input"] = names[:1]
                n.props["tools"] = names[1:]
            elif kind in ("geom.union", "geom.intersection"):
                n.props["input"] = list(names)
            elif names:
                n.props["input"] = names[-1:]
            self.settings.refresh()

    def add_mesh_node(self, kind):
        comp = self.model.component
        if comp is None:
            return
        mesh = comp.child("mesh")
        if mesh.get("sequence") != "user":
            self._push_undo()
            mesh.props["sequence"] = "user"
            if not mesh.children:
                self.model.create("mesh.size", mesh)
                self.model.create("mesh.ftet" if self.sdim() == 3 else "mesh.ftri", mesh).selection.all = True
        if kind in ("mesh.ftet", "mesh.ftri", "mesh.fquad") and any(c.kind == kind for c in mesh.children):
            self.tree.refresh(select=next(c for c in mesh.children if c.kind == kind))
            return
        self.create_node(kind, mesh)

    def set_mesh_size(self, val):
        comp = self.model.component
        if comp is not None:
            self.set_prop(comp.child("mesh"), "size", val, rebuild=True)
            self.tree.select(comp.child("mesh"))

    def add_study(self, key):
        if self.model.component is None:
            QMessageBox.information(self, APP_NAME, "Add a component and physics before adding a study.")
            return
        self._push_undo()
        st = B.add_study(self.model, key)
        self.tree.refresh(select=st)

    def _study_of(self, node: Node | None) -> Node | None:
        if node is not None:
            if node.kind == "study":
                return node
            st = node.ancestor("study")
            if st is not None and st.kind == "study":
                return st
        sts = self.model.studies()
        return sts[0] if sts else None

    def add_step(self, key):
        st = self._study_of(self.tree.current())
        if st is None:
            self.add_study(key)
            return
        self.create_node(f"study.{key}", st, index=next((i for i, c in enumerate(st.children) if c.kind == "study.solvers"),
                                                        len(st.children)))

    def add_sweep(self):
        st = self._study_of(self.tree.current())
        if st is None:
            return
        if st.child("study.sweep"):
            self.tree.select(st.child("study.sweep"))
            return
        rows = self.model.parameter_rows()
        self.create_node("study.sweep", st, index=0, pname=rows[0][0] if rows else "")

    def show_default_solver(self):
        st = self._study_of(self.tree.current())
        if st is None:
            return
        self._push_undo()
        sols = B.ensure_solver_configs(self.model, st)
        self.tree.refresh(select=sols.children[0] if sols.children else sols)

    def add_plot_group(self, kind):
        res = self.model.results
        idx = next((i for i, c in enumerate(res.children) if c.kind == "results.export"), len(res.children))
        pg = self.create_node(kind, res, index=idx)
        if kind == "pgpolar":
            self.create_node("plot.polar", pg, select=False)
            self.tree.refresh(select=pg)
        elif kind != "pg1d":
            sol = self.solution_of(self._dataset_for(pg))
            f = self.create_node("plot.surface", pg, select=False)
            if sol is not None:
                f.props["expr"] = default_expr(sol)
            self.tree.refresh(select=pg)
        else:
            if not pg.get("data"):                    # COMSOL: a new plot group takes the current solution dataset
                ds = self._dataset_for(pg)
                if ds is not None:
                    pg.props["data"] = ds.tag
            cl = next((d for d in self.datasets() if d.kind == "dset.cutline"), None)
            g = self.create_node("plot.linegraph", pg, select=False)
            if cl is not None:
                g.props["data"] = cl.tag
            self.tree.refresh(select=pg)

    def add_plot(self, kind):
        cur = self.tree.current()
        if kind in ("plot.deform", "plot.selection", "plot.colorexpr"):
            if cur is not None and cur.kind.startswith("plot.") and kind in REGISTRY[cur.kind].child_kinds(cur):
                self.create_node(kind, cur)
            else:
                self.statusBar().showMessage(f"Select a plot that supports {REGISTRY[kind].title} first.", 5000)
            return
        pg = cur if cur is not None and cur.kind.startswith("pg") else (cur.ancestor("pg") if cur is not None else None)
        if kind == "plot.polar":
            if pg is None or pg.kind != "pgpolar":
                self.add_plot_group("pgpolar")
            else:
                self.create_node(kind, pg)
            return
        if kind in ("plot.linegraph", "plot.globalgraph", "plot.pointgraph", "plot.tablegraph", "plot.directivity"):
            if pg is None or pg.kind != "pg1d":
                self.add_plot_group("pg1d")
                pg = self.tree.current()
                if kind != "plot.linegraph":
                    for c in list(pg.children):
                        pg.remove(c)
            n = self.create_node(kind, pg)
            ds = [d for d in self.datasets() if d.kind == "dset.solution"]
            if kind == "plot.globalgraph" and ds:
                n.props["data"] = ds[-1].tag
            return
        if pg is None or pg.kind == "pg1d":
            self.add_plot_group("pg3d" if self.sdim() == 3 else "pg2d")
            pg = self.tree.current()
        if kind not in REGISTRY[pg.kind].child_kinds(pg):
            QMessageBox.information(self, APP_NAME, f"{REGISTRY[kind].title} is not available in a {REGISTRY[pg.kind].title}.")
            return
        n = self.create_node(kind, pg)
        sol = self.solution_of(self._dataset_for(pg))
        if sol is not None and kind in ("plot.arrow", "plot.streamline"):
            from .plots import default_vector
            n.props["expr"] = default_vector(sol)
        elif sol is not None and "expr" in n.props and not n.props["expr"]:
            n.props["expr"] = default_expr(sol)
        self.settings.refresh()

    def add_results_child(self, container, kind):
        res = self.model.results
        c = res.child(container)
        self.create_node(kind, c)

    def show_material_browser(self):
        self.dock_mat.show()
        self.dock_mat.raise_()

    def add_library_material(self, name):
        comp = self.model.component
        if comp is None:
            return
        self._push_undo()
        mats = comp.child("materials")
        first = not any(c.kind == "material" for c in mats.children)
        from ..core.materials import make_material
        n = make_material(self.model, mats, name, all_domains=first)
        if not first:
            n.selection.entities = []
        self.tree.refresh(select=n)
        self.messages.add(f"Added material '{name}'" + (" to all domains" if first else "; select its domains"))

    def add_blank_material(self):
        comp = self.model.component
        if comp is not None:
            self.create_node("material", comp.child("materials"))

    # ================================================================== jobs
    def _start_job(self, fn, title, on_done, on_event=None, node=None):
        if self.job is not None:
            # queue it (a newer request with the same title replaces an older queued one)
            self._queue = [q for q in getattr(self, "_queue", []) if q[1] != title] + [(fn, title, on_done, on_event, node)]
            self.statusBar().showMessage(f"{title}: queued", 3000)
            return True
        self.job = Job(fn, self)
        # bound methods of this QObject => queued delivery on the GUI thread
        self._job_cb = (title, on_done, on_event, node)
        self.job.done.connect(self._on_job_done)
        self.job.failed.connect(self._on_job_failed)
        self.job.event.connect(self._on_job_event)
        self.pbar.setRange(0, 0)
        self.pbar.show()
        self.statusBar().showMessage(title + "...")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self._job_title, self._job_stage, self._stopping = title, "", False
        self._job_started = time.time()
        self.job_timer.start(1000)
        self.checkpoint()                               # a crash during the job must not lose unsaved edits
        try:
            self.recovery.note_job(title)
        except OSError:
            pass
        self.job.start()
        self._update_job_status()
        self.schedule_update()
        return True

    def _on_job_done(self, r):
        if self.job is not None and self.job.cancelled:      # stopped while it ran: drop the result
            self._job_failed(self._job_cb[0], "stopped by the user", "", self._job_cb[3])
        else:
            self._job_done(r, self._job_cb[1])
        self._next_job()

    def _on_job_failed(self, msg, tb):
        self._job_failed(self._job_cb[0], msg, tb, self._job_cb[3])
        self._next_job()

    def _next_job(self):
        q = getattr(self, "_queue", [])
        if q and self.job is None:
            fn, title, on_done, on_event, node = q.pop(0)
            self._start_job(fn, title, on_done, on_event, node)
        if self.job is None:
            try:
                self.recovery.note_job(None)
            except OSError:
                pass
            self._update_job_status()
            self.schedule_update()

    def busy(self) -> bool:
        # a job counts as busy until its done/failed handler has run on the GUI thread
        return self.job is not None or bool(getattr(self, "_queue", []))

    def _on_job_event(self, kind, payload):
        cb = self._job_cb[2]
        if cb is not None:
            cb(kind, payload)

    def _job_done(self, r, on_done):
        QApplication.restoreOverrideCursor()
        self.pbar.hide()
        self.statusBar().clearMessage()
        self.job = None
        try:
            on_done(r)
        except Exception as exc:
            self.messages.add(f"Error: {exc}", "error")
            traceback.print_exc()

    def _job_failed(self, title, msg, tb, node=None):
        QApplication.restoreOverrideCursor()
        self.pbar.hide()
        self.job = None
        self._pending_after_geo = None
        self.messages.add(f"{title} failed: {msg}", "error", node=node)
        if tb:
            self.logpane.add(tb)
        self.statusBar().showMessage(f"{title} failed", 6000)
        if "stopped by the user" not in msg and not self.no_modal:
            QMessageBox.warning(self, APP_NAME, f"{title} failed:\n\n{msg}")

    def stop_job(self):
        """Stop: drop the queued jobs and stop the running one (a solver is killed; a geometry or mesh build, which
        cannot be interrupted, finishes in the background and its result is discarded)."""
        dropped = len(getattr(self, "_queue", []))
        self._queue = []
        self._pending_after_geo = None
        if self.job is not None:
            self.job.cancelled = True
            self._stopping = True
            self._update_job_status()
            if self.job.runner is not None:
                self.job.runner.stop()
        if dropped:
            self.messages.add(f"Stopped; {dropped} queued job(s) cancelled.")

    # -- geometry
    def build_geometry(self, upto: Node | None = None, silent=False, then=None):
        comp = self.model.component
        if comp is None:
            return
        model_snapshot = self._snapshot()
        upto_tag = upto.tag if upto is not None else None
        rev = self.rev

        def work(emit):
            g = model_snapshot.component.child("geometry")
            u = model_snapshot.by_tag(upto_tag) if upto_tag else None
            return build_geometry(model_snapshot, g, upto=u)

        def done(res: GeometryResult):
            with self.model.scoped(comp):         # the component it was built for, whichever is selected now
                self._geo_done(res, comp, rev, upto, silent, then)
        self._start_job(work, "Building geometry", done, node=comp.child("geometry"))

    def _geo_done(self, res, comp, rev, upto, silent, then):
        self.geo = res
        current = self.rev == rev            # the model was not edited while the geometry was built
        self.geo_stale = upto is not None or not current
        if not current:
            self.messages.add("The model changed while the geometry was built; build it again.", "warning")
        elif upto is None:
            with self.model.scoped(comp):         # the same remapping the build did on its copy (its messages
                apply_named_selections(self.model, res)   # are in res.messages)
            self.tree.update_marks()
        for m in res.messages:
            self.messages.add(m, "warning")
        if not silent:
            self.messages.add(f"Geometry: {res.count('domain')} domains, {res.count('boundary')} boundaries"
                              + (f", {res.count('edge')} edges" if res.sdim == 3 else "")
                              + f", {res.count('point')} points.")
        cur = self.tree.current()
        if then is not None:
            then()
        elif cur is not None:
            self._show_geometry_for(cur)
        else:
            self.graphics.show_geometry(res)
        if cur is not None and cur.kind in ("geometry", "geom.finalize"):
            self.settings.refresh()

    def build_selected(self):
        cur = self.tree.current()
        if cur is None:
            return
        if cur.kind.startswith("geom.") and cur.kind != "geom.finalize":
            self.build_geometry(upto=cur if cur.parent.kind == "geometry" else cur.parent)
        elif cur.kind.startswith("mesh"):
            self.build_mesh()
        elif cur.kind.startswith("study"):
            self.compute(self._study_of(cur))
        elif cur.kind.startswith(("pg", "plot.")):
            self.plot(cur if cur.kind.startswith("pg") else cur.ancestor("pg"))
        else:
            self.build_geometry()

    def build_all_context(self):
        cur = self.tree.current()
        if cur is None:
            return
        if cur.kind.startswith("mesh"):
            self.build_mesh()
        elif cur.kind.startswith("study"):
            self.compute(self._study_of(cur))
        elif cur.kind.startswith(("pg", "plot.")):
            self.plot(cur if cur.kind.startswith("pg") else cur.ancestor("pg"))
        elif cur.kind.startswith("eval."):
            self.evaluate_node(cur)
        else:
            self.build_geometry()

    # -- mesh
    def build_mesh(self, then=None):
        if self.model.component is None:
            return
        model_snapshot = self._snapshot()
        rev = self.rev
        comp = self.model.component

        def work(emit):
            return build_mesh(model_snapshot)

        def done(res):
            with self.model.scoped(comp):
                self._mesh_done(res, rev, then)
        self._start_job(work, "Building mesh", done, node=comp.child("mesh"))

    def _mesh_done(self, res, rev, then):
        self.mesh = res
        if self.rev == rev:                  # else the model changed while it was meshed: the mesh is stale
            self.mesh_stale = False
            self.geo_stale = False
        else:
            self.messages.add("The model changed while the mesh was built; build it again.", "warning")
        st = res.stats
        self.messages.add(f"Complete mesh consists of {st['elements']} domain elements and {st['boundary_elements']} "
                          f"boundary elements ({st['nodes']} nodes, order {st['order']}). "
                          f"Minimum element quality: {st['min_quality']:.4g}. Time: {st['time']:.2f} s.")
        if then is not None:
            then()
        else:
            self.show_mesh()
        cur = self.tree.current()
        if cur is not None and cur.kind == "mesh":
            self.settings.refresh()

    def clear_mesh(self):
        self.mesh = None
        self.mesh_stale = True
        self.messages.add("Mesh cleared.")
        if self.geo is not None:
            self.graphics.show_geometry(self.geo)

    def mesh_statistics(self):
        if self.mesh is None or self.mesh_stale:
            self.build_mesh(then=self.mesh_statistics)
            return
        st = self.mesh.stats
        rows = [["Number of nodes", st["nodes"]], ["Domain elements", st["elements"]],
                ["Boundary elements", st["boundary_elements"]], ["Element order", st["order"]],
                ["Minimum element quality", st["min_quality"]], ["Average element quality", st["avg_quality"]]]
        rows += [[f"{k}", v] for k, v in st["types"].items()]
        self.table.show_table("Mesh Statistics", ["Property", "Value"], rows)
        self.dock_table.raise_()
        self.show_mesh(quality=True)

    # -- compute
    def compute(self, study: Node | None = None):
        study = study or self._study_of(self.tree.current())
        if study is None:
            QMessageBox.information(self, APP_NAME, "There is no study to compute. Add one from the Study tab.")
            return
        if self.inst is None:
            QMessageBox.warning(self, APP_NAME, "Elmer was not found. Set its installation folder in File > Preferences.")
            return
        self.dock_mat.hide()
        model_snapshot = Model.from_dict(json.loads(json.dumps(self.model.to_dict())))
        rev = self.rev
        st_snap = model_snapshot.by_tag(study.tag)
        mesh_cache = {t: sl["mesh"] for t, sl in self._slots.items() if sl["mesh"] is not None and not sl["mesh_stale"]}
        self.conv.reset()
        if study.get("convplot", True) and self.central.indexOf(self.conv) < 0:
            self.central.addTab(self.conv, icons.icon("convergence"), "Convergence Plot 1")
        self.progress.start(f"{study.label}: Compute")
        self.dock_prog.raise_()
        self.logpane.add(f"===== {study.label} =====")
        holder = {}

        def work(emit):
            r = StudyRunner(model_snapshot, st_snap, self.workdir, self.inst, on_event=emit, mesh_cache=mesh_cache)
            holder["runner"] = r
            self.job.runner = r
            return r.run()

        self._conv_dirty = False
        self._sweep = ""

        def on_event(kind, p):
            if kind == "log":
                self.logpane.add(p)
            elif kind == "conv":
                self.conv.add(p)
                self._conv_dirty = True
                self.progress.set(f"Solver: {p[1]}", None, f"{p[3]:.2e}", f"iteration {p[4]}" if len(p) > 4 else None)
            elif kind == "progress":
                frac, text = p
                self.pbar.setRange(0, 100)
                self.pbar.setValue(int(frac * 100))
                self.progress.set(f"{study.label}: Compute", frac, None, text)
                self._job_stage = text if not self._sweep else f"{self._sweep} - {text}"
                self.statusBar().showMessage(f"{study.label}: {text}")
            elif kind == "sweep":
                self._sweep = p
                self.progress.set("Parametric sweep", None, None, p)
            elif kind == "message":
                self.messages.add(p)
            elif kind == "warning":
                self.messages.add(p, "warning")
            elif kind == "error_line":
                self.messages.add(p, "error")
            elif kind == "mesh":
                c = self.model.by_tag(getattr(p, "comp_tag", "")) or self.model.component
                with self.model.scoped(c):
                    self.mesh = p
                    self.mesh_stale = self.rev != rev    # meshed from the model as it was when Compute started

        def done(folders):
            self._flush_log()
            self.sol_cache = {k: v for k, v in self.sol_cache.items() if not k.startswith(os.path.join(self.workdir, study.tag))}
            ds = B.solution_dataset(self.model, study)
            steps = B.study_steps(study)
            step_kind = steps[-1].kind.split(".", 1)[1] if steps else "stationary"
            made = B.default_plots(self.model, study, step_kind) if study.get("plots", True) else []
            for n in self.model.root.walk():
                n.meta.pop("needs_update", None)
            self.dirty = True
            self._update_title()
            self.progress.set(f"{study.label}: Compute", 1.0, "Done")
            self.progress.finish_all()
            self.dock_msg.raise_()
            target = made[0] if made else next((pg for pg in self.plot_groups() if pg.get("data") == ds.tag), None)
            self.tree.refresh(select=target or study)
            if target is not None:
                self.plot(target)
            sol = self.solution_of(ds)
            if sol is not None and sol.kind == "study.eigen":
                cols, rows = evaluate_derived(Node("eval.global"), self.model, sol, [], None)
                self.table.show_table(f"{study.label}: Eigenfrequencies", cols, rows)
                self.dock_table.raise_()
            if self.central.indexOf(self.conv) >= 0:
                self.conv.redraw()
            self.central.setCurrentIndex(0)

        self._start_job(work, f"Computing {study.label}", done, on_event, node=study)

    def _flush_log(self):
        self.logpane.flush()
        if getattr(self, "_conv_dirty", False):
            self._conv_dirty = False
            if self.central.indexOf(self.conv) >= 0:
                self.conv.redraw()
                if self.job is not None:
                    self.central.setCurrentWidget(self.conv)

    def clear_solutions(self):
        import shutil
        for st in self.model.studies():
            shutil.rmtree(os.path.join(self.workdir, st.tag), ignore_errors=True)
        self.sol_cache.clear()
        self.messages.add("All solutions cleared.")
        if self.geo is not None:
            self.graphics.show_geometry(self.geo)

    # -- results
    def plot(self, pg: Node | None):
        if pg is None:
            return
        self.current_plot = pg
        ds = self._dataset_for(pg)
        if pg.kind in ("pg1d", "pgpolar"):
            def get_sol(dset, sel):
                sol = self.solution_of(dset if dset is not None else ds)
                if sol is None:
                    return None, -1, 1.0
                return sol, sol.frame_index(pg.get("solnum", "last")), sol.scale
            try:
                draw_1d(self.plot1d.fig, pg, self.model, get_sol, pg.get("solnum", "last"),
                        get_sweep=lambda d: (self.sweep_solutions(d), *self.sweep_info(d)),
                        get_table=lambda tag: self.model.by_tag(tag) if tag else None)
            except Exception as exc:
                self.messages.add(f"{pg.label}: {exc}", "error")
            self.plot1d.draw()
            self.gstack.setCurrentIndex(1)
            self.central.setCurrentIndex(0)
            return
        sol, rev = self.view_of(ds)
        if sol is None:
            self.statusBar().showMessage("No solution available: compute the study first.", 5000)
            if self.geo is not None:
                self.graphics.show_geometry(self.geo)
            return
        fi = sol.frame_index(pg.get("solnum", "last"))
        views = {}
        for f in pg.children:                      # plots with their own dataset (grid, curve, ...)
            tag = f.props.get("data", "fromparent")
            if f.enabled and tag not in ("", "fromparent", None) and (ds is None or tag != ds.tag):
                v = self.solution_of(self.model.by_tag(tag))
                if v is not None:
                    views[f.tag] = v
        try:
            scene = build_scene(pg, sol, fi, sol.sdim, self.is_axi(), revolution=rev, views=views, model=self.model)
        except Exception as exc:
            self.messages.add(f"{pg.label}: {exc}", "error")
            traceback.print_exc()
            return
        self.gstack.setCurrentIndex(0)
        self.central.setCurrentIndex(0)
        self.graphics.show_scene(scene)

    def evaluate_node(self, node: Node):
        ds = self._dataset_for(node)
        sol = self.solution_of(ds)
        if sol is None:
            QMessageBox.information(self, APP_NAME, "No solution available: compute the study first.")
            return
        sel = list(range(len(sol.frames))) if node.get("solnum", "last") == "all" else [sol.frame_index("last")]
        ents = None
        points = None
        if node.selection is not None and self.geo is not None:
            ents = node.selection.resolve(self.geo.numbers.get(node.selection.level, []))
        if node.kind == "eval.point":
            ents = None
            if node.selection is not None and node.selection.entities:   # geometry points -> their mesh nodes
                base = sol
                while not isinstance(base, Solution):
                    base = base.base
                pn = base.topo.point_nodes
                points = [list(base.topo.nodes[pn[e]][:max(2, base.sdim)] * base.scale)
                          for e in sorted(node.selection.entities) if e in pn]
                if not points:
                    self.messages.add(f"{node.label}: the selected points are not mesh vertices; remesh the geometry.",
                                      "warning")
        sweep = self.sweep_solutions(ds)
        sw_label, sw_unit = self.sweep_info(ds)
        try:
            cols, rows = evaluate_derived(node, self.model, sol, sel, ents, sweep or None, sw_label, sw_unit,
                                          points=points)
        except Exception as exc:
            self.messages.add(f"{node.label}: {exc}", "error")
            return
        self.table.show_table(node.label, cols, rows)
        self.dock_table.raise_()
        tables = self.model.results.child("results.tables")
        tbl = next((t for t in tables.children if t.meta.get("source") == node.tag), None)
        if tbl is None:
            tbl = self.model.create("table", tables)
            tbl.meta["source"] = node.tag
            self.tree.refresh(select=node)
        tbl.props["columns"] = cols
        tbl.props["rows"] = [[float(x) if isinstance(x, (int, float)) else str(x) for x in r] for r in rows]

    def export_node(self, node: Node):
        fn = str(node.get("file", "")).strip()
        if not fn:
            ext = {"export.image": "png", "export.data": "csv", "export.vtu": "vtu", "export.report": "html"}[node.kind]
            fn, _ = QFileDialog.getSaveFileName(self, "Export", f"{node.label}.{ext}")
            if not fn:
                return
            node.props["file"] = fn
            self.settings.refresh()
        try:
            if node.kind == "export.image":
                pg = self.model.by_tag(node.get("source", "")) or self.current_plot
                if pg is not None:
                    self.plot(pg)
                self.graphics.snapshot(fn, int(node.get("w", 1600)), int(node.get("h", 1000)))
            elif node.kind == "export.data":
                sol = self.solution_of(self._dataset_for(node))
                exprs = [e.strip() for e in str(node.get("expr", "")).split(",") if e.strip()] or [default_expr(sol)]
                fi = sol.frame_index("last")
                import numpy as np
                P = sol.topo.nodes * sol.scale
                cols = [P[:, 0], P[:, 1]] + ([P[:, 2]] if sol.sdim == 3 else []) + [sol.evaluate(e, fi) for e in exprs]
                hdr = "x,y," + ("z," if sol.sdim == 3 else "") + ",".join(exprs)
                np.savetxt(fn, np.column_stack(cols), delimiter=",", header=hdr, comments="% ")
            elif node.kind == "export.vtu":
                import shutil
                sol = self.solution_of(self._dataset_for(node))
                shutil.copy(sol.frames[sol.frame_index("last")].file, fn)
            elif node.kind == "export.report":
                self.write_report(fn, node.get("images", True))
            self.messages.add(f"Exported: {fn}")
        except Exception as exc:
            self.messages.add(f"Export failed: {exc}", "error")

    def write_report(self, fn, images=True):
        import html
        d = os.path.dirname(fn)
        parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(self.model.root.label)}</title>"
                 "<style>body{font-family:Segoe UI,Arial;margin:30px;color:#222} h1{color:#2559a4} "
                 "table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:3px 8px} th{background:#eef0f3}</style>"
                 f"</head><body><h1>{html.escape(self.model.root.label)}</h1>"
                 f"<p>Generated by {APP_NAME} {__version__}</p>"]
        rows = self.model.parameter_rows()
        if rows:
            vals = self.model.parameters()
            parts.append("<h2>Parameters</h2><table><tr><th>Name</th><th>Expression</th><th>Value (SI)</th></tr>")
            for n, e in rows:
                parts.append(f"<tr><td>{n}</td><td>{html.escape(e)}</td><td>{vals.get(n, float('nan')):.6g}</td></tr>")
            parts.append("</table>")
        comp = self.model.component
        if comp is not None:
            parts.append("<h2>Model</h2><ul>")
            for c in comp.children:
                parts.append(f"<li>{html.escape(c.label)}<ul>" + "".join(f"<li>{html.escape(x.label)}</li>" for x in c.children)
                             + "</ul></li>")
            parts.append("</ul>")
        if images:
            parts.append("<h2>Results</h2>")
            for i, pg in enumerate(self.plot_groups()):
                if pg.kind in ("pg1d", "pgpolar"):
                    continue
                self.plot(pg)
                QApplication.processEvents()
                img = os.path.join(d, f"report_img{i + 1}.png")
                self.graphics.snapshot(img, 1200, 800)
                parts.append(f"<h3>{html.escape(pg.label)}</h3><img src='{os.path.basename(img)}' width='900'>")
        parts.append("</body></html>")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

    # ================================================================== Elmer utilities
    def _sif_for(self, study: Node):
        steps = B.study_steps(study)
        if not steps:
            raise SifError("The study has no steps")
        mesh = self.mesh if (self.mesh is not None and not self.mesh_stale) else build_mesh(self.model)
        self.mesh, self.mesh_stale = mesh, False
        return mesh, build_sif(self.model, study, steps[0], mesh.geometry, len(mesh.nodes))

    def show_sif(self, study: Node | None = None):
        study = study if isinstance(study, Node) else self._study_of(self.tree.current())
        if study is None:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            mesh, sif = self._sif_for(study)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        QApplication.restoreOverrideCursor()
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Elmer Solver Input File - {study.label}")
        dlg.resize(760, 680)
        v = QVBoxLayout(dlg)
        te = QPlainTextEdit(sif.text)
        te.setReadOnly(True)
        te.setObjectName("MessagesView")
        te.setLineWrapMode(QPlainTextEdit.NoWrap)
        v.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        save = bb.addButton("Save As...", QDialogButtonBox.ActionRole)
        save.clicked.connect(lambda: self._save_text(sif.text, "case.sif"))
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        self._last_sif = sif.text
        if self.no_modal:
            return
        dlg.exec()

    def _save_text(self, text, default):
        fn, _ = QFileDialog.getSaveFileName(self, "Save As", default)
        if fn:
            with open(fn, "w") as f:
                f.write(text)

    def export_case(self):
        study = self._study_of(self.tree.current())
        if study is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Export Elmer case to folder")
        if not d:
            return
        try:
            mesh, sif = self._sif_for(study)
            write_case(d, mesh.to_elmer(), sif.text)
            self.messages.add(f"Elmer case written to {d} (run: ElmerSolver case.sif)")
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))

    def open_case_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        study = self._study_of(self.tree.current())
        path = os.path.join(self.workdir, study.tag) if study else self.workdir
        if not os.path.isdir(path):
            path = self.workdir
        self.last_url = QUrl.fromLocalFile(path).toString()
        self.statusBar().showMessage(f"Case folder: {path}", 5000)
        if not self.no_modal:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def run_script(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Run Python Script", "", "Python (*.py)")
        if not fn:
            return
        self._push_undo()
        ns = {"model": self.model, "builder": B, "B": B, "app": self, "units": units}
        try:
            with open(fn, encoding="utf-8") as f:
                exec(compile(f.read(), fn, "exec"), ns)
            self.messages.add(f"Ran script {fn}")
        except Exception as exc:
            self.messages.add(f"Script error: {exc}", "error")
            self.logpane.add(traceback.format_exc())
        self.geo_stale = self.mesh_stale = True
        self.tree.refresh()

    def preferences(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences")
        dlg.resize(620, 220)
        f = QFormLayout(dlg)
        row = QHBoxLayout()
        le = QLineEdit(self.inst.home if self.inst else (self.qs.value("elmer_home") or ""))
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: (lambda d: d and le.setText(d))(QFileDialog.getExistingDirectory(dlg, "Elmer installation folder")))
        det = QPushButton("Detect")
        info = QLabel(self.inst.version if self.inst and self.inst.version else "")

        def detect():
            inst = find_elmer(le.text().strip() or None)
            if inst:
                le.setText(inst.home)
                info.setText((elmer_version(inst) or inst.solver) + "\n" + _caps_text(inst))
            else:
                info.setText("<span style='color:#c0392b'>ElmerSolver not found</span>")
        det.clicked.connect(detect)
        row.addWidget(le, 1)
        row.addWidget(b)
        row.addWidget(det)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f.addRow("Elmer installation or build folder:", w)
        f.addRow("", info)
        cb = QCheckBox("Save solutions in the model file")
        cb.setChecked(self.qs.value("save_solutions", True, bool))
        f.addRow("", cb)
        # desktop and graphics appearance (COMSOL: Preferences > User Interface and Graphics)
        font = QDoubleSpinBox()
        font.setRange(7.0, 16.0)
        font.setSingleStep(0.5)
        font.setSuffix(" pt")
        font.setValue(float(self.qs.value("font_pt", theme.BASE_PT)))
        font.setToolTip("Base text size of the desktop and the plots; display scaling (100-200 %) applies on top. "
                        "Takes effect when Elmer Studio starts again.")
        f.addRow("Font size:", font)
        gth = QComboBox()
        for key, text in (("light", "Light gradient"), ("white", "White"), ("dark", "Dark")):
            gth.addItem(text, key)
        gth.setCurrentIndex(max(0, gth.findData(self.graphics.theme_name)))
        f.addRow("Graphics theme:", gth)
        eth = QComboBox()
        for key, text in (("white", "White (readable in documents)"), ("graphics", "Same as the Graphics theme")):
            eth.addItem(text, key)
        eth.setCurrentIndex(max(0, eth.findData(self.graphics.export_theme)))
        f.addRow("Image export theme:", eth)
        auto = QSpinBox()
        auto.setRange(0, 3600)
        auto.setSuffix(" s")
        auto.setValue(int(self.qs.value("autosave_s", 60)))
        auto.setToolTip("How often unsaved changes are written to a recovery file (0: only before a computation)")
        f.addRow("Recovery checkpoint every:", auto)
        rec = QCheckBox("Offer to recover unsaved models after a crash")
        rec.setChecked(self.qs.value("offer_recovery", True, bool))
        f.addRow("", rec)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        dlg.resize(620, 380)
        if dlg.exec():
            if abs(font.value() - float(self.qs.value("font_pt", theme.BASE_PT))) > 1e-6:
                self.qs.setValue("font_pt", font.value())
                self.messages.add(f"Font size {font.value():g} pt: restart Elmer Studio to apply it.")
            self.qs.setValue("graphics_theme", gth.currentData())
            self.qs.setValue("export_theme", eth.currentData())
            self.graphics.export_theme = eth.currentData()
            if gth.currentData() != self.graphics.theme_name:
                self.graphics.set_theme(gth.currentData())
            self.qs.setValue("autosave_s", auto.value())
            if auto.value():
                self.autosave_timer.start(auto.value() * 1000)
            else:
                self.autosave_timer.stop()
            self.qs.setValue("offer_recovery", rec.isChecked())
            inst = find_elmer(le.text().strip() or None)
            if inst:
                self.inst = inst
                self.qs.setValue("elmer_home", inst.home)
                self.messages.add(f"Elmer: {inst.solver} {elmer_version(inst)}; {_caps_text(inst)}")
            else:
                QMessageBox.warning(self, APP_NAME, "ElmerSolver was not found in that folder.")
            self.qs.setValue("save_solutions", cb.isChecked())

    def open_docs(self):
        """Ctrl+F1: the documentation site (installation, tutorials, user guide, reference)."""
        self.open_url(DOCS_URL)

    def about(self):
        if self.no_modal:
            self.statusBar().showMessage(f"{APP_NAME} {__version__}", 5000)
            return
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<h3>{APP_NAME} {__version__}</h3><p>Documentation: <a href='{DOCS_URL}'>{DOCS_URL}</a></p>"
                          "<p>A model-builder desktop for the open-source "
                          f"<b>Elmer</b> finite element solver (CSC - IT Center for Science).</p>"
                          f"<p>Geometry and meshing: gmsh / OpenCASCADE. Graphics: VTK. GUI: Qt.</p>"
                          f"<p>Elmer: {self.inst.solver if self.inst else 'not found'}</p>"
                          "<p>Mouse: left-drag rotate (pan in 2D), right-drag pan, wheel zoom; click to select "
                          "entities when a selection list is active.</p>")
