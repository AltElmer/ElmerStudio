"""Main window: ribbon, Model Builder, Settings, Graphics, Messages/Progress/Log/Table."""
from __future__ import annotations

import copy
import glob
import json
import os
import re
import traceback

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, FILE_EXT, __version__
from ..core import builder as B
from ..core import project, units
from ..core.elmer import elmer_version, find_elmer, write_case
from ..core.examples import EXAMPLES
from ..core.geometry import GeometryError, GeometryResult, build_geometry, object_names_before
from ..core.materials import MATPROPS
from ..core.meshing import SIZE_CHOICES, build_mesh
from ..core.model import REGISTRY, Model, Node, is_axisymmetric, space_dim
from ..core.physics import COUPLINGS, PHYSICS, STUDY_TYPES, physics_def
from ..core.results import Solution, compatible_units
from ..core.sif import SifError, build_sif
from ..core.study_runner import StudyCancelled, StudyRunner
from . import icons
from .graphics import GraphicsView
from .material_browser import MaterialBrowser
from .model_builder import ModelBuilder
from .panes import ConvergencePlot, LogPane, MessagesPane, MplCanvas, ProgressPane, TablePane
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

    def run(self):
        try:
            r = self.fn(lambda k, p=None: self.event.emit(k, p))
            self.done.emit(r)
        except StudyCancelled:
            self.failed.emit("Computation stopped by the user.", "")
        except Exception as exc:  # report everything to the GUI thread
            self.failed.emit(str(exc), traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, open_path: str | None = None, show_new=True):
        super().__init__()
        self.qs = QSettings("ElmerStudio", "ElmerStudio")
        self.inst = find_elmer(self.qs.value("elmer_home") or None)
        self.model: Model = B.new_model(None)
        self.workdir = project.new_workdir()
        self.file_path: str | None = None
        self.dirty = False
        self.geo: GeometryResult | None = None
        self.geo_stale = True
        self.mesh = None
        self.mesh_stale = True
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.sol_cache: dict[str, Solution] = {}
        self.job: Job | None = None
        self.active_sel: Node | None = None
        self.current_plot: Node | None = None
        self._pending_after_geo = None
        self.no_modal = False          # tests: never block on modal dialogs/menus
        self.setWindowIcon(icons.icon("app"))
        self.resize(1500, 920)
        self._build_actions()
        self._build_ribbon()
        self._build_panes()
        self._build_statusbar()
        self._default_state = self.saveState()
        st = self.qs.value("window_state")
        if isinstance(st, QByteArray):
            self.restoreState(st)
        geom = self.qs.value("window_geometry")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._flush_log)
        self.log_timer.start(250)
        self.mem_timer = QTimer(self)
        self.mem_timer.timeout.connect(self._update_mem)
        self.mem_timer.start(2000)
        self._set_model(self.model, None)
        self.messages.add(f"{APP_NAME} {__version__}")
        if self.inst:
            v = elmer_version(self.inst)
            self.messages.add(f"Elmer found: {self.inst.solver}" + (f"  [{v}]" if v else ""))
        else:
            self.messages.add("Elmer was not found. Set the installation folder in File > Preferences.", "warning")
        if open_path:
            QTimer.singleShot(0, lambda: self.open_file(open_path))
        elif show_new:
            QTimer.singleShot(150, self.file_new)

    # ================================================================== construction
    def _a(self, name, text, icon_name=None, cb=None, shortcut=None, tip=None, checkable=False, menu=None):
        a = QAction(icons.icon(icon_name) if icon_name else QIcon(), text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.WindowShortcut)
        if tip or shortcut:
            a.setToolTip(f"{tip or text.replace(chr(10), ' ')}" + (f" ({shortcut})" if shortcut else ""))
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
        a("redo", "Redo", "redo", self.redo, "Ctrl+Y")
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
        a("help", "Help", "help", self.about, "F1")
        a("reset_desktop", "Reset\nDesktop", "reset_desktop", self.reset_desktop)
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
        for n in ("new", "open"):
            fm.addAction(self.act[n])
        fm.addMenu(self.recent_menu)
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
            ("1D Plot Group", "plot_group_1d", lambda: self.add_plot_group("pg1d"))])))
        g = t.add_group("Layout")
        g.add_large(self.act["reset_desktop"])

        # --- Definitions
        t = r.add_tab("Definitions")
        g = t.add_group("Definitions")
        g.add_large(self._a("params2", "Parameters", "parameters", lambda: self.add_child_kind("global", "params")))
        g.add_large(self._a("vars2", "Variables", "variables", lambda: self.add_child_kind("definitions", "variables")))
        g = t.add_group("Functions")
        g.add_large(self._a("an", "Analytic", "functions", lambda: self.add_child_kind("global", "func.analytic")))
        g.add_large(self._a("interp", "Interpolation", "functions", lambda: self.add_child_kind("global", "func.interp")))

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
                             (["geom.rectangle", "geom.circle", "geom.ellipse", "geom.polygon"], self.geo2d_actions)):
            for k in kinds:
                nt = REGISTRY[k]
                act = self._a(k, nt.title, nt.icon, (lambda kk=k: self.add_geom(kk)))
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
                     self._a("mesh_fquad", "Free Quad", "free_tri", lambda: self.add_mesh_node("mesh.fquad"))])
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
        g = t.add_group("Plots")
        plots = [self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon, (lambda kk=k: self.add_plot(kk)))
                 for k in ("plot.surface", "plot.volume", "plot.slice", "plot.isosurface", "plot.contour", "plot.arrow",
                           "plot.streamline", "plot.mesh", "plot.linegraph", "plot.globalgraph")]
        g.add_large(plots[0])
        g.add_small(plots[1:4])
        g.add_small(plots[4:7])
        g.add_small(plots[7:10])
        g.add_large(self._a("r_def", "Deformation", "deformation", lambda: self.add_plot("plot.deform")))
        g = t.add_group("Datasets")
        g.add_small([self._a("r_cln", "Cut Line", "cut_line", lambda: self.add_results_child("results.datasets", "dset.cutline")),
                     self._a("r_cpt", "Cut Point", "cut_point", lambda: self.add_results_child("results.datasets", "dset.cutpoint"))])
        g = t.add_group("Evaluation")
        g.add_large(self._a("r_glob", "Global\nEvaluation", "global_evaluation",
                            lambda: self.add_results_child("results.derived", "eval.global")))
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.derived", kk)))
                     for k in ("eval.volint", "eval.surfint", "eval.avg")])
        g.add_small([self._a(f"r_{k}", REGISTRY[k].title, REGISTRY[k].icon,
                             (lambda kk=k: self.add_results_child("results.derived", kk)))
                     for k in ("eval.max", "eval.point")])
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
        self.dock_tree = self._dock("Model Builder", "model_builder", self.tree, Qt.LeftDockWidgetArea)
        self.settings = SettingsWindow(self, self)
        self.settings.actionRequested.connect(self.on_settings_action)
        self.dock_settings = self._dock("Settings", "settings_window", self.settings, Qt.LeftDockWidgetArea)
        self.splitDockWidget(self.dock_tree, self.dock_settings, Qt.Horizontal)
        self.resizeDocks([self.dock_tree, self.dock_settings], [300, 380], Qt.Horizontal)
        # central graphics
        self.central = QTabWidget(self)
        self.central.setDocumentMode(False)
        self.gstack = QStackedWidget()
        self.graphics = GraphicsView(self)
        self.graphics.entityClicked.connect(self.on_entity_clicked)
        self.graphics.status.connect(lambda s: self.statusBar().showMessage(s, 3000))
        self.plot1d = MplCanvas(self)
        self.gstack.addWidget(self.graphics)
        self.gstack.addWidget(self.plot1d)
        self.central.addTab(self.gstack, icons.icon("graphics_window"), "Graphics")
        self.conv = ConvergencePlot(self)
        self.setCentralWidget(self.central)
        # bottom panes
        self.messages = MessagesPane(self)
        self.progress = ProgressPane(self)
        self.logpane = LogPane(self)
        self.table = TablePane(self)
        self.dock_msg = self._dock("Messages", "messages", self.messages, Qt.BottomDockWidgetArea)
        self.dock_prog = self._dock("Progress", "progress", self.progress, Qt.BottomDockWidgetArea)
        self.dock_log = self._dock("Log", "log", self.logpane, Qt.BottomDockWidgetArea)
        self.dock_table = self._dock("Table", "table", self.table, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(self.dock_msg, self.dock_prog)
        self.tabifyDockWidget(self.dock_prog, self.dock_log)
        self.tabifyDockWidget(self.dock_log, self.dock_table)
        self.dock_msg.raise_()
        self.resizeDocks([self.dock_msg], [190], Qt.Vertical)
        self.matbrowser = MaterialBrowser(self)
        self.matbrowser.addRequested.connect(self.add_library_material)
        self.dock_mat = self._dock("Add Material", "materials", self.matbrowser, Qt.RightDockWidgetArea)
        self.dock_mat.hide()
        self.resizeDocks([self.dock_mat], [300], Qt.Horizontal)

    def _dock(self, title, icon_name, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(f"dock_{title.replace(' ', '_')}")
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
        self.mem.setStyleSheet("color:#444; padding: 0 6px;")
        sb.addPermanentWidget(self.pbar)
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
            self.workdir = workdir
        self.geo = None
        self.geo_stale = True
        self.mesh = None
        self.mesh_stale = True
        self.sol_cache.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self.current_plot = None
        self.active_sel = None
        self.tree.set_model(model, self._file_name())
        self.graphics.clear()
        self._update_title()
        self._update_geom_actions()
        self.tree.select(model.root)

    def _file_name(self):
        if self.file_path:
            return os.path.basename(self.file_path)
        return f"{self.model.root.label or 'Untitled'}{FILE_EXT}"

    def _update_title(self):
        self.setWindowTitle(f"{self._file_name()}{' *' if self.dirty else ''} - {APP_NAME}")

    def _update_geom_actions(self):
        sd = self.sdim() if self.model.component else 3
        for a in self.geo3d_actions + self.wp_actions + self.fc_actions:
            a.setEnabled(sd == 3)
        for a in self.geo2d_actions:
            a.setEnabled(sd == 2)
        self.act["add_comp"].setEnabled(self.model.component is None)

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
        if m.component is not None:
            self.build_geometry()

    def file_save(self) -> bool:
        if not self.file_path:
            return self.file_save_as()
        try:
            project.save(self.model, self.file_path, self.workdir, include_solutions=self.qs.value("save_solutions", True, bool))
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Save failed: {exc}")
            return False
        self.dirty = False
        self._update_title()
        self.messages.add(f"Saved file: {self.file_path}")
        self._add_recent(self.file_path)
        return True

    def file_save_as(self) -> bool:
        fn, _ = QFileDialog.getSaveFileName(self, "Save As", os.path.join(self.qs.value("last_dir", ""), self._file_name()),
                                            f"{APP_NAME} model (*{FILE_EXT})")
        if not fn:
            return False
        if not fn.endswith(FILE_EXT):
            fn += FILE_EXT
        self.file_path = fn
        self.model.root.label = os.path.splitext(os.path.basename(fn))[0]
        self.qs.setValue("last_dir", os.path.dirname(fn))
        self.tree.file_name = self._file_name()
        self.tree.refresh()
        return self.file_save()

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

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.stop_job()
        self.qs.setValue("window_state", self.saveState())
        self.qs.setValue("window_geometry", self.saveGeometry())
        try:
            if self.graphics.plotter is not None:
                self.graphics.plotter.close()
        except Exception:
            pass
        ev.accept()

    def reset_desktop(self):
        self.restoreState(self._default_state)
        for d in (self.dock_tree, self.dock_settings, self.dock_msg):
            d.show()

    # ================================================================== undo / mutation
    def _push_undo(self):
        self.undo_stack.append(self.model.snapshot())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.dirty = True
        self._update_title()

    def _restore(self, snap):
        cur = self.tree.current()
        tag = cur.tag if cur else None
        self.model.restore(snap)
        self.geo_stale = True
        self.mesh_stale = True
        self.tree.set_model(self.model, self._file_name())
        n = self.model.by_tag(tag) if tag else None
        self.tree.refresh(select=n or self.model.root)
        self._update_geom_actions()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.model.snapshot())
            self._restore(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.model.snapshot())
            self._restore(self.redo_stack.pop())

    def _mark(self, node: Node):
        if node.kind == "params" or node.kind.startswith("func.") or node.kind == "variables":
            self.geo_stale = True
            self.mesh_stale = True
            return
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

    def set_selection(self, node: Node, all_=None, entities=None):
        if node.selection is None:
            return
        self._push_undo()
        if all_ is not None:
            node.selection.all = all_
            if not all_ and not node.selection.entities and self.geo is not None and False:
                pass
        if entities is not None:
            node.selection.entities = list(entities)
            node.selection.all = False
        self._mark(node)
        self.settings.refresh()

    def remove_entities(self, node, ents):
        if node.selection is None or not ents:
            return
        self.set_selection(node, entities=[e for e in node.selection.entities if e not in ents])

    def add_graphics_selection(self, node):
        pass

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
                if level == "boundary" and node.kind.endswith((".insulation", ".zc", ".ins", ".free", ".wall", ".mi", ".hard", ".nf", ".zf")):
                    pass
                out |= mine & set(sib.selection.resolve(self.geo.numbers.get(level, [])))
        return out

    def on_entity_clicked(self, level, ent, ctrl):
        node = self.active_sel
        if node is None or node.selection is None or node.selection.level != level:
            return
        self._push_undo()
        ents = list(node.selection.entities)
        if ent in ents:
            ents.remove(ent)
        else:
            ents.append(ent)
            ents.sort()
        node.selection.entities = ents
        node.selection.all = False
        self._mark(node)
        self.graphics.set_selection_state(level, ents)
        self.settings.refresh()

    def activate_selection(self, node: Node | None, show: Node | None = None):
        self.active_sel = node
        target = node or show
        if target is None or target.selection is None:
            return
        if self.geo is None:
            return
        level = target.selection.level
        if self.sdim() == 2 and level == "edge":
            level = "boundary"
        ents = target.selection.resolve(self.geo.numbers.get(level, []))
        if self.graphics.mode != "geometry":
            self.graphics.show_geometry(self.geo, level if node else None, ents)
        else:
            self.graphics.set_selection_state(level if node else None, ents)
            self.graphics.level = level if node else None

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
            if kind.startswith("mp.") or kind == "mesh.ftet" or kind == "mesh.ftri" or kind == "mesh.fquad":
                n.selection.all = True
            if kind == "material":
                mats = [c for c in parent.children if c.kind == "material" and c is not n]
                n.selection.all = not mats
        if kind.startswith("plot.") and "data" in n.props:
            n.props["data"] = "fromparent"
        if kind in ("pg3d", "pg2d", "pg1d") or kind.startswith("eval.") or kind.startswith("export.") or kind.startswith("dset.cut"):
            ds = [d for d in self.datasets() if d.kind == "dset.solution" and d is not n]
            if ds and "data" in n.props:
                n.props["data"] = ds[-1].tag
        if kind in ("dset.cutline",) and self.geo is not None:
            b = self.geo.bbox
            n.props.update({"x0": f"{b[0]:.6g}", "y0": f"{(b[1] + b[4]) / 2:.6g}", "z0": f"{(b[2] + b[5]) / 2:.6g}",
                            "x1": f"{b[3]:.6g}", "y1": f"{(b[1] + b[4]) / 2:.6g}", "z1": f"{(b[2] + b[5]) / 2:.6g}"})
        if kind.startswith("study.") and "physics" in n.props:
            n.props["physics"] = {ph.tag: True for ph in B.physics_nodes(self.model)}
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
        self._push_undo()
        parent = node.parent
        idx = parent.children.index(node)
        parent.remove(node)
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
        return B.physics_nodes(self.model)

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
        if k == "params":
            return [("Load from File", "open", "")] if False else []
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

    def solution_of(self, dset: Node | None) -> Solution | None:
        if dset is None:
            return None
        if dset.kind in ("dset.cutline", "dset.cutpoint"):
            dset = self.model.by_tag(dset.get("data", "")) or next((x for x in self.datasets() if x.kind == "dset.solution"), None)
            if dset is None:
                return None
        st = self.model.by_tag(dset.get("study", ""))
        if st is None:
            return None
        folders = sorted((f for f in glob.glob(os.path.join(self.workdir, st.tag, "sol_*_*"))
                          if os.path.isfile(os.path.join(f, "meta.json"))),      # skip solutions still being computed
                         key=lambda p: tuple(int(x) for x in re.findall(r"\d+", os.path.basename(p))))
        if not folders:
            return None
        sweep = int(dset.get("sweepidx", 0) or 0)
        chosen = [f for f in folders if os.path.basename(f).startswith(f"sol_{sweep}_")] or folders
        folder = chosen[-1]
        if folder not in self.sol_cache:
            try:
                self.sol_cache[folder] = Solution(folder)
            except Exception as exc:
                self.messages.add(f"Could not load solution {folder}: {exc}", "error")
                return None
        return self.sol_cache[folder]

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
        if node.kind == "study.solver":
            pass
        return out

    # ================================================================== selection / graphics updates
    def on_node_selected(self, node: Node):
        self.active_sel = None
        self.settings.show_node(node)
        self._update_contextual(node)
        k = node.kind
        if k.startswith("pg"):
            self.plot(node)
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

    def _update_contextual(self, node):
        tabs = {"geometry": "Geometry", "materials": "Materials", "mesh": "Mesh", "study": "Study", "results": "Results"}
        for anc in node.path():
            if anc.kind.startswith("physics."):
                return
            if anc.kind in tabs and node.kind != "root":
                pass

    def _show_geometry_for(self, node: Node):
        if self.model.component is None:
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
        if node.selection is not None:
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
        k = node.kind
        nt = REGISTRY.get(k)

        def add_kind(menu, kind, parent=node):
            t = REGISTRY[kind]
            a = menu.addAction(icons.icon(t.icon), t.title)
            a.triggered.connect(lambda _=False: self.create_node(kind, parent))
            sd = self.sdim() if self.model.component else 3
            if sd not in t.dims and kind.startswith("geom.") and not node.kind == "geom.workplane":
                a.setEnabled(False)

        if k == "root":
            sub = m.addMenu(icons.icon("component"), "Add Component")
            for s, lbl in (("3D", "3D"), ("2Daxi", "2D Axisymmetric"), ("2D", "2D")):
                a = sub.addAction(lbl)
                a.triggered.connect(lambda _=False, ss=s: self.add_component(ss))
                a.setEnabled(self.model.component is None)
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
                                                                 "geom.ellipse", "geom.polygon")]),
                      ("Booleans and Partitions", [x for x in kinds if x in ("geom.union", "geom.difference", "geom.intersection")]),
                      ("Transforms", [x for x in kinds if x in ("geom.move", "geom.rotate", "geom.scale", "geom.mirror",
                                                                 "geom.array", "geom.copy")]),
                      ("Other", [x for x in kinds if x in ("geom.workplane", "geom.extrude", "geom.revolve", "geom.import",
                                                            "geom.fillet3d", "geom.chamfer3d")])]
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
            m.addAction(icons.icon("build_selected"), "Build Selected", self.build_selected)
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
            m.addSeparator()
            if nt is None or nt.renamable:
                m.addAction(icons.icon("rename"), "Rename", self.tree.edit_current, "F2")
            if nt is not None and nt.deletable and not node.meta.get("default"):
                m.addAction(icons.icon("duplicate"), "Duplicate", lambda: self.duplicate_node(node))
                m.addAction(icons.icon("delete"), "Delete", lambda: self.delete_node(node))
            if nt is not None and nt.can_disable and not node.meta.get("default"):
                if node.enabled:
                    m.addAction(icons.icon("disable"), "Disable", lambda: self.set_enabled(node, False))
                else:
                    m.addAction(icons.icon("enable"), "Enable", lambda: self.set_enabled(node, True))
            m.addAction(icons.icon("up"), "Move Up", lambda: self.move_node(node, -1))
            m.addAction(icons.icon("down"), "Move Down", lambda: self.move_node(node, 1))
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
        if self.model.component is not None:
            return
        self._push_undo()
        comp = B.add_component(self.model, sdim)
        self.tree.refresh(select=comp.child("geometry"))
        self._update_geom_actions()

    def add_physics(self, pid):
        if self.model.component is None:
            self.add_component("3D")
        pd = PHYSICS[pid]
        if self.model.component.get("sdim") not in pd.dims:
            QMessageBox.information(self, APP_NAME, f"{pd.name} is not available in this space dimension.")
            return
        self._push_undo()
        ph = B.add_physics(self.model, pid)
        if not self.model.studies():
            pass
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
        if cur is not None and (cur.kind == "geom.workplane" or (cur.parent is not None and cur.parent.kind == "geom.workplane")):
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
        if kind != "pg1d":
            sol = self.solution_of(self._dataset_for(pg))
            f = self.create_node("plot.surface", pg, select=False)
            if sol is not None:
                f.props["expr"] = default_expr(sol)
            self.tree.refresh(select=pg)
        else:
            cl = next((d for d in self.datasets() if d.kind == "dset.cutline"), None)
            g = self.create_node("plot.linegraph", pg, select=False)
            if cl is not None:
                g.props["data"] = cl.tag
            self.tree.refresh(select=pg)

    def add_plot(self, kind):
        cur = self.tree.current()
        if kind == "plot.deform":
            if cur is not None and cur.kind.startswith("plot.") and kind in REGISTRY[cur.kind].child_kinds(cur):
                self.create_node(kind, cur)
            return
        pg = cur if cur is not None and cur.kind.startswith("pg") else (cur.ancestor("pg") if cur is not None else None)
        if kind in ("plot.linegraph", "plot.globalgraph"):
            if pg is None or pg.kind != "pg1d":
                self.add_plot_group("pg1d")
                pg = self.tree.current()
                if kind == "plot.globalgraph":
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
    def _start_job(self, fn, title, on_done, on_event=None, cancellable=False):
        if self.job is not None:
            # queue it (a newer request with the same title replaces an older queued one)
            self._queue = [q for q in getattr(self, "_queue", []) if q[1] != title] + [(fn, title, on_done, on_event)]
            self.statusBar().showMessage(f"{title}: queued", 3000)
            return True
        self.job = Job(fn, self)
        # bound methods of this QObject => queued delivery on the GUI thread
        self._job_cb = (title, on_done, on_event)
        self.job.done.connect(self._on_job_done)
        self.job.failed.connect(self._on_job_failed)
        self.job.event.connect(self._on_job_event)
        self.pbar.setRange(0, 0)
        self.pbar.show()
        self.statusBar().showMessage(title + "...")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self.job.start()
        return True

    def _on_job_done(self, r):
        self._job_done(r, self._job_cb[1])
        self._next_job()

    def _on_job_failed(self, msg, tb):
        self._job_failed(self._job_cb[0], msg, tb)
        self._next_job()

    def _next_job(self):
        q = getattr(self, "_queue", [])
        if q and self.job is None:
            fn, title, on_done, on_event = q.pop(0)
            self._start_job(fn, title, on_done, on_event)

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

    def _job_failed(self, title, msg, tb):
        QApplication.restoreOverrideCursor()
        self.pbar.hide()
        self.job = None
        self._pending_after_geo = None
        self.messages.add(f"{title} failed: {msg}", "error")
        if tb:
            self.logpane.add(tb)
        self.statusBar().showMessage(f"{title} failed", 6000)
        if "stopped by the user" not in msg:
            QMessageBox.warning(self, APP_NAME, f"{title} failed:\n\n{msg}")

    def stop_job(self):
        if self.job is not None and self.job.runner is not None:
            self.job.runner.stop()

    # -- geometry
    def build_geometry(self, upto: Node | None = None, silent=False, then=None):
        comp = self.model.component
        if comp is None:
            return
        model_snapshot = Model.from_dict(json.loads(json.dumps(self.model.to_dict())))
        geom_tag = comp.child("geometry").tag
        upto_tag = upto.tag if upto is not None else None

        def work(emit):
            g = model_snapshot.component.child("geometry")
            u = model_snapshot.by_tag(upto_tag) if upto_tag else None
            return build_geometry(model_snapshot, g, upto=u)

        def done(res: GeometryResult):
            self.geo = res
            self.geo_stale = upto is not None
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
        self._start_job(work, "Building geometry", done)

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
        model_snapshot = Model.from_dict(json.loads(json.dumps(self.model.to_dict())))

        def work(emit):
            return build_mesh(model_snapshot)

        def done(res):
            self.mesh = res
            self.mesh_stale = False
            self.geo_stale = False
            if self.geo is None or self.geo.count("domain") != res.geometry.count("domain"):
                pass
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
        self._start_job(work, "Building mesh", done)

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
        st_snap = model_snapshot.by_tag(study.tag)
        mesh_cache = self.mesh if (self.mesh is not None and not self.mesh_stale) else None
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

        def on_event(kind, p):
            if kind == "log":
                self.logpane.add(p)
            elif kind == "conv":
                self.conv.add(p)
                self._conv_dirty = True
                self.progress.set(f"Solver: {p[1]}", None, f"{p[3]:.2e}")
            elif kind == "progress":
                frac, text = p
                self.pbar.setRange(0, 100)
                self.pbar.setValue(int(frac * 100))
                self.progress.set(f"{study.label}: Compute", frac)
                self.statusBar().showMessage(f"{study.label}: {text}")
            elif kind == "message":
                self.messages.add(p)
            elif kind == "warning":
                self.messages.add(p, "warning")
            elif kind == "error_line":
                self.messages.add(p, "error")
            elif kind == "mesh":
                self.mesh = p
                self.mesh_stale = False

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

        self._start_job(work, f"Computing {study.label}", done, on_event)

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
        if pg.kind == "pg1d":
            def get_sol(dset, sel):
                sol = self.solution_of(dset)
                if sol is None:
                    return None, -1, 1.0
                return sol, sol.frame_index(pg.get("solnum", "last")), sol.scale
            try:
                draw_1d(self.plot1d.fig, pg, self.model, get_sol, pg.get("solnum", "last"))
            except Exception as exc:
                self.messages.add(f"{pg.label}: {exc}", "error")
            self.plot1d.draw()
            self.gstack.setCurrentIndex(1)
            self.central.setCurrentIndex(0)
            return
        sol = self.solution_of(ds)
        if sol is None:
            self.statusBar().showMessage("No solution available: compute the study first.", 5000)
            if self.geo is not None:
                self.graphics.show_geometry(self.geo)
            return
        fi = sol.frame_index(pg.get("solnum", "last"))
        try:
            scene = build_scene(pg, sol, fi, sol.sdim, self.is_axi())
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
        if node.selection is not None and self.geo is not None:
            ents = node.selection.resolve(self.geo.numbers.get(node.selection.level, []))
        try:
            cols, rows = evaluate_derived(node, self.model, sol, sel, ents)
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
                if pg.kind == "pg1d":
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
                info.setText(elmer_version(inst) or inst.solver)
            else:
                info.setText("<span style='color:#c0392b'>ElmerSolver not found</span>")
        det.clicked.connect(detect)
        row.addWidget(le, 1)
        row.addWidget(b)
        row.addWidget(det)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f.addRow("Elmer installation folder (ELMER_HOME):", w)
        f.addRow("", info)
        cb = QCheckBox("Save solutions in the model file")
        cb.setChecked(self.qs.value("save_solutions", True, bool))
        f.addRow("", cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec():
            inst = find_elmer(le.text().strip() or None)
            if inst:
                self.inst = inst
                self.qs.setValue("elmer_home", inst.home)
                self.messages.add(f"Elmer: {inst.solver} {elmer_version(inst)}")
            else:
                QMessageBox.warning(self, APP_NAME, "ElmerSolver was not found in that folder.")
            self.qs.setValue("save_solutions", cb.isChecked())

    def about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<h3>{APP_NAME} {__version__}</h3><p>A model-builder desktop for the open-source "
                          f"<b>Elmer</b> finite element solver (CSC - IT Center for Science).</p>"
                          f"<p>Geometry and meshing: gmsh / OpenCASCADE. Graphics: VTK. GUI: Qt.</p>"
                          f"<p>Elmer: {self.inst.solver if self.inst else 'not found'}</p>"
                          "<p>Mouse: left-drag rotate (pan in 2D), right-drag pan, wheel zoom; click to select "
                          "entities when a selection list is active.</p>")
