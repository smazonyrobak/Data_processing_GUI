from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from pipeline_config import PipelineConfig
from process_utils import close_job, guarded_popen
from repair_tools import (
    RepairError,
    RepairPlan,
    SegmentSpec,
    analyse_segment,
    build_repair_plan,
    parse_ranges,
    parse_values,
    plan_summary,
    write_repaired_run,
)


STAGES = [
    ("ecephys_pipeline", "ecephys_spike_sorting_LNE pipeline"),
    ("custom_ks4", "Custom somatic KS4"),
    ("stimulus_metadata", "Stimulus metadata"),
    ("preprocessing_output", "Build preprocessing output"),
]

LOG_PREFIX = "__PIPELINE_LOG__ "
OUT_PREFIX = "__PIPELINE_OUT__ "


APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR / "processing.ico"
APP_USER_MODEL_ID = "MarcinSzymon.Neuropixels.DataProcessingGUI"


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { background:#161b22; color:#d7e7f5; font-size:10pt; }
        QTabWidget::pane { border:1px solid #2d3a4c; border-radius:6px; }
        QTabBar::tab { background:#111820; border:1px solid #2d3a4c; padding:8px 14px; }
        QTabBar::tab:selected { background:#24415a; }
        QGroupBox { border:1px solid #2d3a4c; border-radius:7px; margin-top:8px; padding:8px; }
        QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget, QTableWidget {
            background:#0f131a; border:1px solid #2d3a4c; border-radius:5px; padding:4px;
        }
        QHeaderView::section { background:#1b2634; color:#d7e7f5; padding:5px; border:1px solid #2d3a4c; }
        QPushButton { background:#24415a; border:1px solid #41627f; border-radius:6px; padding:7px 12px; }
        QPushButton:hover { background:#2d526f; }
        QPushButton:disabled { color:#718092; background:#1b222d; }
        QCheckBox, QRadioButton { spacing:8px; }
        QScrollArea { border:0; }
        QSplitter::handle { background:#2d3a4c; }
        """
    )


class PipelineWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    output = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, cfg: PipelineConfig, stages: list[str], *, allow_existing_output: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.stages = stages
        self.allow_existing_output = allow_existing_output
        self.proc: subprocess.Popen[str] | None = None
        self._stopping = False

    @QtCore.Slot()
    def stop(self) -> None:
        self._stopping = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            close_job(proc)
            try:
                proc.kill()
            except Exception:
                pass

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.cfg.validate_for_run(self.stages, allow_existing_output=self.allow_existing_output)
            config_path = self.cfg.logs_dir / "pipeline_config.json"
            self.cfg.logs_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.save(config_path)
            cmd = [
                self.cfg.processing_python,
                str(APP_DIR / "pipeline_cli.py"),
                "--config",
                str(config_path),
                "--stages",
                ",".join(self.stages),
            ]
            cmd.append("--validated-output")
            self.output.emit(" ".join(cmd))
            env = os.environ.copy()
            env_dir = Path(self.cfg.processing_python).resolve().parent.parent
            env_paths = [
                env_dir,
                env_dir / "Library" / "mingw-w64" / "bin",
                env_dir / "Library" / "usr" / "bin",
                env_dir / "Library" / "bin",
                env_dir / "Scripts",
            ]
            env["PATH"] = os.pathsep.join(str(path) for path in env_paths if path.exists()) + os.pathsep + env.get("PATH", "")
            env["CONDA_PREFIX"] = str(env_dir)
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["MPLBACKEND"] = "Agg"
            proc = guarded_popen(
                cmd,
                cwd=str(APP_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self.proc = proc
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    text = line.rstrip()
                    if text.startswith(LOG_PREFIX):
                        self.log.emit(text[len(LOG_PREFIX) :])
                    elif text.startswith(OUT_PREFIX):
                        self.output.emit(text[len(OUT_PREFIX) :])
                    else:
                        self.output.emit(text)
                code = proc.wait()
            finally:
                close_job(proc)
                self.proc = None
            if self._stopping:
                self.finished.emit(False, "Stopped")
                return
            if code == 0:
                self.finished.emit(True, "Finished")
            else:
                self.finished.emit(False, f"Pipeline failed with exit code {code}")
        except Exception as exc:
            self.output.emit(traceback.format_exc())
            self.finished.emit(False, str(exc))


class PathEdit(QtWidgets.QPlainTextEdit):
    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.setPlainText(text)
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.textChanged.connect(self._sync_display)
        QtCore.QTimer.singleShot(0, self._sync_display)

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:
        self.setPlainText(text)
        self._sync_display()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_display()

    def _sync_display(self) -> None:
        text = self.toPlainText()
        self.setToolTip(text)
        self.document().setTextWidth(max(1, self.viewport().width()))
        height = int(self.document().size().height() + 14)
        target = max(54, min(height, 220))
        if self.height() != target:
            self.setFixedHeight(target)


def add_path_row(form: QtWidgets.QFormLayout, label: str, text: str, browse_title: str, *, file_mode: bool = False) -> tuple[PathEdit, QtWidgets.QPushButton]:
    row = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = PathEdit(text)
    button = QtWidgets.QPushButton("Browse")
    button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
    layout.addWidget(edit, 1)
    layout.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
    form.addRow(label, row)

    def browse() -> None:
        if file_mode:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(row, browse_title, edit.text())
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(row, browse_title, edit.text())
        if path:
            edit.setText(path)

    button.clicked.connect(browse)
    return edit, button


def int_spin(value: int, minimum: int = 0, maximum: int = 999999) -> QtWidgets.QSpinBox:
    widget = QtWidgets.QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(int(value))
    return widget


def float_spin(value: float, minimum: float = 0.0, maximum: float = 999999.0, decimals: int = 4) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(float(value))
    return widget


def combo_box(options: list[str], value: str) -> QtWidgets.QComboBox:
    widget = QtWidgets.QComboBox()
    widget.addItems(options)
    index = widget.findText(value)
    if index >= 0:
        widget.setCurrentIndex(index)
    return widget


def set_combo_value(widget: QtWidgets.QComboBox, value: str) -> None:
    index = widget.findText(value)
    if index >= 0:
        widget.setCurrentIndex(index)
    else:
        widget.addItem(value)
        widget.setCurrentText(value)


def csv_text(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


def parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def dict_edit(data: dict[str, Any]) -> QtWidgets.QPlainTextEdit:
    widget = QtWidgets.QPlainTextEdit(json.dumps(data, indent=2))
    widget.setMaximumHeight(115)
    return widget


def read_json_dict(widget: QtWidgets.QPlainTextEdit) -> dict[str, Any]:
    text = widget.toPlainText().strip()
    return json.loads(text) if text else {}


def check_box(checked: bool = False) -> QtWidgets.QCheckBox:
    widget = QtWidgets.QCheckBox()
    widget.setChecked(bool(checked))
    return widget


def split_arg_string(text: str) -> list[str]:
    return [part.strip() for part in text.split() if part.strip()]


def option_value(tokens: list[str], prefix: str, default: str = "") -> str:
    for token in tokens:
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return default


def option_extras(tokens: list[str], exact: set[str], prefixes: tuple[str, ...]) -> str:
    extras = []
    for token in tokens:
        if token in exact:
            continue
        if any(token.startswith(prefix) for prefix in prefixes):
            continue
        extras.append(token)
    return " ".join(extras)


def parse_extract_string(text: str) -> dict[int, dict[str, str]]:
    parsed: dict[int, dict[str, str]] = {}
    for token in split_arg_string(text):
        if not (token.startswith("-xd=") or token.startswith("-xid=")):
            continue
        edge_type, value = token[1:].split("=", 1)
        parts = [part.strip() for part in value.split(",")]
        if len(parts) < 5:
            continue
        try:
            bit = int(float(parts[3]))
        except ValueError:
            continue
        parsed.setdefault(bit, {})[edge_type] = parts[4]
    return parsed


def extract_string_extras(text: str) -> str:
    extras = []
    for token in split_arg_string(text):
        if token.startswith("-xd=") or token.startswith("-xid="):
            continue
        extras.append(token)
    return " ".join(extras)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neuropixels data processing")
        self.resize(1450, 930)
        if ICON_PATH.exists():
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))

        self.worker_thread: QtCore.QThread | None = None
        self.worker: PipelineWorker | None = None
        self.default_cfg = PipelineConfig()
        self.stage_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.repair_plan: RepairPlan | None = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)

        tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(tabs)

        processing_tab = QtWidgets.QWidget()
        processing_layout = QtWidgets.QVBoxLayout(processing_tab)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        tabs.addTab(processing_tab, "Processing")

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        processing_layout.addWidget(main_splitter)

        self.controls = QtWidgets.QScrollArea()
        self.controls.setWidgetResizable(True)
        controls_body = QtWidgets.QWidget()
        self.controls.setWidget(controls_body)
        controls_layout = QtWidgets.QVBoxLayout(controls_body)
        main_splitter.addWidget(self.controls)

        self.build_path_controls(controls_layout)
        self.build_stage_controls(controls_layout)
        self.build_parameter_controls(controls_layout)
        self.build_button_controls(controls_layout)
        controls_layout.addStretch(1)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setChildrenCollapsible(False)
        main_splitter.addWidget(right)

        log_box = QtWidgets.QGroupBox("Run log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self.run_log = QtWidgets.QPlainTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("Run log")
        log_layout.addWidget(self.run_log)

        console_box = QtWidgets.QGroupBox("Output console")
        console_layout = QtWidgets.QVBoxLayout(console_box)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Output console")
        console_layout.addWidget(self.console)

        right.addWidget(log_box)
        right.addWidget(console_box)
        right.setSizes([360, 520])
        main_splitter.setSizes([880, 570])

        repair_tab = QtWidgets.QWidget()
        tabs.addTab(repair_tab, "Repair")
        self.build_repair_tab(repair_tab)

    def build_repair_tab(self, tab: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left_body = QtWidgets.QWidget()
        left.setWidget(left_body)
        left_layout = QtWidgets.QVBoxLayout(left_body)
        splitter.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        splitter.addWidget(right)
        splitter.setSizes([760, 650])

        intro = QtWidgets.QLabel(
            "Build a repaired stim/cam run without changing the original recordings. "
            "The output folder can later be selected as the Stim/cam run in the Processing tab."
        )
        intro.setWordWrap(True)
        left_layout.addWidget(intro)

        paths_group = QtWidgets.QGroupBox("Repair paths")
        paths_layout = QtWidgets.QVBoxLayout(paths_group)
        paths_form = QtWidgets.QFormLayout()
        paths_layout.addLayout(paths_form)
        left_layout.addWidget(paths_group)

        self.repair_output_folder, _ = add_path_row(
            paths_form,
            "Repair output folder",
            "",
            "Select repaired stim/cam output folder",
        )
        sync_row = QtWidgets.QHBoxLayout()
        self.repair_use_processing_paths_btn = QtWidgets.QPushButton("Use processing paths")
        self.repair_guess_output_btn = QtWidgets.QPushButton("Guess output")
        sync_row.addWidget(self.repair_use_processing_paths_btn)
        sync_row.addWidget(self.repair_guess_output_btn)
        paths_layout.addLayout(sync_row)

        segment_group = QtWidgets.QGroupBox("Source segments")
        segment_layout = QtWidgets.QVBoxLayout(segment_group)
        left_layout.addWidget(segment_group)
        self.repair_segment_table = QtWidgets.QTableWidget(0, 6)
        self.repair_segment_table.setHorizontalHeaderLabels(["Use", "Label", "Folder", "Include key", "Include ranges", "Summary"])
        self.repair_segment_table.horizontalHeader().setStretchLastSection(True)
        self.repair_segment_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.repair_segment_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.repair_segment_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.repair_segment_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        segment_layout.addWidget(self.repair_segment_table)

        segment_buttons = QtWidgets.QHBoxLayout()
        self.repair_add_segment_btn = QtWidgets.QPushButton("Add segment")
        self.repair_add_children_btn = QtWidgets.QPushButton("Add folders from parent")
        self.repair_remove_segment_btn = QtWidgets.QPushButton("Remove selected")
        segment_buttons.addWidget(self.repair_add_segment_btn)
        segment_buttons.addWidget(self.repair_add_children_btn)
        segment_buttons.addWidget(self.repair_remove_segment_btn)
        segment_layout.addLayout(segment_buttons)

        rule_group = QtWidgets.QGroupBox("Repair decisions")
        rule_layout = QtWidgets.QVBoxLayout(rule_group)
        left_layout.addWidget(rule_group)
        rule_form = QtWidgets.QFormLayout()
        rule_layout.addLayout(rule_form)
        self.repair_exclude_steps = QtWidgets.QLineEdit()
        self.repair_exclude_steps.setPlaceholderText("Examples: 1-48, 155, 210-212")
        self.repair_exclude_key = combo_box(["global_step", "source_move_index", "repair_trial_index"], "global_step")
        self.repair_crop_key = combo_box(["repair_trial_index", "global_step", "source_move_index"], "repair_trial_index")
        self.repair_crop_start = int_spin(0, 0, 999999)
        self.repair_crop_end = int_spin(0, 0, 999999)
        rule_form.addRow("Exclude key", self.repair_exclude_key)
        rule_form.addRow("Exclude values", self.repair_exclude_steps)
        rule_form.addRow("Crop key", self.repair_crop_key)
        rule_form.addRow("Crop start (0 = first included)", self.repair_crop_start)
        rule_form.addRow("Crop end (0 = last included)", self.repair_crop_end)

        self.repair_allow_existing_output = QtWidgets.QCheckBox("Allow writing into an existing repair output folder")
        self.repair_allow_existing_output.setToolTip("Known repair files may be replaced. Original source folders are still protected.")
        rule_layout.addWidget(self.repair_allow_existing_output)

        action_row = QtWidgets.QHBoxLayout()
        self.repair_preview_btn = QtWidgets.QPushButton("Preview repair")
        self.repair_write_btn = QtWidgets.QPushButton("Create repaired folder")
        self.repair_use_output_btn = QtWidgets.QPushButton("Use output in Processing tab")
        action_row.addWidget(self.repair_preview_btn)
        action_row.addWidget(self.repair_write_btn)
        action_row.addWidget(self.repair_use_output_btn)
        rule_layout.addLayout(action_row)
        left_layout.addStretch(1)

        summary_group = QtWidgets.QGroupBox("Repair summary")
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        self.repair_summary = QtWidgets.QPlainTextEdit()
        self.repair_summary.setReadOnly(True)
        self.repair_summary.setPlaceholderText("Preview a repair to see exactly which moves will be included, excluded, and used as crop anchors.")
        summary_layout.addWidget(self.repair_summary)
        right.addWidget(summary_group)

        moves_group = QtWidgets.QGroupBox("Included moves preview")
        moves_layout = QtWidgets.QVBoxLayout(moves_group)
        self.repair_moves_table = QtWidgets.QTableWidget(0, 8)
        self.repair_moves_table.setHorizontalHeaderLabels(
            ["Repair", "Global", "Segment", "Source move", "Move label", "Start rel s", "End rel s", "Duration s"]
        )
        self.repair_moves_table.horizontalHeader().setStretchLastSection(True)
        self.repair_moves_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        moves_layout.addWidget(self.repair_moves_table)
        right.addWidget(moves_group)
        right.setSizes([360, 460])

        self.repair_use_processing_paths_btn.clicked.connect(self.use_processing_paths_for_repair)
        self.repair_guess_output_btn.clicked.connect(self.guess_repair_output)
        self.repair_add_segment_btn.clicked.connect(self.add_repair_segment_dialog)
        self.repair_add_children_btn.clicked.connect(self.add_repair_children_dialog)
        self.repair_remove_segment_btn.clicked.connect(self.remove_selected_repair_segments)
        self.repair_preview_btn.clicked.connect(self.preview_repair)
        self.repair_write_btn.clicked.connect(self.write_repair)
        self.repair_use_output_btn.clicked.connect(self.use_repair_output_in_processing_tab)

    def use_processing_paths_for_repair(self) -> None:
        if self.stim_cam_run.text().strip():
            self.add_repair_segment(Path(self.stim_cam_run.text().strip()))
        self.guess_repair_output()

    def guess_repair_output(self) -> None:
        specs = self.repair_segment_specs(show_errors=False)
        if specs:
            base = Path(specs[0].folder)
            self.repair_output_folder.setText(str(base.parent / f"{base.name}_repaired"))
        elif self.stim_cam_run.text().strip():
            base = Path(self.stim_cam_run.text().strip())
            self.repair_output_folder.setText(str(base.parent / f"{base.name}_repaired"))

    def add_repair_segment_dialog(self) -> None:
        start_dir = self.stim_cam_run.text().strip() or str(Path.cwd())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select stim/cam segment folder", start_dir)
        if path:
            self.add_repair_segment(Path(path))

    def add_repair_children_dialog(self) -> None:
        start_dir = self.stim_cam_run.text().strip() or str(Path.cwd())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select parent folder containing stim/cam segments", start_dir)
        if not path:
            return
        parent = Path(path)
        added = 0
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "stimulus_trials.csv").exists() and (child / "events.csv").exists():
                self.add_repair_segment(child)
                added += 1
        if added == 0:
            QtWidgets.QMessageBox.information(self, "No segments found", "No child folders contained both stimulus_trials.csv and events.csv.")
        self.guess_repair_output()

    def add_repair_segment(self, folder: Path) -> None:
        folder = Path(folder)
        existing = {
            self.repair_segment_table.item(row, 2).text()
            for row in range(self.repair_segment_table.rowCount())
            if self.repair_segment_table.item(row, 2) is not None
        }
        if str(folder) in existing:
            return
        row = self.repair_segment_table.rowCount()
        self.repair_segment_table.insertRow(row)

        use_item = QtWidgets.QTableWidgetItem("")
        use_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable)
        use_item.setCheckState(QtCore.Qt.CheckState.Checked)
        self.repair_segment_table.setItem(row, 0, use_item)

        label_item = QtWidgets.QTableWidgetItem(folder.name)
        folder_item = QtWidgets.QTableWidgetItem(str(folder))
        folder_item.setFlags(folder_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        key_item = QtWidgets.QTableWidgetItem("global_step")
        ranges_item = QtWidgets.QTableWidgetItem("")
        summary_item = QtWidgets.QTableWidgetItem("Not scanned")
        summary_item.setFlags(summary_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

        try:
            segment = analyse_segment(SegmentSpec(folder=folder, label=folder.name), row + 1)
            summary_item.setText(
                f"{segment['trial_count']} trials, {segment['completed_move_count']} completed, "
                f"{segment['incomplete_move_count']} incomplete"
            )
        except Exception as exc:
            summary_item.setText(f"Scan failed: {exc}")

        for column, item in enumerate([use_item, label_item, folder_item, key_item, ranges_item, summary_item]):
            self.repair_segment_table.setItem(row, column, item)
        self.guess_repair_output()

    def remove_selected_repair_segments(self) -> None:
        rows = sorted({index.row() for index in self.repair_segment_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.repair_segment_table.removeRow(row)
        self.repair_plan = None

    def _table_text(self, table: QtWidgets.QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _optional_int_text(self, text: str) -> int | None:
        if not text.strip():
            return None
        return int(float(text.strip()))

    def repair_segment_specs(self, *, show_errors: bool = True) -> list[SegmentSpec]:
        specs: list[SegmentSpec] = []
        errors: list[str] = []
        for row in range(self.repair_segment_table.rowCount()):
            use_item = self.repair_segment_table.item(row, 0)
            if use_item is not None and use_item.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            folder_text = self._table_text(self.repair_segment_table, row, 2)
            if not folder_text:
                continue
            try:
                specs.append(
                    SegmentSpec(
                        folder=Path(folder_text),
                        label=self._table_text(self.repair_segment_table, row, 1),
                        include_key=self._table_text(self.repair_segment_table, row, 3) or "global_step",
                        include_ranges=parse_ranges(self._table_text(self.repair_segment_table, row, 4)),
                    )
                )
            except Exception as exc:
                errors.append(f"Segment row {row + 1}: {exc}")
        if errors and show_errors:
            raise RepairError("\n".join(errors))
        return specs

    def build_current_repair_plan(self) -> RepairPlan:
        specs = self.repair_segment_specs()
        output_text = self.repair_output_folder.text().strip()
        if not output_text:
            raise RepairError("Choose a repair output folder.")
        excluded = parse_values(self.repair_exclude_steps.text())
        crop_start = self.repair_crop_start.value() or None
        crop_end = self.repair_crop_end.value() or None
        return build_repair_plan(
            specs,
            Path(output_text),
            exclude_key=self.repair_exclude_key.currentText(),
            exclude_values=excluded,
            crop_key=self.repair_crop_key.currentText(),
            crop_start=crop_start,
            crop_end=crop_end,
        )

    def preview_repair(self) -> None:
        try:
            self.repair_plan = self.build_current_repair_plan()
            self.repair_summary.setPlainText(plan_summary(self.repair_plan))
            self.populate_repair_moves_table(self.repair_plan)
        except Exception as exc:
            self.repair_plan = None
            self.repair_summary.setPlainText(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Repair preview failed", str(exc))

    def populate_repair_moves_table(self, plan: RepairPlan) -> None:
        rows = plan.included_moves[:1000]
        self.repair_moves_table.setRowCount(len(rows))
        for row, move in enumerate(rows):
            values = [
                move.get("repair_trial_index"),
                move.get("global_step"),
                move.get("source_segment_label"),
                move.get("source_move_index"),
                move.get("move_label"),
                move.get("source_start_relative_to_first_camera_frame_s"),
                move.get("source_end_relative_to_first_camera_frame_s"),
                move.get("source_duration_s"),
            ]
            for column, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.3f}"
                else:
                    text = "" if value is None else str(value)
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.repair_moves_table.setItem(row, column, item)
        if len(plan.included_moves) > len(rows):
            self.repair_summary.appendPlainText(f"\nMove table shows first {len(rows)} included moves only.")

    def write_repair(self) -> None:
        try:
            plan = self.build_current_repair_plan()
            if plan.errors:
                self.repair_summary.setPlainText(plan_summary(plan))
                raise RepairError("Repair plan has errors; fix them before writing.")
            write_repaired_run(plan, allow_existing_output=self.repair_allow_existing_output.isChecked())
            self.repair_plan = plan
            self.repair_summary.setPlainText(plan_summary(plan) + "\n\nWrote repaired stim/cam folder.")
            QtWidgets.QMessageBox.information(self, "Repair written", f"Repaired stim/cam folder created:\n{plan.output_folder}")
        except Exception as exc:
            self.repair_summary.appendPlainText("\n\n" + traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Repair write failed", str(exc))

    def use_repair_output_in_processing_tab(self) -> None:
        output_text = self.repair_output_folder.text().strip()
        if not output_text:
            QtWidgets.QMessageBox.warning(self, "No repair output", "Choose or create a repair output folder first.")
            return
        self.stim_cam_run.setText(output_text)
        self.status.setText("Repair output selected as stim/cam run")

    def build_path_controls(self, parent: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Paths")
        layout = QtWidgets.QVBoxLayout(group)
        parent.addWidget(group)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)
        left = QtWidgets.QWidget()
        right = QtWidgets.QWidget()
        left_form = QtWidgets.QFormLayout(left)
        right_form = QtWidgets.QFormLayout(right)
        left_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        right_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([520, 520])

        cfg = self.default_cfg
        bundled_ecephys = APP_DIR / "ecephys_spike_sorting_LNE" / "ecephys_spike_sorting"
        bundled_custom_ks4 = APP_DIR / "Kilosort_state_enhanced"
        self.processing_python, _ = add_path_row(left_form, "Processing Python", cfg.processing_python, "Select processing python.exe", file_mode=True)
        self.spikeglx_run, _ = add_path_row(left_form, "SpikeGLX run (read-only)", cfg.spikeglx_run, "Select SpikeGLX run folder")
        self.stim_cam_run, _ = add_path_row(left_form, "Stim/cam run", cfg.stim_cam_run, "Select stim/cam run folder")
        self.preprocessed_root, _ = add_path_row(left_form, "Output folder", cfg.preprocessed_root, "Select processing output folder")
        self.preprocessed_root.setPlaceholderText("Required: dedicated output parent, outside both input folders")

        self.catgt_exe, _ = add_path_row(right_form, "CatGT", cfg.catgt_exe, "Select CatGT.exe or CatGT folder", file_mode=True)
        self.tprime_exe, _ = add_path_row(right_form, "TPrime", cfg.tprime_exe, "Select TPrime runit.bat, TPrime.exe, or folder", file_mode=True)
        self.cwaves_path, _ = add_path_row(right_form, "C_Waves", cfg.cwaves_path, "Select C_Waves runit.bat or folder", file_mode=False)
        self.ecephys_directory, _ = add_path_row(
            right_form,
            "ecephys package",
            cfg.ecephys_directory or str(bundled_ecephys),
            "Select ecephys_spike_sorting package folder",
        )
        self.npy_matlab_repository, _ = add_path_row(right_form, "npy-matlab repo", cfg.npy_matlab_repository, "Select npy-matlab repository")
        self.kilosort_repository, _ = add_path_row(right_form, "Kilosort repo", cfg.kilosort_repository, "Select Kilosort repository")
        self.kilosort20_repository, _ = add_path_row(right_form, "KS2.0 repo", cfg.kilosort20_repository, "Select KS2.0 repository")
        self.kilosort25_repository, _ = add_path_row(right_form, "KS2.5 repo", cfg.kilosort25_repository, "Select KS2.5 repository")
        self.kilosort30_repository, _ = add_path_row(right_form, "KS3.0 repo", cfg.kilosort30_repository, "Select KS3.0 repository")
        self.custom_kilosort_repository, _ = add_path_row(
            right_form,
            "Custom KS4 repo",
            cfg.custom_kilosort_repository or str(bundled_custom_ks4),
            "Select custom KS4 repository",
        )

        notice = QtWidgets.QLabel(
            "Data inputs are only SpikeGLX run and stim/cam run. CatGT, Kilosort, JSON, logs, and final data are written only inside the selected output folder."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

    def build_stage_controls(self, parent: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Stages")
        layout = QtWidgets.QVBoxLayout(group)
        parent.addWidget(group)
        for key, label in STAGES:
            check = QtWidgets.QCheckBox(label)
            check.setChecked(True)
            self.stage_checks[key] = check
            layout.addWidget(check)

    def build_parameter_controls(self, parent: QtWidgets.QVBoxLayout) -> None:
        cfg = self.default_cfg
        columns = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        columns.setChildrenCollapsible(False)
        parent.addWidget(columns)

        left_widget = QtWidgets.QWidget()
        right_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        right = QtWidgets.QVBoxLayout(right_widget)
        left.setContentsMargins(0, 0, 0, 0)
        right.setContentsMargins(0, 0, 0, 0)
        columns.addWidget(left_widget)
        columns.addWidget(right_widget)
        columns.setSizes([520, 520])

        self.build_run_spec_controls(left, cfg)
        self.build_timing_controls(left, cfg)
        self.build_ecephys_controls(left, cfg)
        self.build_stimulus_output_controls(left, cfg)

        self.build_catgt_controls(right, cfg)
        self.build_kilosort_controls(right, cfg)
        self.build_region_controls(right, cfg)
        left.addStretch(1)
        right.addStretch(1)

    def build_run_spec_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("Run and probes")
        layout = QtWidgets.QVBoxLayout(group)
        parent.addWidget(group)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.gate_index = int_spin(cfg.gate_index)
        self.trial_start = int_spin(cfg.trial_start)
        self.trial_end = int_spin(cfg.trial_end)
        self.n_probes = int_spin(cfg.n_probes, 1, 8)
        self.n_probes.valueChanged.connect(lambda _value: self.update_probe_rows())
        for label, widget in [
            ("Gate index", self.gate_index),
            ("Trial start", self.trial_start),
            ("Trial end", self.trial_end),
            ("Number of probes", self.n_probes),
        ]:
            form.addRow(label, widget)

        self.probe_rows_widget = QtWidgets.QWidget()
        self.probe_rows_layout = QtWidgets.QGridLayout(self.probe_rows_widget)
        self.probe_rows_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.probe_rows_widget)
        self.probe_id_spins: list[QtWidgets.QSpinBox] = []
        self.probe_region_boxes: list[QtWidgets.QComboBox] = []
        self.update_probe_rows(cfg.probe_ids, cfg.brain_regions)

    def update_probe_rows(self, probe_ids: list[int] | None = None, regions: list[str] | None = None) -> None:
        if not hasattr(self, "probe_rows_layout"):
            return
        current_ids = [widget.value() for widget in getattr(self, "probe_id_spins", [])]
        current_regions = [widget.currentText() for widget in getattr(self, "probe_region_boxes", [])]
        probe_ids = list(probe_ids) if probe_ids is not None else current_ids
        regions = list(regions) if regions is not None else current_regions

        while self.probe_rows_layout.count():
            item = self.probe_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.probe_id_spins = []
        self.probe_region_boxes = []
        self.probe_rows_layout.addWidget(QtWidgets.QLabel("Probe"), 0, 0)
        self.probe_rows_layout.addWidget(QtWidgets.QLabel("Probe ID"), 0, 1)
        self.probe_rows_layout.addWidget(QtWidgets.QLabel("Brain region"), 0, 2)
        for row in range(self.n_probes.value()):
            probe_id = probe_ids[row] if row < len(probe_ids) else row
            region = regions[row] if row < len(regions) else (regions[0] if regions else "cortex")
            probe_spin = int_spin(int(probe_id), 0, 999)
            region_box = combo_box(["cortex", "thalamus", "medulla", "default"], "cortex")
            region_box.setEditable(True)
            set_combo_value(region_box, str(region))
            self.probe_rows_layout.addWidget(QtWidgets.QLabel(str(row + 1)), row + 1, 0)
            self.probe_rows_layout.addWidget(probe_spin, row + 1, 1)
            self.probe_rows_layout.addWidget(region_box, row + 1, 2)
            self.probe_id_spins.append(probe_spin)
            self.probe_region_boxes.append(region_box)
        self.probe_rows_layout.setColumnStretch(2, 1)

    def build_timing_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("Synchronization")
        form = QtWidgets.QFormLayout(group)
        parent.addWidget(group)
        self.ni_word = int_spin(cfg.ni_word)
        self.sync_bit = int_spin(cfg.sync_bit, 0, 31)
        self.sync_threshold = int_spin(cfg.sync_threshold)
        self.ap_sync_word = int_spin(cfg.ap_sync_word)
        self.ap_sync_bit = int_spin(cfg.ap_sync_bit, 0, 31)
        self.event_threshold = int_spin(cfg.event_threshold)
        self.tprime_syncperiod_s = float_spin(cfg.tprime_syncperiod_s, 0.001, 60.0, 3)
        self.tprime_reference_stream = combo_box(["imec0", "ni", "probe0", "nidq", "imec1", "imec2", "imec3"], cfg.tprime_reference_stream)
        self.sync_crop_enabled = check_box(False)
        self.sync_crop_start_index = int_spin(0, 0, 100000000)
        self.sync_crop_end_index = int_spin(0, 0, 100000000)
        for label, widget in [
            ("NI word", self.ni_word),
            ("NI sync bit", self.sync_bit),
            ("Sync threshold", self.sync_threshold),
            ("AP sync word", self.ap_sync_word),
            ("AP sync bit", self.ap_sync_bit),
            ("Default event threshold", self.event_threshold),
            ("TPrime sync period (s)", self.tprime_syncperiod_s),
            ("TPrime reference", self.tprime_reference_stream),
        ]:
            form.addRow(label, widget)
        note = QtWidgets.QLabel("Crop the run once, using the sorting crop in seconds under Kilosort. Sync-edge event-file cropping is kept off by the GUI.")
        note.setWordWrap(True)
        form.addRow("Crop policy", note)

    def build_ecephys_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("Package pipeline")
        form = QtWidgets.QFormLayout(group)
        parent.addWidget(group)
        self.log_name = QtWidgets.QLineEdit(cfg.log_name)
        self.run_catgt = check_box(cfg.run_catgt)
        self.run_tprime = check_box(cfg.run_tprime)
        self.ni_present = check_box(cfg.ni_present)
        self.obx_present = check_box(cfg.obx_present)
        self.onebox_streams = QtWidgets.QLineEdit(cfg.onebox_streams)
        self.ks_ver = combo_box(["4", "3.0", "2.5", "2.0"], cfg.ks_ver)
        self.modules = QtWidgets.QLineEdit(csv_text(cfg.modules))
        for label, widget in [
            ("Log file", self.log_name),
            ("Run CatGT", self.run_catgt),
            ("Run TPrime", self.run_tprime),
            ("NI present", self.ni_present),
            ("OneBox present", self.obx_present),
            ("OneBox streams", self.onebox_streams),
            ("Kilosort version", self.ks_ver),
            ("Modules", self.modules),
        ]:
            form.addRow(label, widget)

    def build_catgt_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("CatGT command builder")
        layout = QtWidgets.QVBoxLayout(group)
        parent.addWidget(group)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        tokens = split_arg_string(cfg.catgt_cmd_string)
        self.catgt_prb_fld = check_box("-prb_fld" in tokens)
        self.catgt_out_prb_fld = check_box("-out_prb_fld" in tokens)
        self.process_lf = check_box(cfg.process_lf)
        self.car_mode = combo_box(["gblcar", "gbldmx", "loccar", "None"], cfg.car_mode)
        self.loccar_min = int_spin(cfg.loccar_min, 0, 10000)
        self.loccar_max = int_spin(cfg.loccar_max, 0, 10000)
        self.catgt_ap_filter = QtWidgets.QLineEdit(option_value(tokens, "-apfilter=", "butter,12,300,10000"))
        self.catgt_lf_filter = QtWidgets.QLineEdit(option_value(tokens, "-lffilter=", "butter,12,1,500"))
        self.catgt_gfix = QtWidgets.QLineEdit(option_value(tokens, "-gfix=", "0.4,0.10,0.02"))
        self.catgt_extra_args = QtWidgets.QLineEdit(option_extras(tokens, {"-prb_fld", "-out_prb_fld"}, ("-apfilter=", "-lffilter=", "-gfix=")))
        for label, widget in [
            ("Folder per probe input", self.catgt_prb_fld),
            ("Folder per probe output", self.catgt_out_prb_fld),
            ("Process LF stream", self.process_lf),
            ("CAR mode", self.car_mode),
            ("loccar min um", self.loccar_min),
            ("loccar max um", self.loccar_max),
            ("AP filter", self.catgt_ap_filter),
            ("LF filter", self.catgt_lf_filter),
            ("gfix", self.catgt_gfix),
            ("Extra CatGT args", self.catgt_extra_args),
        ]:
            form.addRow(label, widget)

        self.catgt_raw_override = check_box(False)
        self.catgt_raw_cmd = QtWidgets.QPlainTextEdit(cfg.catgt_cmd_string)
        self.catgt_raw_cmd.setMaximumHeight(72)
        self.catgt_raw_cmd.setEnabled(False)
        self.catgt_preview = QtWidgets.QPlainTextEdit()
        self.catgt_preview.setReadOnly(True)
        self.catgt_preview.setMaximumHeight(72)
        form.addRow("Use raw command", self.catgt_raw_override)
        form.addRow("Raw command", self.catgt_raw_cmd)
        form.addRow("Generated command", self.catgt_preview)

        self.event_rows: dict[str, dict[str, QtWidgets.QWidget]] = {}
        event_box = QtWidgets.QGroupBox("NI / OneBox event extraction")
        event_layout = QtWidgets.QVBoxLayout(event_box)
        layout.addWidget(event_box)
        self.event_grid = QtWidgets.QGridLayout()
        event_layout.addLayout(self.event_grid)
        parsed_extract = parse_extract_string(cfg.ni_obx_extract_string)
        self.build_event_grid(parsed_extract, cfg)

        self.extract_extra_args = QtWidgets.QLineEdit(extract_string_extras(cfg.ni_obx_extract_string))
        self.extract_raw_override = check_box(False)
        self.extract_raw_cmd = QtWidgets.QPlainTextEdit(cfg.ni_obx_extract_string)
        self.extract_raw_cmd.setMaximumHeight(72)
        self.extract_raw_cmd.setEnabled(False)
        self.extract_preview = QtWidgets.QPlainTextEdit()
        self.extract_preview.setReadOnly(True)
        self.extract_preview.setMaximumHeight(72)
        extract_form = QtWidgets.QFormLayout()
        event_layout.addLayout(extract_form)
        extract_form.addRow("Extra extract args", self.extract_extra_args)
        extract_form.addRow("Use raw extract string", self.extract_raw_override)
        extract_form.addRow("Raw extract string", self.extract_raw_cmd)
        extract_form.addRow("Generated extract string", self.extract_preview)

        self.catgt_raw_override.toggled.connect(self.catgt_raw_cmd.setEnabled)
        self.extract_raw_override.toggled.connect(self.extract_raw_cmd.setEnabled)
        self.connect_catgt_preview_signals()
        self.update_catgt_previews()

    def build_event_grid(self, parsed_extract: dict[int, dict[str, str]], cfg: PipelineConfig) -> None:
        labels = ["Event", "Bit", "Rising threshold", "Falling edge", "Falling threshold"]
        for column, label in enumerate(labels):
            self.event_grid.addWidget(QtWidgets.QLabel(label), 0, column)
        specs = [
            ("rotation", "Rotation", cfg.rotation_bit),
            ("switching", "Switching", cfg.switching_bit),
            ("camera", "Camera", cfg.camera_bit),
        ]
        for row, (key, label, bit) in enumerate(specs, start=1):
            edge = parsed_extract.get(int(bit), {})
            bit_spin = int_spin(int(bit), 0, 31)
            rise = QtWidgets.QLineEdit(edge.get("xd", str(cfg.event_threshold)))
            fall_enabled = check_box("xid" in edge)
            fall = QtWidgets.QLineEdit(edge.get("xid", str(cfg.event_threshold)))
            self.event_grid.addWidget(QtWidgets.QLabel(label), row, 0)
            self.event_grid.addWidget(bit_spin, row, 1)
            self.event_grid.addWidget(rise, row, 2)
            self.event_grid.addWidget(fall_enabled, row, 3)
            self.event_grid.addWidget(fall, row, 4)
            self.event_rows[key] = {
                "bit": bit_spin,
                "rise": rise,
                "fall_enabled": fall_enabled,
                "fall": fall,
            }

    def build_kilosort_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("Kilosort and C_Waves")
        form = QtWidgets.QFormLayout(group)
        parent.addWidget(group)
        self.probe_geometry_mode = combo_box(["metadata", "single_shank", "custom_json"], cfg.probe_geometry_mode)
        self.custom_probe_geometry, _ = add_path_row(form, "Custom probe JSON", cfg.custom_probe_geometry, "Select Kilosort probe JSON", file_mode=True)
        self.ks_remDup = int_spin(cfg.ks_remDup, 0, 1)
        self.ks_saveRez = int_spin(cfg.ks_saveRez, 0, 1)
        self.ks_copy_fproc = int_spin(cfg.ks_copy_fproc, 0, 1)
        self.ks_templateRadius_um = int_spin(cfg.ks_templateRadius_um, 1, 10000)
        self.ks_whiteningRadius_um = int_spin(cfg.ks_whiteningRadius_um, 1, 10000)
        self.ks_minfr_goodchannels = float_spin(cfg.ks_minfr_goodchannels, 0.0, 1000.0, 4)
        self.ks_CAR = int_spin(cfg.ks_CAR, 0, 1)
        self.ks_nblocks = int_spin(cfg.ks_nblocks, 1, 1000)
        self.ks_doFilter = int_spin(cfg.ks_doFilter, 0, 1)
        self.ks4_duplicate_spike_ms = float_spin(cfg.ks4_duplicate_spike_ms, 0.0, 1000.0, 4)
        self.ks4_min_template_size_um = int_spin(cfg.ks4_min_template_size_um, 1, 10000)
        self.ks4_det = check_box(cfg.ks4_det)
        self.ks_tmin = float_spin(cfg.ks_tmin, -1.0, 999999.0, 3)
        self.ks_tmax = float_spin(cfg.ks_tmax, -1.0, 999999.0, 3)
        self.ks_CSBseed = int_spin(cfg.ks_CSBseed, 0, 1000000)
        self.ks_LTseed = int_spin(cfg.ks_LTseed, 0, 1000000)
        self.ks_helper_noise_threshold = int_spin(cfg.ks_helper_noise_threshold, 0, 1000000)
        self.c_waves_snr_um = int_spin(cfg.c_waves_snr_um, 1, 10000)
        self.c_waves_calc_half = check_box(cfg.c_waves_calc_half)
        self.include_pc_metrics = check_box(cfg.include_pc_metrics)
        self.noise_template_use_rf = check_box(cfg.noise_template_use_rf)
        self.custom_ks4_reference_duration_s = float_spin(cfg.custom_ks4_reference_duration_s, 0.0, 999999.0, 3)
        self.custom_ks4_run_quality_metrics = check_box(cfg.custom_ks4_run_quality_metrics)
        self.custom_ks4_run_quality_metrics.setChecked(True)
        self.custom_ks4_run_quality_metrics.setEnabled(False)
        self.somatic_fragment_merge_max_depth_um = float_spin(cfg.somatic_fragment_merge_max_depth_um, 0.0, 10000.0, 3)
        self.somatic_fragment_merge_same_shank_only = check_box(cfg.somatic_fragment_merge_same_shank_only)
        self.somatic_fragment_merge_min_soma_similarity = float_spin(cfg.somatic_fragment_merge_min_soma_similarity, 0.0, 1.0, 4)
        self.somatic_fragment_merge_max_isi_violation_fraction = float_spin(cfg.somatic_fragment_merge_max_isi_violation_fraction, 0.0, 1.0, 4)
        self.somatic_fragment_merge_max_duplicate_fraction = float_spin(cfg.somatic_fragment_merge_max_duplicate_fraction, 0.0, 1.0, 4)
        self.somatic_state_group_full_template_similarity = float_spin(cfg.somatic_state_group_full_template_similarity, 0.0, 1.0, 4)
        self.somatic_refractory_ms = float_spin(cfg.somatic_refractory_ms, 0.0, 1000.0, 4)
        self.somatic_duplicate_ms = float_spin(cfg.somatic_duplicate_ms, 0.0, 0.5, 4)
        self.somatic_conflict_ratio_threshold = float_spin(cfg.somatic_conflict_ratio_threshold, 0.0, 1.0, 4)
        self.somatic_max_spikes_per_unit_for_conflict_metrics = int_spin(cfg.somatic_max_spikes_per_unit_for_conflict_metrics, 1, 10000000)
        self.somatic_state_channel_radius = int_spin(cfg.somatic_state_channel_radius, 0, 10000)
        for label, widget in [
            ("Geometry mode", self.probe_geometry_mode),
            ("ks remDup", self.ks_remDup),
            ("ks saveRez", self.ks_saveRez),
            ("ks copy fproc", self.ks_copy_fproc),
            ("Template radius um", self.ks_templateRadius_um),
            ("Whitening radius um", self.ks_whiteningRadius_um),
            ("Min FR good channels", self.ks_minfr_goodchannels),
            ("KS CAR", self.ks_CAR),
            ("KS nblocks", self.ks_nblocks),
            ("KS doFilter", self.ks_doFilter),
            ("KS4 duplicate spike ms", self.ks4_duplicate_spike_ms),
            ("KS4 min template um", self.ks4_min_template_size_um),
            ("KS4 deterministic", self.ks4_det),
            ("Sort start s (ks_tmin)", self.ks_tmin),
            ("Sort end s (ks_tmax, -1 = end)", self.ks_tmax),
            ("KS CSB seed", self.ks_CSBseed),
            ("KS LT seed", self.ks_LTseed),
            ("KS helper noise threshold", self.ks_helper_noise_threshold),
            ("C_Waves SNR um", self.c_waves_snr_um),
            ("C_Waves half run", self.c_waves_calc_half),
            ("Include PC metrics", self.include_pc_metrics),
            ("Noise RF classifier", self.noise_template_use_rf),
            ("Custom KS4 duration s", self.custom_ks4_reference_duration_s),
            ("Write unit QC outputs", self.custom_ks4_run_quality_metrics),
            ("Somatic merge max depth um", self.somatic_fragment_merge_max_depth_um),
            ("Somatic merge same shank", self.somatic_fragment_merge_same_shank_only),
            ("Somatic min soma similarity", self.somatic_fragment_merge_min_soma_similarity),
            ("Somatic max ISI fraction", self.somatic_fragment_merge_max_isi_violation_fraction),
            ("Somatic max duplicate fraction", self.somatic_fragment_merge_max_duplicate_fraction),
            ("Somatic state template similarity", self.somatic_state_group_full_template_similarity),
            ("Somatic refractory ms", self.somatic_refractory_ms),
            ("Somatic duplicate ms", self.somatic_duplicate_ms),
            ("Somatic conflict ratio", self.somatic_conflict_ratio_threshold),
            ("Somatic conflict sample cap", self.somatic_max_spikes_per_unit_for_conflict_metrics),
            ("Somatic state channel radius", self.somatic_state_channel_radius),
        ]:
            form.addRow(label, widget)

    def build_region_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("Region parameter dictionaries")
        form = QtWidgets.QFormLayout(group)
        parent.addWidget(group)
        self.ref_per_ms_by_region = dict_edit(cfg.ref_per_ms_by_region)
        self.ks_th2_by_region = dict_edit(cfg.ks_th2_by_region)
        self.ks_th3_by_region = dict_edit(cfg.ks_th3_by_region)
        self.ks_th4_by_region = dict_edit(cfg.ks_th4_by_region)
        form.addRow("Ref period ms", self.ref_per_ms_by_region)
        form.addRow("KS2 thresholds", self.ks_th2_by_region)
        form.addRow("KS3 thresholds", self.ks_th3_by_region)
        form.addRow("KS4 thresholds", self.ks_th4_by_region)

    def build_stimulus_output_controls(self, parent: QtWidgets.QVBoxLayout, cfg: PipelineConfig) -> None:
        group = QtWidgets.QGroupBox("Stimulus metadata and final output")
        form = QtWidgets.QFormLayout(group)
        parent.addWidget(group)
        self.drop_first_switching = int_spin(cfg.drop_first_switching_intervals, 0, 1000)
        self.drop_first_rotation = int_spin(cfg.drop_first_rotation_intervals, 0, 1000)
        self.create_aux_timepoints = check_box(cfg.create_aux_timepoints)
        self.event_ex_param_str = QtWidgets.QLineEdit(cfg.event_ex_param_str)
        for label, widget in [
            ("Drop first switching intervals", self.drop_first_switching),
            ("Drop first rotation intervals", self.drop_first_rotation),
            ("Create aux timepoints", self.create_aux_timepoints),
            ("PSTH event extract", self.event_ex_param_str),
        ]:
            form.addRow(label, widget)

    def connect_catgt_preview_signals(self) -> None:
        widgets = [
            self.catgt_prb_fld,
            self.catgt_out_prb_fld,
            self.process_lf,
            self.catgt_raw_override,
            self.extract_raw_override,
        ]
        for widget in widgets:
            widget.toggled.connect(lambda _checked: self.update_catgt_previews())
        line_edits = [
            self.catgt_ap_filter,
            self.catgt_lf_filter,
            self.catgt_gfix,
            self.catgt_extra_args,
            self.extract_extra_args,
        ]
        for widget in line_edits:
            widget.textChanged.connect(lambda _text: self.update_catgt_previews())
        for row in self.event_rows.values():
            row["bit"].valueChanged.connect(lambda _value: self.update_catgt_previews())
            row["rise"].textChanged.connect(lambda _text: self.update_catgt_previews())
            row["fall_enabled"].toggled.connect(lambda _checked: self.update_catgt_previews())
            row["fall"].textChanged.connect(lambda _text: self.update_catgt_previews())
        self.ni_word.valueChanged.connect(lambda _value: self.update_catgt_previews())
        self.event_threshold.valueChanged.connect(lambda _value: self.update_catgt_previews())

    def build_catgt_cmd_string(self) -> str:
        parts = []
        if self.catgt_prb_fld.isChecked():
            parts.append("-prb_fld")
        if self.catgt_out_prb_fld.isChecked():
            parts.append("-out_prb_fld")
        if self.catgt_ap_filter.text().strip():
            parts.append(f"-apfilter={self.catgt_ap_filter.text().strip()}")
        if self.process_lf.isChecked() and self.catgt_lf_filter.text().strip():
            parts.append(f"-lffilter={self.catgt_lf_filter.text().strip()}")
        if self.catgt_gfix.text().strip():
            parts.append(f"-gfix={self.catgt_gfix.text().strip()}")
        if self.catgt_extra_args.text().strip():
            parts.append(self.catgt_extra_args.text().strip())
        return " ".join(parts)

    def build_extract_string(self) -> str:
        parts = []
        for key in ["rotation", "switching", "camera"]:
            row = self.event_rows[key]
            bit = row["bit"].value()
            rise = row["rise"].text().strip() or str(self.event_threshold.value())
            parts.append(f"-xd={self.ni_word.value()},0,-1,{bit},{rise}")
            if row["fall_enabled"].isChecked():
                fall = row["fall"].text().strip() or rise
                parts.append(f"-xid={self.ni_word.value()},0,-1,{bit},{fall}")
        if self.extract_extra_args.text().strip():
            parts.append(self.extract_extra_args.text().strip())
        return " ".join(parts)

    def update_catgt_previews(self) -> None:
        if hasattr(self, "catgt_preview"):
            self.catgt_preview.setPlainText(self.build_catgt_cmd_string())
        if hasattr(self, "extract_preview"):
            self.extract_preview.setPlainText(self.build_extract_string())

    def build_button_controls(self, parent: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Run")
        layout = QtWidgets.QVBoxLayout(group)
        parent.addWidget(group)

        row = QtWidgets.QHBoxLayout()
        self.run_selected_btn = QtWidgets.QPushButton("Run selected stages")
        self.run_all_btn = QtWidgets.QPushButton("Run all")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.run_selected_btn)
        row.addWidget(self.run_all_btn)
        row.addWidget(self.stop_btn)
        layout.addLayout(row)

        self.resume_output = QtWidgets.QCheckBox("Resume existing output folder")
        self.resume_output.setToolTip("Allow continuing a partially completed run in the selected output folder.")
        layout.addWidget(self.resume_output)

        row = QtWidgets.QHBoxLayout()
        self.save_cfg_btn = QtWidgets.QPushButton("Save config")
        self.load_cfg_btn = QtWidgets.QPushButton("Load config")
        row.addWidget(self.save_cfg_btn)
        row.addWidget(self.load_cfg_btn)
        layout.addLayout(row)

        self.status = QtWidgets.QLabel("Idle")
        self.status.setStyleSheet("color:#9fb4c8;")
        layout.addWidget(self.status)

        self.run_selected_btn.clicked.connect(lambda: self.start_run(self.selected_stage_keys()))
        self.run_all_btn.clicked.connect(lambda: self.start_run([key for key, _label in STAGES]))
        self.stop_btn.clicked.connect(self.stop_current_run)
        self.save_cfg_btn.clicked.connect(self.save_config_dialog)
        self.load_cfg_btn.clicked.connect(self.load_config_dialog)

    def parse_probe_ids(self) -> list[int]:
        return [widget.value() for widget in self.probe_id_spins]

    def parse_brain_regions(self) -> list[str]:
        return [widget.currentText().strip() or "default" for widget in self.probe_region_boxes]

    def collect_config(self) -> PipelineConfig:
        catgt_cmd_string = self.catgt_raw_cmd.toPlainText().strip() if self.catgt_raw_override.isChecked() else self.build_catgt_cmd_string()
        ni_obx_extract_string = self.extract_raw_cmd.toPlainText().strip() if self.extract_raw_override.isChecked() else self.build_extract_string()
        return PipelineConfig(
            processing_python=self.processing_python.text().strip(),
            spikeglx_run=self.spikeglx_run.text().strip(),
            stim_cam_run=self.stim_cam_run.text().strip(),
            catgt_exe=self.catgt_exe.text().strip(),
            tprime_exe=self.tprime_exe.text().strip(),
            cwaves_path=self.cwaves_path.text().strip(),
            preprocessed_root=self.preprocessed_root.text().strip(),
            catgt_dest="",
            json_directory="",
            kilosort_output_tmp="",
            ecephys_directory=self.ecephys_directory.text().strip(),
            npy_matlab_repository=self.npy_matlab_repository.text().strip(),
            kilosort_repository=self.kilosort_repository.text().strip(),
            kilosort20_repository=self.kilosort20_repository.text().strip(),
            kilosort25_repository=self.kilosort25_repository.text().strip(),
            kilosort30_repository=self.kilosort30_repository.text().strip(),
            custom_kilosort_repository=self.custom_kilosort_repository.text().strip(),
            custom_ks4_reference_duration_s=self.custom_ks4_reference_duration_s.value(),
            custom_ks4_run_quality_metrics=True,
            somatic_fragment_merge_max_depth_um=self.somatic_fragment_merge_max_depth_um.value(),
            somatic_fragment_merge_same_shank_only=self.somatic_fragment_merge_same_shank_only.isChecked(),
            somatic_fragment_merge_min_soma_similarity=self.somatic_fragment_merge_min_soma_similarity.value(),
            somatic_fragment_merge_max_isi_violation_fraction=self.somatic_fragment_merge_max_isi_violation_fraction.value(),
            somatic_fragment_merge_max_duplicate_fraction=self.somatic_fragment_merge_max_duplicate_fraction.value(),
            somatic_state_group_full_template_similarity=self.somatic_state_group_full_template_similarity.value(),
            somatic_refractory_ms=self.somatic_refractory_ms.value(),
            somatic_duplicate_ms=self.somatic_duplicate_ms.value(),
            somatic_conflict_ratio_threshold=self.somatic_conflict_ratio_threshold.value(),
            somatic_max_spikes_per_unit_for_conflict_metrics=self.somatic_max_spikes_per_unit_for_conflict_metrics.value(),
            somatic_state_channel_radius=self.somatic_state_channel_radius.value(),
            gate_index=self.gate_index.value(),
            trial_start=self.trial_start.value(),
            trial_end=self.trial_end.value(),
            n_probes=self.n_probes.value(),
            probe_ids=self.parse_probe_ids(),
            brain_regions=self.parse_brain_regions(),
            onebox_streams=self.onebox_streams.text().strip(),
            ni_word=self.ni_word.value(),
            sync_bit=self.sync_bit.value(),
            rotation_bit=self.event_rows["rotation"]["bit"].value(),
            switching_bit=self.event_rows["switching"]["bit"].value(),
            camera_bit=self.event_rows["camera"]["bit"].value(),
            tprime_syncperiod_s=self.tprime_syncperiod_s.value(),
            tprime_reference_stream=self.tprime_reference_stream.currentText(),
            ap_sync_word=self.ap_sync_word.value(),
            ap_sync_bit=self.ap_sync_bit.value(),
            sync_threshold=self.sync_threshold.value(),
            event_threshold=self.event_threshold.value(),
            sync_crop_enabled=False,
            sync_crop_start_index=0,
            sync_crop_end_index=0,
            drop_first_switching_intervals=self.drop_first_switching.value(),
            drop_first_rotation_intervals=self.drop_first_rotation.value(),
            catgt_ap_filter=self.catgt_ap_filter.text().strip(),
            catgt_loccar_um=f"{self.loccar_min.value()},{self.loccar_max.value()}",
            catgt_gfix=self.catgt_gfix.text().strip(),
            log_name=self.log_name.text().strip(),
            run_catgt=self.run_catgt.isChecked(),
            run_tprime=self.run_tprime.isChecked(),
            ni_present=self.ni_present.isChecked(),
            obx_present=self.obx_present.isChecked(),
            car_mode=self.car_mode.currentText(),
            loccar_min=self.loccar_min.value(),
            loccar_max=self.loccar_max.value(),
            process_lf=self.process_lf.isChecked(),
            catgt_cmd_string=catgt_cmd_string,
            ni_obx_extract_string=ni_obx_extract_string,
            create_aux_timepoints=self.create_aux_timepoints.isChecked(),
            event_ex_param_str=self.event_ex_param_str.text().strip(),
            probe_geometry_mode=self.probe_geometry_mode.currentText(),
            custom_probe_geometry=self.custom_probe_geometry.text().strip(),
            ks_ver=self.ks_ver.currentText(),
            modules=parse_csv_text(self.modules.text()),
            ref_per_ms_by_region=read_json_dict(self.ref_per_ms_by_region),
            ks_th2_by_region=read_json_dict(self.ks_th2_by_region),
            ks_th3_by_region=read_json_dict(self.ks_th3_by_region),
            ks_th4_by_region=read_json_dict(self.ks_th4_by_region),
            ks_remDup=self.ks_remDup.value(),
            ks_saveRez=self.ks_saveRez.value(),
            ks_copy_fproc=self.ks_copy_fproc.value(),
            ks_templateRadius_um=self.ks_templateRadius_um.value(),
            ks_whiteningRadius_um=self.ks_whiteningRadius_um.value(),
            ks_minfr_goodchannels=self.ks_minfr_goodchannels.value(),
            ks_CAR=self.ks_CAR.value(),
            ks_nblocks=self.ks_nblocks.value(),
            ks_doFilter=self.ks_doFilter.value(),
            ks4_duplicate_spike_ms=self.ks4_duplicate_spike_ms.value(),
            ks4_min_template_size_um=self.ks4_min_template_size_um.value(),
            ks4_det=self.ks4_det.isChecked(),
            ks_tmin=self.ks_tmin.value(),
            ks_tmax=self.ks_tmax.value(),
            ks_CSBseed=self.ks_CSBseed.value(),
            ks_LTseed=self.ks_LTseed.value(),
            ks_helper_noise_threshold=self.ks_helper_noise_threshold.value(),
            c_waves_snr_um=self.c_waves_snr_um.value(),
            c_waves_calc_half=self.c_waves_calc_half.isChecked(),
            include_pc_metrics=self.include_pc_metrics.isChecked(),
            noise_template_use_rf=self.noise_template_use_rf.isChecked(),
        )

    def apply_config(self, cfg: PipelineConfig) -> None:
        self.processing_python.setText(cfg.processing_python)
        self.spikeglx_run.setText(cfg.spikeglx_run)
        self.stim_cam_run.setText(cfg.stim_cam_run)
        self.catgt_exe.setText(cfg.catgt_exe)
        self.tprime_exe.setText(cfg.tprime_exe)
        self.cwaves_path.setText(cfg.cwaves_path)
        self.preprocessed_root.setText(cfg.preprocessed_root)
        self.ecephys_directory.setText(cfg.ecephys_directory)
        self.npy_matlab_repository.setText(cfg.npy_matlab_repository)
        self.kilosort_repository.setText(cfg.kilosort_repository)
        self.kilosort20_repository.setText(cfg.kilosort20_repository)
        self.kilosort25_repository.setText(cfg.kilosort25_repository)
        self.kilosort30_repository.setText(cfg.kilosort30_repository)
        self.custom_kilosort_repository.setText(cfg.custom_kilosort_repository or str(APP_DIR / "Kilosort_state_enhanced"))
        self.gate_index.setValue(cfg.gate_index)
        self.trial_start.setValue(cfg.trial_start)
        self.trial_end.setValue(cfg.trial_end)
        self.n_probes.blockSignals(True)
        self.n_probes.setValue(cfg.n_probes)
        self.n_probes.blockSignals(False)
        self.update_probe_rows(cfg.probe_ids, cfg.brain_regions)
        self.onebox_streams.setText(cfg.onebox_streams)
        self.ni_word.setValue(cfg.ni_word)
        self.sync_bit.setValue(cfg.sync_bit)
        self.event_rows["rotation"]["bit"].setValue(cfg.rotation_bit)
        self.event_rows["switching"]["bit"].setValue(cfg.switching_bit)
        self.event_rows["camera"]["bit"].setValue(cfg.camera_bit)
        self.tprime_syncperiod_s.setValue(cfg.tprime_syncperiod_s)
        set_combo_value(self.tprime_reference_stream, cfg.tprime_reference_stream)
        self.ap_sync_word.setValue(cfg.ap_sync_word)
        self.ap_sync_bit.setValue(cfg.ap_sync_bit)
        self.sync_threshold.setValue(cfg.sync_threshold)
        self.event_threshold.setValue(cfg.event_threshold)
        self.sync_crop_enabled.setChecked(False)
        self.sync_crop_start_index.setValue(0)
        self.sync_crop_end_index.setValue(0)
        self.drop_first_switching.setValue(cfg.drop_first_switching_intervals)
        self.drop_first_rotation.setValue(cfg.drop_first_rotation_intervals)
        catgt_tokens = split_arg_string(cfg.catgt_cmd_string)
        self.catgt_prb_fld.setChecked("-prb_fld" in catgt_tokens)
        self.catgt_out_prb_fld.setChecked("-out_prb_fld" in catgt_tokens)
        self.catgt_ap_filter.setText(option_value(catgt_tokens, "-apfilter=", cfg.catgt_ap_filter))
        self.catgt_lf_filter.setText(option_value(catgt_tokens, "-lffilter=", "butter,12,1,500"))
        self.catgt_gfix.setText(option_value(catgt_tokens, "-gfix=", cfg.catgt_gfix))
        self.catgt_extra_args.setText(option_extras(catgt_tokens, {"-prb_fld", "-out_prb_fld"}, ("-apfilter=", "-lffilter=", "-gfix=")))
        self.catgt_raw_override.setChecked(False)
        self.catgt_raw_cmd.setPlainText(cfg.catgt_cmd_string)
        self.log_name.setText(cfg.log_name)
        self.run_catgt.setChecked(cfg.run_catgt)
        self.run_tprime.setChecked(cfg.run_tprime)
        self.ni_present.setChecked(cfg.ni_present)
        self.obx_present.setChecked(cfg.obx_present)
        set_combo_value(self.car_mode, cfg.car_mode)
        self.loccar_min.setValue(cfg.loccar_min)
        self.loccar_max.setValue(cfg.loccar_max)
        self.process_lf.setChecked(cfg.process_lf)
        parsed_extract = parse_extract_string(cfg.ni_obx_extract_string)
        for key in ["rotation", "switching", "camera"]:
            bit = self.event_rows[key]["bit"].value()
            edge = parsed_extract.get(int(bit), {})
            self.event_rows[key]["rise"].setText(edge.get("xd", str(cfg.event_threshold)))
            self.event_rows[key]["fall_enabled"].setChecked("xid" in edge)
            self.event_rows[key]["fall"].setText(edge.get("xid", str(cfg.event_threshold)))
        self.extract_extra_args.setText(extract_string_extras(cfg.ni_obx_extract_string))
        self.extract_raw_override.setChecked(False)
        self.extract_raw_cmd.setPlainText(cfg.ni_obx_extract_string)
        self.create_aux_timepoints.setChecked(cfg.create_aux_timepoints)
        self.event_ex_param_str.setText(cfg.event_ex_param_str)
        set_combo_value(self.probe_geometry_mode, cfg.probe_geometry_mode)
        self.custom_probe_geometry.setText(cfg.custom_probe_geometry)
        set_combo_value(self.ks_ver, cfg.ks_ver)
        self.modules.setText(csv_text(cfg.modules))
        self.ref_per_ms_by_region.setPlainText(json.dumps(cfg.ref_per_ms_by_region, indent=2))
        self.ks_th2_by_region.setPlainText(json.dumps(cfg.ks_th2_by_region, indent=2))
        self.ks_th3_by_region.setPlainText(json.dumps(cfg.ks_th3_by_region, indent=2))
        self.ks_th4_by_region.setPlainText(json.dumps(cfg.ks_th4_by_region, indent=2))
        self.ks_remDup.setValue(cfg.ks_remDup)
        self.ks_saveRez.setValue(cfg.ks_saveRez)
        self.ks_copy_fproc.setValue(cfg.ks_copy_fproc)
        self.ks_templateRadius_um.setValue(cfg.ks_templateRadius_um)
        self.ks_whiteningRadius_um.setValue(cfg.ks_whiteningRadius_um)
        self.ks_minfr_goodchannels.setValue(cfg.ks_minfr_goodchannels)
        self.ks_CAR.setValue(cfg.ks_CAR)
        self.ks_nblocks.setValue(cfg.ks_nblocks)
        self.ks_doFilter.setValue(cfg.ks_doFilter)
        self.ks4_duplicate_spike_ms.setValue(cfg.ks4_duplicate_spike_ms)
        self.ks4_min_template_size_um.setValue(cfg.ks4_min_template_size_um)
        self.ks4_det.setChecked(cfg.ks4_det)
        self.ks_tmin.setValue(cfg.ks_tmin)
        self.ks_tmax.setValue(cfg.ks_tmax)
        self.ks_CSBseed.setValue(cfg.ks_CSBseed)
        self.ks_LTseed.setValue(cfg.ks_LTseed)
        self.ks_helper_noise_threshold.setValue(cfg.ks_helper_noise_threshold)
        self.c_waves_snr_um.setValue(cfg.c_waves_snr_um)
        self.c_waves_calc_half.setChecked(cfg.c_waves_calc_half)
        self.include_pc_metrics.setChecked(cfg.include_pc_metrics)
        self.noise_template_use_rf.setChecked(cfg.noise_template_use_rf)
        self.custom_ks4_reference_duration_s.setValue(cfg.custom_ks4_reference_duration_s)
        self.custom_ks4_run_quality_metrics.setChecked(True)
        self.somatic_fragment_merge_max_depth_um.setValue(cfg.somatic_fragment_merge_max_depth_um)
        self.somatic_fragment_merge_same_shank_only.setChecked(cfg.somatic_fragment_merge_same_shank_only)
        self.somatic_fragment_merge_min_soma_similarity.setValue(cfg.somatic_fragment_merge_min_soma_similarity)
        self.somatic_fragment_merge_max_isi_violation_fraction.setValue(cfg.somatic_fragment_merge_max_isi_violation_fraction)
        self.somatic_fragment_merge_max_duplicate_fraction.setValue(cfg.somatic_fragment_merge_max_duplicate_fraction)
        self.somatic_state_group_full_template_similarity.setValue(cfg.somatic_state_group_full_template_similarity)
        self.somatic_refractory_ms.setValue(cfg.somatic_refractory_ms)
        self.somatic_duplicate_ms.setValue(cfg.somatic_duplicate_ms)
        self.somatic_conflict_ratio_threshold.setValue(cfg.somatic_conflict_ratio_threshold)
        self.somatic_max_spikes_per_unit_for_conflict_metrics.setValue(cfg.somatic_max_spikes_per_unit_for_conflict_metrics)
        self.somatic_state_channel_radius.setValue(cfg.somatic_state_channel_radius)
        self.update_catgt_previews()

    def selected_stage_keys(self) -> list[str]:
        return [key for key, check in self.stage_checks.items() if check.isChecked()]

    def append_log(self, text: str) -> None:
        self.run_log.appendPlainText(text)

    def append_output(self, text: str) -> None:
        self.console.appendPlainText(text)

    def start_run(self, stage_keys: list[str]) -> None:
        if not stage_keys:
            QtWidgets.QMessageBox.warning(self, "No stages selected", "Select at least one stage to run.")
            return
        try:
            cfg = self.collect_config()
            allow_existing_output = self.resume_output.isChecked()
            cfg.validate_for_run(stage_keys, allow_existing_output=allow_existing_output)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid configuration", str(exc))
            return

        self.run_log.clear()
        self.console.clear()
        self.set_running(True)
        self.worker_thread = QtCore.QThread(self)
        self.worker = PipelineWorker(cfg, stage_keys, allow_existing_output=allow_existing_output)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.output.connect(self.append_output)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def set_running(self, running: bool) -> None:
        self.run_selected_btn.setEnabled(not running)
        self.run_all_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.save_cfg_btn.setEnabled(not running)
        self.load_cfg_btn.setEnabled(not running)
        self.resume_output.setEnabled(not running)
        self.status.setText("Running" if running else "Idle")

    def stop_current_run(self) -> None:
        if self.worker is None:
            return
        self.status.setText("Stopping")
        self.worker.stop()

    def on_worker_finished(self, ok: bool, message: str) -> None:
        self.set_running(False)
        self.status.setText("Finished" if ok else message)
        self.append_log(message)
        if not ok and message != "Stopped":
            QtWidgets.QMessageBox.critical(self, "Pipeline failed", message)
        self.worker = None
        self.worker_thread = None

    def save_config_dialog(self) -> None:
        cfg = self.collect_config()
        default = str(APP_DIR / "pipeline_config.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save pipeline config", default, "JSON files (*.json)")
        if path:
            cfg.save(Path(path))
            self.status.setText(f"Saved config: {path}")

    def load_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load pipeline config", str(APP_DIR), "JSON files (*.json)")
        if path:
            self.apply_config(PipelineConfig.load(Path(path)))
            self.status.setText(f"Loaded config: {path}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_current_run()
        super().closeEvent(event)


def main() -> int:
    set_windows_app_id()
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)
    if ICON_PATH.exists():
        app.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
