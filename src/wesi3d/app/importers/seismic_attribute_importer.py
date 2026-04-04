#!/usr/bin/env python3
"""
Standalone SEG-Y volume import dialog.

This module can be imported by the main viewer or run directly for
dialog debugging and reuse in small helper tools.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from PySide6 import QtCore, QtGui, QtWidgets

from wesi3d.utils.constants import INLINE_FIELD, XLINE_FIELD


@dataclass(frozen=True)
class SegyImportOptions:
    path: str
    file_type: str
    name: str
    target_category: str
    interval_inline: int
    interval_xline: int
    interval_sample: int
    step_inline: float
    step_xline: float
    step_sample: float
    inline_field: int
    xline_field: int
    x_field: int
    y_field: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class SeismicAttributeImportDialog(QtWidgets.QDialog):
    CELL_WIDTH = 96
    COMBO_WIDTH = 120
    BUTTON_WIDTH = 96
    ROW_SPACING = 8
    SECTION_SPACING = 14
    HORIZONTAL_MARGIN = 12
    DATA_RANGE_COLUMNS = 4
    TABLE_WIDTH = CELL_WIDTH * DATA_RANGE_COLUMNS + ROW_SPACING * (DATA_RANGE_COLUMNS - 1)
    FILE_EDIT_WIDTH = TABLE_WIDTH - BUTTON_WIDTH - ROW_SPACING
    FORM_LABEL_WIDTH = 96
    SCAN_HEADER_MIN = 0
    SCAN_HEADER_MAX = 239
    SCAN_SUPPORTED_FILE_TYPES = {"segy", "su"}

    @classmethod
    def _header_label(cls, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setFixedWidth(cls.CELL_WIDTH)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return label

    @staticmethod
    def _form_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setFixedWidth(SeismicAttributeImportDialog.FORM_LABEL_WIDTH)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return label

    @classmethod
    def _new_row_widget(cls, widgets: list[QtWidgets.QWidget]) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(cls.ROW_SPACING)
        for widget in widgets:
            row.addWidget(widget)
        row.addStretch(1)
        return container

    def _new_line_edit(
        self,
        text: str = "",
        *,
        validator: QtGui.QValidator | None = None,
        fixed_width: int | None = None,
    ) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(text)
        edit.setFixedWidth(self.CELL_WIDTH if fixed_width is None else fixed_width)
        edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if validator is not None:
            edit.setValidator(validator)
        return edit

    def _new_button(self, text: str, *, width: int | None = None) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setFixedWidth(self.BUTTON_WIDTH if width is None else width)
        button.setAutoDefault(False)
        button.setDefault(False)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        return button

    @classmethod
    def _new_combo(cls, items: list[tuple[str, str]], *, width: int, current_value: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setFixedWidth(width)
        combo.setMinimumContentsLength(8)
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        for label, value in items:
            combo.addItem(label, value)
        for index in range(combo.count()):
            if str(combo.itemData(index)) == current_value:
                combo.setCurrentIndex(index)
                break
        return combo

    def _build_table_section_widget(
        self,
        section_name: str,
        column_names: list[str],
        row_names: list[str],
        cells: dict[str, list[QtWidgets.QWidget]],
    ) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(self.ROW_SPACING)
        grid.setVerticalSpacing(4)

        grid.addWidget(self._form_label(section_name), 0, 0)
        for column_index, column_name in enumerate(column_names, start=1):
            grid.addWidget(self._header_label(column_name), 0, column_index)

        for row_index, row_name in enumerate(row_names, start=1):
            grid.addWidget(self._form_label(row_name), row_index, 0)
            for column_index, cell_widget in enumerate(cells[row_name], start=1):
                grid.addWidget(cell_widget, row_index, column_index)

        grid.setColumnMinimumWidth(0, self.FORM_LABEL_WIDTH)
        grid.setColumnStretch(len(column_names) + 1, 1)
        return widget

    def _header_map_values_for_scan(self) -> dict[str, int] | None:
        values: dict[str, int] = {}
        for key, widget in (
            ("inline", self.inline_field_edit),
            ("xline", self.xline_field_edit),
            ("x", self.x_field_edit),
            ("y", self.y_field_edit),
        ):
            text = widget.text().strip()
            if not text:
                return None
            try:
                value = int(text)
            except ValueError:
                return None
            if not (self.SCAN_HEADER_MIN <= value <= self.SCAN_HEADER_MAX):
                return None
            values[key] = value
        return values

    def _scan_ready_state(self) -> tuple[bool, str]:
        path_text = self.path_edit.text().strip()
        if not path_text:
            return False, "input file is empty"

        file_type = str(self.file_type_combo.currentData())
        if file_type not in self.SCAN_SUPPORTED_FILE_TYPES:
            return False, f"file type {file_type!r} does not support scan"

        header_values = self._header_map_values_for_scan()
        if header_values is None:
            return False, "header map is incomplete or out of range"

        return True, "ready"

    def _perform_scan(self, trigger: str) -> None:
        self._append_info(f"scan requested by {trigger}")
        self._append_info(f"path={self.path_edit.text().strip()}")
        self._append_info(f"file_type={self.file_type_combo.currentData()}")
        self._append_info(f"header_map={self._header_map_values_for_scan()}")

    def _maybe_scan(self, trigger: str) -> None:
        ready, reason = self._scan_ready_state()
        self._append_info(f"maybe_scan trigger={trigger} ready={ready} reason={reason}")
        if ready:
            self._perform_scan(trigger)

    def _on_input_file_changed(self, _text: str) -> None:
        self._maybe_scan("input_file_changed")

    def _on_file_type_changed(self, _index: int) -> None:
        self._update_header_field_state()
        self._maybe_scan("file_type_changed")

    def _on_header_map_changed(self, _text: str) -> None:
        self._maybe_scan("header_map_changed")

    def _append_info(self, message: str) -> None:
        line = f"[SeismicAttributeImportDialog] {message}"
        print(line, flush=True)
        self.info_log.appendPlainText(line)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        target_category: str = "seismic",
        initial_values: dict[str, object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Seismic/Attribute Data")
        self.setModal(True)
        initial_values = {} if initial_values is None else dict(initial_values)

        int_validator = QtGui.QIntValidator(1, 10**9, self)
        float_validator = QtGui.QDoubleValidator(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(self.HORIZONTAL_MARGIN, 12, self.HORIZONTAL_MARGIN, 12)
        layout.setSpacing(0)

        top_form = QtWidgets.QFormLayout()
        top_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        top_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        top_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        top_form.setHorizontalSpacing(8)
        top_form.setVerticalSpacing(4)
        top_form.setContentsMargins(0, 0, 0, 0)

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setFixedWidth(self.FILE_EDIT_WIDTH)
        self.path_edit.setText(str(initial_values.get("path", "")))
        browse_button = self._new_button("Browse")
        browse_button.clicked.connect(self._browse_path)
        self.file_type_combo = self._new_combo(
            [("Segy", "segy"), ("Su", "su"), ("Binary", "binary")],
            width=self.COMBO_WIDTH,
            current_value=str(initial_values.get("file_type", "segy")),
        )
        top_form.addRow(self._form_label("Input File"), self._new_row_widget([self.path_edit, browse_button]))
        top_form.addRow(self._form_label("File Type"), self.file_type_combo)
        layout.addLayout(top_form)
        layout.addSpacing(self.SECTION_SPACING)

        self.inline_field_edit = self._new_line_edit(str(initial_values.get("inline_field", 17)), validator=int_validator)
        self.xline_field_edit = self._new_line_edit(str(initial_values.get("xline_field", 21)), validator=int_validator)
        self.x_field_edit = self._new_line_edit(str(initial_values.get("x_field", 73)), validator=int_validator)
        self.y_field_edit = self._new_line_edit(str(initial_values.get("y_field", 77)), validator=int_validator)
        self._header_field_edits = [
            self.inline_field_edit,
            self.xline_field_edit,
            self.x_field_edit,
            self.y_field_edit,
        ]
        layout.addWidget(
            self._build_table_section_widget(
                "Header Map",
                ["Inline", "Cxline", "X", "Y"],
                ["Position"],
                {"Position": [self.inline_field_edit, self.xline_field_edit, self.x_field_edit, self.y_field_edit]},
            )
        )
        layout.addSpacing(self.SECTION_SPACING)

        self.begin_inline_edit = self._new_line_edit(str(initial_values.get("begin_inline", "")), validator=float_validator)
        self.begin_xline_edit = self._new_line_edit(str(initial_values.get("begin_xline", "")), validator=float_validator)
        self.begin_sample_edit = self._new_line_edit(str(initial_values.get("begin_sample", "")), validator=float_validator)
        self.end_inline_edit = self._new_line_edit(str(initial_values.get("end_inline", "")), validator=float_validator)
        self.end_xline_edit = self._new_line_edit(str(initial_values.get("end_xline", "")), validator=float_validator)
        self.end_sample_edit = self._new_line_edit(str(initial_values.get("end_sample", "")), validator=float_validator)
        self.step_inline_edit = self._new_line_edit(str(initial_values.get("step_inline", "")), validator=float_validator)
        self.step_xline_edit = self._new_line_edit(str(initial_values.get("step_xline", "")), validator=float_validator)
        self.step_sample_edit = self._new_line_edit(str(initial_values.get("step_sample", "")), validator=float_validator)
        self.spacing_inline_edit = self._new_line_edit(str(initial_values.get("spacing_inline", "")), validator=int_validator)
        self.spacing_xline_edit = self._new_line_edit(str(initial_values.get("spacing_xline", "")), validator=int_validator)
        self.spacing_sample_edit = self._new_line_edit(str(initial_values.get("spacing_sample", "")), validator=int_validator)
        layout.addWidget(
            self._build_table_section_widget(
                "Data Range",
                ["Begin", "End", "Step", "Spacing"],
                ["Inline", "Cxline", "Sample"],
                {
                    "Inline": [self.begin_inline_edit, self.end_inline_edit, self.step_inline_edit, self.spacing_inline_edit],
                    "Cxline": [self.begin_xline_edit, self.end_xline_edit, self.step_xline_edit, self.spacing_xline_edit],
                    "Sample": [self.begin_sample_edit, self.end_sample_edit, self.step_sample_edit, self.spacing_sample_edit],
                },
            )
        )
        layout.addSpacing(self.SECTION_SPACING)

        self.p0_x_edit = self._new_line_edit(str(initial_values.get("p0_x", "")))
        self.p0_y_edit = self._new_line_edit(str(initial_values.get("p0_y", "")))
        self.p1_x_edit = self._new_line_edit(str(initial_values.get("p1_x", "")))
        self.p1_y_edit = self._new_line_edit(str(initial_values.get("p1_y", "")))
        self.p3_x_edit = self._new_line_edit(str(initial_values.get("p3_x", "")))
        self.p3_y_edit = self._new_line_edit(str(initial_values.get("p3_y", "")))
        layout.addWidget(
            self._build_table_section_widget(
                "Grid Points",
                ["X", "Y"],
                ["P0", "P1", "P3"],
                {"P0": [self.p0_x_edit, self.p0_y_edit], "P1": [self.p1_x_edit, self.p1_y_edit], "P3": [self.p3_x_edit, self.p3_y_edit]},
            )
        )
        layout.addSpacing(self.SECTION_SPACING)

        self.output_name_edit = QtWidgets.QLineEdit()
        self.output_name_edit.setFixedWidth(self.FILE_EDIT_WIDTH)
        self.output_name_edit.setText(str(initial_values.get("output_file", initial_values.get("name", ""))))
        output_browse_button = self._new_button("Browse")
        output_browse_button.clicked.connect(self._browse_output_path)

        output_form = QtWidgets.QFormLayout()
        output_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        output_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        output_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        output_form.setHorizontalSpacing(8)
        output_form.setVerticalSpacing(4)
        output_form.setContentsMargins(0, 0, 0, 0)
        output_form.addRow(self._form_label("Output File"), self._new_row_widget([self.output_name_edit, output_browse_button]))
        layout.addLayout(output_form)

        self.target_combo = self._new_combo(
            [("Seismic", "seismic"), ("Attribute", "attribute")],
            width=self.COMBO_WIDTH,
            current_value=str(initial_values.get("target_category", target_category)),
        )
        cancel_button = self._new_button("Cancel")
        ok_button = self._new_button("OK")
        cancel_button.clicked.connect(self.reject)
        ok_button.clicked.connect(self.accept)

        target_cancel_row = QtWidgets.QWidget()
        target_cancel_layout = QtWidgets.QHBoxLayout(target_cancel_row)
        target_cancel_layout.setContentsMargins(0, 0, 0, 0)
        target_cancel_layout.setSpacing(self.ROW_SPACING)
        target_cancel_layout.addWidget(self.target_combo)
        target_cancel_layout.addStretch(1)
        target_cancel_layout.addWidget(cancel_button)
        target_cancel_row.setFixedWidth(self.FILE_EDIT_WIDTH)
        output_as_row = self._new_row_widget([target_cancel_row, ok_button])

        output_as_form = QtWidgets.QFormLayout()
        output_as_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        output_as_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        output_as_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        output_as_form.setHorizontalSpacing(8)
        output_as_form.setVerticalSpacing(4)
        output_as_form.setContentsMargins(0, 0, 0, 0)
        output_as_form.addRow(self._form_label("Output As"), output_as_row)
        layout.addLayout(output_as_form)
        layout.addSpacing(self.SECTION_SPACING)

        info_container = QtWidgets.QWidget()
        info_container.setContentsMargins(0, 0, 0, 0)
        info_container.setFixedWidth(self.FORM_LABEL_WIDTH + self.TABLE_WIDTH)
        info_layout = QtWidgets.QHBoxLayout(info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(0)
        self.info_log = QtWidgets.QPlainTextEdit()
        self.info_log.setReadOnly(True)
        self.info_log.setMinimumHeight(96)
        self.info_log.setPlaceholderText("Info")
        info_layout.addWidget(self.info_log)
        layout.addWidget(info_container)
        self.setFixedWidth(self.FORM_LABEL_WIDTH + self.TABLE_WIDTH + self.HORIZONTAL_MARGIN * 2)

        self.path_edit.textChanged.connect(self._on_input_file_changed)
        self.file_type_combo.currentIndexChanged.connect(self._on_file_type_changed)
        for widget in self._header_field_edits:
            widget.textChanged.connect(self._on_header_map_changed)
        self._update_header_field_state()

    def _browse_path(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Select Input File", "", "SEG-Y Files (*.sgy *.segy *.su);;All Files (*)")
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        if dialog.exec():
            files = dialog.selectedFiles()
            if files:
                self.path_edit.setText(files[0])

    def _browse_output_path(self) -> None:
        start_path = self.output_name_edit.text().strip() or self.path_edit.text().strip()
        dialog = QtWidgets.QFileDialog(self, "Select Output File", start_path, "All Files (*)")
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        if dialog.exec():
            files = dialog.selectedFiles()
            if files:
                self.output_name_edit.setText(files[0])

    def _update_header_field_state(self) -> None:
        enabled = str(self.file_type_combo.currentData()) != "binary"
        for widget in self._header_field_edits:
            widget.setEnabled(enabled)

    def options(self) -> SegyImportOptions | None:
        path_text = self.path_edit.text().strip()
        if not path_text:
            return None
        try:
            return SegyImportOptions(
                path=path_text,
                file_type=str(self.file_type_combo.currentData()),
                name=self.output_name_edit.text().strip() or Path(path_text).stem,
                target_category=str(self.target_combo.currentData()),
                interval_inline=max(1, int(self.spacing_inline_edit.text().strip() or "1")),
                interval_xline=max(1, int(self.spacing_xline_edit.text().strip() or "1")),
                interval_sample=max(1, int(self.spacing_sample_edit.text().strip() or "1")),
                step_inline=float(self.step_inline_edit.text().strip() or "1"),
                step_xline=float(self.step_xline_edit.text().strip() or "1"),
                step_sample=float(self.step_sample_edit.text().strip() or "1"),
                inline_field=int(self.inline_field_edit.text().strip() or str(INLINE_FIELD)),
                xline_field=int(self.xline_field_edit.text().strip() or str(XLINE_FIELD)),
                x_field=int(self.x_field_edit.text().strip() or "181"),
                y_field=int(self.y_field_edit.text().strip() or "185"),
            )
        except ValueError:
            return None

    def values(self) -> dict[str, object] | None:
        options = self.options()
        return None if options is None else options.as_dict()


def main() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    dialog = SeismicAttributeImportDialog()
    result = dialog.exec()
    if result == int(QtWidgets.QDialog.DialogCode.Accepted):
        options = dialog.options()
        if options is not None:
            print(options.as_dict(), flush=True)
            return 0
    return 1 if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
