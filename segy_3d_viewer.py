#!/usr/bin/env python3
"""
3D SEG-Y viewer for a velocity cube.

Default behavior:
- open ./vel.sgy
- read inline / crossline from trace header bytes 189 / 193
- downsample a large cube to a manageable volume
- display three orthogonal slices with VTK
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import segyio
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: segyio\n"
        "Install with: pip install segyio vtk numpy"
    ) from exc

try:
    import vtk
    from vtk.util import numpy_support
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: vtk\n"
        "Install with: pip install vtk"
    ) from exc
try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing VTK Qt bridge\n"
        "Install a VTK build with Qt support."
    ) from exc

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PySide6\n"
        "Install with: pip install PySide6"
    ) from exc

INLINE_FIELD = 189
XLINE_FIELD = 193


@dataclass
class SegyGeometry:
    inlines: np.ndarray
    xlines: np.ndarray
    sample_axis: np.ndarray
    trace_index_grid: np.ndarray


@dataclass
class AttributeVolume:
    name: str
    image: vtk.vtkImageData
    lut: vtk.vtkLookupTable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEG-Y slice viewer")
    parser.add_argument(
        "segy_path",
        nargs="?",
        default="vel.sgy",
        help="Path to SEG-Y file. Defaults to ./vel.sgy",
    )
    parser.add_argument(
        "--debug-ui",
        action="store_true",
        help="Print extra diagnostics for VTK window startup",
    )
    parser.add_argument(
        "--interval-inline",
        type=int,
        default=4,
        help="Inline downsampling interval, e.g. 4 means take every 4th inline",
    )
    parser.add_argument(
        "--interval-xline",
        type=int,
        default=4,
        help="Crossline downsampling interval, e.g. 4 means take every 4th crossline",
    )
    parser.add_argument(
        "--interval-sample",
        type=int,
        default=4,
        help="Sample downsampling interval, e.g. 4 means take every 4th sample",
    )
    parser.add_argument(
        "--step-inline",
        type=float,
        default=20.0,
        help="Displayed inline spacing in the 3D scene",
    )
    parser.add_argument(
        "--step-xline",
        type=float,
        default=20.0,
        help="Displayed crossline spacing in the 3D scene",
    )
    parser.add_argument(
        "--step-sample",
        type=float,
        default=10.0,
        help="Displayed sample spacing in the 3D scene",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Clip symmetric amplitudes/velocities by percentile for rendering",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.85,
        help="Opacity for the displayed slices",
    )
    return parser.parse_args()


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug-ui] {message}", flush=True)


def normalize_macos_gui_env(debug_ui: bool) -> None:
    if sys.platform != "darwin":
        return
    display = os.environ.get("DISPLAY")
    if not display:
        return
    debug_log(debug_ui, f"unsetting DISPLAY for macOS GUI startup: {display}")
    os.environ.pop("DISPLAY", None)


def validate_interval(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def detect_regular_grid(iline_vals: np.ndarray, xline_vals: np.ndarray) -> np.ndarray:
    unique_ilines = np.unique(iline_vals)
    unique_xlines = np.unique(xline_vals)
    ni = len(unique_ilines)
    nx = len(unique_xlines)
    ntraces = len(iline_vals)

    if ni * nx != ntraces:
        raise ValueError(
            "Trace geometry is not a full regular inline/xline grid. "
            f"unique_ilines={ni}, unique_xlines={nx}, traces={ntraces}"
        )

    sort_idx = np.lexsort((xline_vals, iline_vals))
    sorted_ilines = iline_vals[sort_idx].reshape(ni, nx)
    sorted_xlines = xline_vals[sort_idx].reshape(ni, nx)

    if not np.array_equal(sorted_ilines[:, 0], unique_ilines):
        raise ValueError(
            "Inline headers do not form a consistent regular grid."
        )
    if not np.array_equal(sorted_xlines[0, :], unique_xlines):
        raise ValueError(
            "Crossline headers do not form a consistent regular grid."
        )

    return sort_idx.reshape(ni, nx)


def load_geometry(segy_path: Path) -> SegyGeometry:
    with segyio.open(str(segy_path), "r", strict=False, ignore_geometry=True) as segy:
        iline_vals = np.asarray(segy.attributes(INLINE_FIELD)[:], dtype=np.int64)
        xline_vals = np.asarray(segy.attributes(XLINE_FIELD)[:], dtype=np.int64)
        sample_axis = np.asarray(segy.samples, dtype=np.float32)

    unique_ilines = np.unique(iline_vals)
    unique_xlines = np.unique(xline_vals)
    trace_index_grid = detect_regular_grid(iline_vals, xline_vals)

    return SegyGeometry(
        inlines=unique_ilines,
        xlines=unique_xlines,
        sample_axis=sample_axis,
        trace_index_grid=trace_index_grid,
    )


def build_downsampled_volume(
    segy_path: Path,
    geometry: SegyGeometry,
    interval_inline: int,
    interval_xline: int,
    interval_sample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    il_idx = np.arange(
        0,
        len(geometry.inlines),
        validate_interval("interval_inline", interval_inline),
        dtype=np.int64,
    )
    xl_idx = np.arange(
        0,
        len(geometry.xlines),
        validate_interval("interval_xline", interval_xline),
        dtype=np.int64,
    )
    z_idx = np.arange(
        0,
        len(geometry.sample_axis),
        validate_interval("interval_sample", interval_sample),
        dtype=np.int64,
    )

    volume = np.empty((len(xl_idx), len(il_idx), len(z_idx)), dtype=np.float32, order="F")

    with segyio.open(str(segy_path), "r", strict=False, ignore_geometry=True) as segy:
        for out_y, il_pos in enumerate(il_idx):
            trace_indices = geometry.trace_index_grid[il_pos, xl_idx]
            for out_x, trace_idx in enumerate(trace_indices):
                trace = np.asarray(segy.trace[int(trace_idx)], dtype=np.float32)
                volume[out_x, out_y, :] = trace[z_idx]

    return (
        volume,
        geometry.inlines[il_idx],
        geometry.xlines[xl_idx],
        geometry.sample_axis[z_idx],
    )


def create_vtk_image(
    volume: np.ndarray,
    xlines: np.ndarray,
    inlines: np.ndarray,
    samples: np.ndarray,
    step_xline: float,
    step_inline: float,
    step_sample: float,
) -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(volume.shape[0], volume.shape[1], volume.shape[2])

    image.SetSpacing(float(step_xline), float(step_inline), float(step_sample))
    image.SetOrigin(0.0, 0.0, 0.0)

    vtk_array = numpy_support.numpy_to_vtk(
        volume.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName("velocity")
    image.GetPointData().SetScalars(vtk_array)
    return image


def create_lookup_table(image: vtk.vtkImageData, clip_percentile: float) -> vtk.vtkLookupTable:
    scalars = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
    lower = float(np.percentile(scalars, 100.0 - clip_percentile))
    upper = float(np.percentile(scalars, clip_percentile))
    if lower == upper:
        lower = float(np.min(scalars))
        upper = float(np.max(scalars))
    if lower == upper:
        upper = lower + 1.0

    lut = vtk.vtkLookupTable()
    lut.SetRange(lower, upper)
    lut.SetNumberOfTableValues(256)
    lut.Build()
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            local = t / 0.5
            r = 0.10 + local * (0.88 - 0.10)
            g = 0.18 + local * (0.90 - 0.18)
            b = 0.48 + local * (0.92 - 0.48)
        else:
            local = (t - 0.5) / 0.5
            r = 0.88 + local * (0.70 - 0.88)
            g = 0.90 + local * (0.16 - 0.90)
            b = 0.92 + local * (0.12 - 0.92)
        lut.SetTableValue(i, r, g, b, 1.0)
    return lut


def set_slice_index(actor: vtk.vtkImageActor, image: vtk.vtkImageData, orientation: str, slice_index: int) -> None:
    extent = list(image.GetExtent())
    if orientation == "xline":
        extent[0] = slice_index
        extent[1] = slice_index
    elif orientation == "inline":
        extent[2] = slice_index
        extent[3] = slice_index
    elif orientation == "sample":
        extent[4] = slice_index
        extent[5] = slice_index
    else:
        raise ValueError(f"Unknown slice orientation: {orientation}")
    actor.SetDisplayExtent(*extent)


class SliceActorBundle:
    def __init__(
        self,
        orientation: str,
        image: vtk.vtkImageData,
        slice_index: int,
        lut: vtk.vtkLookupTable,
        opacity: float,
    ) -> None:
        self.orientation = orientation
        self.image = image
        self.mapper = vtk.vtkImageMapToColors()
        self.mapper.SetInputData(image)
        self.mapper.SetLookupTable(lut)
        self.mapper.Update()

        self.actor = vtk.vtkImageActor()
        self.actor.GetMapper().SetInputConnection(self.mapper.GetOutputPort())
        self.actor.InterpolateOn()
        self.actor.ForceOpaqueOff()
        self.actor.SetOpacity(opacity)
        self.slice_index = 0
        self.set_index(slice_index)

    def set_image(self, image: vtk.vtkImageData, lut: vtk.vtkLookupTable) -> None:
        self.image = image
        self.mapper.SetInputData(image)
        self.mapper.SetLookupTable(lut)
        self.mapper.Update()
        self.set_index(self.slice_index)

    def set_index(self, slice_index: int) -> None:
        self.slice_index = int(slice_index)
        set_slice_index(self.actor, self.image, self.orientation, self.slice_index)


def create_outline(image: vtk.vtkImageData) -> vtk.vtkActor:
    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(image)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(outline.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.95, 0.95, 0.95)
    return actor


def create_axis_label_actor(text: str, x: float, y: float, z: float) -> vtk.vtkBillboardTextActor3D:
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(x, y, z)
    prop = actor.GetTextProperty()
    prop.SetFontSize(28)
    prop.SetColor(0.92, 0.92, 0.92)
    prop.SetJustificationToCentered()
    prop.SetVerticalJustificationToCentered()
    return actor


def create_axis_labels(
    image: vtk.vtkImageData,
    xlines: np.ndarray,
    inlines: np.ndarray,
    samples: np.ndarray,
    step_xline: float,
    step_inline: float,
    step_sample: float,
    axis_interval_xy: int = 100,
    axis_interval_z: int = 1000,
) -> list[vtk.vtkBillboardTextActor3D]:
    bounds = image.GetBounds()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    labels: list[vtk.vtkBillboardTextActor3D] = []
    margin_xy = max(step_inline, step_xline) * 0.6
    margin_z = max(step_sample, 1.0) * 1.2

    labels.append(
        create_axis_label_actor("Crossline", (x_min + x_max) * 0.5, y_min - margin_xy * 1.8, z_min - margin_z)
    )
    labels.append(
        create_axis_label_actor("Inline", x_min - margin_xy * 1.8, (y_min + y_max) * 0.5, z_min - margin_z)
    )
    labels.append(
        create_axis_label_actor("Sample", x_min - margin_xy * 1.8, y_min - margin_xy * 1.8, (z_min + z_max) * 0.5)
    )
    return labels


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3f}"


def configure_default_camera(renderer: vtk.vtkRenderer, image: vtk.vtkImageData) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = image.GetBounds()
    center = (
        (x_min + x_max) * 0.5,
        (y_min + y_max) * 0.5,
        (z_min + z_max) * 0.5,
    )
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    distance = max(span_x, span_y, span_z) * 2.2

    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(center[0] + distance, center[1] + distance, center[2])
    camera.SetViewUp(0.0, 0.0, -1.0)
    renderer.ResetCameraClippingRange()
    camera.Dolly(2.0)
    renderer.ResetCameraClippingRange()


class AxisControl(QtWidgets.QGroupBox):
    value_changed = QtCore.Signal(int)

    def __init__(self, title: str, values: np.ndarray, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.values = values

        layout = QtWidgets.QVBoxLayout(self)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setTracking(True)
        layout.addWidget(self.slider)

        row = QtWidgets.QHBoxLayout()
        self.index_label = QtWidgets.QLabel("")
        self.value_edit = QtWidgets.QLineEdit()
        self.value_edit.setValidator(QtGui.QDoubleValidator())
        self.go_button = QtWidgets.QPushButton("Set")
        row.addWidget(self.index_label)
        row.addWidget(self.value_edit)
        row.addWidget(self.go_button)
        layout.addLayout(row)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.go_button.clicked.connect(self._apply_value)
        self.value_edit.returnPressed.connect(self._apply_value)

        self.set_index(len(values) // 2)

    def _on_slider_changed(self, index: int) -> None:
        self.index_label.setText(f"index={index}")
        self.value_edit.setText(format_value(self.values[index]))
        self.value_changed.emit(index)

    def _apply_value(self) -> None:
        text = self.value_edit.text().strip()
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            return
        index = int(np.argmin(np.abs(self.values.astype(np.float64) - value)))
        self.set_index(index)

    def set_index(self, index: int) -> None:
        index = max(0, min(index, len(self.values) - 1))
        was_blocked = self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(was_blocked)
        self.index_label.setText(f"index={index}")
        self.value_edit.setText(format_value(self.values[index]))
        self.value_changed.emit(index)


class SliceUpdater:
    def __init__(
        self,
        interactor: vtk.vtkRenderWindowInteractor,
        image: vtk.vtkImageData,
        bundles: dict[str, SliceActorBundle],
        overlay: vtk.vtkTextActor,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        segy_path: Path,
        initial_attribute_name: str,
        initial_lut: vtk.vtkLookupTable,
        opacity: float,
    ) -> None:
        self.interactor = interactor
        self.image = image
        self.bundles = bundles
        self.overlay = overlay
        self.xlines = xlines
        self.inlines = inlines
        self.samples = samples
        self.segy_path = segy_path
        self.opacity = opacity
        self.indices = {
            "xline": len(xlines) // 2,
            "inline": len(inlines) // 2,
            "sample": len(samples) // 2,
        }
        self.attributes: dict[str, AttributeVolume] = {
            initial_attribute_name: AttributeVolume(
                name=initial_attribute_name,
                image=image,
                lut=initial_lut,
            )
        }
        self.current_attribute_name = initial_attribute_name
        self.update_overlay()

    def current_text(self) -> str:
        return (
            f"{self.segy_path.name}\n"
            f"Attribute: {self.current_attribute_name}\n"
            f"Crossline: {format_value(self.xlines[self.indices['xline']])}\n"
            f"Inline: {format_value(self.inlines[self.indices['inline']])}\n"
            f"Sample: {format_value(self.samples[self.indices['sample']])}"
        )

    def attribute_names(self) -> list[str]:
        return list(self.attributes.keys())

    def current_attribute(self) -> AttributeVolume:
        return self.attributes[self.current_attribute_name]

    def current_scalar_range(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.current_attribute().image.GetScalarRange())

    def set_index(self, orientation: str, index: int, render: bool = True) -> None:
        max_index = {
            "xline": len(self.xlines) - 1,
            "inline": len(self.inlines) - 1,
            "sample": len(self.samples) - 1,
        }[orientation]
        index = max(0, min(index, max_index))
        self.indices[orientation] = index
        self.bundles[orientation].set_index(index)
        self.update_overlay()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_attribute(self, name: str, render: bool = True) -> None:
        if name not in self.attributes:
            raise KeyError(f"Unknown attribute: {name}")
        attr = self.attributes[name]
        self.current_attribute_name = name
        self.image = attr.image
        for orientation, bundle in self.bundles.items():
            bundle.set_image(attr.image, attr.lut)
            bundle.set_index(self.indices[orientation])
        self.update_overlay()
        if render:
            self.interactor.GetRenderWindow().Render()

    def extract_range_attribute(self, min_value: float, max_value: float) -> str:
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        source = self.current_attribute()
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(source.image)
        threshold.ThresholdBetween(min_value, max_value)
        threshold.ReplaceInOff()
        threshold.ReplaceOutOn()
        threshold.SetOutValue(0.0)
        threshold.SetOutputScalarTypeToFloat()
        threshold.Update()

        output = vtk.vtkImageData()
        output.DeepCopy(threshold.GetOutput())
        lut = create_lookup_table(output, 99.0)

        base_name = f"{source.name}_range_{format_value(min_value)}_{format_value(max_value)}"
        new_name = base_name
        suffix = 1
        while new_name in self.attributes:
            suffix += 1
            new_name = f"{base_name}_{suffix}"

        self.attributes[new_name] = AttributeVolume(
            name=new_name,
            image=output,
            lut=lut,
        )
        return new_name

    def update_overlay(self) -> None:
        self.overlay.SetInput(self.current_text())


class SegyViewerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        updater: SliceUpdater,
        vtk_widget: QVTKRenderWindowInteractor,
        render_window: vtk.vtkRenderWindow,
        renderer: vtk.vtkRenderer,
        image: vtk.vtkImageData,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        debug_ui: bool,
    ) -> None:
        super().__init__()
        self.updater = updater
        self.vtk_widget = vtk_widget
        self.render_window = render_window
        self.renderer = renderer
        self.image = image
        self.debug_ui = debug_ui
        self._vtk_initialized = False
        self._render_pending = False

        self.setWindowTitle("SEG-Y Slice Viewer")
        self.resize(2200, 1400)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.vtk_widget.setMinimumSize(1200, 900)
        layout.addWidget(self.vtk_widget, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(420)
        layout.addWidget(panel)
        panel_layout = QtWidgets.QVBoxLayout(panel)

        header = QtWidgets.QLabel("SEG-Y Slice Prototype")
        header_font = QtGui.QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        panel_layout.addWidget(header)

        self.info_label = QtWidgets.QLabel("")
        info_font = QtGui.QFont("Menlo")
        info_font.setPointSize(11)
        self.info_label.setFont(info_font)
        self.info_label.setWordWrap(True)
        panel_layout.addWidget(self.info_label)

        note = QtWidgets.QLabel(
            "Use the controls below to move slices, switch attributes,\n"
            "extract value ranges, and reset the default view."
        )
        note.setWordWrap(True)
        panel_layout.addWidget(note)

        self.reset_view_button = QtWidgets.QPushButton("Reset 视角")
        self.reset_view_button.clicked.connect(self.reset_view)
        panel_layout.addWidget(self.reset_view_button)

        attributes_group = QtWidgets.QGroupBox("Attributes")
        attributes_layout = QtWidgets.QVBoxLayout(attributes_group)
        self.attributes_list = QtWidgets.QListWidget()
        attributes_layout.addWidget(self.attributes_list)
        panel_layout.addWidget(attributes_group)

        range_group = QtWidgets.QGroupBox("Extract Range")
        range_layout = QtWidgets.QGridLayout(range_group)
        range_layout.addWidget(QtWidgets.QLabel("Min"), 0, 0)
        range_layout.addWidget(QtWidgets.QLabel("Max"), 1, 0)
        self.range_min_edit = QtWidgets.QLineEdit()
        self.range_max_edit = QtWidgets.QLineEdit()
        validator = QtGui.QDoubleValidator()
        self.range_min_edit.setValidator(validator)
        self.range_max_edit.setValidator(validator)
        range_layout.addWidget(self.range_min_edit, 0, 1)
        range_layout.addWidget(self.range_max_edit, 1, 1)
        self.extract_button = QtWidgets.QPushButton("提取为新属性")
        self.extract_button.clicked.connect(self.extract_attribute)
        range_layout.addWidget(self.extract_button, 2, 0, 1, 2)
        panel_layout.addWidget(range_group)

        self.xline_control = AxisControl("Crossline", xlines)
        self.inline_control = AxisControl("Inline", inlines)
        self.sample_control = AxisControl("Sample", samples)
        panel_layout.addWidget(self.xline_control)
        panel_layout.addWidget(self.inline_control)
        panel_layout.addWidget(self.sample_control)
        panel_layout.addStretch(1)

        self.xline_control.value_changed.connect(lambda index: self._set_index("xline", index))
        self.inline_control.value_changed.connect(lambda index: self._set_index("inline", index))
        self.sample_control.value_changed.connect(lambda index: self._set_index("sample", index))
        self.attributes_list.currentTextChanged.connect(self.change_attribute)

        self.refresh_attributes()
        self.refresh_info()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        debug_log(
            self.debug_ui,
            f"window showEvent: visible={self.isVisible()} active={self.isActiveWindow()} geometry={self.geometry().getRect()}",
        )
        if self._vtk_initialized:
            return
        self._vtk_initialized = True
        self.vtk_widget.Initialize()
        QtCore.QTimer.singleShot(0, self._first_render)

    def _first_render(self) -> None:
        self.render_window.Render()
        debug_log(self.debug_ui, "embedded vtk render completed")

    def schedule_render(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        QtCore.QTimer.singleShot(16, self._flush_render)

    def _flush_render(self) -> None:
        self._render_pending = False
        self.render_window.Render()

    def reset_view(self) -> None:
        configure_default_camera(self.renderer, self.image)
        self.schedule_render()
        debug_log(self.debug_ui, "camera reset to default view")

    def _set_index(self, orientation: str, index: int) -> None:
        self.updater.set_index(orientation, index, render=False)
        self.refresh_info()
        self.schedule_render()

    def refresh_info(self) -> None:
        self.info_label.setText(self.updater.current_text())
        min_value, max_value = self.updater.current_scalar_range()
        if not self.range_min_edit.hasFocus():
            self.range_min_edit.setText(format_value(min_value))
        if not self.range_max_edit.hasFocus():
            self.range_max_edit.setText(format_value(max_value))

    def refresh_attributes(self) -> None:
        selected = self.updater.current_attribute_name
        self.attributes_list.blockSignals(True)
        self.attributes_list.clear()
        self.attributes_list.addItems(self.updater.attribute_names())
        items = self.attributes_list.findItems(selected, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.attributes_list.setCurrentItem(items[0])
        self.attributes_list.blockSignals(False)

    def change_attribute(self, name: str) -> None:
        if not name:
            return
        self.updater.set_attribute(name, render=False)
        self.refresh_info()
        self.schedule_render()

    def extract_attribute(self) -> None:
        min_text = self.range_min_edit.text().strip()
        max_text = self.range_max_edit.text().strip()
        if not min_text or not max_text:
            return
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return
        new_name = self.updater.extract_range_attribute(min_value, max_value)
        self.refresh_attributes()
        items = self.attributes_list.findItems(new_name, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.attributes_list.setCurrentItem(items[0])
        self.schedule_render()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.render_window.Finalize()
        super().closeEvent(event)


def launch_vtk_viewer(
    segy_path: Path,
    image: vtk.vtkImageData,
    xlines: np.ndarray,
    inlines: np.ndarray,
    samples: np.ndarray,
    step_xline: float,
    step_inline: float,
    step_sample: float,
    clip_percentile: float,
    opacity_scale: float,
    debug_ui: bool = False,
) -> int:
    normalize_macos_gui_env(debug_ui)

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    debug_log(debug_ui, f"qt platform={app.platformName()} DISPLAY={os.environ.get('DISPLAY')}")

    initial_attribute_name = "seismic"
    lut = create_lookup_table(image, clip_percentile)
    dims = image.GetDimensions()
    xline_bundle = SliceActorBundle("xline", image, dims[0] // 2, lut, opacity_scale)
    inline_bundle = SliceActorBundle("inline", image, dims[1] // 2, lut, opacity_scale)
    sample_bundle = SliceActorBundle("sample", image, dims[2] // 2, lut, opacity_scale)
    outline_actor = create_outline(image)
    axis_texts = create_axis_labels(
        image,
        xlines,
        inlines,
        samples,
        step_xline,
        step_inline,
        step_sample,
    )

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.08, 0.10, 0.14)
    renderer.AddActor(xline_bundle.actor)
    renderer.AddActor(inline_bundle.actor)
    renderer.AddActor(sample_bundle.actor)
    renderer.AddActor(outline_actor)
    for actor in axis_texts:
        renderer.AddActor(actor)

    overlay = vtk.vtkTextActor()
    overlay.GetTextProperty().SetFontSize(20)
    overlay.GetTextProperty().SetColor(0.95, 0.95, 0.95)
    overlay.SetDisplayPosition(20, 20)
    renderer.AddViewProp(overlay)

    vtk_widget = QVTKRenderWindowInteractor()
    render_window = vtk_widget.GetRenderWindow()
    render_window.SetWindowName(f"SEG-Y Slice Viewer - {segy_path.name}")
    render_window.AddRenderer(renderer)

    interactor = render_window.GetInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    updater = SliceUpdater(
        interactor=interactor,
        image=image,
        bundles={
            "xline": xline_bundle,
            "inline": inline_bundle,
            "sample": sample_bundle,
        },
        overlay=overlay,
        xlines=xlines,
        inlines=inlines,
        samples=samples,
        segy_path=segy_path,
        initial_attribute_name=initial_attribute_name,
        initial_lut=lut,
        opacity=opacity_scale,
    )

    configure_default_camera(renderer, image)
    debug_log(debug_ui, f"vtk render window created: {type(render_window).__name__}")
    debug_log(debug_ui, f"control ranges: xline={len(xlines)} inline={len(inlines)} sample={len(samples)}")

    window = SegyViewerWindow(
        updater=updater,
        vtk_widget=vtk_widget,
        render_window=render_window,
        renderer=renderer,
        image=image,
        xlines=xlines,
        inlines=inlines,
        samples=samples,
        debug_ui=debug_ui,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    debug_log(debug_ui, f"window shown: visible={window.isVisible()} active={window.isActiveWindow()}")

    if owns_app:
        return app.exec()
    return 0


def main() -> int:
    args = parse_args()
    segy_path = Path(args.segy_path).expanduser().resolve()
    if not segy_path.exists():
        raise SystemExit(f"SEG-Y file not found: {segy_path}")

    geometry = load_geometry(segy_path)
    volume, ilines, xlines, samples = build_downsampled_volume(
        segy_path=segy_path,
        geometry=geometry,
        interval_inline=args.interval_inline,
        interval_xline=args.interval_xline,
        interval_sample=args.interval_sample,
    )

    image = create_vtk_image(
        volume,
        xlines,
        ilines,
        samples,
        step_xline=args.step_xline,
        step_inline=args.step_inline,
        step_sample=args.step_sample,
    )
    dims_text = (
        f"xlines={len(xlines)}, inlines={len(ilines)}, samples={len(samples)}"
    )

    print(f"SEG-Y: {segy_path}")
    print(
        "Original grid: "
        f"inlines={len(geometry.inlines)}, "
        f"xlines={len(geometry.xlines)}, "
        f"samples={len(geometry.sample_axis)}"
    )
    print(f"Displayed grid: {dims_text}")
    print(
        "Intervals: "
        f"inline={args.interval_inline}, "
        f"xline={args.interval_xline}, "
        f"sample={args.interval_sample}"
    )
    print(
        "Steps: "
        f"inline={args.step_inline}, "
        f"xline={args.step_xline}, "
        f"sample={args.step_sample}"
    )

    return launch_vtk_viewer(
        image=image,
        xlines=xlines,
        inlines=ilines,
        samples=samples,
        step_xline=args.step_xline,
        step_inline=args.step_inline,
        step_sample=args.step_sample,
        clip_percentile=args.clip_percentile,
        opacity_scale=args.opacity,
        segy_path=segy_path,
        debug_ui=args.debug_ui,
    )


if __name__ == "__main__":
    sys.exit(main())
