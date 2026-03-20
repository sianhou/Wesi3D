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


def axis_labels(values: np.ndarray, interval: int) -> list[int]:
    start = int(values[0])
    end = int(values[-1])
    label_start = (start // interval) * interval
    if label_start < start:
        label_start += interval
    label_end = (end // interval) * interval
    if label_end < label_start:
        return [start, end] if start != end else [start]
    return list(range(label_start, label_end + 1, interval))


def value_to_world(value: float, values: np.ndarray, step: float) -> float:
    if len(values) <= 1:
        return 0.0
    value_min = float(values[0])
    value_max = float(values[-1])
    if value_max == value_min:
        return 0.0
    scale = ((value - value_min) / (value_max - value_min)) * (len(values) - 1)
    return float(scale * step)


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


def create_slice_actor(
    image: vtk.vtkImageData,
    orientation: str,
    slice_index: int,
    lut: vtk.vtkLookupTable,
    opacity: float,
) -> vtk.vtkImageActor:
    mapper = vtk.vtkImageMapToColors()
    mapper.SetInputData(image)
    mapper.SetLookupTable(lut)
    mapper.Update()

    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    actor.InterpolateOn()

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
    actor.ForceOpaqueOff()
    actor.SetOpacity(opacity)
    return actor


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

    for xline in axis_labels(xlines, axis_interval_xy):
        x = value_to_world(float(xline), xlines, step_xline)
        labels.append(create_axis_label_actor(str(xline), x, y_min - margin_xy, z_min))
        labels.append(create_axis_label_actor(str(xline), x, y_max + margin_xy, z_min))
        labels.append(create_axis_label_actor(str(xline), x, y_min - margin_xy, z_max))
        labels.append(create_axis_label_actor(str(xline), x, y_max + margin_xy, z_max))

    for inline in axis_labels(inlines, axis_interval_xy):
        y = value_to_world(float(inline), inlines, step_inline)
        labels.append(create_axis_label_actor(str(inline), x_min - margin_xy, y, z_min))
        labels.append(create_axis_label_actor(str(inline), x_max + margin_xy, y, z_min))
        labels.append(create_axis_label_actor(str(inline), x_min - margin_xy, y, z_max))
        labels.append(create_axis_label_actor(str(inline), x_max + margin_xy, y, z_max))

    for sample in axis_labels(samples, axis_interval_z):
        z = value_to_world(float(sample), samples, step_sample)
        labels.append(create_axis_label_actor(str(sample), x_min - margin_xy, y_min - margin_xy, z))
        labels.append(create_axis_label_actor(str(sample), x_max + margin_xy, y_min - margin_xy, z))
        labels.append(create_axis_label_actor(str(sample), x_min - margin_xy, y_max + margin_xy, z))
        labels.append(create_axis_label_actor(str(sample), x_max + margin_xy, y_max + margin_xy, z))

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
        actors: dict[str, vtk.vtkImageActor],
        overlay: vtk.vtkTextActor,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        segy_path: Path,
    ) -> None:
        self.interactor = interactor
        self.image = image
        self.actors = actors
        self.overlay = overlay
        self.xlines = xlines
        self.inlines = inlines
        self.samples = samples
        self.segy_path = segy_path
        self.indices = {
            "xline": len(xlines) // 2,
            "inline": len(inlines) // 2,
            "sample": len(samples) // 2,
        }
        self.update_overlay()

    def current_text(self) -> str:
        return (
            f"{self.segy_path.name}\n"
            f"Crossline: {format_value(self.xlines[self.indices['xline']])}\n"
            f"Inline: {format_value(self.inlines[self.indices['inline']])}\n"
            f"Sample: {format_value(self.samples[self.indices['sample']])}"
        )

    def set_index(self, orientation: str, index: int) -> None:
        max_index = {
            "xline": len(self.xlines) - 1,
            "inline": len(self.inlines) - 1,
            "sample": len(self.samples) - 1,
        }[orientation]
        index = max(0, min(index, max_index))
        self.indices[orientation] = index
        set_slice_index(self.actors[orientation], self.image, orientation, index)
        self.update_overlay()
        self.interactor.GetRenderWindow().Render()

    def update_overlay(self) -> None:
        self.overlay.SetInput(self.current_text())


class ControlPanelWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        updater: SliceUpdater,
        interactor: vtk.vtkRenderWindowInteractor,
        render_window: vtk.vtkRenderWindow,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        debug_ui: bool,
    ) -> None:
        super().__init__()
        self.updater = updater
        self.interactor = interactor
        self.render_window = render_window
        self.debug_ui = debug_ui

        self.setWindowTitle("SEG-Y Control Panel")
        self.resize(520, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        header = QtWidgets.QLabel("SEG-Y Slice Prototype")
        header_font = QtGui.QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        self.info_label = QtWidgets.QLabel("")
        info_font = QtGui.QFont("Menlo")
        info_font.setPointSize(11)
        self.info_label.setFont(info_font)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        note = QtWidgets.QLabel(
            "VTK uses a separate native window.\n"
            "Use the controls below to move the three orthogonal slices."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.xline_control = AxisControl("Crossline", xlines)
        self.inline_control = AxisControl("Inline", inlines)
        self.sample_control = AxisControl("Sample", samples)
        layout.addWidget(self.xline_control)
        layout.addWidget(self.inline_control)
        layout.addWidget(self.sample_control)
        layout.addStretch(1)

        self.xline_control.value_changed.connect(lambda index: self._set_index("xline", index))
        self.inline_control.value_changed.connect(lambda index: self._set_index("inline", index))
        self.sample_control.value_changed.connect(lambda index: self._set_index("sample", index))

        self.event_timer = QtCore.QTimer(self)
        self.event_timer.timeout.connect(self._pump_vtk_events)
        self.event_timer.start(16)

        self.refresh_info()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        debug_log(
            self.debug_ui,
            f"panel showEvent: visible={self.isVisible()} active={self.isActiveWindow()} geometry={self.geometry().getRect()}",
        )
        QtCore.QTimer.singleShot(0, self._show_vtk_window)

    def _show_vtk_window(self) -> None:
        self.render_window.Render()
        debug_log(self.debug_ui, "vtk native window render completed")

    def _set_index(self, orientation: str, index: int) -> None:
        self.updater.set_index(orientation, index)
        self.refresh_info()

    def refresh_info(self) -> None:
        self.info_label.setText(self.updater.current_text())

    def _pump_vtk_events(self) -> None:
        if hasattr(self.interactor, "ProcessEvents"):
            try:
                self.interactor.ProcessEvents()
            except Exception as exc:
                debug_log(self.debug_ui, f"ProcessEvents failed: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.event_timer.stop()
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

    lut = create_lookup_table(image, clip_percentile)
    dims = image.GetDimensions()
    xline_actor = create_slice_actor(image, "xline", dims[0] // 2, lut, opacity_scale)
    inline_actor = create_slice_actor(image, "inline", dims[1] // 2, lut, opacity_scale)
    sample_actor = create_slice_actor(image, "sample", dims[2] // 2, lut, opacity_scale)
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
    renderer.AddActor(xline_actor)
    renderer.AddActor(inline_actor)
    renderer.AddActor(sample_actor)
    renderer.AddActor(outline_actor)
    for actor in axis_texts:
        renderer.AddActor(actor)

    overlay = vtk.vtkTextActor()
    overlay.GetTextProperty().SetFontSize(20)
    overlay.GetTextProperty().SetColor(0.95, 0.95, 0.95)
    overlay.SetDisplayPosition(20, 20)
    renderer.AddViewProp(overlay)

    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(f"SEG-Y Slice Viewer - {segy_path.name}")
    render_window.SetSize(1600, 960)
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    updater = SliceUpdater(
        interactor=interactor,
        image=image,
        actors={
            "xline": xline_actor,
            "inline": inline_actor,
            "sample": sample_actor,
        },
        overlay=overlay,
        xlines=xlines,
        inlines=inlines,
        samples=samples,
        segy_path=segy_path,
    )

    renderer.ResetCamera()
    debug_log(debug_ui, f"vtk render window created: {type(render_window).__name__}")
    interactor.Initialize()
    render_window.Render()
    debug_log(debug_ui, f"control ranges: xline={len(xlines)} inline={len(inlines)} sample={len(samples)}")

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    debug_log(debug_ui, f"qt platform={app.platformName()} DISPLAY={os.environ.get('DISPLAY')}")

    panel = ControlPanelWindow(
        updater=updater,
        interactor=interactor,
        render_window=render_window,
        xlines=xlines,
        inlines=inlines,
        samples=samples,
        debug_ui=debug_ui,
    )
    panel.show()
    panel.raise_()
    panel.activateWindow()
    debug_log(debug_ui, f"panel shown: visible={panel.isVisible()} active={panel.isActiveWindow()}")

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
