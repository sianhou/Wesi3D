#!/usr/bin/env python3
"""
3D SEG-Y viewer for a velocity cube.

Default behavior:
- open data/raw/vel.sgy
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

from ..config import DEFAULT_VIEWER_CONFIG
from ..data.attribute_data import (
    AttributeVolume,
    RenderSpacing,
    create_lookup_table_from_scalars,
    load_attribute_from_volume,
)
from ..data.volume_data import load_segy_geometry, read_segy_volume
from ..processing.control_points import (
    ControlPoint,
    apply_master_point_z_move,
    extract_control_points,
    master_control_points,
)
from ..processing.volume_processing import extract_connected_components, extract_range_volume
from ..utils.constants import APP_NAME
from ..utils.formatting import format_value


@dataclass
class HorizonSurface:
    name: str
    actor: vtk.vtkActor
    mapper: vtk.vtkPolyDataMapper
    polydata: vtk.vtkPolyData
    lut: vtk.vtkLookupTable
    component_index: int
    voxel_count: int
    scalar_range: tuple[float, float]
    opacity: float = 0.55
    visible: bool = True
    component_mask: np.ndarray | None = None
    source_attribute_name: str = ""


@dataclass
class ControlPointSet:
    name: str
    actor: vtk.vtkActor
    sphere_source: vtk.vtkSphereSource
    polydata: vtk.vtkPolyData
    points: list[ControlPoint]
    horizon_name: str
    source_attribute_name: str
    source_horizon_name: str
    original_horizon_mask: np.ndarray
    display_scale: float = 1.0
    visible: bool = True

    @property
    def master_points(self) -> list[ControlPoint]:
        return master_control_points(self.points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEG-Y slice viewer")
    parser.add_argument(
        "segy_path",
        nargs="?",
        default=str(DEFAULT_VIEWER_CONFIG.default_segy_path),
        help="Path to SEG-Y file. Defaults to project data/raw/vel.sgy",
    )
    parser.add_argument(
        "--debug-ui",
        action="store_true",
        help="Print extra diagnostics for VTK window startup",
    )
    parser.add_argument(
        "--interval-inline",
        type=int,
        default=DEFAULT_VIEWER_CONFIG.interval_inline,
        help="Inline downsampling interval, e.g. 4 means take every 4th inline",
    )
    parser.add_argument(
        "--interval-xline",
        type=int,
        default=DEFAULT_VIEWER_CONFIG.interval_xline,
        help="Crossline downsampling interval, e.g. 4 means take every 4th crossline",
    )
    parser.add_argument(
        "--interval-sample",
        type=int,
        default=DEFAULT_VIEWER_CONFIG.interval_sample,
        help="Sample downsampling interval, e.g. 4 means take every 4th sample",
    )
    parser.add_argument(
        "--step-inline",
        type=float,
        default=DEFAULT_VIEWER_CONFIG.step_inline,
        help="Displayed inline spacing in the 3D scene",
    )
    parser.add_argument(
        "--step-xline",
        type=float,
        default=DEFAULT_VIEWER_CONFIG.step_xline,
        help="Displayed crossline spacing in the 3D scene",
    )
    parser.add_argument(
        "--step-sample",
        type=float,
        default=DEFAULT_VIEWER_CONFIG.step_sample,
        help="Displayed sample spacing in the 3D scene",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=DEFAULT_VIEWER_CONFIG.clip_percentile,
        help="Clip symmetric amplitudes/velocities by percentile for rendering",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=DEFAULT_VIEWER_CONFIG.opacity,
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


def create_horizon_surface_actor(
    mask: np.ndarray,
    scalar_values: np.ndarray,
    spacing: RenderSpacing,
    clip_percentile: float,
) -> tuple[vtk.vtkActor, vtk.vtkPolyData, vtk.vtkPolyDataMapper, vtk.vtkLookupTable, tuple[float, float]]:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    padded_values = np.pad(np.asarray(scalar_values, dtype=np.float32), 1, mode="edge")

    def _make_vtk_image(array: np.ndarray, array_type: int) -> vtk.vtkImageData:
        image = vtk.vtkImageData()
        image.SetDimensions(*array.shape)
        image.SetSpacing(float(spacing.xline), float(spacing.inline), float(spacing.sample))
        image.SetOrigin(-float(spacing.xline), -float(spacing.inline), -float(spacing.sample))
        flat = np.ascontiguousarray(array.ravel(order="F"))
        scalars = numpy_support.numpy_to_vtk(
            flat,
            deep=True,
            array_type=array_type,
        )
        image.GetPointData().SetScalars(scalars)
        return image

    image = _make_vtk_image(padded, vtk.VTK_UNSIGNED_CHAR)
    value_image = _make_vtk_image(padded_values, vtk.VTK_FLOAT)

    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, 0.5)

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surface.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.08)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()

    probe = vtk.vtkProbeFilter()
    probe.SetInputConnection(normals.GetOutputPort())
    probe.SetSourceData(value_image)
    probe.Update()

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(probe.GetOutput())
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError("Empty horizon surface generated from component mask.")

    point_scalars = polydata.GetPointData().GetScalars()
    scalar_array = numpy_support.vtk_to_numpy(point_scalars)
    scalar_range = (float(np.min(scalar_array)), float(np.max(scalar_array)))
    lut = create_lookup_table_from_scalars(scalar_array, clip_percentile)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    mapper.SetScalarRange(lut.GetRange())
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.55)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().EdgeVisibilityOff()
    return actor, polydata, mapper, lut, scalar_range


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


def create_control_point_actor(
    points: list[ControlPoint],
    spacing: RenderSpacing,
    display_scale: float = 1.0,
) -> tuple[vtk.vtkActor, vtk.vtkPolyData, vtk.vtkSphereSource]:
    vtk_points = vtk.vtkPoints()
    values = vtk.vtkFloatArray()
    values.SetName("value")
    kinds = vtk.vtkUnsignedCharArray()
    kinds.SetName("kind")

    for point in points:
        vtk_points.InsertNextPoint(
            float(point.xline_index) * float(spacing.xline),
            float(point.inline_index) * float(spacing.inline),
            float(point.sample_index) * float(spacing.sample),
        )
        values.InsertNextValue(float(point.value))
        kinds.InsertNextValue(1 if point.kind == "surface" else 0)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.GetPointData().AddArray(values)
    polydata.GetPointData().AddArray(kinds)
    polydata.GetPointData().SetActiveScalars("value")

    sphere = vtk.vtkSphereSource()
    radius = max(min(spacing.xline, spacing.inline, spacing.sample) * 0.18 * float(display_scale), 1.5)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(polydata)
    mapper.SetSourceConnection(sphere.GetOutputPort())
    mapper.ScalingOff()
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.98, 0.80, 0.20)
    actor.GetProperty().SetOpacity(0.95)
    return actor, polydata, sphere


def create_horizon_surface_from_control_points(
    points: list[ControlPoint],
    spacing: RenderSpacing,
    clip_percentile: float,
) -> tuple[vtk.vtkActor, vtk.vtkPolyData, vtk.vtkPolyDataMapper, vtk.vtkLookupTable, tuple[float, float]]:
    if len(points) < 4:
        raise ValueError("At least 4 control points are required to rebuild a closed surface.")

    vtk_points = vtk.vtkPoints()
    scalars = vtk.vtkFloatArray()
    scalars.SetName("value")
    for point in points:
        vtk_points.InsertNextPoint(
            float(point.xline_index) * float(spacing.xline),
            float(point.inline_index) * float(spacing.inline),
            float(point.sample_index) * float(spacing.sample),
        )
        scalars.InsertNextValue(float(point.value))

    point_cloud = vtk.vtkPolyData()
    point_cloud.SetPoints(vtk_points)
    point_cloud.GetPointData().SetScalars(scalars)

    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(point_cloud)
    delaunay.SetTolerance(0.001)
    delaunay.SetAlpha(0.0)

    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(delaunay.GetOutputPort())

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(surface.GetOutputPort())

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(clean.GetOutputPort())
    smooth.SetNumberOfIterations(25)
    smooth.BoundarySmoothingOn()
    smooth.FeatureEdgeSmoothingOff()
    smooth.SetPassBand(0.06)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputConnection(smooth.GetOutputPort())
    interpolator.SetSourceData(point_cloud)
    interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(interpolator.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(normals.GetOutput())
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError("Empty rebuilt horizon surface.")

    point_scalars = polydata.GetPointData().GetScalars()
    scalar_array = numpy_support.vtk_to_numpy(point_scalars)
    scalar_range = (float(np.min(scalar_array)), float(np.max(scalar_array)))
    lut = create_lookup_table_from_scalars(scalar_array, clip_percentile)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    mapper.SetScalarRange(lut.GetRange())
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.55)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().EdgeVisibilityOff()
    return actor, polydata, mapper, lut, scalar_range


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
    distance = max(span_x, span_y, span_z) * (4.4 / 1.5)

    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    diagonal = distance / np.sqrt(2.0)
    camera.SetPosition(center[0] + diagonal, center[1] - diagonal, center[2] + distance)
    camera.SetViewUp(0.0, 0.0, -1.0)
    renderer.ResetCameraClippingRange()
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


class ExtractRangeDialog(QtWidgets.QDialog):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract Range")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        validator = QtGui.QDoubleValidator(self)
        self.min_edit = QtWidgets.QLineEdit(format_value(min_value))
        self.min_edit.setValidator(validator)
        self.max_edit = QtWidgets.QLineEdit(format_value(max_value))
        self.max_edit.setValidator(validator)
        form.addRow("Min", self.min_edit)
        form.addRow("Max", self.max_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, float] | None:
        min_text = self.min_edit.text().strip()
        max_text = self.max_edit.text().strip()
        if not min_text or not max_text:
            return None
        try:
            return float(min_text), float(max_text)
        except ValueError:
            return None


class ExtractHorizonDialog(QtWidgets.QDialog):
    def __init__(self, min_voxels: int = 1, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract Horizons")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.min_voxels_edit = QtWidgets.QLineEdit(str(max(1, min_voxels)))
        self.min_voxels_edit.setValidator(QtGui.QIntValidator(1, 10**9, self))
        form.addRow("Min Voxels", self.min_voxels_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> int | None:
        text = self.min_voxels_edit.text().strip()
        if not text:
            return None
        try:
            return max(1, int(text))
        except ValueError:
            return None


class ExtractControlPointsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract Control Points")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        int_validator = QtGui.QIntValidator(1, 10**9, self)
        self.surface_xline_edit = QtWidgets.QLineEdit("8")
        self.surface_inline_edit = QtWidgets.QLineEdit("8")
        self.interior_xline_edit = QtWidgets.QLineEdit("8")
        self.interior_inline_edit = QtWidgets.QLineEdit("8")
        self.interior_sample_edit = QtWidgets.QLineEdit("8")
        for widget in (
            self.surface_xline_edit,
            self.surface_inline_edit,
            self.interior_xline_edit,
            self.interior_inline_edit,
            self.interior_sample_edit,
        ):
            widget.setValidator(int_validator)
        form.addRow("Surface Xline Interval", self.surface_xline_edit)
        form.addRow("Surface Inline Interval", self.surface_inline_edit)
        form.addRow("Interior Xline Interval", self.interior_xline_edit)
        form.addRow("Interior Inline Interval", self.interior_inline_edit)
        form.addRow("Interior Sample Interval", self.interior_sample_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, int] | None:
        fields = {
            "surface_xline_interval": self.surface_xline_edit.text().strip(),
            "surface_inline_interval": self.surface_inline_edit.text().strip(),
            "interior_xline_interval": self.interior_xline_edit.text().strip(),
            "interior_inline_interval": self.interior_inline_edit.text().strip(),
            "interior_sample_interval": self.interior_sample_edit.text().strip(),
        }
        try:
            return {key: max(1, int(value)) for key, value in fields.items() if value}
        except ValueError:
            return None


class EditMasterPointDialog(QtWidgets.QDialog):
    def __init__(self, surface_points: list[ControlPoint], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Master Point")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.master_combo = QtWidgets.QComboBox()
        for point in surface_points:
            label = (
                f"#{point.master_index} "
                f"xline={point.xline_index}, inline={point.inline_index}, sample={point.sample_index}"
            )
            self.master_combo.addItem(label, point.master_index)
        self.delta_z_edit = QtWidgets.QLineEdit("1")
        self.delta_z_edit.setValidator(QtGui.QDoubleValidator(self))
        form.addRow("Master Point", self.master_combo)
        form.addRow("Delta Z (samples)", self.delta_z_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[int, float] | None:
        data = self.master_combo.currentData()
        text = self.delta_z_edit.text().strip()
        if data is None or not text:
            return None
        try:
            return int(data), float(text)
        except ValueError:
            return None


class SliceUpdater:
    def __init__(
        self,
        interactor: vtk.vtkRenderWindowInteractor,
        renderer: vtk.vtkRenderer,
        bundles: dict[str, SliceActorBundle],
        overlay: vtk.vtkTextActor,
        segy_path: Path,
        initial_attribute: AttributeVolume,
        spacing: RenderSpacing,
        clip_percentile: float,
        opacity: float,
    ) -> None:
        self.interactor = interactor
        self.renderer = renderer
        self.image = initial_attribute.image
        self.bundles = bundles
        self.overlay = overlay
        self.segy_path = segy_path
        self.spacing = spacing
        self.clip_percentile = clip_percentile
        self.opacity = opacity
        self.attributes: dict[str, AttributeVolume] = {
            initial_attribute.name: initial_attribute
        }
        self.horizons: dict[str, HorizonSurface] = {}
        self.control_point_sets: dict[str, ControlPointSet] = {}
        self.current_horizon_name: str | None = None
        self.current_control_point_set_name: str | None = None
        self.current_attribute_name = initial_attribute.name
        self._sync_axes_from_current_attribute()
        self.indices = {
            "xline": len(self.xlines) // 2,
            "inline": len(self.inlines) // 2,
            "sample": len(self.samples) // 2,
        }
        self.update_overlay()

    def _sync_axes_from_current_attribute(self) -> None:
        volume_data = self.current_attribute().volume_data
        self.xlines = volume_data.xlines
        self.inlines = volume_data.inlines
        self.samples = volume_data.samples

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

    def current_attribute_display_range(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.current_attribute().lut.GetRange())

    def horizon_names(self) -> list[str]:
        return list(self.horizons.keys())

    def control_point_set_names(self) -> list[str]:
        return list(self.control_point_sets.keys())

    def current_attribute_opacity(self) -> float:
        return float(self.current_attribute().opacity)

    def current_horizon_scalar_range(self) -> tuple[float, float] | None:
        if self.current_horizon_name is None:
            return None
        return self.horizons[self.current_horizon_name].scalar_range

    def current_horizon_opacity(self) -> float | None:
        if self.current_horizon_name is None:
            return None
        return float(self.horizons[self.current_horizon_name].opacity)

    def current_horizon(self) -> HorizonSurface | None:
        if self.current_horizon_name is None:
            return None
        return self.horizons[self.current_horizon_name]

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
        self._sync_axes_from_current_attribute()
        self.indices["xline"] = max(0, min(self.indices["xline"], len(self.xlines) - 1))
        self.indices["inline"] = max(0, min(self.indices["inline"], len(self.inlines) - 1))
        self.indices["sample"] = max(0, min(self.indices["sample"], len(self.samples) - 1))
        for orientation, bundle in self.bundles.items():
            bundle.set_image(attr.image, attr.lut)
            bundle.actor.SetOpacity(attr.opacity)
            bundle.set_index(self.indices[orientation])
        self.update_overlay()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_attribute_display_range(self, min_value: float, max_value: float, render: bool = True) -> None:
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        attr = self.current_attribute()
        attr.lut.SetRange(float(min_value), float(max_value))
        attr.lut.Build()
        for bundle in self.bundles.values():
            bundle.mapper.Update()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_attribute_opacity(self, opacity: float, render: bool = True) -> None:
        opacity = max(0.0, min(1.0, float(opacity)))
        attr = self.current_attribute()
        attr.opacity = opacity
        for bundle in self.bundles.values():
            bundle.actor.SetOpacity(opacity)
        if render:
            self.interactor.GetRenderWindow().Render()

    def extract_range_attribute(self, min_value: float, max_value: float) -> str:
        source = self.current_attribute()
        base_name = f"{source.name}_range_{format_value(min_value)}_{format_value(max_value)}"
        new_name = base_name
        suffix = 1
        while new_name in self.attributes:
            suffix += 1
            new_name = f"{base_name}_{suffix}"

        output_volume = extract_range_volume(
            source.volume_data,
            min_value=min_value,
            max_value=max_value,
            name=new_name,
        )
        self.attributes[new_name] = load_attribute_from_volume(
            output_volume,
            name=new_name,
            spacing=self.spacing,
            clip_percentile=self.clip_percentile,
            opacity=source.opacity,
        )
        return new_name

    def extract_envelope_horizons(self, min_voxels: int = 1) -> list[str]:
        source = self.current_attribute()
        components = extract_connected_components(
            source.volume_data,
            min_voxels=min_voxels,
        )
        new_names: list[str] = []
        for component in components:
            color = (
                0.35 + 0.45 * ((component.index * 37) % 100) / 100.0,
                0.40 + 0.35 * ((component.index * 53) % 100) / 100.0,
                0.45 + 0.40 * ((component.index * 71) % 100) / 100.0,
            )
            try:
                actor, polydata, mapper, lut, scalar_range = create_horizon_surface_actor(
                    component.mask,
                    source.volume_data.data,
                    self.spacing,
                    self.clip_percentile,
                )
            except ValueError:
                continue
            base_name = f"{source.name}_component_{component.index}_horizon"
            new_name = base_name
            suffix = 1
            while new_name in self.horizons:
                suffix += 1
                new_name = f"{base_name}_{suffix}"
            horizon = HorizonSurface(
                name=new_name,
                actor=actor,
                mapper=mapper,
                polydata=polydata,
                lut=lut,
                component_index=component.index,
                voxel_count=component.voxel_count,
                scalar_range=scalar_range,
                component_mask=np.array(component.mask, copy=True),
                source_attribute_name=source.name,
            )
            self.horizons[new_name] = horizon
            self.renderer.AddActor(actor)
            new_names.append(new_name)
        if new_names:
            self.set_current_horizon(new_names[0], render=False)
        return new_names

    def set_current_horizon(self, name: str | None, render: bool = True) -> None:
        self.current_horizon_name = name
        for horizon_name, horizon in self.horizons.items():
            prop = horizon.actor.GetProperty()
            if horizon_name == name:
                prop.SetOpacity(min(1.0, horizon.opacity + 0.18) if horizon.visible else 0.0)
                prop.SetLineWidth(2.5)
                prop.SetAmbient(0.25)
                prop.SetSpecular(0.35)
            else:
                prop.SetOpacity(horizon.opacity if horizon.visible else 0.0)
                prop.SetLineWidth(1.0)
                prop.SetAmbient(0.10)
                prop.SetSpecular(0.15)
            horizon.actor.SetVisibility(horizon.visible)
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_horizon_visibility(self, name: str, visible: bool, render: bool = True) -> None:
        horizon = self.horizons[name]
        horizon.visible = bool(visible)
        horizon.actor.SetVisibility(horizon.visible)
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()

    def extract_control_points_for_current_horizon(self, **intervals: int) -> str | None:
        horizon = self.current_horizon()
        if horizon is None or horizon.component_mask is None:
            return None
        source_attribute = self.attributes.get(horizon.source_attribute_name)
        if source_attribute is None:
            source_attribute = self.current_attribute()
        points = extract_control_points(
            source_attribute.volume_data,
            horizon.component_mask,
            **intervals,
        )
        if not points:
            return None
        actor, polydata, sphere_source = create_control_point_actor(points, self.spacing)
        base_name = f"{horizon.name}_control_points"
        new_name = base_name
        suffix = 1
        while new_name in self.control_point_sets:
            suffix += 1
            new_name = f"{base_name}_{suffix}"
        point_set = ControlPointSet(
            name=new_name,
            actor=actor,
            sphere_source=sphere_source,
            polydata=polydata,
            points=points,
            horizon_name=horizon.name,
            source_attribute_name=source_attribute.name,
            source_horizon_name=horizon.name,
            original_horizon_mask=np.array(horizon.component_mask, copy=True),
            display_scale=1.0,
        )
        self.control_point_sets[new_name] = point_set
        self.renderer.AddActor(actor)
        self.set_current_control_point_set(new_name, render=False)
        return new_name

    def set_current_control_point_set(self, name: str | None, render: bool = True) -> None:
        self.current_control_point_set_name = name
        for point_set_name, point_set in self.control_point_sets.items():
            prop = point_set.actor.GetProperty()
            if point_set_name == name:
                prop.SetOpacity(1.0 if point_set.visible else 0.0)
                prop.SetColor(1.0, 0.88, 0.18)
            else:
                prop.SetOpacity(0.82 if point_set.visible else 0.0)
                prop.SetColor(0.95, 0.70, 0.15)
            point_set.actor.SetVisibility(point_set.visible)
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_set_visibility(self, name: str, visible: bool, render: bool = True) -> None:
        point_set = self.control_point_sets[name]
        point_set.visible = bool(visible)
        point_set.actor.SetVisibility(point_set.visible)
        self.set_current_control_point_set(self.current_control_point_set_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()

    def current_control_point_set(self) -> ControlPointSet | None:
        if self.current_control_point_set_name is None:
            return None
        return self.control_point_sets[self.current_control_point_set_name]

    def edit_current_control_point_set_master(self, master_index: int, delta_sample: float) -> bool:
        point_set = self.current_control_point_set()
        if point_set is None:
            return False
        source_attribute = self.attributes.get(point_set.source_attribute_name)
        if source_attribute is None:
            return False
        point_set.points = apply_master_point_z_move(
            point_set.points,
            master_index,
            delta_sample,
            source_attribute.volume_data,
        )
        actor, polydata, sphere_source = create_control_point_actor(
            point_set.points,
            self.spacing,
            display_scale=point_set.display_scale,
        )
        point_set.actor.SetMapper(actor.GetMapper())
        point_set.sphere_source = sphere_source
        point_set.polydata = polydata
        self.set_current_control_point_set(self.current_control_point_set_name, render=False)
        return True

    def rebuild_current_horizon_from_control_points(self) -> str | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        source_attribute = self.attributes.get(point_set.source_attribute_name)
        if source_attribute is None:
            return None
        master_points = point_set.master_points
        if len(master_points) < 4:
            return None
        base_horizon = self.horizons.get(point_set.source_horizon_name)
        if base_horizon is None:
            return None
        try:
            actor, polydata, mapper, lut, scalar_range = create_horizon_surface_from_control_points(
                master_points,
                self.spacing,
                self.clip_percentile,
            )
        except ValueError:
            return None

        base_name = f"{point_set.name}_rebuilt_horizon"
        new_name = base_name
        suffix = 1
        while new_name in self.horizons:
            suffix += 1
            new_name = f"{base_name}_{suffix}"
        horizon = HorizonSurface(
            name=new_name,
            actor=actor,
            mapper=mapper,
            polydata=polydata,
            lut=lut,
            component_index=base_horizon.component_index,
            voxel_count=len(master_points),
            scalar_range=scalar_range,
            opacity=base_horizon.opacity,
            visible=True,
            component_mask=np.array(point_set.original_horizon_mask, copy=True),
            source_attribute_name=source_attribute.name,
        )
        self.horizons[new_name] = horizon
        self.renderer.AddActor(actor)
        self.set_current_horizon(new_name, render=False)
        return new_name

    def current_control_point_display_scale(self) -> float | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        return float(point_set.display_scale)

    def set_control_point_display_scale(self, scale: float, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        point_set.display_scale = max(0.2, min(5.0, float(scale)))
        radius = max(
            min(self.spacing.xline, self.spacing.inline, self.spacing.sample)
            * 0.18
            * point_set.display_scale,
            1.5,
        )
        point_set.sphere_source.SetRadius(radius)
        point_set.sphere_source.Update()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_horizon_display_range(self, min_value: float, max_value: float, render: bool = True) -> None:
        if self.current_horizon_name is None:
            return
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        horizon = self.horizons[self.current_horizon_name]
        horizon.scalar_range = (float(min_value), float(max_value))
        horizon.lut.SetRange(*horizon.scalar_range)
        horizon.lut.Build()
        horizon.mapper.SetScalarRange(horizon.scalar_range)
        horizon.mapper.Update()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_horizon_opacity(self, opacity: float, render: bool = True) -> None:
        if self.current_horizon_name is None:
            return
        opacity = max(0.0, min(1.0, float(opacity)))
        horizon = self.horizons[self.current_horizon_name]
        horizon.opacity = opacity
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()

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

        self.setWindowTitle(APP_NAME)
        self.resize(2200, 1400)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        data_panel = QtWidgets.QWidget()
        data_panel.setMinimumWidth(280)
        data_panel.setMaximumWidth(360)
        data_layout = QtWidgets.QVBoxLayout(data_panel)

        data_header = QtWidgets.QLabel("Data")
        data_header_font = QtGui.QFont()
        data_header_font.setPointSize(18)
        data_header_font.setBold(True)
        data_header.setFont(data_header_font)
        data_layout.addWidget(data_header)

        seismic_group = QtWidgets.QGroupBox("Seismic")
        seismic_layout = QtWidgets.QVBoxLayout(seismic_group)
        self.seismic_list = QtWidgets.QListWidget()
        seismic_layout.addWidget(self.seismic_list)
        data_layout.addWidget(seismic_group)

        attributes_group = QtWidgets.QGroupBox("Attributes")
        attributes_layout = QtWidgets.QVBoxLayout(attributes_group)
        self.attributes_list = QtWidgets.QListWidget()
        attributes_layout.addWidget(self.attributes_list)
        data_layout.addWidget(attributes_group)

        horizons_group = QtWidgets.QGroupBox("Horizons")
        horizons_layout = QtWidgets.QVBoxLayout(horizons_group)
        self.horizons_list = QtWidgets.QListWidget()
        horizons_layout.addWidget(self.horizons_list)
        data_layout.addWidget(horizons_group)

        wells_group = QtWidgets.QGroupBox("Control Points")
        wells_layout = QtWidgets.QVBoxLayout(wells_group)
        self.control_points_list = QtWidgets.QListWidget()
        wells_layout.addWidget(self.control_points_list)
        data_layout.addWidget(wells_group)
        data_layout.addStretch(1)
        layout.addWidget(data_panel, stretch=0)

        viewer_panel = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_header = QtWidgets.QLabel("3D View")
        viewer_header_font = QtGui.QFont()
        viewer_header_font.setPointSize(18)
        viewer_header_font.setBold(True)
        viewer_header.setFont(viewer_header_font)
        viewer_layout.addWidget(viewer_header)
        self.vtk_widget.setMinimumSize(1000, 900)
        viewer_layout.addWidget(self.vtk_widget, stretch=1)
        layout.addWidget(viewer_panel, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(360)
        layout.addWidget(panel, stretch=0)
        panel_layout = QtWidgets.QVBoxLayout(panel)

        header = QtWidgets.QLabel("Operations")
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
            "extract value ranges, build horizon envelopes, and reset the view."
        )
        note.setWordWrap(True)
        panel_layout.addWidget(note)

        self.reset_view_button = QtWidgets.QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        panel_layout.addWidget(self.reset_view_button)

        attribute_display_group = QtWidgets.QGroupBox("Attribute Display")
        attribute_display_layout = QtWidgets.QGridLayout(attribute_display_group)
        attribute_display_layout.addWidget(QtWidgets.QLabel("Min"), 0, 0)
        attribute_display_layout.addWidget(QtWidgets.QLabel("Max"), 1, 0)
        self.attribute_display_min_edit = QtWidgets.QLineEdit()
        self.attribute_display_max_edit = QtWidgets.QLineEdit()
        range_validator = QtGui.QDoubleValidator()
        self.attribute_display_min_edit.setValidator(range_validator)
        self.attribute_display_max_edit.setValidator(range_validator)
        attribute_display_layout.addWidget(self.attribute_display_min_edit, 0, 1)
        attribute_display_layout.addWidget(self.attribute_display_max_edit, 1, 1)
        self.apply_attribute_display_button = QtWidgets.QPushButton("Apply Attribute Display")
        self.apply_attribute_display_button.clicked.connect(self.apply_attribute_display)
        attribute_display_layout.addWidget(self.apply_attribute_display_button, 2, 0, 1, 2)
        attribute_display_layout.addWidget(QtWidgets.QLabel("Opacity"), 3, 0)
        self.attribute_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.attribute_opacity_slider.setRange(0, 100)
        self.attribute_opacity_slider.valueChanged.connect(self.change_attribute_opacity)
        attribute_display_layout.addWidget(self.attribute_opacity_slider, 3, 1)
        panel_layout.addWidget(attribute_display_group)

        range_group = QtWidgets.QGroupBox("Extract Range")
        range_layout = QtWidgets.QVBoxLayout(range_group)
        self.extract_button = QtWidgets.QPushButton("Open Range Extraction")
        self.extract_button.clicked.connect(self.open_extract_range_dialog)
        range_layout.addWidget(self.extract_button)
        panel_layout.addWidget(range_group)

        envelope_group = QtWidgets.QGroupBox("Extract Horizon Envelopes")
        envelope_layout = QtWidgets.QVBoxLayout(envelope_group)
        self.extract_envelope_button = QtWidgets.QPushButton("Open Horizon Extraction")
        self.extract_envelope_button.clicked.connect(self.open_extract_horizon_dialog)
        envelope_layout.addWidget(self.extract_envelope_button)
        panel_layout.addWidget(envelope_group)

        control_points_group = QtWidgets.QGroupBox("Extract Control Points")
        control_points_layout = QtWidgets.QVBoxLayout(control_points_group)
        self.extract_control_points_button = QtWidgets.QPushButton("Open Control Point Extraction")
        self.extract_control_points_button.clicked.connect(self.open_extract_control_points_dialog)
        control_points_layout.addWidget(self.extract_control_points_button)
        self.edit_master_point_button = QtWidgets.QPushButton("Edit Master Point")
        self.edit_master_point_button.clicked.connect(self.open_edit_master_point_dialog)
        control_points_layout.addWidget(self.edit_master_point_button)
        self.rebuild_horizon_button = QtWidgets.QPushButton("Rebuild Horizon from Control Points")
        self.rebuild_horizon_button.clicked.connect(self.rebuild_horizon_from_control_points)
        control_points_layout.addWidget(self.rebuild_horizon_button)
        control_points_layout.addWidget(QtWidgets.QLabel("Control Point Display Size"))
        self.control_point_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.control_point_size_slider.setRange(20, 300)
        self.control_point_size_slider.valueChanged.connect(self.change_control_point_size)
        control_points_layout.addWidget(self.control_point_size_slider)
        panel_layout.addWidget(control_points_group)

        horizon_display_group = QtWidgets.QGroupBox("Horizon Display")
        horizon_display_layout = QtWidgets.QGridLayout(horizon_display_group)
        horizon_display_layout.addWidget(QtWidgets.QLabel("Min"), 0, 0)
        horizon_display_layout.addWidget(QtWidgets.QLabel("Max"), 1, 0)
        self.horizon_display_min_edit = QtWidgets.QLineEdit()
        self.horizon_display_max_edit = QtWidgets.QLineEdit()
        self.horizon_display_min_edit.setValidator(range_validator)
        self.horizon_display_max_edit.setValidator(range_validator)
        horizon_display_layout.addWidget(self.horizon_display_min_edit, 0, 1)
        horizon_display_layout.addWidget(self.horizon_display_max_edit, 1, 1)
        self.apply_horizon_display_button = QtWidgets.QPushButton("Apply Horizon Display")
        self.apply_horizon_display_button.clicked.connect(self.apply_horizon_display)
        horizon_display_layout.addWidget(self.apply_horizon_display_button, 2, 0, 1, 2)
        horizon_display_layout.addWidget(QtWidgets.QLabel("Opacity"), 3, 0)
        self.horizon_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.horizon_opacity_slider.setRange(0, 100)
        self.horizon_opacity_slider.valueChanged.connect(self.change_horizon_opacity)
        horizon_display_layout.addWidget(self.horizon_opacity_slider, 3, 1)
        panel_layout.addWidget(horizon_display_group)

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
        self.horizons_list.currentItemChanged.connect(self.change_horizon)
        self.horizons_list.itemChanged.connect(self.toggle_horizon_visibility)
        self.control_points_list.currentItemChanged.connect(self.change_control_point_set)
        self.control_points_list.itemChanged.connect(self.toggle_control_point_set_visibility)

        self.seismic_list.addItem("Seismic")
        self.refresh_attributes()
        self.refresh_horizons()
        self.refresh_control_points()
        self.refresh_info()
        self.refresh_display_controls()

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

    def refresh_display_controls(self) -> None:
        attr_min, attr_max = self.updater.current_attribute_display_range()
        if not self.attribute_display_min_edit.hasFocus():
            self.attribute_display_min_edit.setText(format_value(attr_min))
        if not self.attribute_display_max_edit.hasFocus():
            self.attribute_display_max_edit.setText(format_value(attr_max))
        self.attribute_opacity_slider.blockSignals(True)
        self.attribute_opacity_slider.setValue(int(round(self.updater.current_attribute_opacity() * 100.0)))
        self.attribute_opacity_slider.blockSignals(False)

        horizon_range = self.updater.current_horizon_scalar_range()
        horizon_opacity = self.updater.current_horizon_opacity()
        has_horizon = horizon_range is not None and horizon_opacity is not None
        self.horizon_display_min_edit.setEnabled(has_horizon)
        self.horizon_display_max_edit.setEnabled(has_horizon)
        self.apply_horizon_display_button.setEnabled(has_horizon)
        self.horizon_opacity_slider.setEnabled(has_horizon)
        if has_horizon:
            if not self.horizon_display_min_edit.hasFocus():
                self.horizon_display_min_edit.setText(format_value(horizon_range[0]))
            if not self.horizon_display_max_edit.hasFocus():
                self.horizon_display_max_edit.setText(format_value(horizon_range[1]))
            self.horizon_opacity_slider.blockSignals(True)
            self.horizon_opacity_slider.setValue(int(round(horizon_opacity * 100.0)))
            self.horizon_opacity_slider.blockSignals(False)
        else:
            self.horizon_display_min_edit.clear()
            self.horizon_display_max_edit.clear()
            self.horizon_opacity_slider.blockSignals(True)
            self.horizon_opacity_slider.setValue(0)
            self.horizon_opacity_slider.blockSignals(False)

        has_control_points = self.updater.current_control_point_set() is not None
        self.edit_master_point_button.setEnabled(has_control_points)
        self.rebuild_horizon_button.setEnabled(has_control_points)
        self.control_point_size_slider.setEnabled(has_control_points)
        control_point_scale = self.updater.current_control_point_display_scale()
        self.control_point_size_slider.blockSignals(True)
        self.control_point_size_slider.setValue(
            int(round((1.0 if control_point_scale is None else control_point_scale) * 100.0))
        )
        self.control_point_size_slider.blockSignals(False)

    def refresh_attributes(self) -> None:
        selected = self.updater.current_attribute_name
        self.attributes_list.blockSignals(True)
        self.attributes_list.clear()
        self.attributes_list.addItems(self.updater.attribute_names())
        items = self.attributes_list.findItems(selected, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.attributes_list.setCurrentItem(items[0])
        self.attributes_list.blockSignals(False)

    def refresh_horizons(self) -> None:
        selected = self.updater.current_horizon_name
        self.horizons_list.blockSignals(True)
        self.horizons_list.clear()
        for name in self.updater.horizon_names():
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if self.updater.horizons[name].visible
                else QtCore.Qt.CheckState.Unchecked
            )
            self.horizons_list.addItem(item)
            if name == selected:
                self.horizons_list.setCurrentItem(item)
        self.horizons_list.blockSignals(False)
        self.refresh_display_controls()

    def refresh_control_points(self) -> None:
        selected = self.updater.current_control_point_set_name
        self.control_points_list.blockSignals(True)
        self.control_points_list.clear()
        for name in self.updater.control_point_set_names():
            point_set = self.updater.control_point_sets[name]
            label = f"{name} ({len(point_set.points)} pts, {len(point_set.master_points)} masters)"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if point_set.visible
                else QtCore.Qt.CheckState.Unchecked
            )
            self.control_points_list.addItem(item)
            if name == selected:
                self.control_points_list.setCurrentItem(item)
        self.control_points_list.blockSignals(False)

    def change_attribute(self, name: str) -> None:
        if not name:
            return
        self.updater.set_attribute(name, render=False)
        self.image = self.updater.image
        self.refresh_info()
        self.refresh_display_controls()
        self.schedule_render()

    def change_horizon(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        del previous
        name = None if current is None else current.text()
        self.updater.set_current_horizon(name, render=False)
        self.refresh_display_controls()
        self.schedule_render()

    def toggle_horizon_visibility(self, item: QtWidgets.QListWidgetItem) -> None:
        self.updater.set_horizon_visibility(
            item.text(),
            item.checkState() == QtCore.Qt.CheckState.Checked,
            render=False,
        )
        self.refresh_display_controls()
        self.schedule_render()

    def change_control_point_set(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        del previous
        name = None if current is None else current.data(QtCore.Qt.ItemDataRole.UserRole)
        self.updater.set_current_control_point_set(name, render=False)
        self.refresh_display_controls()
        self.schedule_render()

    def toggle_control_point_set_visibility(self, item: QtWidgets.QListWidgetItem) -> None:
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.updater.set_control_point_set_visibility(
            str(name),
            item.checkState() == QtCore.Qt.CheckState.Checked,
            render=False,
        )
        self.refresh_display_controls()
        self.schedule_render()

    def change_control_point_size(self, value: int) -> None:
        self.updater.set_control_point_display_scale(value / 100.0, render=False)
        self.schedule_render()

    def apply_attribute_display(self) -> None:
        min_text = self.attribute_display_min_edit.text().strip()
        max_text = self.attribute_display_max_edit.text().strip()
        if not min_text or not max_text:
            return
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return
        self.updater.set_attribute_display_range(min_value, max_value, render=False)
        self.schedule_render()

    def change_attribute_opacity(self, value: int) -> None:
        self.updater.set_attribute_opacity(value / 100.0, render=False)
        self.schedule_render()

    def apply_horizon_display(self) -> None:
        min_text = self.horizon_display_min_edit.text().strip()
        max_text = self.horizon_display_max_edit.text().strip()
        if not min_text or not max_text:
            return
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return
        self.updater.set_horizon_display_range(min_value, max_value, render=False)
        self.schedule_render()

    def change_horizon_opacity(self, value: int) -> None:
        self.updater.set_horizon_opacity(value / 100.0, render=False)
        self.schedule_render()

    def open_extract_range_dialog(self) -> None:
        min_value, max_value = self.updater.current_scalar_range()
        dialog = ExtractRangeDialog(min_value, max_value, self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        min_value, max_value = values
        new_name = self.updater.extract_range_attribute(min_value, max_value)
        self.refresh_attributes()
        items = self.attributes_list.findItems(new_name, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.attributes_list.setCurrentItem(items[0])
        self.refresh_display_controls()
        self.schedule_render()

    def open_extract_horizon_dialog(self) -> None:
        dialog = ExtractHorizonDialog(1, self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        min_voxels = dialog.values()
        if min_voxels is None:
            return
        new_names = self.updater.extract_envelope_horizons(min_voxels=min_voxels)
        if not new_names:
            QtWidgets.QMessageBox.information(
                self,
                "No Envelopes",
                "No envelope components met the minimum voxel threshold.",
            )
            return
        self.refresh_horizons()
        items = self.horizons_list.findItems(new_names[0], QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.horizons_list.setCurrentItem(items[0])
        self.refresh_display_controls()
        self.schedule_render()

    def open_extract_control_points_dialog(self) -> None:
        if self.updater.current_horizon_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Horizon",
                "Select a horizon before extracting control points.",
            )
            return
        dialog = ExtractControlPointsDialog(self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        new_name = self.updater.extract_control_points_for_current_horizon(**values)
        if new_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Control Points",
                "No control points were generated for the current horizon.",
            )
            return
        self.refresh_control_points()
        for row in range(self.control_points_list.count()):
            item = self.control_points_list.item(row)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == new_name:
                self.control_points_list.setCurrentItem(item)
                break
        self.refresh_display_controls()
        self.schedule_render()

    def open_edit_master_point_dialog(self) -> None:
        point_set = self.updater.current_control_point_set()
        if point_set is None:
            return
        surface_points = [point for point in point_set.points if point.kind == "surface" and point.master_index is not None]
        if not surface_points:
            QtWidgets.QMessageBox.information(
                self,
                "No Master Points",
                "The current control point set does not contain editable master points.",
            )
            return
        dialog = EditMasterPointDialog(surface_points, self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        master_index, delta_sample = values
        if not self.updater.edit_current_control_point_set_master(master_index, delta_sample):
            QtWidgets.QMessageBox.information(
                self,
                "Edit Failed",
                "Failed to update the selected master point.",
            )
            return
        self.refresh_control_points()
        self.refresh_display_controls()
        self.schedule_render()

    def rebuild_horizon_from_control_points(self) -> None:
        new_name = self.updater.rebuild_current_horizon_from_control_points()
        if new_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "Rebuild Failed",
                "Failed to rebuild a horizon from the current control point set.",
            )
            return
        self.refresh_horizons()
        items = self.horizons_list.findItems(new_name, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.horizons_list.setCurrentItem(items[0])
        self.schedule_render()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.render_window.Finalize()
        super().closeEvent(event)


def launch_vtk_viewer(
    segy_path: Path,
    initial_attribute: AttributeVolume,
    spacing: RenderSpacing,
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

    image = initial_attribute.image
    xlines = initial_attribute.volume_data.xlines
    inlines = initial_attribute.volume_data.inlines
    samples = initial_attribute.volume_data.samples
    lut = initial_attribute.lut
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
        spacing.xline,
        spacing.inline,
        spacing.sample,
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
        renderer=renderer,
        bundles={
            "xline": xline_bundle,
            "inline": inline_bundle,
            "sample": sample_bundle,
        },
        overlay=overlay,
        segy_path=segy_path,
        initial_attribute=initial_attribute,
        spacing=spacing,
        clip_percentile=clip_percentile,
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

    geometry = load_segy_geometry(segy_path)
    volume_data = read_segy_volume(
        segy_path=segy_path,
        geometry=geometry,
        interval_inline=args.interval_inline,
        interval_xline=args.interval_xline,
        interval_sample=args.interval_sample,
    )
    spacing = RenderSpacing(
        xline=args.step_xline,
        inline=args.step_inline,
        sample=args.step_sample,
    )
    initial_attribute = load_attribute_from_volume(
        volume_data,
        name="seismic",
        spacing=spacing,
        clip_percentile=args.clip_percentile,
        opacity=args.opacity,
    )
    dims_text = (
        f"xlines={len(volume_data.xlines)}, "
        f"inlines={len(volume_data.inlines)}, "
        f"samples={len(volume_data.samples)}"
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
        initial_attribute=initial_attribute,
        spacing=spacing,
        clip_percentile=args.clip_percentile,
        opacity_scale=args.opacity,
        segy_path=segy_path,
        debug_ui=args.debug_ui,
    )


if __name__ == "__main__":
    sys.exit(main())
