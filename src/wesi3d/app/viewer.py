#!/usr/bin/env python3
"""
3D SEG-Y viewer for a velocity cube.

Default behavior:
- start an empty viewer
- load SEG-Y volumes on demand
- display three orthogonal slices with VTK
"""

from __future__ import annotations

import argparse
import json
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

from ..config import DEFAULT_VIEWER_CONFIG, DERIVED_DATA_DIR
from ..data.volume_data import VolumeData
from ..data.attribute_data import (
    AttributeVolume,
    DEFAULT_COLORMAP_NAME,
    RenderSpacing,
    apply_colormap_preset,
    available_colormap_names,
    create_lookup_table_from_scalars,
    load_attribute_from_volume,
)
from ..data.volume_data import load_segy_geometry, read_segy_volume
from .data_panel import DataPanelItem, DataPanelWidget
from ..processing.control_points import (
    ControlPoint,
    MasterMove,
    apply_master_point_z_moves,
    extract_control_points,
    master_control_points,
)
from ..utils.constants import INLINE_FIELD, XLINE_FIELD
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
    color: tuple[float, float, float] = (0.70, 0.88, 0.96)
    opacity: float = 0.55
    visible: bool = True
    component_mask: np.ndarray | None = None
    source_attribute_name: str = ""
    xlines: np.ndarray | None = None
    inlines: np.ndarray | None = None
    samples: np.ndarray | None = None
    control_point_set: ControlPointSet | None = None
    base_polydata: vtk.vtkPolyData | None = None


@dataclass
class ControlPointSet:
    name: str
    actor: vtk.vtkActor
    sphere_source: vtk.vtkSphereSource
    polydata: vtk.vtkPolyData
    master_actor: vtk.vtkActor
    master_sphere_source: vtk.vtkSphereSource
    master_polydata: vtk.vtkPolyData
    linked_master_actor: vtk.vtkActor
    linked_master_sphere_source: vtk.vtkSphereSource
    linked_master_polydata: vtk.vtkPolyData
    selected_master_actor: vtk.vtkActor
    selected_master_sphere_source: vtk.vtkSphereSource
    selected_master_polydata: vtk.vtkPolyData
    points: list[ControlPoint]
    horizon_name: str
    source_attribute_name: str
    xlines: np.ndarray
    inlines: np.ndarray
    samples: np.ndarray
    value_attribute_name: str | None
    use_attribute_colormap: bool
    source_horizon_name: str
    original_horizon_mask: np.ndarray
    display_scale: float = 1.0
    link_radius: float = 8.0
    rebuild_smoothness: float = 0.55
    visible: bool = True

    @property
    def master_points(self) -> list[ControlPoint]:
        return master_control_points(self.points)

    def master_point_by_index(self, master_index: int) -> ControlPoint | None:
        for point in self.master_points:
            if point.master_index == master_index:
                return point
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEG-Y slice viewer")
    parser.add_argument("segy_path", nargs="?", default=None, help="Optional path to a SEG-Y file.")
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


def create_scalar_bar_actor() -> vtk.vtkScalarBarActor:
    actor = vtk.vtkScalarBarActor()
    actor.SetNumberOfLabels(5)
    actor.SetMaximumWidthInPixels(110)
    actor.SetMaximumHeightInPixels(500)
    actor.SetWidth(0.10)
    actor.SetHeight(0.60)
    actor.SetPosition(0.88, 0.18)
    actor.SetUnconstrainedFontSize(True)
    title_prop = actor.GetTitleTextProperty()
    title_prop.SetColor(0.95, 0.95, 0.95)
    title_prop.SetFontSize(18)
    label_prop = actor.GetLabelTextProperty()
    label_prop.SetColor(0.95, 0.95, 0.95)
    label_prop.SetFontSize(15)
    actor.SetVisibility(False)
    return actor


def create_placeholder_image() -> tuple[vtk.vtkImageData, vtk.vtkLookupTable]:
    volume = VolumeData(
        data=np.zeros((1, 1, 1), dtype=np.float32),
        xlines=np.asarray([0.0], dtype=np.float32),
        inlines=np.asarray([0.0], dtype=np.float32),
        samples=np.asarray([0.0], dtype=np.float32),
        name="empty",
    )
    attribute = load_attribute_from_volume(
        volume,
        name="empty",
        spacing=RenderSpacing(1.0, 1.0, 1.0),
        clip_percentile=99.0,
        opacity=0.0,
    )
    return attribute.image, attribute.lut


def create_horizon_surface_actor(
    mask: np.ndarray,
    scalar_values: np.ndarray,
    spacing: RenderSpacing,
    clip_percentile: float,
    smoothing: float = 0.55,
    surface_color: tuple[float, float, float] = (0.82, 0.95, 1.0),
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
    smooth_factor = max(0.0, min(1.0, float(smoothing)))
    smoother.SetNumberOfIterations(int(round(8 + smooth_factor * 36)))
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.18 - smooth_factor * 0.14)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()

    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputConnection(smoother.GetOutputPort())
    fill_holes.SetHoleSize(
        max(float(spacing.xline), float(spacing.inline), float(spacing.sample)) * (6.0 + smooth_factor * 14.0)
    )

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(fill_holes.GetOutputPort())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
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
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.55)
    actor.GetProperty().SetColor(*surface_color)
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
    value_lut: vtk.vtkLookupTable | None = None,
    use_attribute_colormap: bool = False,
) -> tuple[
    vtk.vtkActor,
    vtk.vtkPolyData,
    vtk.vtkSphereSource,
    vtk.vtkActor,
    vtk.vtkPolyData,
    vtk.vtkSphereSource,
    vtk.vtkActor,
    vtk.vtkPolyData,
    vtk.vtkSphereSource,
    vtk.vtkActor,
    vtk.vtkPolyData,
    vtk.vtkSphereSource,
]:
    def _make_polydata(source_points: list[ControlPoint]) -> vtk.vtkPolyData:
        vtk_points = vtk.vtkPoints()
        values = vtk.vtkFloatArray()
        values.SetName("value")
        kinds = vtk.vtkUnsignedCharArray()
        kinds.SetName("kind")

        for point in source_points:
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
        return polydata

    def _make_actor(
        polydata: vtk.vtkPolyData,
        *,
        radius_factor: float,
        theta: int,
        phi: int,
        color: tuple[float, float, float],
        opacity: float,
        scalar_coloring: bool = False,
    ) -> tuple[vtk.vtkActor, vtk.vtkSphereSource]:
        sphere = vtk.vtkSphereSource()
        radius = max(
            min(spacing.xline, spacing.inline, spacing.sample) * radius_factor * float(display_scale),
            2.0,
        )
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(theta)
        sphere.SetPhiResolution(phi)

        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(polydata)
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.ScalingOff()
        if scalar_coloring and value_lut is not None:
            mapper.SetLookupTable(value_lut)
            mapper.SetUseLookupTableScalarRange(True)
            mapper.SetScalarRange(value_lut.GetRange())
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("value")
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        return actor, sphere

    polydata = _make_polydata(points)
    actor, sphere = _make_actor(
        polydata,
        radius_factor=0.28,
        theta=16,
        phi=16,
        color=(1.0, 0.88, 0.28),
        opacity=0.92,
        scalar_coloring=use_attribute_colormap and value_lut is not None,
    )

    master_points = master_control_points(points)
    master_polydata = _make_polydata(master_points)
    master_actor, master_sphere = _make_actor(
        master_polydata,
        radius_factor=0.44,
        theta=20,
        phi=20,
        color=(1.0, 0.34, 0.22),
        opacity=0.98,
    )
    linked_master_polydata = _make_polydata([])
    linked_master_actor, linked_master_sphere = _make_actor(
        linked_master_polydata,
        radius_factor=0.508,
        theta=22,
        phi=22,
        color=(0.16, 0.82, 1.0),
        opacity=0.96,
    )
    linked_master_actor.SetVisibility(False)
    selected_master_polydata = _make_polydata([])
    selected_master_actor, selected_master_sphere = _make_actor(
        selected_master_polydata,
        radius_factor=0.66,
        theta=24,
        phi=24,
        color=(0.20, 1.0, 0.32),
        opacity=1.0,
    )
    selected_master_actor.SetVisibility(False)
    return (
        actor,
        polydata,
        sphere,
        master_actor,
        master_polydata,
        master_sphere,
        linked_master_actor,
        linked_master_polydata,
        linked_master_sphere,
        selected_master_actor,
        selected_master_polydata,
        selected_master_sphere,
    )


def create_horizon_surface_from_control_points(
    base_polydata: vtk.vtkPolyData,
    points: list[ControlPoint],
    spacing: RenderSpacing,
    clip_percentile: float,
    smoothing: float = 0.55,
) -> tuple[vtk.vtkActor, vtk.vtkPolyData, vtk.vtkPolyDataMapper, vtk.vtkLookupTable, tuple[float, float]]:
    if len(points) < 4:
        raise ValueError("At least 4 master points are required to deform a horizon.")
    if base_polydata.GetNumberOfPoints() == 0:
        raise ValueError("The source horizon surface is empty.")

    master_points = [point for point in points if point.kind == "surface" and point.master_index is not None]
    if len(master_points) < 4:
        raise ValueError("At least 4 master points are required to deform a horizon.")

    source_landmarks = vtk.vtkPoints()
    target_landmarks = vtk.vtkPoints()
    has_deformation = False
    for point in master_points:
        source_z = float(point.sample_index - point.dz) * float(spacing.sample)
        target_z = float(point.sample_index) * float(spacing.sample)
        x = float(point.xline_index) * float(spacing.xline)
        y = float(point.inline_index) * float(spacing.inline)
        source_landmarks.InsertNextPoint(x, y, source_z)
        target_landmarks.InsertNextPoint(x, y, target_z)
        if abs(target_z - source_z) > 1e-9:
            has_deformation = True
    if not has_deformation:
        raise ValueError("The master points do not contain any deformation yet.")

    smooth_factor = max(0.0, min(1.0, float(smoothing)))
    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(source_landmarks)
    transform.SetTargetLandmarks(target_landmarks)
    transform.SetBasisToR()

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(base_polydata)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(transform_filter.GetOutputPort())

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(clean.GetOutputPort())

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(triangulate.GetOutputPort())
    smooth.SetNumberOfIterations(int(round(6 + smooth_factor * 18)))
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.SetPassBand(0.24 - smooth_factor * 0.14)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    surface_polydata = vtk.vtkPolyData()
    surface_polydata.DeepCopy(normals.GetOutput())
    if surface_polydata.GetNumberOfPoints() == 0 or surface_polydata.GetNumberOfPolys() == 0:
        raise ValueError("Empty rebuilt horizon surface.")

    point_scalars = surface_polydata.GetPointData().GetScalars()
    if point_scalars is None:
        points_data = surface_polydata.GetPoints()
        if points_data is None or points_data.GetData() is None:
            scalar_array = np.asarray([0.0, 1.0], dtype=np.float32)
        else:
            point_xyz = np.asarray(numpy_support.vtk_to_numpy(points_data.GetData()), dtype=np.float32)
            if point_xyz.size == 0:
                scalar_array = np.asarray([0.0, 1.0], dtype=np.float32)
            else:
                scalar_array = np.asarray(point_xyz[:, 2], dtype=np.float32)
    else:
        scalar_array = numpy_support.vtk_to_numpy(point_scalars)
    scalar_range = (float(np.min(scalar_array)), float(np.max(scalar_array)))
    lut = create_lookup_table_from_scalars(scalar_array, clip_percentile)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface_polydata)
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    mapper.SetScalarRange(lut.GetRange())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.55)
    actor.GetProperty().SetColor(0.82, 0.95, 1.0)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().EdgeVisibilityOff()
    return actor, surface_polydata, mapper, lut, scalar_range


def clone_polydata(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    cloned = vtk.vtkPolyData()
    cloned.DeepCopy(polydata)
    return cloned


def polydata_to_payload(polydata: vtk.vtkPolyData) -> dict[str, np.ndarray]:
    points_data = polydata.GetPoints()
    points = (
        np.empty((0, 3), dtype=np.float32)
        if points_data is None or points_data.GetData() is None
        else np.asarray(numpy_support.vtk_to_numpy(points_data.GetData()), dtype=np.float32)
    )
    polys = polydata.GetPolys()
    if polys is None:
        offsets = np.asarray([0], dtype=np.int64)
        connectivity = np.empty((0,), dtype=np.int64)
    else:
        offsets_array = polys.GetOffsetsArray()
        connectivity_array = polys.GetConnectivityArray()
        if offsets_array is None or connectivity_array is None:
            offsets = np.asarray([0], dtype=np.int64)
            connectivity = np.empty((0,), dtype=np.int64)
        else:
            offsets = np.asarray(numpy_support.vtk_to_numpy(offsets_array), dtype=np.int64)
            connectivity = np.asarray(numpy_support.vtk_to_numpy(connectivity_array), dtype=np.int64)
    return {
        "points": points,
        "polys_offsets": offsets,
        "polys_connectivity": connectivity,
    }


def polydata_from_payload(
    points: np.ndarray,
    polys_offsets: np.ndarray,
    polys_connectivity: np.ndarray,
) -> vtk.vtkPolyData:
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    if len(points) > 0:
        vtk_points.SetData(
            numpy_support.numpy_to_vtk(np.asarray(points, dtype=np.float32), deep=True)
        )
    polydata.SetPoints(vtk_points)

    cell_array = vtk.vtkCellArray()
    offsets = np.asarray(polys_offsets, dtype=np.int64).ravel()
    connectivity = np.asarray(polys_connectivity, dtype=np.int64).ravel()
    if offsets.size > 0 and connectivity.size > 0:
        vtk_offsets = numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True)
        vtk_connectivity = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)
        cell_array.SetData(vtk_offsets, vtk_connectivity)
    polydata.SetPolys(cell_array)
    return polydata


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
        self.values = np.asarray(values)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setTracking(True)
        self.index_label = QtWidgets.QLabel("")
        self.index_label.setMinimumWidth(60)
        self.value_edit = QtWidgets.QLineEdit()
        self.value_edit.setMaximumWidth(88)
        self.value_edit.setValidator(QtGui.QDoubleValidator())
        self.go_button = QtWidgets.QPushButton("Set")
        self.go_button.setMaximumWidth(52)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.index_label)
        layout.addWidget(self.value_edit)
        layout.addWidget(self.go_button)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.go_button.clicked.connect(self._apply_value)
        self.value_edit.returnPressed.connect(self._apply_value)

        self.set_values(self.values)

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

    def set_values(self, values: np.ndarray, index: int | None = None) -> None:
        self.values = np.asarray(values)
        self.slider.blockSignals(True)
        self.slider.setEnabled(len(self.values) > 0)
        self.go_button.setEnabled(len(self.values) > 0)
        self.value_edit.setEnabled(len(self.values) > 0)
        self.slider.setRange(0, max(0, len(self.values) - 1))
        self.slider.blockSignals(False)
        if len(self.values) == 0:
            self.index_label.setText("index=-")
            self.value_edit.clear()
            return
        self.set_index(len(self.values) // 2 if index is None else index)


class ColorMapControlWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QtWidgets.QLabel("Preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(available_colormap_names())
        preset_row.addWidget(self.preset_combo, stretch=1)
        layout.addLayout(preset_row)

        range_row = QtWidgets.QHBoxLayout()
        range_row.setSpacing(6)
        range_row.addWidget(QtWidgets.QLabel("Min"))
        self.min_edit = QtWidgets.QLineEdit()
        self.min_edit.setMaximumWidth(72)
        range_row.addWidget(self.min_edit)
        range_row.addWidget(QtWidgets.QLabel("Max"))
        self.max_edit = QtWidgets.QLineEdit()
        self.max_edit.setMaximumWidth(72)
        range_row.addWidget(self.max_edit)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setMaximumWidth(60)
        range_row.addWidget(self.apply_button)
        layout.addLayout(range_row)

        validator = QtGui.QDoubleValidator(self)
        self.min_edit.setValidator(validator)
        self.max_edit.setValidator(validator)

    def set_range(self, value_range: tuple[float, float] | None) -> None:
        if value_range is None:
            if not self.min_edit.hasFocus():
                self.min_edit.clear()
            if not self.max_edit.hasFocus():
                self.max_edit.clear()
            return
        if not self.min_edit.hasFocus():
            self.min_edit.setText(format_value(value_range[0]))
        if not self.max_edit.hasFocus():
            self.max_edit.setText(format_value(value_range[1]))

    def set_current_preset(self, name: str | None) -> None:
        target = DEFAULT_COLORMAP_NAME if name is None else name
        index = self.preset_combo.findText(target)
        if index < 0:
            index = self.preset_combo.findText(DEFAULT_COLORMAP_NAME)
        self.preset_combo.blockSignals(True)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.preset_combo.setEnabled(enabled)
        self.min_edit.setEnabled(enabled)
        self.max_edit.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)


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
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
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
            return {
                key: max(1, int(value))
                for key, value in fields.items()
                if value
            }
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


class CopyControlPointValuesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        horizon_names: list[str],
        attribute_names: list[str],
        selected_horizon_name: str | None = None,
        selected_attribute_name: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Copy Attribute Values To Control Points")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.horizon_combo = QtWidgets.QComboBox()
        for name in horizon_names:
            self.horizon_combo.addItem(name, name)
        if selected_horizon_name is not None:
            index = self.horizon_combo.findData(selected_horizon_name)
            if index >= 0:
                self.horizon_combo.setCurrentIndex(index)
        form.addRow("Horizon", self.horizon_combo)

        self.attribute_combo = QtWidgets.QComboBox()
        for name in attribute_names:
            self.attribute_combo.addItem(name, name)
        if selected_attribute_name is not None:
            index = self.attribute_combo.findData(selected_attribute_name)
            if index >= 0:
                self.attribute_combo.setCurrentIndex(index)
        form.addRow("Attribute", self.attribute_combo)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str] | None:
        horizon_name = self.horizon_combo.currentData()
        attribute_name = self.attribute_combo.currentData()
        if horizon_name is None or attribute_name is None:
            return None
        return str(horizon_name), str(attribute_name)


class LoadSeismicDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None, target_category: str = "seismic") -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Seismic Data")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.path_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_path)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_button)
        form.addRow("SEG-Y", path_row)

        self.name_edit = QtWidgets.QLineEdit()
        form.addRow("Name", self.name_edit)

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("地震", "seismic")
        self.target_combo.addItem("属性", "attribute")
        index = 0 if target_category == "seismic" else 1
        self.target_combo.setCurrentIndex(index)
        form.addRow("导入为", self.target_combo)

        int_validator = QtGui.QIntValidator(1, 10**9, self)
        float_validator = QtGui.QDoubleValidator(self)

        self.interval_inline_edit = QtWidgets.QLineEdit(str(DEFAULT_VIEWER_CONFIG.interval_inline))
        self.interval_xline_edit = QtWidgets.QLineEdit(str(DEFAULT_VIEWER_CONFIG.interval_xline))
        self.interval_sample_edit = QtWidgets.QLineEdit(str(DEFAULT_VIEWER_CONFIG.interval_sample))
        for widget in (self.interval_inline_edit, self.interval_xline_edit, self.interval_sample_edit):
            widget.setValidator(int_validator)
        form.addRow("Inline 抽稀", self.interval_inline_edit)
        form.addRow("Xline 抽稀", self.interval_xline_edit)
        form.addRow("Sample 抽稀", self.interval_sample_edit)

        self.step_inline_edit = QtWidgets.QLineEdit(format_value(DEFAULT_VIEWER_CONFIG.step_inline))
        self.step_xline_edit = QtWidgets.QLineEdit(format_value(DEFAULT_VIEWER_CONFIG.step_xline))
        self.step_sample_edit = QtWidgets.QLineEdit(format_value(DEFAULT_VIEWER_CONFIG.step_sample))
        for widget in (self.step_inline_edit, self.step_xline_edit, self.step_sample_edit):
            widget.setValidator(float_validator)
        form.addRow("Inline Step", self.step_inline_edit)
        form.addRow("Xline Step", self.step_xline_edit)
        form.addRow("Sample Step", self.step_sample_edit)

        self.inline_field_edit = QtWidgets.QLineEdit(str(INLINE_FIELD))
        self.xline_field_edit = QtWidgets.QLineEdit(str(XLINE_FIELD))
        for widget in (self.inline_field_edit, self.xline_field_edit):
            widget.setValidator(QtGui.QIntValidator(1, 10**9, self))
        form.addRow("Inline 道头位置", self.inline_field_edit)
        form.addRow("Xline 道头位置", self.xline_field_edit)

        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select SEG-Y File",
            "",
            "SEG-Y Files (*.sgy *.segy);;All Files (*)",
        )
        if path:
            self.path_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(path).stem)

    def values(self) -> dict[str, object] | None:
        path_text = self.path_edit.text().strip()
        if not path_text:
            return None
        try:
            return {
                "path": path_text,
                "name": self.name_edit.text().strip() or Path(path_text).stem,
                "target_category": str(self.target_combo.currentData()),
                "interval_inline": max(1, int(self.interval_inline_edit.text().strip() or "1")),
                "interval_xline": max(1, int(self.interval_xline_edit.text().strip() or "1")),
                "interval_sample": max(1, int(self.interval_sample_edit.text().strip() or "1")),
                "step_inline": float(self.step_inline_edit.text().strip() or "1"),
                "step_xline": float(self.step_xline_edit.text().strip() or "1"),
                "step_sample": float(self.step_sample_edit.text().strip() or "1"),
                "inline_field": int(self.inline_field_edit.text().strip() or str(INLINE_FIELD)),
                "xline_field": int(self.xline_field_edit.text().strip() or str(XLINE_FIELD)),
            }
        except ValueError:
            return None


class SliceUpdater:
    def __init__(
        self,
        interactor: vtk.vtkRenderWindowInteractor,
        renderer: vtk.vtkRenderer,
        bundles: dict[str, SliceActorBundle],
        overlay: vtk.vtkTextActor,
        scalar_bar_actor: vtk.vtkScalarBarActor,
        segy_path: Path | None,
        initial_attribute: AttributeVolume | None,
        spacing: RenderSpacing,
        clip_percentile: float,
        opacity: float,
    ) -> None:
        self.interactor = interactor
        self.renderer = renderer
        self.bundles = bundles
        self.overlay = overlay
        self.scalar_bar_actor = scalar_bar_actor
        self.segy_path = segy_path
        self.spacing = spacing
        self.clip_percentile = clip_percentile
        self.opacity = opacity
        self.attributes: dict[str, AttributeVolume] = {}
        self.horizons: dict[str, HorizonSurface] = {}
        self.current_horizon_name: str | None = None
        self.current_attribute_name: str | None = None
        self.last_rebuild_error: str | None = None
        self.image = bundles["xline"].image
        self.xlines = np.asarray([0.0], dtype=np.float32)
        self.inlines = np.asarray([0.0], dtype=np.float32)
        self.samples = np.asarray([0.0], dtype=np.float32)
        self.indices = {
            "xline": 0,
            "inline": 0,
            "sample": 0,
        }
        if initial_attribute is not None:
            self.attributes[initial_attribute.name] = initial_attribute
            self.current_attribute_name = initial_attribute.name
            self.image = initial_attribute.image
            self._sync_axes_from_current_attribute()
            self.indices = {
                "xline": len(self.xlines) // 2,
                "inline": len(self.inlines) // 2,
                "sample": len(self.samples) // 2,
            }
        self.update_overlay()
        self.refresh_scalar_bar()

    @staticmethod
    def _unique_name(existing: dict[str, object], base_name: str) -> str:
        new_name = base_name
        suffix = 1
        while new_name in existing:
            suffix += 1
            new_name = f"{base_name}_{suffix}"
        return new_name

    def _sync_axes_from_current_attribute(self) -> None:
        attribute = self.current_attribute()
        if attribute is None:
            self.xlines = np.asarray([0.0], dtype=np.float32)
            self.inlines = np.asarray([0.0], dtype=np.float32)
            self.samples = np.asarray([0.0], dtype=np.float32)
            return
        volume_data = attribute.volume_data
        self.xlines = volume_data.xlines
        self.inlines = volume_data.inlines
        self.samples = volume_data.samples

    def current_text(self) -> str:
        if not self.has_attribute_data():
            return "No data loaded\nUse 'Load Seismic Data' to import a SEG-Y volume."
        return (
            f"{self.segy_path.name if self.segy_path is not None else 'No File'}\n"
            f"Attribute: {self.current_attribute_name or '-'}\n"
            f"Crossline: {format_value(self.xlines[self.indices['xline']])}\n"
            f"Inline: {format_value(self.inlines[self.indices['inline']])}\n"
            f"Sample: {format_value(self.samples[self.indices['sample']])}"
        )

    def attribute_names(self) -> list[str]:
        return list(self.attributes.keys())

    def has_attribute_data(self) -> bool:
        return self.current_attribute_name is not None and self.current_attribute_name in self.attributes

    def current_attribute(self) -> AttributeVolume | None:
        if not self.has_attribute_data():
            return None
        return self.attributes[self.current_attribute_name]

    def current_scalar_range(self) -> tuple[float, float]:
        attribute = self.current_attribute()
        if attribute is None:
            return (0.0, 0.0)
        return tuple(float(v) for v in attribute.image.GetScalarRange())

    def current_attribute_display_range(self) -> tuple[float, float] | None:
        attribute = self.current_attribute()
        if attribute is None:
            return None
        return tuple(float(v) for v in attribute.lut.GetRange())

    def current_attribute_colormap_name(self) -> str | None:
        attribute = self.current_attribute()
        if attribute is None:
            return None
        return attribute.colormap_name

    def horizon_names(self) -> list[str]:
        return list(self.horizons.keys())

    def current_attribute_opacity(self) -> float:
        attribute = self.current_attribute()
        if attribute is None:
            return float(self.opacity)
        return float(attribute.opacity)

    def current_horizon_scalar_range(self) -> tuple[float, float] | None:
        return None

    def current_horizon_opacity(self) -> float | None:
        if self.current_horizon_name is None:
            return None
        return float(self.horizons[self.current_horizon_name].opacity)

    def current_horizon(self) -> HorizonSurface | None:
        if self.current_horizon_name is None:
            return None
        return self.horizons[self.current_horizon_name]

    def set_index(self, orientation: str, index: int, render: bool = True) -> None:
        if not self.has_attribute_data():
            return
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
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_attribute_display_range(self, min_value: float, max_value: float, render: bool = True) -> None:
        if not self.has_attribute_data():
            return
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        attr = self.current_attribute()
        if attr is None:
            return
        attr.lut.SetRange(float(min_value), float(max_value))
        attr.lut.Build()
        for bundle in self.bundles.values():
            bundle.mapper.Update()
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def _set_attribute_colormap(self, attribute: AttributeVolume, colormap_name: str) -> None:
        current_range = tuple(float(v) for v in attribute.lut.GetRange())
        attribute.colormap_name = (
            colormap_name if colormap_name in available_colormap_names() else DEFAULT_COLORMAP_NAME
        )
        apply_colormap_preset(attribute.lut, attribute.colormap_name)
        attribute.lut.SetRange(*current_range)
        attribute.lut.Build()
        attribute.volume_data = attribute.volume_data.with_data(
            attribute.volume_data.data,
            metadata={**attribute.volume_data.metadata, "colormap_name": attribute.colormap_name},
        )

    def set_attribute_colormap(self, colormap_name: str, render: bool = True) -> None:
        attribute = self.current_attribute()
        if attribute is None:
            return
        self._set_attribute_colormap(attribute, colormap_name)
        for bundle in self.bundles.values():
            bundle.mapper.Update()
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_attribute_opacity(self, opacity: float, render: bool = True) -> None:
        if not self.has_attribute_data():
            return
        opacity = max(0.0, min(1.0, float(opacity)))
        attr = self.current_attribute()
        if attr is None:
            return
        attr.opacity = opacity
        for bundle in self.bundles.values():
            bundle.actor.SetOpacity(opacity)
        if render:
            self.interactor.GetRenderWindow().Render()

    def add_attribute_volume(
        self,
        volume_data: VolumeData,
        *,
        name: str | None = None,
        opacity: float | None = None,
        select: bool = False,
    ) -> str:
        base_name = volume_data.name if name is None else name
        new_name = self._unique_name(self.attributes, base_name)
        current_attribute = self.current_attribute()
        source_opacity = (
            current_attribute.opacity
            if current_attribute is not None and opacity is None
            else float(self.opacity if opacity is None else opacity)
        )
        self.attributes[new_name] = load_attribute_from_volume(
            volume_data,
            name=new_name,
            spacing=self.spacing,
            clip_percentile=self.clip_percentile,
            opacity=source_opacity,
        )
        if select:
            self.set_attribute(new_name, render=False)
        self.refresh_scalar_bar()
        return new_name

    def extract_range_attribute(self, min_value: float, max_value: float) -> str:
        source = self.current_attribute()
        if source is None:
            raise ValueError("No current attribute.")
        new_name = self._unique_name(
            self.attributes,
            f"{source.name}_range_{format_value(min_value)}_{format_value(max_value)}",
        )

        output_volume = extract_range_volume(
            source.volume_data,
            min_value=min_value,
            max_value=max_value,
            name=new_name,
        )
        return self.add_attribute_volume(output_volume, name=new_name, opacity=source.opacity)

    def extract_envelope_horizons(self, min_voxels: int = 1) -> list[str]:
        source = self.current_attribute()
        if source is None:
            return []
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
            base_name = f"{source.name}_component_{component.index}_horizon"
            try:
                new_name = self.add_horizon(
                    base_name,
                    component_mask=np.asarray(component.mask, dtype=bool),
                    xlines=np.array(source.volume_data.xlines, copy=True),
                    inlines=np.array(source.volume_data.inlines, copy=True),
                    samples=np.array(source.volume_data.samples, copy=True),
                    scalar_values=np.asarray(source.volume_data.data, dtype=np.float32),
                    source_attribute_name=source.name,
                    component_index=component.index,
                    voxel_count=component.voxel_count,
                    opacity=0.55,
                    color=color,
                    visible=True,
                    select=False,
                )
            except ValueError:
                continue
            new_names.append(new_name)
        if new_names:
            self.set_current_horizon(new_names[0], render=False)
        return new_names

    def set_current_horizon(self, name: str | None, render: bool = True) -> None:
        self.current_horizon_name = name
        for horizon_name, horizon in self.horizons.items():
            prop = horizon.actor.GetProperty()
            prop.SetColor(*horizon.color)
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
            point_set = horizon.control_point_set
            if point_set is not None:
                is_current = horizon_name == name
                point_set.actor.GetProperty().SetOpacity((0.96 if is_current else 0.68) if point_set.visible else 0.0)
                if not point_set.use_attribute_colormap:
                    point_set.actor.GetProperty().SetColor(*((1.0, 0.90, 0.30) if is_current else (0.94, 0.74, 0.28)))
                point_set.master_actor.GetProperty().SetOpacity((1.0 if is_current else 0.88) if point_set.visible else 0.0)
                point_set.master_actor.GetProperty().SetColor(*((1.0, 0.24, 0.16) if is_current else (0.96, 0.40, 0.26)))
                point_set.linked_master_actor.GetProperty().SetOpacity((0.98 if is_current else 0.84) if point_set.visible else 0.0)
                point_set.actor.SetVisibility(point_set.visible)
                point_set.master_actor.SetVisibility(point_set.visible)
                point_set.linked_master_actor.SetVisibility(False)
                point_set.selected_master_actor.SetVisibility(False)
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_horizon_visibility(self, name: str, visible: bool, render: bool = True) -> None:
        horizon = self.horizons[name]
        horizon.visible = bool(visible)
        horizon.actor.SetVisibility(horizon.visible)
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()

    def extract_control_points_for_current_horizon(
        self,
        **intervals: int,
    ) -> str | None:
        horizon = self.current_horizon()
        if (
            horizon is None
            or horizon.component_mask is None
            or horizon.xlines is None
            or horizon.inlines is None
            or horizon.samples is None
        ):
            return None
        points = extract_control_points(
            horizon.xlines,
            horizon.inlines,
            horizon.samples,
            horizon.component_mask,
            **intervals,
        )
        if not points:
            return None
        horizon_name = self.set_control_points_for_horizon(
            horizon.name,
            points=points,
            source_attribute_name="",
            xlines=np.array(horizon.xlines, copy=True),
            inlines=np.array(horizon.inlines, copy=True),
            samples=np.array(horizon.samples, copy=True),
            value_attribute_name=None,
            use_attribute_colormap=False,
            source_horizon_name=horizon.name,
            original_horizon_mask=np.array(horizon.component_mask, copy=True),
            display_scale=1.0,
            visible=True,
        )
        return horizon_name

    def current_control_point_set(self) -> ControlPointSet | None:
        horizon = self.current_horizon()
        if horizon is None:
            return None
        return horizon.control_point_set

    def edit_current_control_point_set_master(self, master_index: int, delta_sample: float) -> bool:
        return self.edit_current_control_point_set_masters(
            [MasterMove(master_index=int(master_index), delta_sample=float(delta_sample))]
        )

    def edit_current_control_point_set_masters(self, moves: list[MasterMove]) -> bool:
        point_set = self.current_control_point_set()
        if point_set is None:
            return False
        point_set.points = apply_master_point_z_moves(
            point_set.points,
            moves,
            point_set.samples,
            value_volume_data=self._control_point_value_volume_data(point_set),
        )
        self._refresh_control_point_set_actor(point_set)
        self.set_current_horizon(self.current_horizon_name, render=False)
        return True

    def _control_point_value_volume_data(self, point_set: ControlPointSet) -> VolumeData | None:
        if point_set.value_attribute_name is None:
            return None
        attribute = self.attributes.get(point_set.value_attribute_name)
        if attribute is None:
            return None
        return attribute.volume_data

    def _control_point_value_lut(self, point_set: ControlPointSet) -> vtk.vtkLookupTable | None:
        if not point_set.use_attribute_colormap or point_set.value_attribute_name is None:
            return None
        attribute = self.attributes.get(point_set.value_attribute_name)
        if attribute is None:
            return None
        return attribute.lut

    def _refresh_control_point_values(self, point_set: ControlPointSet) -> None:
        value_volume_data = self._control_point_value_volume_data(point_set)
        refreshed_points: list[ControlPoint] = []
        for point in point_set.points:
            value = (
                0.0
                if value_volume_data is None
                else float(
                    value_volume_data.data[
                        int(point.xline_index),
                        int(point.inline_index),
                        int(point.sample_index),
                    ]
                )
            )
            refreshed_points.append(
                ControlPoint(
                    xline_index=int(point.xline_index),
                    inline_index=int(point.inline_index),
                    sample_index=int(point.sample_index),
                    xline=float(point.xline),
                    inline=float(point.inline),
                    sample=float(point.sample),
                    value=float(value),
                    kind=point.kind,
                    base_sample_index=point.base_sample_index,
                    master_index=point.master_index,
                    dz=float(point.dz),
                )
            )
        point_set.points = refreshed_points

    def _refresh_control_point_set_actor(self, point_set: ControlPointSet) -> None:
        (
            actor,
            polydata,
            sphere_source,
            master_actor,
            master_polydata,
            master_sphere_source,
            linked_master_actor,
            linked_master_polydata,
            linked_master_sphere_source,
            selected_master_actor,
            selected_master_polydata,
            selected_master_sphere_source,
        ) = create_control_point_actor(
            point_set.points,
            self.spacing,
            display_scale=point_set.display_scale,
            value_lut=self._control_point_value_lut(point_set),
            use_attribute_colormap=point_set.use_attribute_colormap,
        )
        point_set.actor.SetMapper(actor.GetMapper())
        point_set.master_actor.SetMapper(master_actor.GetMapper())
        point_set.linked_master_actor.SetMapper(linked_master_actor.GetMapper())
        point_set.selected_master_actor.SetMapper(selected_master_actor.GetMapper())
        point_set.sphere_source = sphere_source
        point_set.polydata = polydata
        point_set.master_sphere_source = master_sphere_source
        point_set.master_polydata = master_polydata
        point_set.linked_master_sphere_source = linked_master_sphere_source
        point_set.linked_master_polydata = linked_master_polydata
        point_set.selected_master_sphere_source = selected_master_sphere_source
        point_set.selected_master_polydata = selected_master_polydata
        self.set_control_point_display_scale(point_set.display_scale, render=False)

    def update_current_horizon_from_control_points(self) -> bool:
        self.last_rebuild_error = None
        current_horizon = self.current_horizon()
        point_set = self.current_control_point_set()
        if current_horizon is None or point_set is None:
            self.last_rebuild_error = "No current horizon or control-point set is available."
            return False
        return self._apply_control_point_deformation_to_horizon(current_horizon.name, point_set)

    def _apply_control_point_deformation_to_horizon(self, horizon_name: str, point_set: ControlPointSet) -> bool:
        horizon = self.horizons.get(horizon_name)
        if horizon is None:
            return False
        master_points = point_set.master_points
        if len(master_points) < 4:
            self.last_rebuild_error = "At least 4 master points are required to rebuild a horizon."
            return False
        base_polydata = horizon.base_polydata or horizon.polydata
        try:
            actor, polydata, mapper, lut, scalar_range = create_horizon_surface_from_control_points(
                base_polydata,
                master_points,
                self.spacing,
                self.clip_percentile,
                smoothing=point_set.rebuild_smoothness,
            )
        except ValueError as exc:
            self.last_rebuild_error = str(exc)
            return False

        horizon.actor.SetMapper(actor.GetMapper())
        horizon.polydata = polydata
        horizon.mapper = mapper
        horizon.lut = lut
        horizon.scalar_range = scalar_range
        horizon.actor.GetProperty().SetColor(*horizon.color)
        horizon.actor.SetVisibility(horizon.visible)
        return True

    def add_horizon(
        self,
        name: str,
        *,
        component_mask: np.ndarray,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        scalar_values: np.ndarray | None = None,
        source_attribute_name: str = "",
        component_index: int = 0,
        voxel_count: int | None = None,
        opacity: float = 0.55,
        color: tuple[float, float, float] = (0.82, 0.95, 1.0),
        visible: bool = True,
        select: bool = False,
    ) -> str:
        mask_array = np.asarray(component_mask, dtype=bool)
        actor, polydata, mapper, lut, scalar_range = create_horizon_surface_actor(
            mask_array,
            np.zeros(mask_array.shape, dtype=np.float32) if scalar_values is None else np.asarray(scalar_values, dtype=np.float32),
            self.spacing,
            self.clip_percentile,
        )
        new_name = self._unique_name(self.horizons, name)
        horizon = HorizonSurface(
            name=new_name,
            actor=actor,
            mapper=mapper,
            polydata=polydata,
            lut=lut,
            component_index=int(component_index),
            voxel_count=int(np.count_nonzero(mask_array) if voxel_count is None else voxel_count),
            scalar_range=scalar_range,
            color=tuple(float(v) for v in color),
            opacity=float(opacity),
            visible=bool(visible),
            component_mask=np.array(mask_array, copy=True),
            source_attribute_name=source_attribute_name,
            xlines=np.array(xlines, copy=True),
            inlines=np.array(inlines, copy=True),
            samples=np.array(samples, copy=True),
            control_point_set=None,
            base_polydata=clone_polydata(polydata),
        )
        self.horizons[new_name] = horizon
        self.renderer.AddActor(actor)
        self.set_current_horizon(new_name if select else self.current_horizon_name, render=False)
        return new_name

    def _build_control_point_set(
        self,
        name: str,
        *,
        points: list[ControlPoint],
        horizon_name: str,
        source_attribute_name: str,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        value_attribute_name: str | None,
        use_attribute_colormap: bool,
        source_horizon_name: str,
        original_horizon_mask: np.ndarray,
        display_scale: float = 1.0,
        link_radius: float | None = None,
        visible: bool = True,
    ) -> ControlPointSet:
        (
            actor,
            polydata,
            sphere_source,
            master_actor,
            master_polydata,
            master_sphere_source,
            linked_master_actor,
            linked_master_polydata,
            linked_master_sphere_source,
            selected_master_actor,
            selected_master_polydata,
            selected_master_sphere_source,
        ) = create_control_point_actor(
            points,
            self.spacing,
            display_scale=display_scale,
            value_lut=None
            if not use_attribute_colormap or value_attribute_name is None or value_attribute_name not in self.attributes
            else self.attributes[value_attribute_name].lut,
            use_attribute_colormap=use_attribute_colormap,
        )
        return ControlPointSet(
            name=name,
            actor=actor,
            sphere_source=sphere_source,
            polydata=polydata,
            master_actor=master_actor,
            master_sphere_source=master_sphere_source,
            master_polydata=master_polydata,
            linked_master_actor=linked_master_actor,
            linked_master_sphere_source=linked_master_sphere_source,
            linked_master_polydata=linked_master_polydata,
            selected_master_actor=selected_master_actor,
            selected_master_sphere_source=selected_master_sphere_source,
            selected_master_polydata=selected_master_polydata,
            points=list(points),
            horizon_name=horizon_name,
            source_attribute_name=source_attribute_name,
            xlines=np.array(xlines, copy=True),
            inlines=np.array(inlines, copy=True),
            samples=np.array(samples, copy=True),
            value_attribute_name=value_attribute_name,
            use_attribute_colormap=bool(use_attribute_colormap),
            source_horizon_name=source_horizon_name,
            original_horizon_mask=np.array(original_horizon_mask, copy=True),
            display_scale=float(display_scale),
            link_radius=float(
                (8.0 * min(self.spacing.xline, self.spacing.inline)) if link_radius is None else link_radius
            ),
            rebuild_smoothness=0.55,
            visible=bool(visible),
        )

    def _add_control_point_actors(self, point_set: ControlPointSet | None) -> None:
        if point_set is None:
            return
        self.renderer.AddActor(point_set.actor)
        self.renderer.AddActor(point_set.master_actor)
        self.renderer.AddActor(point_set.linked_master_actor)
        self.renderer.AddActor(point_set.selected_master_actor)

    def _remove_control_point_actors(self, point_set: ControlPointSet | None) -> None:
        if point_set is None:
            return
        self.renderer.RemoveActor(point_set.actor)
        self.renderer.RemoveActor(point_set.master_actor)
        self.renderer.RemoveActor(point_set.linked_master_actor)
        self.renderer.RemoveActor(point_set.selected_master_actor)

    def set_control_points_for_horizon(
        self,
        horizon_name: str,
        *,
        points: list[ControlPoint],
        source_attribute_name: str,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        value_attribute_name: str | None,
        use_attribute_colormap: bool,
        source_horizon_name: str,
        original_horizon_mask: np.ndarray,
        display_scale: float = 1.0,
        link_radius: float | None = None,
        visible: bool = True,
    ) -> str:
        horizon = self.horizons[horizon_name]
        existing_smoothness = 0.55 if horizon.control_point_set is None else horizon.control_point_set.rebuild_smoothness
        self._remove_control_point_actors(horizon.control_point_set)
        horizon.control_point_set = self._build_control_point_set(
            name=f"{horizon.name}_control_points",
            points=points,
            horizon_name=horizon.name,
            source_attribute_name=source_attribute_name,
            xlines=xlines,
            inlines=inlines,
            samples=samples,
            value_attribute_name=value_attribute_name,
            use_attribute_colormap=use_attribute_colormap,
            source_horizon_name=source_horizon_name,
            original_horizon_mask=original_horizon_mask,
            display_scale=display_scale,
            link_radius=link_radius,
            visible=visible,
        )
        horizon.control_point_set.rebuild_smoothness = existing_smoothness
        self._add_control_point_actors(horizon.control_point_set)
        self.set_current_horizon(horizon_name, render=False)
        return horizon_name

    def remove_attribute(self, name: str) -> bool:
        if name not in self.attributes or len(self.attributes) <= 1:
            return False
        was_current = self.current_attribute_name == name
        del self.attributes[name]
        if was_current:
            next_name = next(iter(self.attributes), None)
            if next_name is None:
                self.current_attribute_name = None
                self._sync_axes_from_current_attribute()
                self.update_overlay()
            else:
                self.set_attribute(next_name, render=False)
        return True

    def remove_horizon(self, name: str) -> bool:
        horizon = self.horizons.pop(name, None)
        if horizon is None:
            return False
        self._remove_control_point_actors(horizon.control_point_set)
        self.renderer.RemoveActor(horizon.actor)
        if self.current_horizon_name == name:
            next_name = next(iter(self.horizons), None)
            self.set_current_horizon(next_name, render=False)
        return True

    def current_control_point_display_scale(self) -> float | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        return float(point_set.display_scale)

    def current_control_point_link_radius(self) -> float | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        return float(point_set.link_radius)

    def current_control_point_rebuild_smoothness(self) -> float | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        return float(point_set.rebuild_smoothness)

    def current_control_point_value_attribute_name(self) -> str | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        return point_set.value_attribute_name

    def current_control_point_colormap_name(self) -> str | None:
        point_set = self.current_control_point_set()
        if point_set is None or point_set.value_attribute_name is None:
            return None
        attribute = self.attributes.get(point_set.value_attribute_name)
        if attribute is None:
            return None
        return attribute.colormap_name

    def current_control_point_use_attribute_colormap(self) -> bool:
        point_set = self.current_control_point_set()
        if point_set is None:
            return False
        return bool(point_set.use_attribute_colormap)

    def current_control_point_colormap_range(self) -> tuple[float, float] | None:
        point_set = self.current_control_point_set()
        if point_set is None or point_set.value_attribute_name is None:
            return None
        attribute = self.attributes.get(point_set.value_attribute_name)
        if attribute is None:
            return None
        return tuple(float(v) for v in attribute.lut.GetRange())

    def set_control_point_display_scale(self, scale: float, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        point_set.display_scale = max(0.4, min(8.0, float(scale)))
        base_radius = max(
            min(self.spacing.xline, self.spacing.inline, self.spacing.sample)
            * 0.28
            * point_set.display_scale,
            2.0,
        )
        master_radius = max(
            min(self.spacing.xline, self.spacing.inline, self.spacing.sample)
            * 0.44
            * point_set.display_scale,
            3.0,
        )
        point_set.sphere_source.SetRadius(base_radius)
        point_set.sphere_source.Update()
        point_set.master_sphere_source.SetRadius(master_radius)
        point_set.master_sphere_source.Update()
        point_set.linked_master_sphere_source.SetRadius(master_radius * 1.15)
        point_set.linked_master_sphere_source.Update()
        point_set.selected_master_sphere_source.SetRadius(master_radius * 1.3)
        point_set.selected_master_sphere_source.Update()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_link_radius(self, radius: float, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        min_spacing = min(self.spacing.xline, self.spacing.inline)
        point_set.link_radius = max(min_spacing, float(radius))
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_rebuild_smoothness(self, smoothness: float, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        point_set.rebuild_smoothness = max(0.0, min(1.0, float(smoothness)))
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_value_attribute(self, value_attribute_name: str | None, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        point_set.value_attribute_name = value_attribute_name
        self._refresh_control_point_values(point_set)
        self._refresh_control_point_set_actor(point_set)
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()

    def copy_attribute_values_to_control_points(
        self,
        horizon_name: str,
        attribute_name: str,
        render: bool = True,
    ) -> bool:
        horizon = self.horizons.get(horizon_name)
        if horizon is None or horizon.control_point_set is None:
            return False
        if attribute_name not in self.attributes:
            return False
        point_set = horizon.control_point_set
        point_set.value_attribute_name = attribute_name
        self._refresh_control_point_values(point_set)
        self._refresh_control_point_set_actor(point_set)
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()
        return True

    def set_control_point_use_attribute_colormap(self, enabled: bool, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        point_set.use_attribute_colormap = bool(enabled)
        self._refresh_control_point_set_actor(point_set)
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_colormap(self, colormap_name: str, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None or point_set.value_attribute_name is None:
            return
        attribute = self.attributes.get(point_set.value_attribute_name)
        if attribute is None:
            return
        self._set_attribute_colormap(attribute, colormap_name)
        self._refresh_control_point_set_actor(point_set)
        if self.current_attribute_name == attribute.name:
            for bundle in self.bundles.values():
                bundle.mapper.Update()
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_colormap_range(self, min_value: float, max_value: float, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None or point_set.value_attribute_name is None:
            return
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        attribute = self.attributes.get(point_set.value_attribute_name)
        if attribute is None:
            return
        attribute.lut.SetRange(float(min_value), float(max_value))
        attribute.lut.Build()
        if self.current_attribute_name == point_set.value_attribute_name:
            for bundle in self.bundles.values():
                bundle.mapper.Update()
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_horizon_display_range(self, min_value: float, max_value: float, render: bool = True) -> None:
        return

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

    def refresh_scalar_bar(self) -> None:
        lut: vtk.vtkLookupTable | None = None
        title = ""
        point_set = self.current_control_point_set()
        if (
            point_set is not None
            and point_set.use_attribute_colormap
            and point_set.value_attribute_name is not None
            and point_set.value_attribute_name in self.attributes
        ):
            attribute = self.attributes[point_set.value_attribute_name]
            lut = attribute.lut
            title = f"Control Points\n{attribute.name}"
        else:
            attribute = self.current_attribute()
            if attribute is not None:
                lut = attribute.lut
                title = f"Attribute\n{attribute.name}"
        if lut is None:
            self.scalar_bar_actor.SetVisibility(False)
            return
        self.scalar_bar_actor.SetLookupTable(lut)
        self.scalar_bar_actor.SetTitle(title)
        self.scalar_bar_actor.SetVisibility(True)
        self.scalar_bar_actor.Modified()


class SegyViewerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        updater: SliceUpdater,
        vtk_widget: QVTKRenderWindowInteractor,
        render_window: vtk.vtkRenderWindow,
        renderer: vtk.vtkRenderer,
        outline_actor: vtk.vtkActor,
        axis_texts: list[vtk.vtkBillboardTextActor3D],
        debug_ui: bool,
    ) -> None:
        super().__init__()
        self.updater = updater
        self.vtk_widget = vtk_widget
        self.render_window = render_window
        self.renderer = renderer
        self.image = updater.image
        self.outline_actor = outline_actor
        self.axis_texts = axis_texts
        self.debug_ui = debug_ui
        self._vtk_initialized = False
        self._render_pending = False
        self._selected_master_point_index: int | None = None
        self._linked_master_point_indices: set[int] = set()
        self._prop_picker = vtk.vtkPropPicker()
        self.data_panel = DataPanelWidget(self)
        self._selected_data_item: tuple[str, str] | None = None

        self.setWindowTitle(APP_NAME)
        self.resize(1920, 1400)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.data_panel.setMinimumWidth(300)
        self.data_panel.setMaximumWidth(380)
        layout.addWidget(self.data_panel, stretch=0)

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
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        self.load_seismic_button = QtWidgets.QPushButton("Load Seismic Data")
        self.load_seismic_button.clicked.connect(self.open_load_seismic_dialog)
        panel_layout.addWidget(self.load_seismic_button)

        self.extract_button = QtWidgets.QPushButton("Open Range Extraction")
        self.extract_button.clicked.connect(self.open_extract_range_dialog)
        panel_layout.addWidget(self.extract_button)

        self.extract_envelope_button = QtWidgets.QPushButton("Open Horizon Extraction")
        self.extract_envelope_button.clicked.connect(self.open_extract_horizon_dialog)
        panel_layout.addWidget(self.extract_envelope_button)

        self.extract_control_points_button = QtWidgets.QPushButton("Extract Control Point")
        self.extract_control_points_button.clicked.connect(self.open_extract_control_points_dialog)
        panel_layout.addWidget(self.extract_control_points_button)

        attribute_display_group = ColorMapControlWidget("Attribute Colormap")
        self.attribute_colormap_widget = attribute_display_group
        self.attribute_colormap_widget.apply_button.clicked.connect(self.apply_attribute_display)
        self.attribute_colormap_widget.preset_combo.currentTextChanged.connect(self.change_attribute_colormap)
        attribute_display_layout = attribute_display_group.layout()

        attribute_opacity_row = QtWidgets.QHBoxLayout()
        attribute_opacity_row.setSpacing(6)
        attribute_opacity_row.addWidget(QtWidgets.QLabel("Opacity"))
        self.attribute_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.attribute_opacity_slider.setRange(0, 100)
        self.attribute_opacity_slider.valueChanged.connect(self.change_attribute_opacity)
        attribute_opacity_row.addWidget(self.attribute_opacity_slider, stretch=1)
        attribute_display_layout.addLayout(attribute_opacity_row)
        panel_layout.addWidget(attribute_display_group)

        horizon_display_group = QtWidgets.QGroupBox("Horizon Display")
        horizon_display_layout = QtWidgets.QVBoxLayout(horizon_display_group)
        horizon_display_layout.setContentsMargins(8, 6, 8, 6)
        horizon_display_layout.setSpacing(6)
        self.horizon_color_label = QtWidgets.QLabel("Fixed color rendering")
        horizon_display_layout.addWidget(self.horizon_color_label)

        horizon_opacity_row = QtWidgets.QHBoxLayout()
        horizon_opacity_row.setSpacing(6)
        horizon_opacity_row.addWidget(QtWidgets.QLabel("Opacity"))
        self.horizon_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.horizon_opacity_slider.setRange(0, 100)
        self.horizon_opacity_slider.valueChanged.connect(self.change_horizon_opacity)
        horizon_opacity_row.addWidget(self.horizon_opacity_slider, stretch=1)
        horizon_display_layout.addLayout(horizon_opacity_row)
        panel_layout.addWidget(horizon_display_group)

        self.xline_control = AxisControl("Crossline", np.asarray([0.0], dtype=np.float32))
        self.inline_control = AxisControl("Inline", np.asarray([0.0], dtype=np.float32))
        self.sample_control = AxisControl("Sample", np.asarray([0.0], dtype=np.float32))
        panel_layout.addWidget(self.xline_control)
        panel_layout.addWidget(self.inline_control)
        panel_layout.addWidget(self.sample_control)

        control_point_tools_group = QtWidgets.QGroupBox("Control Point Tools")
        control_point_tools_layout = QtWidgets.QVBoxLayout(control_point_tools_group)
        control_point_tools_layout.setContentsMargins(8, 6, 8, 6)
        control_point_tools_layout.setSpacing(6)
        control_point_tools_row = QtWidgets.QHBoxLayout()
        control_point_tools_row.setSpacing(6)
        self.selected_master_label = QtWidgets.QLabel("Master: none")
        self.selected_master_label.setMinimumWidth(86)
        control_point_tools_row.addWidget(self.selected_master_label)
        self.edit_master_point_button = QtWidgets.QPushButton("Edit")
        self.edit_master_point_button.clicked.connect(self.open_edit_master_point_dialog)
        control_point_tools_row.addWidget(self.edit_master_point_button)
        self.update_horizon_button = QtWidgets.QPushButton("Update")
        self.update_horizon_button.clicked.connect(self.update_horizon_from_control_points)
        control_point_tools_row.addWidget(self.update_horizon_button)
        self.move_master_up_button = QtWidgets.QPushButton("Z+")
        self.move_master_up_button.clicked.connect(lambda: self.move_selected_master_point(1.0))
        control_point_tools_row.addWidget(self.move_master_up_button)
        self.move_master_down_button = QtWidgets.QPushButton("Z-")
        self.move_master_down_button.clicked.connect(lambda: self.move_selected_master_point(-1.0))
        control_point_tools_row.addWidget(self.move_master_down_button)
        control_point_tools_layout.addLayout(control_point_tools_row)

        control_point_size_row = QtWidgets.QHBoxLayout()
        control_point_size_row.setSpacing(6)
        control_point_size_row.addWidget(QtWidgets.QLabel("Size"))
        self.control_point_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.control_point_size_slider.setRange(40, 800)
        self.control_point_size_slider.valueChanged.connect(self.change_control_point_size)
        control_point_size_row.addWidget(self.control_point_size_slider, stretch=1)
        control_point_tools_layout.addLayout(control_point_size_row)

        control_point_link_row = QtWidgets.QHBoxLayout()
        control_point_link_row.setSpacing(6)
        control_point_link_row.addWidget(QtWidgets.QLabel("Link Radius"))
        self.control_point_link_radius_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.control_point_link_radius_slider.setRange(1, 120)
        self.control_point_link_radius_slider.valueChanged.connect(self.change_control_point_link_radius)
        control_point_link_row.addWidget(self.control_point_link_radius_slider, stretch=1)
        control_point_tools_layout.addLayout(control_point_link_row)

        control_point_value_row = QtWidgets.QHBoxLayout()
        control_point_value_row.setSpacing(6)
        self.copy_control_point_values_button = QtWidgets.QPushButton("Copy Values")
        self.copy_control_point_values_button.clicked.connect(self.open_copy_control_point_values_dialog)
        control_point_value_row.addWidget(self.copy_control_point_values_button)
        self.control_point_value_attribute_label = QtWidgets.QLabel("Value Attr: none")
        control_point_value_row.addWidget(self.control_point_value_attribute_label, stretch=1)
        control_point_tools_layout.addLayout(control_point_value_row)

        self.control_point_use_colormap_checkbox = QtWidgets.QCheckBox("Use Attribute Colormap")
        self.control_point_use_colormap_checkbox.toggled.connect(self.toggle_control_point_colormap)
        control_point_tools_layout.addWidget(self.control_point_use_colormap_checkbox)

        self.control_point_colormap_widget = ColorMapControlWidget("Control Point Colormap")
        self.control_point_colormap_widget.apply_button.clicked.connect(self.apply_control_point_colormap_range)
        self.control_point_colormap_widget.preset_combo.currentTextChanged.connect(self.change_control_point_colormap)
        control_point_tools_layout.addWidget(self.control_point_colormap_widget)

        control_point_smooth_row = QtWidgets.QHBoxLayout()
        control_point_smooth_row.setSpacing(6)
        control_point_smooth_row.addWidget(QtWidgets.QLabel("Smooth"))
        self.control_point_smoothness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.control_point_smoothness_slider.setRange(0, 100)
        self.control_point_smoothness_slider.valueChanged.connect(self.change_control_point_smoothness)
        control_point_smooth_row.addWidget(self.control_point_smoothness_slider, stretch=1)
        control_point_tools_layout.addLayout(control_point_smooth_row)
        panel_layout.addWidget(control_point_tools_group)

        panel_layout.addStretch(1)

        self.reset_view_button = QtWidgets.QPushButton("Reset")
        self.reset_view_button.clicked.connect(self.reset_view)
        panel_layout.addWidget(self.reset_view_button)

        self.xline_control.value_changed.connect(lambda index: self._set_index("xline", index))
        self.inline_control.value_changed.connect(lambda index: self._set_index("inline", index))
        self.sample_control.value_changed.connect(lambda index: self._set_index("sample", index))
        self.data_panel.item_activated.connect(self.activate_data_item)
        self.data_panel.category_load_requested.connect(self.load_data_for_category)
        self.data_panel.item_store_requested.connect(self.store_data_item)
        self.data_panel.item_unload_requested.connect(self.unload_data_item)

        self.refresh_axis_controls()
        self.refresh_data_panel()
        self.refresh_info()
        self.refresh_display_controls()
        self.vtk_widget.installEventFilter(self)

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

    def _is_shift_pressed(self) -> bool:
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        return bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

    def _is_ctrl_pressed(self) -> bool:
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        return bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.vtk_widget:
            if (
                event.type() == QtCore.QEvent.Type.MouseButtonPress
                and isinstance(event, QtGui.QMouseEvent)
                and event.button() == QtCore.Qt.MouseButton.LeftButton
            ):
                if self.pick_master_control_point(
                    int(event.position().x()),
                    int(event.position().y()),
                    toggle_linked=self._is_ctrl_pressed(),
                ):
                    self.schedule_render()
                    return True
            if (
                event.type() == QtCore.QEvent.Type.Wheel
                and isinstance(event, QtGui.QWheelEvent)
                and self._selected_master_point_index is not None
                and self._is_shift_pressed()
            ):
                delta = event.angleDelta().y()
                if delta > 0:
                    self.move_selected_master_point(1.0)
                    return True
                if delta < 0:
                    self.move_selected_master_point(-1.0)
                    return True
        return super().eventFilter(watched, event)

    def schedule_render(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        QtCore.QTimer.singleShot(16, self._flush_render)

    def _flush_render(self) -> None:
        self._render_pending = False
        self.render_window.Render()

    def reset_view(self) -> None:
        if not self.updater.has_attribute_data():
            return
        configure_default_camera(self.renderer, self.image)
        self.schedule_render()
        debug_log(self.debug_ui, "camera reset to default view")

    def _set_index(self, orientation: str, index: int) -> None:
        self.updater.set_index(orientation, index, render=False)
        self.refresh_info()
        self.schedule_render()

    def refresh_info(self) -> None:
        return

    def refresh_axis_controls(self) -> None:
        self.xline_control.set_values(self.updater.xlines, self.updater.indices["xline"])
        self.inline_control.set_values(self.updater.inlines, self.updater.indices["inline"])
        self.sample_control.set_values(self.updater.samples, self.updater.indices["sample"])

    def refresh_scene_guides(self) -> None:
        if self.outline_actor is not None:
            self.renderer.RemoveActor(self.outline_actor)
        for actor in self.axis_texts:
            self.renderer.RemoveActor(actor)
        self.outline_actor = create_outline(self.updater.image)
        self.axis_texts = create_axis_labels(
            self.updater.image,
            self.updater.xlines,
            self.updater.inlines,
            self.updater.samples,
            self.updater.spacing.xline,
            self.updater.spacing.inline,
            self.updater.spacing.sample,
        )
        self.renderer.AddActor(self.outline_actor)
        for actor in self.axis_texts:
            self.renderer.AddActor(actor)

    def refresh_display_controls(self) -> None:
        attribute_range = self.updater.current_attribute_display_range()
        has_attribute = attribute_range is not None
        self.attribute_colormap_widget.set_controls_enabled(has_attribute)
        self.attribute_opacity_slider.setEnabled(has_attribute)
        if has_attribute:
            self.attribute_colormap_widget.set_range(attribute_range)
            self.attribute_colormap_widget.set_current_preset(self.updater.current_attribute_colormap_name())
        else:
            self.attribute_colormap_widget.set_range(None)
        self.attribute_opacity_slider.blockSignals(True)
        self.attribute_opacity_slider.setValue(int(round(self.updater.current_attribute_opacity() * 100.0)))
        self.attribute_opacity_slider.blockSignals(False)

        horizon_opacity = self.updater.current_horizon_opacity()
        has_horizon = horizon_opacity is not None
        self.horizon_opacity_slider.setEnabled(has_horizon)
        if has_horizon:
            self.horizon_opacity_slider.blockSignals(True)
            self.horizon_opacity_slider.setValue(int(round(horizon_opacity * 100.0)))
            self.horizon_opacity_slider.blockSignals(False)
        else:
            self.horizon_opacity_slider.blockSignals(True)
            self.horizon_opacity_slider.setValue(0)
            self.horizon_opacity_slider.blockSignals(False)

        has_control_points = self.updater.current_control_point_set() is not None
        self.edit_master_point_button.setEnabled(has_control_points)
        self.update_horizon_button.setEnabled(has_control_points)
        self.control_point_size_slider.setEnabled(has_control_points)
        self.control_point_link_radius_slider.setEnabled(has_control_points)
        self.copy_control_point_values_button.setEnabled(bool(self.updater.attribute_names()) and any(
            horizon.control_point_set is not None for horizon in self.updater.horizons.values()
        ))
        has_value_attribute = self.updater.current_control_point_value_attribute_name() is not None
        self.control_point_use_colormap_checkbox.setEnabled(has_control_points and has_value_attribute)
        self.control_point_smoothness_slider.setEnabled(has_control_points)
        has_selected_master = has_control_points and self._selected_master_point_index is not None
        self.move_master_up_button.setEnabled(has_selected_master)
        self.move_master_down_button.setEnabled(has_selected_master)
        if has_selected_master:
            linked_count = len(self._linked_master_point_indices)
            self.selected_master_label.setText(f"Master: #{self._selected_master_point_index} (+{linked_count})")
        else:
            self.selected_master_label.setText("Master: none")
        control_point_scale = self.updater.current_control_point_display_scale()
        self.control_point_size_slider.blockSignals(True)
        self.control_point_size_slider.setValue(
            int(round((1.0 if control_point_scale is None else control_point_scale) * 100.0))
        )
        self.control_point_size_slider.blockSignals(False)
        link_radius = self.updater.current_control_point_link_radius()
        min_spacing = min(self.updater.spacing.xline, self.updater.spacing.inline)
        self.control_point_link_radius_slider.blockSignals(True)
        self.control_point_link_radius_slider.setValue(
            int(round((min_spacing if link_radius is None else link_radius) / max(min_spacing, 1e-6)))
        )
        self.control_point_link_radius_slider.blockSignals(False)
        selected_value_attribute = self.updater.current_control_point_value_attribute_name()
        self.control_point_value_attribute_label.setText(
            f"Value Attr: {selected_value_attribute or 'none'}"
        )
        self.control_point_use_colormap_checkbox.blockSignals(True)
        self.control_point_use_colormap_checkbox.setChecked(
            has_value_attribute and self.updater.current_control_point_use_attribute_colormap()
        )
        self.control_point_use_colormap_checkbox.blockSignals(False)
        colormap_range = self.updater.current_control_point_colormap_range()
        has_colormap_attribute = colormap_range is not None
        self.control_point_colormap_widget.set_controls_enabled(has_control_points and has_colormap_attribute)
        if colormap_range is not None:
            self.control_point_colormap_widget.set_range(colormap_range)
            self.control_point_colormap_widget.set_current_preset(self.updater.current_control_point_colormap_name())
        else:
            self.control_point_colormap_widget.set_range(None)
        smoothness = self.updater.current_control_point_rebuild_smoothness()
        self.control_point_smoothness_slider.blockSignals(True)
        self.control_point_smoothness_slider.setValue(
            int(round((0.55 if smoothness is None else smoothness) * 100.0))
        )
        self.control_point_smoothness_slider.blockSignals(False)

        self.extract_button.setEnabled(has_attribute)
        self.extract_envelope_button.setEnabled(has_attribute)
        self.reset_view_button.setEnabled(has_attribute)

    def refresh_data_panel(self) -> None:
        seismic_items: list[DataPanelItem] = []
        attribute_items: list[DataPanelItem] = []
        for name, attribute in self.updater.attributes.items():
            panel_category = attribute.volume_data.metadata.get("panel_category", "seismic" if name == "seismic" else "attribute")
            entry = DataPanelItem(category=str(panel_category), name=name, label=name)
            if panel_category == "seismic":
                seismic_items.append(entry)
            else:
                attribute_items.append(entry)
        items = {
            "seismic": seismic_items,
            "attribute": attribute_items,
            "horizon": [
                DataPanelItem(
                    category="horizon",
                    name=name,
                    label=(
                        f"{name} ({self.updater.horizons[name].voxel_count} voxels"
                        f", {0 if self.updater.horizons[name].control_point_set is None else len(self.updater.horizons[name].control_point_set.points)} pts)"
                    ),
                )
                for name in self.updater.horizon_names()
            ],
            "well": [],
        }
        self.data_panel.set_items(items)
        if self._selected_data_item is not None:
            selected_category, selected_name = self._selected_data_item
            self.data_panel.select_item(selected_category, selected_name)
        self.refresh_display_controls()

    def activate_data_item(self, category: str, name: str) -> None:
        self._selected_data_item = (category, name)
        if category in {"seismic", "attribute"}:
            self.updater.set_attribute(name, render=False)
            self.image = self.updater.image
            self.refresh_axis_controls()
            self.refresh_scene_guides()
        elif category == "horizon":
            self.updater.set_current_horizon(name, render=False)
            self._selected_master_point_index = None
            self._linked_master_point_indices.clear()
            self._refresh_selected_master_actor()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def pick_master_control_point(self, display_x: int, display_y: int, *, toggle_linked: bool = False) -> bool:
        point_set = self.updater.current_control_point_set()
        if point_set is None or not point_set.visible:
            self._selected_master_point_index = None
            self._linked_master_point_indices.clear()
            self._refresh_selected_master_actor()
            self.refresh_display_controls()
            return False

        vtk_display_x, vtk_display_y = self._qt_to_vtk_display(display_x, display_y)
        picked_actor = None
        if self._prop_picker.Pick(float(vtk_display_x), float(vtk_display_y), 0.0, self.renderer) != 0:
            picked_actor = self._prop_picker.GetActor()
            if picked_actor not in {point_set.master_actor, point_set.linked_master_actor, point_set.selected_master_actor}:
                picked_actor = None

        selected_master: ControlPoint | None = None
        best_distance2 = float("inf")
        for point in point_set.master_points:
            world_x = float(point.xline_index) * float(self.updater.spacing.xline)
            world_y = float(point.inline_index) * float(self.updater.spacing.inline)
            world_z = float(point.sample_index) * float(self.updater.spacing.sample)
            self.renderer.SetWorldPoint(world_x, world_y, world_z, 1.0)
            self.renderer.WorldToDisplay()
            display_point = self.renderer.GetDisplayPoint()
            distance2 = (float(display_point[0]) - float(vtk_display_x)) ** 2 + (
                float(display_point[1]) - float(vtk_display_y)
            ) ** 2
            if distance2 < best_distance2:
                best_distance2 = distance2
                selected_master = point

        threshold2 = 36.0 * 36.0 if picked_actor is not None else 22.0 * 22.0
        if selected_master is None or selected_master.master_index is None or best_distance2 > threshold2:
            return False

        selected_master_index = int(selected_master.master_index)
        if toggle_linked and self._selected_master_point_index is not None:
            if selected_master_index != self._selected_master_point_index:
                selected_main = point_set.master_point_by_index(self._selected_master_point_index)
                if selected_main is not None:
                    selected_column = (int(selected_main.xline_index), int(selected_main.inline_index))
                    target_column = (int(selected_master.xline_index), int(selected_master.inline_index))
                    if target_column != selected_column:
                        same_column_indices = {
                            int(point.master_index)
                            for point in point_set.master_points
                            if point.master_index is not None
                            and int(point.xline_index) == target_column[0]
                            and int(point.inline_index) == target_column[1]
                        }
                        if selected_master_index in self._linked_master_point_indices:
                            self._linked_master_point_indices.discard(selected_master_index)
                        else:
                            self._linked_master_point_indices.difference_update(same_column_indices)
                            self._linked_master_point_indices.add(selected_master_index)
        else:
            self._selected_master_point_index = selected_master_index
            self._refresh_linked_master_points()
        self._refresh_selected_master_actor()
        self.refresh_display_controls()
        return True

    def _refresh_linked_master_points(self) -> None:
        point_set = self.updater.current_control_point_set()
        if point_set is None or self._selected_master_point_index is None:
            self._linked_master_point_indices.clear()
            return
        selected_point = point_set.master_point_by_index(self._selected_master_point_index)
        if selected_point is None:
            self._linked_master_point_indices.clear()
            return

        radius = max(float(point_set.link_radius), min(self.updater.spacing.xline, self.updater.spacing.inline))
        linked_by_column: dict[tuple[int, int], tuple[float, int]] = {}
        selected_column = (int(selected_point.xline_index), int(selected_point.inline_index))
        selected_sample = float(selected_point.sample_index)
        selected_xy = np.asarray(
            [
                float(selected_point.xline_index) * float(self.updater.spacing.xline),
                float(selected_point.inline_index) * float(self.updater.spacing.inline),
            ],
            dtype=np.float64,
        )
        for point in point_set.master_points:
            if point.master_index is None or point.master_index == self._selected_master_point_index:
                continue
            point_column = (int(point.xline_index), int(point.inline_index))
            if point_column == selected_column:
                continue
            point_xy = np.asarray(
                [
                    float(point.xline_index) * float(self.updater.spacing.xline),
                    float(point.inline_index) * float(self.updater.spacing.inline),
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(point_xy - selected_xy))
            if distance <= radius:
                z_distance = abs(float(point.sample_index) - selected_sample)
                existing = linked_by_column.get(point_column)
                if existing is None or z_distance < existing[0]:
                    linked_by_column[point_column] = (z_distance, int(point.master_index))
        self._linked_master_point_indices = {master_index for _, master_index in linked_by_column.values()}

    def _qt_to_vtk_display(self, display_x: int, display_y: int) -> tuple[int, int]:
        dpr = float(self.vtk_widget.devicePixelRatioF())
        vtk_x = int(round(float(display_x) * dpr))
        _, render_height = self.render_window.GetSize()
        vtk_y = max(0, int(render_height) - 1 - int(round(float(display_y) * dpr)))
        return vtk_x, vtk_y

    def _refresh_selected_master_actor(self) -> None:
        point_set = self.updater.current_control_point_set()
        if point_set is None or self._selected_master_point_index is None:
            if point_set is not None:
                point_set.linked_master_actor.SetVisibility(False)
                point_set.selected_master_actor.SetVisibility(False)
            return

        linked_points = [
            point
            for point in point_set.master_points
            if point.master_index is not None and int(point.master_index) in self._linked_master_point_indices
        ]
        selected_point = point_set.master_point_by_index(self._selected_master_point_index)
        if selected_point is None:
            point_set.linked_master_actor.SetVisibility(False)
            point_set.selected_master_actor.SetVisibility(False)
            return

        linked_polydata = vtk.vtkPolyData()
        linked_vtk_points = vtk.vtkPoints()
        for linked_point in linked_points:
            linked_vtk_points.InsertNextPoint(
                float(linked_point.xline_index) * float(self.updater.spacing.xline),
                float(linked_point.inline_index) * float(self.updater.spacing.inline),
                float(linked_point.sample_index) * float(self.updater.spacing.sample),
            )
        linked_polydata.SetPoints(linked_vtk_points)
        point_set.linked_master_polydata = linked_polydata
        linked_mapper = vtk.vtkGlyph3DMapper()
        linked_mapper.SetInputData(linked_polydata)
        linked_mapper.SetSourceConnection(point_set.linked_master_sphere_source.GetOutputPort())
        linked_mapper.ScalingOff()
        linked_mapper.ScalarVisibilityOff()
        point_set.linked_master_actor.SetMapper(linked_mapper)
        point_set.linked_master_actor.SetVisibility(point_set.visible and len(linked_points) > 0)

        selected_polydata = vtk.vtkPolyData()
        selected_vtk_points = vtk.vtkPoints()
        selected_vtk_points.InsertNextPoint(
            float(selected_point.xline_index) * float(self.updater.spacing.xline),
            float(selected_point.inline_index) * float(self.updater.spacing.inline),
            float(selected_point.sample_index) * float(self.updater.spacing.sample),
        )
        selected_polydata.SetPoints(selected_vtk_points)
        point_set.selected_master_polydata = selected_polydata
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(selected_polydata)
        mapper.SetSourceConnection(point_set.selected_master_sphere_source.GetOutputPort())
        mapper.ScalingOff()
        mapper.ScalarVisibilityOff()
        point_set.selected_master_actor.SetMapper(mapper)
        point_set.selected_master_actor.SetVisibility(point_set.visible)

    def move_selected_master_point(self, delta_sample: float) -> None:
        if self._selected_master_point_index is None:
            return
        moves = self._linked_moves_for_delta(delta_sample)
        if not self.updater.edit_current_control_point_set_masters(moves):
            QtWidgets.QMessageBox.information(
                self,
                "Edit Failed",
                self.updater.last_rebuild_error or "Failed to move the selected master point.",
            )
            return
        self._refresh_selected_master_actor()
        self.refresh_data_panel()
        self.refresh_display_controls()
        self.schedule_render()

    def _linked_moves_for_delta(self, delta_sample: float) -> list[MasterMove]:
        point_set = self.updater.current_control_point_set()
        if point_set is None or self._selected_master_point_index is None:
            return []
        selected_point = point_set.master_point_by_index(self._selected_master_point_index)
        if selected_point is None:
            return []

        min_spacing = min(self.updater.spacing.xline, self.updater.spacing.inline)
        radius = max(float(point_set.link_radius), min_spacing)
        selected_xy = np.asarray(
            [
                float(selected_point.xline_index) * float(self.updater.spacing.xline),
                float(selected_point.inline_index) * float(self.updater.spacing.inline),
            ],
            dtype=np.float64,
        )

        moves = [MasterMove(master_index=int(self._selected_master_point_index), delta_sample=float(delta_sample))]
        for linked_index in sorted(self._linked_master_point_indices):
            linked_point = point_set.master_point_by_index(linked_index)
            if linked_point is None:
                continue
            linked_xy = np.asarray(
                [
                    float(linked_point.xline_index) * float(self.updater.spacing.xline),
                    float(linked_point.inline_index) * float(self.updater.spacing.inline),
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(linked_xy - selected_xy))
            weight = max(0.0, 1.0 - distance / radius)
            if weight <= 1e-6:
                continue
            moves.append(MasterMove(master_index=int(linked_index), delta_sample=float(delta_sample) * weight))
        return moves

    def open_load_seismic_dialog(self, target_category: str = "seismic") -> None:
        dialog = LoadSeismicDialog(self, target_category=target_category)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        self.load_segy_volume(values)

    def load_segy_volume(self, values: dict[str, object]) -> None:
        segy_path = Path(str(values["path"])).expanduser().resolve()
        if not segy_path.exists():
            QtWidgets.QMessageBox.information(self, "Missing File", f"SEG-Y file not found:\n{segy_path}")
            return

        geometry = load_segy_geometry(
            segy_path,
            inline_field=int(values["inline_field"]),
            xline_field=int(values["xline_field"]),
        )
        volume_data = read_segy_volume(
            segy_path=segy_path,
            geometry=geometry,
            interval_inline=int(values["interval_inline"]),
            interval_xline=int(values["interval_xline"]),
            interval_sample=int(values["interval_sample"]),
            inline_field=int(values["inline_field"]),
            xline_field=int(values["xline_field"]),
            name=str(values["name"]),
        )
        category = str(values["target_category"])
        volume_data = volume_data.with_data(
            volume_data.data,
            metadata={
                **volume_data.metadata,
                "panel_category": category,
            },
        )

        self.updater.spacing = RenderSpacing(
            xline=float(values["step_xline"]),
            inline=float(values["step_inline"]),
            sample=float(values["step_sample"]),
        )
        self.updater.segy_path = segy_path
        name = self.updater.add_attribute_volume(
            volume_data,
            name=str(values["name"]),
            opacity=self.updater.opacity,
            select=True,
        )
        self.image = self.updater.image
        self._selected_data_item = (category, name)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.updater.set_index("xline", len(self.updater.xlines) // 2, render=False)
        self.updater.set_index("inline", len(self.updater.inlines) // 2, render=False)
        self.updater.set_index("sample", len(self.updater.samples) // 2, render=False)
        self.refresh_axis_controls()
        configure_default_camera(self.renderer, self.updater.image)
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    @staticmethod
    def _volume_payload(volume_data: VolumeData, opacity: float | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": volume_data.name,
            "data": np.asarray(volume_data.data, dtype=np.float32),
            "xlines": np.asarray(volume_data.xlines),
            "inlines": np.asarray(volume_data.inlines),
            "samples": np.asarray(volume_data.samples),
            "metadata_json": json.dumps(volume_data.metadata),
        }
        if opacity is not None:
            payload["opacity"] = np.array([float(opacity)], dtype=np.float32)
        return payload

    @staticmethod
    def _volume_from_payload(payload: dict[str, object]) -> tuple[VolumeData, float | None]:
        metadata_text = str(SegyViewerWindow._payload_scalar(payload.get("metadata_json", "{}")))
        volume = VolumeData(
            data=np.asarray(payload["data"], dtype=np.float32),
            xlines=np.asarray(payload["xlines"]),
            inlines=np.asarray(payload["inlines"]),
            samples=np.asarray(payload["samples"]),
            name=str(SegyViewerWindow._payload_scalar(payload["name"])),
            metadata=json.loads(metadata_text),
        )
        opacity = None
        if "opacity" in payload:
            opacity_values = np.asarray(payload["opacity"]).ravel()
            if opacity_values.size:
                opacity = float(opacity_values[0])
        return volume, opacity

    @staticmethod
    def _load_npz_payload(path: str) -> dict[str, object]:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    @staticmethod
    def _payload_scalar(value: object) -> object:
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value

    def _default_output_dir(self, category: str) -> Path:
        path = DERIVED_DATA_DIR / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_data_for_category(self, category: str) -> None:
        if category == "seismic":
            self.open_load_seismic_dialog(target_category=category)
            return

        if category == "attribute":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Attribute",
                str(self._default_output_dir(category)),
                "Wesi3D Volume (*.npz)",
            )
            if not path:
                return
            payload = self._load_npz_payload(path)
            volume, opacity = self._volume_from_payload(payload)
            volume = volume.with_data(
                volume.data,
                metadata={**volume.metadata, "panel_category": "attribute"},
            )
            name = self.updater.add_attribute_volume(volume, name=volume.name, opacity=opacity, select=True)
            self.image = self.updater.image
            self._selected_data_item = ("attribute", name)
            self.refresh_axis_controls()
            self.refresh_scene_guides()
            configure_default_camera(self.renderer, self.updater.image)
            self.refresh_data_panel()
            self.schedule_render()
            return

        if category == "horizon":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Horizon",
                str(self._default_output_dir(category)),
                "Wesi3D Horizon (*.npz)",
            )
            if not path:
                return
            payload = self._load_npz_payload(path)
            if "xlines" in payload and "inlines" in payload and "samples" in payload:
                xlines = np.asarray(payload["xlines"])
                inlines = np.asarray(payload["inlines"])
                samples = np.asarray(payload["samples"])
                scalar_values = np.zeros(
                    (
                        len(xlines),
                        len(inlines),
                        len(samples),
                    ),
                    dtype=np.float32,
                )
            else:
                legacy_volume, _ = self._volume_from_payload(
                    {
                        "name": self._payload_scalar(payload["source_volume_name"]),
                        "data": payload["source_volume_data"],
                        "xlines": payload["source_volume_xlines"],
                        "inlines": payload["source_volume_inlines"],
                        "samples": payload["source_volume_samples"],
                        "metadata_json": self._payload_scalar(payload["source_volume_metadata_json"]),
                    }
                )
                xlines = np.asarray(legacy_volume.xlines)
                inlines = np.asarray(legacy_volume.inlines)
                samples = np.asarray(legacy_volume.samples)
                scalar_values = np.asarray(legacy_volume.data, dtype=np.float32)
            if "polydata_points" in payload and "polydata_polys_offsets" in payload and "polydata_polys_connectivity" in payload:
                polydata = polydata_from_payload(
                    np.asarray(payload["polydata_points"], dtype=np.float32),
                    np.asarray(payload["polydata_polys_offsets"], dtype=np.int64),
                    np.asarray(payload["polydata_polys_connectivity"], dtype=np.int64),
                )
                actor = vtk.vtkActor()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                mapper.ScalarVisibilityOff()
                actor.SetMapper(mapper)
                actor.GetProperty().SetOpacity(float(np.asarray(payload["opacity"]).ravel()[0]))
                actor.GetProperty().SetInterpolationToPhong()
                actor.GetProperty().EdgeVisibilityOff()
                actor.GetProperty().SetColor(
                    *tuple(
                        np.asarray(
                            payload.get("color_rgb", np.array([0.82, 0.95, 1.0], dtype=np.float32)),
                            dtype=np.float32,
                        ).ravel()[:3]
                    )
                )
                name = self.updater._unique_name(
                    self.updater.horizons,
                    str(self._payload_scalar(payload["name"])),
                )
                base_polydata = (
                    polydata_from_payload(
                        np.asarray(payload["base_polydata_points"], dtype=np.float32),
                        np.asarray(payload["base_polydata_polys_offsets"], dtype=np.int64),
                        np.asarray(payload["base_polydata_polys_connectivity"], dtype=np.int64),
                    )
                    if "base_polydata_points" in payload
                    else clone_polydata(polydata)
                )
                horizon = HorizonSurface(
                    name=name,
                    actor=actor,
                    mapper=mapper,
                    polydata=polydata,
                    lut=create_lookup_table_from_scalars(
                        np.asarray([0.0, 1.0], dtype=np.float32),
                        self.updater.clip_percentile,
                    ),
                    component_index=int(np.asarray(payload["component_index"]).ravel()[0]),
                    voxel_count=int(np.asarray(payload["voxel_count"]).ravel()[0]),
                    scalar_range=(0.0, 1.0),
                    color=tuple(
                        np.asarray(
                            payload.get("color_rgb", np.array([0.82, 0.95, 1.0], dtype=np.float32)),
                            dtype=np.float32,
                        ).ravel()[:3]
                    ),
                    opacity=float(np.asarray(payload["opacity"]).ravel()[0]),
                    visible=True,
                    component_mask=np.asarray(payload["component_mask"], dtype=bool),
                    source_attribute_name=str(self._payload_scalar(payload.get("source_attribute_name", np.array("")))),
                    xlines=np.array(xlines, copy=True),
                    inlines=np.array(inlines, copy=True),
                    samples=np.array(samples, copy=True),
                    control_point_set=None,
                    base_polydata=base_polydata,
                )
                self.updater.horizons[name] = horizon
                self.renderer.AddActor(actor)
                self.updater.set_current_horizon(name, render=False)
            else:
                name = self.updater.add_horizon(
                    str(self._payload_scalar(payload["name"])),
                    component_mask=np.asarray(payload["component_mask"], dtype=bool),
                    xlines=xlines,
                    inlines=inlines,
                    samples=samples,
                    scalar_values=scalar_values,
                    source_attribute_name="",
                    component_index=int(np.asarray(payload["component_index"]).ravel()[0]),
                    voxel_count=int(np.asarray(payload["voxel_count"]).ravel()[0]),
                    opacity=float(np.asarray(payload["opacity"]).ravel()[0]),
                    color=tuple(np.asarray(payload.get("color_rgb", np.array([0.82, 0.95, 1.0], dtype=np.float32)), dtype=np.float32).ravel()[:3]),
                    visible=True,
                    select=True,
                )
            if bool(int(np.asarray(payload.get("has_control_points", np.array([0], dtype=np.uint8))).ravel()[0])):
                points = [
                    ControlPoint(**item)
                    for item in json.loads(str(self._payload_scalar(payload["control_points_json"])))
                ]
                value_attribute_text = str(
                    self._payload_scalar(payload.get("control_point_value_attribute_name", np.array("")))
                ).strip()
                self.updater.set_control_points_for_horizon(
                    name,
                    points=points,
                    source_attribute_name="",
                    xlines=np.array(xlines, copy=True),
                    inlines=np.array(inlines, copy=True),
                    samples=np.array(samples, copy=True),
                    value_attribute_name=None if not value_attribute_text else value_attribute_text,
                    use_attribute_colormap=bool(
                        int(np.asarray(payload.get("control_point_use_attribute_colormap", np.array([0], dtype=np.uint8))).ravel()[0])
                    ),
                    source_horizon_name=str(self._payload_scalar(payload.get("control_point_source_horizon_name", np.array(name)))),
                    original_horizon_mask=np.asarray(
                        payload.get("control_point_original_horizon_mask", payload["component_mask"]),
                        dtype=bool,
                    ),
                    display_scale=float(np.asarray(payload.get("control_point_display_scale", np.array([1.0], dtype=np.float32))).ravel()[0]),
                    link_radius=float(np.asarray(payload.get("control_point_link_radius", np.array([8.0 * min(self.updater.spacing.xline, self.updater.spacing.inline)], dtype=np.float32))).ravel()[0]),
                    visible=bool(int(np.asarray(payload.get("control_point_visible", np.array([1], dtype=np.uint8))).ravel()[0])),
                )
                loaded_point_set = self.updater.horizons[name].control_point_set
                if loaded_point_set is not None:
                    loaded_point_set.rebuild_smoothness = float(
                        np.asarray(payload.get("control_point_rebuild_smoothness", np.array([0.55], dtype=np.float32))).ravel()[0]
                    )
            self.refresh_data_panel()
            self.activate_data_item("horizon", name)
            return

        QtWidgets.QMessageBox.information(self, "Not Implemented", "井数据加载尚未实现。")

    def store_data_item(self, category: str, name: str) -> None:
        if category == "seismic":
            attribute = self.updater.attributes[name]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Store Seismic Data",
                str(self._default_output_dir(category) / f"{attribute.name}.npz"),
                "Wesi3D Volume (*.npz)",
            )
            if path:
                np.savez_compressed(path, **self._volume_payload(attribute.volume_data, attribute.opacity))
            return

        if category == "attribute":
            attribute = self.updater.attributes[name]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Store Attribute",
                str(self._default_output_dir(category) / f"{attribute.name}.npz"),
                "Wesi3D Volume (*.npz)",
            )
            if path:
                np.savez_compressed(path, **self._volume_payload(attribute.volume_data, attribute.opacity))
            return

        if category == "horizon":
            horizon = self.updater.horizons[name]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Store Horizon",
                str(self._default_output_dir(category) / f"{name}.npz"),
                "Wesi3D Horizon (*.npz)",
            )
            if path:
                point_set = horizon.control_point_set
                polydata_payload = polydata_to_payload(horizon.polydata)
                base_polydata_payload = polydata_to_payload(horizon.base_polydata or horizon.polydata)
                payload = {
                    "name": np.array(name),
                    "component_mask": np.asarray(horizon.component_mask, dtype=np.uint8),
                    "xlines": np.asarray(horizon.xlines if horizon.xlines is not None else [], dtype=np.float32),
                    "inlines": np.asarray(horizon.inlines if horizon.inlines is not None else [], dtype=np.float32),
                    "samples": np.asarray(horizon.samples if horizon.samples is not None else [], dtype=np.float32),
                    "component_index": np.array([horizon.component_index], dtype=np.int32),
                    "voxel_count": np.array([horizon.voxel_count], dtype=np.int64),
                    "opacity": np.array([horizon.opacity], dtype=np.float32),
                    "color_rgb": np.asarray(horizon.color, dtype=np.float32),
                    "source_attribute_name": np.array(horizon.source_attribute_name),
                    "polydata_points": polydata_payload["points"],
                    "polydata_polys_offsets": polydata_payload["polys_offsets"],
                    "polydata_polys_connectivity": polydata_payload["polys_connectivity"],
                    "base_polydata_points": base_polydata_payload["points"],
                    "base_polydata_polys_offsets": base_polydata_payload["polys_offsets"],
                    "base_polydata_polys_connectivity": base_polydata_payload["polys_connectivity"],
                    "has_control_points": np.array([0 if point_set is None else 1], dtype=np.uint8),
                }
                if point_set is not None:
                    payload.update(
                        {
                            "control_points_json": np.array(json.dumps([vars(point) for point in point_set.points])),
                            "control_point_display_scale": np.array([point_set.display_scale], dtype=np.float32),
                            "control_point_link_radius": np.array([point_set.link_radius], dtype=np.float32),
                            "control_point_rebuild_smoothness": np.array([point_set.rebuild_smoothness], dtype=np.float32),
                            "control_point_value_attribute_name": np.array(
                                "" if point_set.value_attribute_name is None else point_set.value_attribute_name
                            ),
                            "control_point_use_attribute_colormap": np.array(
                                [1 if point_set.use_attribute_colormap else 0], dtype=np.uint8
                            ),
                            "control_point_visible": np.array([1 if point_set.visible else 0], dtype=np.uint8),
                            "control_point_source_horizon_name": np.array(point_set.source_horizon_name),
                            "control_point_original_horizon_mask": np.asarray(
                                point_set.original_horizon_mask,
                                dtype=np.uint8,
                            ),
                        }
                    )
                np.savez_compressed(path, **payload)
            return

        QtWidgets.QMessageBox.information(self, "Not Implemented", "井数据存储尚未实现。")

    def unload_data_item(self, category: str, name: str) -> None:
        removed = False
        if category in {"seismic", "attribute"}:
            removed = self.updater.remove_attribute(name)
            self.image = self.updater.image
        elif category == "horizon":
            removed = self.updater.remove_horizon(name)
        else:
            QtWidgets.QMessageBox.information(self, "Not Implemented", "井数据卸载尚未实现。")
            return

        if not removed:
            QtWidgets.QMessageBox.information(self, "Unload Failed", "当前数据无法卸载。")
            return

        current_attribute = self.updater.current_attribute()
        fallback_category = str(
            current_attribute.volume_data.metadata.get(
                "panel_category",
                "seismic" if self.updater.current_attribute_name == "seismic" else "attribute",
            )
        )
        self._selected_data_item = (fallback_category, self.updater.current_attribute_name)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def change_control_point_size(self, value: int) -> None:
        self.updater.set_control_point_display_scale(value / 100.0, render=False)
        self.schedule_render()

    def change_control_point_link_radius(self, value: int) -> None:
        radius = float(value) * min(self.updater.spacing.xline, self.updater.spacing.inline)
        self.updater.set_control_point_link_radius(radius, render=False)
        self._refresh_linked_master_points()
        self._refresh_selected_master_actor()
        self.refresh_display_controls()
        self.schedule_render()

    def open_copy_control_point_values_dialog(self) -> None:
        horizon_names = [
            name
            for name, horizon in self.updater.horizons.items()
            if horizon.control_point_set is not None
        ]
        attribute_names = self.updater.attribute_names()
        if not horizon_names:
            QtWidgets.QMessageBox.information(
                self,
                "No Control Points",
                "No horizon currently contains control points.",
            )
            return
        if not attribute_names:
            QtWidgets.QMessageBox.information(
                self,
                "No Attribute",
                "Load an attribute before copying values to control points.",
            )
            return
        current_horizon_name = self.updater.current_horizon_name if self.updater.current_horizon_name in horizon_names else horizon_names[0]
        dialog = CopyControlPointValuesDialog(
            horizon_names,
            attribute_names,
            selected_horizon_name=current_horizon_name,
            selected_attribute_name=self.updater.current_attribute_name,
            parent=self,
        )
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        horizon_name, attribute_name = values
        if not self.updater.copy_attribute_values_to_control_points(horizon_name, attribute_name, render=False):
            QtWidgets.QMessageBox.information(
                self,
                "Copy Failed",
                "Failed to copy attribute values to the selected control points.",
            )
            return
        self.updater.set_current_horizon(horizon_name, render=False)
        self._selected_data_item = ("horizon", horizon_name)
        self._refresh_selected_master_actor()
        self.refresh_data_panel()
        self.refresh_display_controls()
        self.schedule_render()

    def toggle_control_point_colormap(self, checked: bool) -> None:
        self.updater.set_control_point_use_attribute_colormap(bool(checked), render=False)
        self._refresh_selected_master_actor()
        self.refresh_display_controls()
        self.schedule_render()

    def apply_control_point_colormap_range(self) -> None:
        min_text = self.control_point_colormap_widget.min_edit.text().strip()
        max_text = self.control_point_colormap_widget.max_edit.text().strip()
        if not min_text or not max_text:
            return
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return
        self.updater.set_control_point_colormap_range(min_value, max_value, render=False)
        self.refresh_display_controls()
        self.schedule_render()

    def change_control_point_colormap(self, name: str) -> None:
        if not name:
            return
        self.updater.set_control_point_colormap(name, render=False)
        self.refresh_display_controls()
        self.schedule_render()

    def change_control_point_smoothness(self, value: int) -> None:
        self.updater.set_control_point_rebuild_smoothness(value / 100.0, render=False)
        self.refresh_display_controls()
        self.schedule_render()

    def update_horizon_from_control_points(self) -> None:
        if not self.updater.update_current_horizon_from_control_points():
            QtWidgets.QMessageBox.information(
                self,
                "Update Failed",
                self.updater.last_rebuild_error or "Failed to update the current horizon from control points.",
            )
            return
        current_name = self.updater.current_horizon_name
        if current_name is not None:
            self._selected_data_item = ("horizon", current_name)
        self.refresh_data_panel()
        self.refresh_display_controls()
        self.schedule_render()

    def apply_attribute_display(self) -> None:
        min_text = self.attribute_colormap_widget.min_edit.text().strip()
        max_text = self.attribute_colormap_widget.max_edit.text().strip()
        if not min_text or not max_text:
            return
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return
        self.updater.set_attribute_display_range(min_value, max_value, render=False)
        self.schedule_render()

    def change_attribute_colormap(self, name: str) -> None:
        if not name:
            return
        self.updater.set_attribute_colormap(name, render=False)
        self.refresh_display_controls()
        self.schedule_render()

    def change_attribute_opacity(self, value: int) -> None:
        self.updater.set_attribute_opacity(value / 100.0, render=False)
        self.schedule_render()

    def apply_horizon_display(self) -> None:
        return

    def change_horizon_opacity(self, value: int) -> None:
        self.updater.set_horizon_opacity(value / 100.0, render=False)
        self.schedule_render()

    def open_extract_range_dialog(self) -> None:
        if not self.updater.has_attribute_data():
            return
        min_value, max_value = self.updater.current_scalar_range()
        dialog = ExtractRangeDialog(min_value, max_value, self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        min_value, max_value = values
        new_name = self.updater.extract_range_attribute(min_value, max_value)
        self.updater.set_attribute(new_name, render=False)
        self._selected_data_item = ("attribute", new_name)
        self.image = self.updater.image
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_extract_horizon_dialog(self) -> None:
        if not self.updater.has_attribute_data():
            return
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
        self.updater.set_current_horizon(new_names[0], render=False)
        self._selected_data_item = ("horizon", new_names[0])
        self.refresh_data_panel()
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
        new_name = self.updater.extract_control_points_for_current_horizon(
            **{str(key): int(value) for key, value in values.items()},
        )
        if new_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Control Points",
                "No control points were generated for the current horizon.",
            )
            return
        self.updater.set_current_horizon(new_name, render=False)
        self._selected_data_item = ("horizon", new_name)
        self._selected_master_point_index = None
        self._linked_master_point_indices.clear()
        self._refresh_selected_master_actor()
        self.refresh_display_controls()
        self.refresh_data_panel()
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
        self._selected_master_point_index = master_index
        self._refresh_linked_master_points()
        if not self.updater.edit_current_control_point_set_masters(self._linked_moves_for_delta(delta_sample)):
            QtWidgets.QMessageBox.information(
                self,
                "Edit Failed",
                self.updater.last_rebuild_error or "Failed to update the selected master point.",
            )
            return
        self._refresh_selected_master_actor()
        self.refresh_data_panel()
        self.schedule_render()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.render_window.Finalize()
        super().closeEvent(event)


def launch_vtk_viewer(
    segy_path: Path | None,
    initial_attribute: AttributeVolume | None,
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

    if initial_attribute is None:
        image, lut = create_placeholder_image()
        xlines = np.asarray([0.0], dtype=np.float32)
        inlines = np.asarray([0.0], dtype=np.float32)
        samples = np.asarray([0.0], dtype=np.float32)
    else:
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
    scalar_bar_actor = create_scalar_bar_actor()
    renderer.AddActor2D(scalar_bar_actor)

    overlay = vtk.vtkTextActor()
    overlay.GetTextProperty().SetFontSize(20)
    overlay.GetTextProperty().SetColor(0.95, 0.95, 0.95)
    overlay.SetDisplayPosition(20, 20)
    renderer.AddViewProp(overlay)

    vtk_widget = QVTKRenderWindowInteractor()
    render_window = vtk_widget.GetRenderWindow()
    render_window.SetWindowName(f"SEG-Y Slice Viewer - {segy_path.name if segy_path is not None else 'Empty'}")
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
        scalar_bar_actor=scalar_bar_actor,
        segy_path=segy_path,
        initial_attribute=initial_attribute,
        spacing=spacing,
        clip_percentile=clip_percentile,
        opacity=opacity_scale,
    )

    if initial_attribute is not None:
        configure_default_camera(renderer, image)
    debug_log(debug_ui, f"vtk render window created: {type(render_window).__name__}")
    debug_log(debug_ui, f"control ranges: xline={len(xlines)} inline={len(inlines)} sample={len(samples)}")

    window = SegyViewerWindow(
        updater=updater,
        vtk_widget=vtk_widget,
        render_window=render_window,
        renderer=renderer,
        outline_actor=outline_actor,
        axis_texts=axis_texts,
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
    segy_path = None if args.segy_path is None else Path(args.segy_path).expanduser().resolve()
    spacing = RenderSpacing(
        xline=args.step_xline,
        inline=args.step_inline,
        sample=args.step_sample,
    )
    initial_attribute = None
    if segy_path is not None:
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
        initial_attribute = load_attribute_from_volume(
            volume_data.with_data(volume_data.data, metadata={**volume_data.metadata, "panel_category": "seismic"}),
            name="seismic",
            spacing=spacing,
            clip_percentile=args.clip_percentile,
            opacity=args.opacity,
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
