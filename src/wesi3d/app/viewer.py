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
from .data_panel import DataPanelItem, DataPanelWidget, ProjectPanelWidget
from .importers import SeismicAttributeImportDialog, build_import_request, execute_import
from ..processing.control_points import (
    ControlPoint,
    MasterMove,
    apply_master_point_z_moves,
    extract_control_points,
    master_control_points,
    rebuild_mask_from_master_points,
)
from ..utils.constants import INLINE_FIELD, XLINE_FIELD
from ..processing.volume_processing import (
    extract_connected_components,
    extract_range_volume,
    interpolate_control_point_values_to_volume,
)
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
    value_colormap_name: str = DEFAULT_COLORMAP_NAME
    value_color_range: tuple[float, float] | None = None
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


@dataclass
class ScatterDataSet:
    name: str
    actor: vtk.vtkActor
    polydata: vtk.vtkPolyData
    mapper: vtk.vtkPolyDataMapper
    lut: vtk.vtkLookupTable
    value_range: tuple[float, float]
    inlines: np.ndarray
    crosslines: np.ndarray
    z_values: np.ndarray
    values: np.ndarray
    source_path: Path
    visible: bool = True


@dataclass
class PolygonDataSet:
    name: str
    actor: vtk.vtkActor
    polydata: vtk.vtkPolyData
    mapper: vtk.vtkPolyDataMapper
    point_actor: vtk.vtkActor
    point_polydata: vtk.vtkPolyData
    point_mapper: vtk.vtkPolyDataMapper
    color_rgb: tuple[int, int, int]
    grid_points: np.ndarray
    z_values: np.ndarray
    source_path: Path
    visible: bool = True


@dataclass
class ModelSurfaceDataSet:
    name: str
    actor: vtk.vtkActor
    polydata: vtk.vtkPolyData
    mapper: vtk.vtkPolyDataMapper
    source_polygon_name: str
    dip_source_path: Path
    direction_source_path: Path
    visible: bool = True


@dataclass
class GridDefinition:
    inline_start: float
    inline_end: float
    crossline_start: float
    crossline_end: float
    sample_start: float
    sample_end: float
    inline_size: float
    crossline_size: float
    sample_size: float
    datum: float = 0.0
    inline_step: float | None = None
    crossline_step: float | None = None
    sample_step: float | None = None

    @staticmethod
    def _axis_values(start: float, end: float, step_size: float) -> np.ndarray:
        start_value = float(start)
        end_value = float(end)
        size_value = max(1e-6, abs(float(step_size)))
        if start_value == end_value:
            return np.asarray([start_value], dtype=np.float32)
        step = size_value if end_value >= start_value else -size_value
        stop = end_value + step
        values = np.arange(start_value, stop, step, dtype=np.float32)
        if values.size == 0 or not np.isclose(values[-1], end_value):
            values = np.append(values, np.float32(end_value))
        return np.asarray(values, dtype=np.float32)

    @property
    def inline_values(self) -> np.ndarray:
        return self._axis_values(
            self.inline_start,
            self.inline_end,
            self.inline_size if self.inline_step is None else self.inline_step,
        )

    @property
    def crossline_values(self) -> np.ndarray:
        return self._axis_values(
            self.crossline_start,
            self.crossline_end,
            self.crossline_size if self.crossline_step is None else self.crossline_step,
        )

    @property
    def sample_values(self) -> np.ndarray:
        return self._axis_values(
            self.sample_start,
            self.sample_end,
            self.sample_size if self.sample_step is None else self.sample_step,
        )

    @property
    def inline_display_spacing(self) -> float:
        return float((self.inline_step if self.inline_step is not None else 1.0) * self.inline_size)

    @property
    def crossline_display_spacing(self) -> float:
        return float((self.crossline_step if self.crossline_step is not None else 1.0) * self.crossline_size)

    @property
    def sample_display_spacing(self) -> float:
        return float((self.sample_step if self.sample_step is not None else 1.0) * self.sample_size)

    def as_dict(self) -> dict[str, float]:
        return {
            "inline_start": float(self.inline_start),
            "inline_end": float(self.inline_end),
            "crossline_start": float(self.crossline_start),
            "crossline_end": float(self.crossline_end),
            "sample_start": float(self.sample_start),
            "sample_end": float(self.sample_end),
            "inline_size": float(self.inline_size),
            "crossline_size": float(self.crossline_size),
            "sample_size": float(self.sample_size),
            "datum": float(self.datum),
            "inline_step": float(self.inline_size if self.inline_step is None else self.inline_step),
            "crossline_step": float(self.crossline_size if self.crossline_step is None else self.crossline_step),
            "sample_step": float(self.sample_size if self.sample_step is None else self.sample_step),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GridDefinition:
        return cls(
            inline_start=float(payload["inline_start"]),
            inline_end=float(payload["inline_end"]),
            crossline_start=float(payload["crossline_start"]),
            crossline_end=float(payload["crossline_end"]),
            sample_start=float(payload["sample_start"]),
            sample_end=float(payload["sample_end"]),
            inline_size=float(payload["inline_size"]),
            crossline_size=float(payload["crossline_size"]),
            sample_size=float(payload["sample_size"]),
            datum=float(payload.get("datum", 0.0)),
            inline_step=float(payload.get("inline_step", payload["inline_size"])),
            crossline_step=float(payload.get("crossline_step", payload["crossline_size"])),
            sample_step=float(payload.get("sample_step", payload["sample_size"])),
        )

    def to_json_file(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target


GEOMAP_POINT0 = (517888.79, 4598260.61, 2000.0, 1200.0)
GEOMAP_POINT1 = (501208.58, 4636806.30, 2000.0, 3300.0)
GEOMAP_POINT3 = (554598.98, 4614146.52, 4000.0, 1200.0)


def default_geomap_grid_definition() -> GridDefinition:
    return GridDefinition(
        inline_start=min(GEOMAP_POINT0[2], GEOMAP_POINT3[2]),
        inline_end=max(GEOMAP_POINT0[2], GEOMAP_POINT3[2]),
        crossline_start=min(GEOMAP_POINT0[3], GEOMAP_POINT1[3]),
        crossline_end=max(GEOMAP_POINT0[3], GEOMAP_POINT1[3]),
        sample_start=0.0,
        sample_end=1.0,
        inline_size=100.0,
        crossline_size=100.0,
        sample_size=1.0,
    )


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


def create_scatter_actor(
    inlines: np.ndarray,
    crosslines: np.ndarray,
    values: np.ndarray,
    *,
    z_values: np.ndarray | None = None,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
) -> tuple[vtk.vtkActor, vtk.vtkPolyData, vtk.vtkPolyDataMapper, vtk.vtkLookupTable, tuple[float, float]]:
    inline_array = np.asarray(inlines, dtype=np.float32).ravel()
    crossline_array = np.asarray(crosslines, dtype=np.float32).ravel()
    value_array = np.asarray(values, dtype=np.float32).ravel()
    z_array = np.zeros_like(value_array, dtype=np.float32) if z_values is None else np.asarray(z_values, dtype=np.float32).ravel()
    if not (inline_array.size and crossline_array.size and value_array.size):
        raise ValueError("Scatter file is empty.")
    if not (inline_array.size == crossline_array.size == value_array.size == z_array.size):
        raise ValueError("Scatter columns must have the same length.")

    points_xyz = np.column_stack(
        [
            crossline_array,
            inline_array,
            z_array,
        ]
    ).astype(np.float32, copy=False)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points_xyz, deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    cells = vtk.vtkCellArray()
    for index in range(points_xyz.shape[0]):
        cells.InsertNextCell(1)
        cells.InsertCellPoint(index)
    polydata.SetVerts(cells)

    scalars = numpy_support.numpy_to_vtk(value_array, deep=True)
    scalars.SetName("value")
    polydata.GetPointData().SetScalars(scalars)

    value_min = float(np.min(value_array))
    value_max = float(np.max(value_array))
    if value_min == value_max:
        value_max = value_min + 1.0
    value_range = (value_min, value_max)

    lut = vtk.vtkLookupTable()
    lut.SetRange(*value_range)
    apply_colormap_preset(lut, colormap_name)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    mapper.SetScalarRange(lut.GetRange())
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToPoints()
    actor.GetProperty().SetPointSize(6.0)
    actor.GetProperty().SetOpacity(0.95)
    return actor, polydata, mapper, lut, value_range


def load_geomap_polygons(path: Path) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    polygons: list[tuple[tuple[int, int, int], np.ndarray]] = []
    current_points: list[tuple[float, float]] = []
    current_color = (240, 210, 120)
    with path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            if line_index == 0:
                continue
            if line.startswith("##"):
                if len(current_points) >= 2:
                    polygons.append((current_color, np.asarray(current_points, dtype=np.float32)))
                current_points = []
                color_parts = line[2:].strip().split()
                if len(color_parts) >= 3:
                    try:
                        current_color = tuple(int(max(0, min(255, int(part)))) for part in color_parts[:3])  # type: ignore[assignment]
                    except ValueError:
                        current_color = (240, 210, 120)
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                current_points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(current_points) >= 2:
        polygons.append((current_color, np.asarray(current_points, dtype=np.float32)))
    return polygons


def polygon_intersects_grid(polygon_points: np.ndarray, grid_definition: GridDefinition) -> bool:
    inline_min = min(float(grid_definition.inline_start), float(grid_definition.inline_end))
    inline_max = max(float(grid_definition.inline_start), float(grid_definition.inline_end))
    crossline_min = min(float(grid_definition.crossline_start), float(grid_definition.crossline_end))
    crossline_max = max(float(grid_definition.crossline_start), float(grid_definition.crossline_end))
    for point in np.asarray(polygon_points, dtype=np.float32):
        inl = float(point[0])
        cxl = float(point[1])
        if inline_min <= inl <= inline_max and crossline_min <= cxl <= crossline_max:
            return True
    return False


def normalize_polygon_grid_points(polygon_points: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon_points, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    deduplicated: list[np.ndarray] = []
    for point in points:
        if not deduplicated or not np.allclose(point, deduplicated[-1]):
            deduplicated.append(point)
    normalized = np.asarray(deduplicated, dtype=np.float32)

    while normalized.shape[0] >= 2 and np.allclose(normalized[0], normalized[-1]):
        normalized = np.asarray(normalized[:-1], dtype=np.float32)

    # Remove a repeated suffix that duplicates the polygon prefix using a
    # prefix-function scan instead of quadratic tail searching.
    while normalized.shape[0] >= 3:
        point_keys = [tuple(float(value) for value in point) for point in normalized]
        prefix_lengths = [0] * len(point_keys)
        for index in range(1, len(point_keys)):
            prefix_length = prefix_lengths[index - 1]
            while prefix_length > 0 and point_keys[index] != point_keys[prefix_length]:
                prefix_length = prefix_lengths[prefix_length - 1]
            if point_keys[index] == point_keys[prefix_length]:
                prefix_length += 1
            prefix_lengths[index] = prefix_length
        repeated_suffix_length = prefix_lengths[-1]
        if repeated_suffix_length <= 0:
            break
        normalized = np.asarray(normalized[:-repeated_suffix_length], dtype=np.float32)

    return np.asarray(normalized, dtype=np.float32)


@dataclass
class ElevSurface:
    inlines: np.ndarray
    crosslines: np.ndarray
    values: np.ndarray

    def sample(self, inline_value: float, crossline_value: float) -> float:
        inline_axis = np.asarray(self.inlines, dtype=np.float32)
        crossline_axis = np.asarray(self.crosslines, dtype=np.float32)
        value_grid = np.asarray(self.values, dtype=np.float32)

        inline_clamped = float(np.clip(inline_value, float(inline_axis[0]), float(inline_axis[-1])))
        crossline_clamped = float(np.clip(crossline_value, float(crossline_axis[0]), float(crossline_axis[-1])))

        if inline_axis.size == 1:
            inline_low = inline_high = 0
            inline_weight = 0.0
        else:
            inline_high = int(np.searchsorted(inline_axis, inline_clamped, side="left"))
            if inline_high <= 0:
                inline_low, inline_high = 0, 1
            elif inline_high >= inline_axis.size:
                inline_low, inline_high = inline_axis.size - 2, inline_axis.size - 1
            else:
                inline_low = inline_high - 1
            inline_span = float(inline_axis[inline_high] - inline_axis[inline_low])
            inline_weight = 0.0 if abs(inline_span) <= 1e-6 else (inline_clamped - float(inline_axis[inline_low])) / inline_span

        if crossline_axis.size == 1:
            crossline_low = crossline_high = 0
            crossline_weight = 0.0
        else:
            crossline_high = int(np.searchsorted(crossline_axis, crossline_clamped, side="left"))
            if crossline_high <= 0:
                crossline_low, crossline_high = 0, 1
            elif crossline_high >= crossline_axis.size:
                crossline_low, crossline_high = crossline_axis.size - 2, crossline_axis.size - 1
            else:
                crossline_low = crossline_high - 1
            crossline_span = float(crossline_axis[crossline_high] - crossline_axis[crossline_low])
            crossline_weight = (
                0.0
                if abs(crossline_span) <= 1e-6
                else (crossline_clamped - float(crossline_axis[crossline_low])) / crossline_span
            )

        v00 = float(value_grid[inline_low, crossline_low])
        v01 = float(value_grid[inline_low, crossline_high])
        v10 = float(value_grid[inline_high, crossline_low])
        v11 = float(value_grid[inline_high, crossline_high])
        top = (1.0 - crossline_weight) * v00 + crossline_weight * v01
        bottom = (1.0 - crossline_weight) * v10 + crossline_weight * v11
        return float((1.0 - inline_weight) * top + inline_weight * bottom)


def build_elev_surface(inlines: np.ndarray, crosslines: np.ndarray, values: np.ndarray) -> ElevSurface:
    inline_array = np.asarray(inlines, dtype=np.float32).ravel()
    crossline_array = np.asarray(crosslines, dtype=np.float32).ravel()
    value_array = np.asarray(values, dtype=np.float32).ravel()
    if not (inline_array.size == crossline_array.size == value_array.size):
        raise ValueError("Elev file columns must have the same length.")
    inline_axis = np.unique(inline_array)
    crossline_axis = np.unique(crossline_array)
    if inline_axis.size * crossline_axis.size != value_array.size:
        raise ValueError("Elev file must define a full inline/cxline grid.")
    order = np.lexsort((crossline_array, inline_array))
    sorted_inlines = inline_array[order]
    sorted_crosslines = crossline_array[order]
    if not np.allclose(sorted_inlines.reshape(inline_axis.size, crossline_axis.size), inline_axis[:, None]):
        raise ValueError("Elev file inline axis is not a regular grid.")
    if not np.allclose(sorted_crosslines.reshape(inline_axis.size, crossline_axis.size), crossline_axis[None, :]):
        raise ValueError("Elev file crossline axis is not a regular grid.")
    value_grid = value_array[order].reshape(inline_axis.size, crossline_axis.size)
    return ElevSurface(inline_axis, crossline_axis, value_grid)


def resample_closed_curve(
    grid_points: np.ndarray,
    z_values: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = normalize_polygon_grid_points(grid_points)
    point_z = np.asarray(z_values, dtype=np.float32).ravel()
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if point_z.size != points.shape[0]:
        point_z = np.zeros(points.shape[0], dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, sample_count, axis=0), np.repeat(point_z, sample_count, axis=0)

    closed_points = np.vstack([points, points[0]])
    closed_z = np.append(point_z, point_z[0])
    segment_lengths = np.linalg.norm(np.diff(closed_points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(points[:1], sample_count, axis=0), np.repeat(point_z[:1], sample_count, axis=0)

    targets = np.linspace(0.0, total_length, int(max(3, sample_count)) + 1, dtype=np.float32)[:-1]
    sample_points = np.zeros((targets.size, 2), dtype=np.float32)
    sample_z = np.zeros(targets.size, dtype=np.float32)
    for index, target in enumerate(targets):
        segment_index = int(np.searchsorted(cumulative, float(target), side="right") - 1)
        segment_index = max(0, min(segment_index, segment_lengths.size - 1))
        start_distance = float(cumulative[segment_index])
        length = float(segment_lengths[segment_index])
        weight = 0.0 if length <= 1e-6 else (float(target) - start_distance) / length
        sample_points[index] = (
            (1.0 - weight) * closed_points[segment_index] + weight * closed_points[segment_index + 1]
        )
        sample_z[index] = float((1.0 - weight) * closed_z[segment_index] + weight * closed_z[segment_index + 1])
    return sample_points, sample_z


def smooth_closed_curve(points_xyz: np.ndarray, iterations: int) -> np.ndarray:
    smoothed = np.asarray(points_xyz, dtype=np.float32)
    for _ in range(max(0, int(iterations))):
        previous_points = np.roll(smoothed, 1, axis=0)
        next_points = np.roll(smoothed, -1, axis=0)
        smoothed = 0.25 * previous_points + 0.5 * smoothed + 0.25 * next_points
    return np.asarray(smoothed, dtype=np.float32)


def smooth_closed_scalar(values: np.ndarray, iterations: int) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float32).ravel()
    for _ in range(max(0, int(iterations))):
        previous_values = np.roll(smoothed, 1)
        next_values = np.roll(smoothed, -1)
        smoothed = 0.25 * previous_values + 0.5 * smoothed + 0.25 * next_values
    return np.asarray(smoothed, dtype=np.float32)


def smooth_closed_angles_deg(values_deg: np.ndarray, iterations: int) -> np.ndarray:
    angles_rad = np.deg2rad(np.asarray(values_deg, dtype=np.float32).ravel())
    sin_values = smooth_closed_scalar(np.sin(angles_rad).astype(np.float32, copy=False), iterations)
    cos_values = smooth_closed_scalar(np.cos(angles_rad).astype(np.float32, copy=False), iterations)
    return np.asarray((np.rad2deg(np.arctan2(sin_values, cos_values)) + 360.0) % 360.0, dtype=np.float32)


def resample_closed_xyz_curve(points_xyz: np.ndarray, sample_count: int) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.shape[0] >= 2 and np.allclose(points[0], points[-1]):
        points = np.asarray(points[:-1], dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, max(3, int(sample_count)), axis=0)

    closed_points = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed_points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(points[:1], max(3, int(sample_count)), axis=0)

    targets = np.linspace(0.0, total_length, int(max(3, sample_count)) + 1, dtype=np.float32)[:-1]
    resampled = np.zeros((targets.size, 3), dtype=np.float32)
    for index, target in enumerate(targets):
        segment_index = int(np.searchsorted(cumulative, float(target), side="right") - 1)
        segment_index = max(0, min(segment_index, segment_lengths.size - 1))
        start_distance = float(cumulative[segment_index])
        length = float(segment_lengths[segment_index])
        weight = 0.0 if length <= 1e-6 else (float(target) - start_distance) / length
        resampled[index] = (
            (1.0 - weight) * closed_points[segment_index] + weight * closed_points[segment_index + 1]
        )
    return np.asarray(resampled, dtype=np.float32)


def _segment_intersection_2d(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
    point_d: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    ax, ay = float(point_a[0]), float(point_a[1])
    bx, by = float(point_b[0]), float(point_b[1])
    cx, cy = float(point_c[0]), float(point_c[1])
    dx, dy = float(point_d[0]), float(point_d[1])

    abx, aby = bx - ax, by - ay
    cdx, cdy = dx - cx, dy - cy
    denominator = abx * cdy - aby * cdx
    if abs(denominator) <= 1e-8:
        return None

    acx, acy = cx - ax, cy - ay
    t = (acx * cdy - acy * cdx) / denominator
    u = (acx * aby - acy * abx) / denominator
    if not (1e-5 < t < 1.0 - 1e-5 and 1e-5 < u < 1.0 - 1e-5):
        return None
    intersection = np.asarray([ax + t * abx, ay + t * aby], dtype=np.float32)
    return intersection, float(t), float(u)


def _closed_loop_length(points_xyz: np.ndarray) -> float:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.shape[0] < 2:
        return 0.0
    closed_points = np.vstack([points, points[0]])
    return float(np.sum(np.linalg.norm(np.diff(closed_points, axis=0), axis=1)))


def find_self_intersection_loop(points_xyz: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    point_total = points.shape[0]
    if point_total < 4:
        return None

    for start_index in range(point_total):
        next_start = (start_index + 1) % point_total
        seg_a0 = points[start_index, :2]
        seg_a1 = points[next_start, :2]
        for end_index in range(start_index + 2, point_total):
            next_end = (end_index + 1) % point_total
            if start_index == 0 and next_end == 0:
                continue
            seg_b0 = points[end_index, :2]
            seg_b1 = points[next_end, :2]
            hit = _segment_intersection_2d(seg_a0, seg_a1, seg_b0, seg_b1)
            if hit is None:
                continue
            intersection_xy, t_ab, t_cd = hit
            intersection_z_ab = float(points[start_index, 2] + t_ab * (points[next_start, 2] - points[start_index, 2]))
            intersection_z_cd = float(points[end_index, 2] + t_cd * (points[next_end, 2] - points[end_index, 2]))
            intersection_xyz = np.asarray(
                [float(intersection_xy[0]), float(intersection_xy[1]), 0.5 * (intersection_z_ab + intersection_z_cd)],
                dtype=np.float32,
            )

            candidate_a = np.vstack([intersection_xyz, points[next_start : end_index + 1], intersection_xyz])
            candidate_b = np.vstack([intersection_xyz, points[next_end:], points[: start_index + 1], intersection_xyz])
            valid_candidates = [candidate for candidate in (candidate_a, candidate_b) if candidate.shape[0] >= 4]
            if not valid_candidates:
                return None
            selected = min(valid_candidates, key=_closed_loop_length)
            return np.asarray(selected[:-1], dtype=np.float32)
    return None


def idw_interpolate_scatter(
    source_inlines: np.ndarray,
    source_crosslines: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    *,
    power: float = 2.0,
    k_neighbors: int = 8,
) -> np.ndarray:
    inlines = np.asarray(source_inlines, dtype=np.float32).ravel()
    crosslines = np.asarray(source_crosslines, dtype=np.float32).ravel()
    values = np.asarray(source_values, dtype=np.float32).ravel()
    targets = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    if not (inlines.size == crosslines.size == values.size):
        raise ValueError("Scatter interpolation inputs must have the same length.")
    if inlines.size == 0:
        raise ValueError("Scatter interpolation source is empty.")

    deltas_inline = targets[:, None, 0] - inlines[None, :]
    deltas_crossline = targets[:, None, 1] - crosslines[None, :]
    distances2 = deltas_inline * deltas_inline + deltas_crossline * deltas_crossline
    neighbor_count = max(1, min(int(k_neighbors), values.size))
    neighbor_indices = np.argpartition(distances2, neighbor_count - 1, axis=1)[:, :neighbor_count]
    neighbor_distances2 = np.take_along_axis(distances2, neighbor_indices, axis=1)
    neighbor_values = values[neighbor_indices]

    exact_mask = neighbor_distances2 <= 1e-12
    weights = 1.0 / np.maximum(neighbor_distances2, 1e-12) ** (float(power) * 0.5)
    weights[exact_mask] = 0.0
    weight_sum = np.sum(weights, axis=1)
    interpolated = np.sum(weights * neighbor_values, axis=1) / np.maximum(weight_sum, 1e-12)
    if np.any(exact_mask):
        exact_rows = np.any(exact_mask, axis=1)
        exact_cols = np.argmax(exact_mask[exact_rows], axis=1)
        interpolated[exact_rows] = neighbor_values[exact_rows, exact_cols]
    return np.asarray(interpolated, dtype=np.float32)


def interpolate_direction_scatter(
    source_inlines: np.ndarray,
    source_crosslines: np.ndarray,
    source_directions_deg: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    directions_rad = np.deg2rad(np.asarray(source_directions_deg, dtype=np.float32).ravel())
    sin_values = np.sin(directions_rad).astype(np.float32, copy=False)
    cos_values = np.cos(directions_rad).astype(np.float32, copy=False)
    interp_sin = idw_interpolate_scatter(source_inlines, source_crosslines, sin_values, target_points)
    interp_cos = idw_interpolate_scatter(source_inlines, source_crosslines, cos_values, target_points)
    return np.asarray((np.rad2deg(np.arctan2(interp_sin, interp_cos)) + 360.0) % 360.0, dtype=np.float32)


def build_extruded_polygon_surface(
    polygon_points: np.ndarray,
    polygon_z_values: np.ndarray,
    dip_inlines: np.ndarray,
    dip_crosslines: np.ndarray,
    dip_values: np.ndarray,
    direction_inlines: np.ndarray,
    direction_crosslines: np.ndarray,
    direction_values: np.ndarray,
    *,
    sample_count: int,
    layer_step: float,
    target_depth: float,
    smooth_iterations: int,
) -> vtk.vtkPolyData:
    boundary_points, boundary_z = resample_closed_curve(polygon_points, polygon_z_values, sample_count)
    if boundary_points.shape[0] < 3:
        raise ValueError("Polygon must contain at least 3 points.")

    point_total = boundary_points.shape[0]
    current_layer = np.zeros((point_total, 3), dtype=np.float32)
    current_layer[:, 0] = boundary_points[:, 1]
    current_layer[:, 1] = boundary_points[:, 0]
    current_layer[:, 2] = boundary_z

    surface_layers = [np.asarray(current_layer, dtype=np.float32)]
    closed_surface = False
    target_depth_value = float(target_depth)
    vertical_step = max(1e-3, float(layer_step))
    smooth_iteration_count = max(0, int(smooth_iterations))
    stabilization_iterations = max(0, smooth_iteration_count + 1) if smooth_iteration_count > 0 else 0

    max_iterations = max(1, int(np.ceil(max(target_depth_value - float(np.min(boundary_z)), 0.0) / vertical_step)) + 2)
    for _ in range(max_iterations):
        remaining_depth = np.maximum(target_depth_value - current_layer[:, 2], 0.0).astype(np.float32, copy=False)
        if not np.any(remaining_depth > 1e-4):
            break

        current_boundary_points = np.column_stack([current_layer[:, 1], current_layer[:, 0]]).astype(np.float32, copy=False)
        dips_deg = np.clip(
            idw_interpolate_scatter(dip_inlines, dip_crosslines, dip_values, current_boundary_points),
            1.0,
            89.0,
        )
        directions_deg = interpolate_direction_scatter(
            direction_inlines,
            direction_crosslines,
            direction_values,
            current_boundary_points,
        )
        if stabilization_iterations > 0:
            dips_deg = np.clip(smooth_closed_scalar(dips_deg, stabilization_iterations), 1.0, 89.0)
            directions_deg = smooth_closed_angles_deg(directions_deg, stabilization_iterations)

        directions_rad = np.deg2rad(directions_deg)
        dips_rad = np.deg2rad(dips_deg)
        vertical_step_vector = np.minimum(np.full(point_total, vertical_step, dtype=np.float32), remaining_depth)
        horizontal_step = vertical_step_vector / np.maximum(np.tan(dips_rad), 1e-3)

        boundary_loop = np.vstack([current_boundary_points, current_boundary_points[0]])
        edge_lengths = np.linalg.norm(np.diff(boundary_loop, axis=0), axis=1)
        local_spacing = np.minimum(edge_lengths, np.roll(edge_lengths, 1))
        max_horizontal_step = np.maximum(0.35 * local_spacing, vertical_step_vector * 0.5)
        horizontal_step = np.minimum(horizontal_step, max_horizontal_step.astype(np.float32, copy=False))

        next_layer = np.array(current_layer, copy=True)
        next_layer[:, 0] = next_layer[:, 0] + horizontal_step * np.sin(directions_rad)
        next_layer[:, 1] = next_layer[:, 1] + horizontal_step * np.cos(directions_rad)
        next_layer[:, 2] = np.minimum(next_layer[:, 2] + vertical_step_vector, target_depth_value)
        if smooth_iteration_count > 0:
            next_layer = smooth_closed_curve(next_layer, smooth_iteration_count)
        next_layer[:, 2] = np.minimum(next_layer[:, 2], target_depth_value)

        intersection_loop = find_self_intersection_loop(next_layer)
        if intersection_loop is not None and intersection_loop.shape[0] >= 3:
            closed_loop = resample_closed_xyz_curve(intersection_loop, point_total)
            if smooth_iteration_count > 0:
                closed_loop = smooth_closed_curve(closed_loop, max(2, smooth_iteration_count + 2))
            next_layer = resample_closed_xyz_curve(closed_loop, point_total)
            closed_surface = True
            surface_layers.append(np.asarray(next_layer, dtype=np.float32))
            break

        surface_layers.append(np.asarray(next_layer, dtype=np.float32))
        current_layer = next_layer

    surface_points = np.stack(surface_layers, axis=0)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(surface_points.reshape(-1, 3), deep=True))

    polys = vtk.vtkCellArray()
    for layer_index in range(surface_points.shape[0] - 1):
        row_offset = layer_index * point_total
        next_row_offset = (layer_index + 1) * point_total
        for point_index in range(point_total):
            next_point_index = (point_index + 1) % point_total
            tri_a = vtk.vtkTriangle()
            tri_a.GetPointIds().SetId(0, row_offset + point_index)
            tri_a.GetPointIds().SetId(1, row_offset + next_point_index)
            tri_a.GetPointIds().SetId(2, next_row_offset + point_index)
            polys.InsertNextCell(tri_a)

            tri_b = vtk.vtkTriangle()
            tri_b.GetPointIds().SetId(0, next_row_offset + point_index)
            tri_b.GetPointIds().SetId(1, row_offset + next_point_index)
            tri_b.GetPointIds().SetId(2, next_row_offset + next_point_index)
            polys.InsertNextCell(tri_b)

    side_polydata = vtk.vtkPolyData()
    side_polydata.SetPoints(vtk_points)
    side_polydata.SetPolys(polys)

    combined_polydata = side_polydata
    if closed_surface:
        append = vtk.vtkAppendPolyData()
        append.AddInputData(side_polydata)
        for cap_points in (surface_points[0], surface_points[-1][::-1]):
            cap_points_vtk = vtk.vtkPoints()
            cap_points_vtk.SetData(numpy_support.numpy_to_vtk(np.asarray(cap_points, dtype=np.float32), deep=True))
            contour = vtk.vtkPolyData()
            contour.SetPoints(cap_points_vtk)
            lines = vtk.vtkCellArray()
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(cap_points.shape[0] + 1)
            for point_index in range(cap_points.shape[0]):
                polyline.GetPointIds().SetId(point_index, point_index)
            polyline.GetPointIds().SetId(cap_points.shape[0], 0)
            lines.InsertNextCell(polyline)
            contour.SetLines(lines)
            triangulator = vtk.vtkContourTriangulator()
            triangulator.SetInputData(contour)
            triangulator.Update()
            append.AddInputData(triangulator.GetOutput())
        append.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(append.GetOutputPort())
        cleaner.Update()
        if smooth_iteration_count > 0:
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(cleaner.GetOutputPort())
            smoother.SetNumberOfIterations(max(12, smooth_iteration_count * 6))
            smoother.SetRelaxationFactor(0.08)
            smoother.FeatureEdgeSmoothingOff()
            smoother.BoundarySmoothingOn()
            smoother.Update()
            combined_polydata = vtk.vtkPolyData()
            combined_polydata.DeepCopy(smoother.GetOutput())
        else:
            combined_polydata = vtk.vtkPolyData()
            combined_polydata.DeepCopy(cleaner.GetOutput())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(combined_polydata)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def create_model_surface_actor(
    polydata: vtk.vtkPolyData,
    *,
    color: tuple[float, float, float] = (0.93, 0.72, 0.38),
) -> tuple[vtk.vtkActor, vtk.vtkPolyDataMapper]:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(0.76)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().EdgeVisibilityOff()
    return actor, mapper


def sample_polydata_surface_depth(
    polydata: vtk.vtkPolyData,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    sample_min: float,
    sample_max: float,
) -> np.ndarray:
    locator = vtk.vtkOBBTree()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    sampled = np.full((len(inline_values), len(crossline_values)), np.nan, dtype=np.float32)
    line_start_z = float(min(sample_min, sample_max))
    line_end_z = float(max(sample_min, sample_max))

    for inline_index, inline_value in enumerate(np.asarray(inline_values, dtype=np.float32)):
        for crossline_index, crossline_value in enumerate(np.asarray(crossline_values, dtype=np.float32)):
            points = vtk.vtkPoints()
            cell_ids = vtk.vtkIdList()
            hit_count = locator.IntersectWithLine(
                (float(crossline_value), float(inline_value), line_start_z),
                (float(crossline_value), float(inline_value), line_end_z),
                points,
                cell_ids,
            )
            if hit_count <= 0 or points.GetNumberOfPoints() == 0:
                continue
            z_hits = [float(points.GetPoint(hit_index)[2]) for hit_index in range(points.GetNumberOfPoints())]
            # For closed/partially closed surfaces we want the deepest envelope
            # intersection, not the first shallow hit on a side wall.
            sampled[inline_index, crossline_index] = float(np.max(z_hits))
    return sampled


def extract_slice_polygons(
    polydata: vtk.vtkPolyData,
    z_value: float,
) -> list[np.ndarray]:
    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        return []

    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, float(z_value))
    plane.SetNormal(0.0, 0.0, 1.0)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(polydata)
    cutter.SetCutFunction(plane)
    cutter.Update()

    stripped_input = cutter.GetOutput()
    if stripped_input.GetNumberOfPoints() == 0 or stripped_input.GetNumberOfLines() == 0:
        return []

    stripper = vtk.vtkStripper()
    stripper.SetInputData(stripped_input)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    stripped = stripper.GetOutput()
    points = stripped.GetPoints()
    lines = stripped.GetLines()
    if points is None or lines is None or stripped.GetNumberOfLines() == 0:
        return []

    polygons: list[np.ndarray] = []
    id_list = vtk.vtkIdList()
    lines.InitTraversal()
    while lines.GetNextCell(id_list):
        if id_list.GetNumberOfIds() < 3:
            continue
        polygon_points: list[tuple[float, float]] = []
        for point_index in range(id_list.GetNumberOfIds()):
            x_value, y_value, _ = points.GetPoint(id_list.GetId(point_index))
            polygon_points.append((float(x_value), float(y_value)))
        polygon = np.asarray(polygon_points, dtype=np.float32)
        if polygon.shape[0] >= 2 and np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-4:
            polygon = polygon[:-1]
        if polygon.shape[0] < 3:
            continue
        polygons.append(polygon)
    return polygons


def polygon_mask_on_grid(
    polygon_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return np.zeros(grid_x.shape, dtype=bool)

    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    query_x = np.asarray(grid_x, dtype=np.float32).ravel()
    query_y = np.asarray(grid_y, dtype=np.float32).ravel()
    inside = np.zeros(query_x.shape, dtype=bool)

    prev_index = polygon.shape[0] - 1
    for curr_index in range(polygon.shape[0]):
        x0 = float(x_coords[prev_index])
        y0 = float(y_coords[prev_index])
        x1 = float(x_coords[curr_index])
        y1 = float(y_coords[curr_index])
        crosses = (y0 > query_y) != (y1 > query_y)
        if abs(y1 - y0) <= 1e-12:
            prev_index = curr_index
            continue
        x_intersections = ((x1 - x0) * (query_y - y0) / (y1 - y0)) + x0
        inside ^= crosses & (query_x <= x_intersections)
        prev_index = curr_index

    return inside.reshape(grid_x.shape)


def polygon_signed_area_xy(polygon_xy: np.ndarray) -> float:
    polygon = np.asarray(polygon_xy, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return 0.0
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    return 0.5 * float(
        np.sum(x_values * np.roll(y_values, -1) - np.roll(x_values, -1) * y_values)
    )


def fill_model_volume_from_surfaces(
    sample_values: np.ndarray,
    elev_depths: np.ndarray,
    model_masks: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    samples = np.asarray(sample_values, dtype=np.float32).ravel()
    output = np.zeros((elev_depths.shape[0], elev_depths.shape[1], samples.size), dtype=np.float32)
    sample_min = float(np.min(samples))
    sample_max = float(np.max(samples))

    for inline_index in range(elev_depths.shape[0]):
        for crossline_index in range(elev_depths.shape[1]):
            top_depth = float(elev_depths[inline_index, crossline_index])
            if np.isnan(top_depth):
                continue
            top_depth = min(max(top_depth, sample_min), sample_max)
            output[inline_index, crossline_index, samples <= top_depth] = -1.0

    for model_id, model_mask in model_masks:
        mask_array = np.asarray(model_mask, dtype=bool)
        if mask_array.shape != output.shape:
            if mask_array.ndim == 3 and mask_array.shape == (output.shape[1], output.shape[0], output.shape[2]):
                mask_array = np.transpose(mask_array, (1, 0, 2))
            else:
                raise ValueError(
                    f"model mask shape {mask_array.shape} does not match output shape {output.shape}"
                )
        output[mask_array] = float(model_id)

    return output


def build_model_mask_volume(
    sample_values: np.ndarray,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    polydata: vtk.vtkPolyData,
    *,
    debug_label: str = "model",
) -> np.ndarray:
    samples = np.asarray(sample_values, dtype=np.float32).ravel()
    output = np.zeros((len(inline_values), len(crossline_values), samples.size), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(
        np.asarray(crossline_values, dtype=np.float32),
        np.asarray(inline_values, dtype=np.float32),
        indexing="xy",
    )

    points_data = polydata.GetPoints()
    if points_data is None or points_data.GetData() is None:
        return output
    surface_points = np.asarray(numpy_support.vtk_to_numpy(points_data.GetData()), dtype=np.float32)
    if surface_points.size == 0:
        return output
    z_min = float(np.min(surface_points[:, 2]))
    z_max = float(np.max(surface_points[:, 2]))
    dx = float(abs(crossline_values[1] - crossline_values[0])) if len(crossline_values) > 1 else 1.0
    dy = float(abs(inline_values[1] - inline_values[0])) if len(inline_values) > 1 else 1.0
    cell_area = max(dx * dy, 1e-6)
    print(
        f"[Build Model Mask] source summary: name={debug_label} "
        f"points={polydata.GetNumberOfPoints()} polys={polydata.GetNumberOfPolys()} "
        f"z_range=({z_min:.3f}, {z_max:.3f}) "
        f"grid=({len(crossline_values)}x{len(inline_values)}x{len(samples)}) "
        f"cell_size=({dx:.3f}x{dy:.3f})",
        flush=True,
    )

    logged_slice_count = 0
    active_slice_count = 0
    total_slice_voxels = 0

    for sample_index, sample_value in enumerate(samples):
        if sample_value < z_min - 1e-4 or sample_value > z_max + 1e-4:
            continue
        polygons = extract_slice_polygons(polydata, float(sample_value))
        if not polygons:
            continue
        active_slice_count += 1
        slice_mask = np.zeros((len(inline_values), len(crossline_values)), dtype=bool)
        polygon_vertex_counts: list[int] = []
        polygon_bbox_summaries: list[str] = []
        polygon_area_summaries: list[str] = []
        for polygon in polygons:
            polygon_vertex_counts.append(int(polygon.shape[0]))
            polygon_area = abs(polygon_signed_area_xy(polygon))
            polygon_area_summaries.append(
                f"{polygon_area:.1f}->{polygon_area / cell_area:.2f}cells"
            )
            polygon_bbox_summaries.append(
                f"({float(np.min(polygon[:, 0])):.1f},{float(np.max(polygon[:, 0])):.1f})x"
                f"({float(np.min(polygon[:, 1])):.1f},{float(np.max(polygon[:, 1])):.1f})"
            )
            slice_mask ^= polygon_mask_on_grid(polygon, grid_x, grid_y)
        filled_cells = int(np.count_nonzero(slice_mask))
        total_slice_voxels += filled_cells
        output[:, :, sample_index][slice_mask] = 1.0
        should_log = (
            logged_slice_count < 12
            or filled_cells <= 3
            or (polygons and max(polygon_vertex_counts, default=0) >= 100)
            or sample_index in {0, len(samples) // 2, len(samples) - 1}
        )
        if should_log:
            print(
                f"[Build Model Mask] slice sample_idx={sample_index} z={float(sample_value):.3f} "
                f"polygons={len(polygons)} vertices={polygon_vertex_counts} "
                f"filled_cells={filled_cells} areas={polygon_area_summaries[:4]} "
                f"bboxes={polygon_bbox_summaries[:4]}",
                flush=True,
            )
            logged_slice_count += 1

    filled_voxels = int(np.count_nonzero(output))
    print(
        f"[Build Model Mask] summary: active_slices={active_slice_count} "
        f"slice_hits={total_slice_voxels} filled_voxels={filled_voxels}",
        flush=True,
    )

    return output


def clip_polygon_to_grid(polygon_points: np.ndarray, grid_definition: GridDefinition) -> np.ndarray:
    points = normalize_polygon_grid_points(polygon_points)
    if points.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)

    inline_min = min(float(grid_definition.inline_start), float(grid_definition.inline_end))
    inline_max = max(float(grid_definition.inline_start), float(grid_definition.inline_end))
    crossline_min = min(float(grid_definition.crossline_start), float(grid_definition.crossline_end))
    crossline_max = max(float(grid_definition.crossline_start), float(grid_definition.crossline_end))

    def inside(point: np.ndarray, edge: str) -> bool:
        if edge == "left":
            return float(point[1]) >= crossline_min
        if edge == "right":
            return float(point[1]) <= crossline_max
        if edge == "bottom":
            return float(point[0]) >= inline_min
        return float(point[0]) <= inline_max

    def intersect(start: np.ndarray, end: np.ndarray, edge: str) -> np.ndarray:
        start_inline = float(start[0])
        start_crossline = float(start[1])
        end_inline = float(end[0])
        end_crossline = float(end[1])
        delta_inline = end_inline - start_inline
        delta_crossline = end_crossline - start_crossline

        if edge == "left":
            target_crossline = crossline_min
            t = 0.0 if abs(delta_crossline) <= 1e-6 else (target_crossline - start_crossline) / delta_crossline
            return np.asarray([start_inline + t * delta_inline, target_crossline], dtype=np.float32)
        if edge == "right":
            target_crossline = crossline_max
            t = 0.0 if abs(delta_crossline) <= 1e-6 else (target_crossline - start_crossline) / delta_crossline
            return np.asarray([start_inline + t * delta_inline, target_crossline], dtype=np.float32)
        if edge == "bottom":
            target_inline = inline_min
            t = 0.0 if abs(delta_inline) <= 1e-6 else (target_inline - start_inline) / delta_inline
            return np.asarray([target_inline, start_crossline + t * delta_crossline], dtype=np.float32)
        target_inline = inline_max
        t = 0.0 if abs(delta_inline) <= 1e-6 else (target_inline - start_inline) / delta_inline
        return np.asarray([target_inline, start_crossline + t * delta_crossline], dtype=np.float32)

    clipped = np.asarray(points, dtype=np.float32)
    for edge in ("left", "right", "bottom", "top"):
        if clipped.shape[0] == 0:
            break
        output: list[np.ndarray] = []
        for index, end_point in enumerate(clipped):
            start_point = clipped[index - 1]
            start_inside = inside(start_point, edge)
            end_inside = inside(end_point, edge)
            if start_inside and end_inside:
                output.append(np.asarray(end_point, dtype=np.float32))
            elif start_inside and not end_inside:
                output.append(intersect(start_point, end_point, edge))
            elif not start_inside and end_inside:
                output.append(intersect(start_point, end_point, edge))
                output.append(np.asarray(end_point, dtype=np.float32))
        clipped = normalize_polygon_grid_points(np.asarray(output, dtype=np.float32)) if output else np.zeros((0, 2), dtype=np.float32)
    return np.asarray(clipped, dtype=np.float32)


def polygon_grid_to_xyz(polygon_points: np.ndarray, z_values: np.ndarray | None = None) -> np.ndarray:
    normalized_points = normalize_polygon_grid_points(polygon_points)
    if normalized_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xyz_points = np.zeros((normalized_points.shape[0], 3), dtype=np.float32)
    z_array = (
        np.zeros(normalized_points.shape[0], dtype=np.float32)
        if z_values is None
        else np.asarray(z_values, dtype=np.float32).ravel()
    )
    if z_array.size != normalized_points.shape[0]:
        raise ValueError("Polygon z values must match polygon point count.")
    for index, point in enumerate(normalized_points):
        xyz_points[index] = (float(point[1]), float(point[0]), float(z_array[index]))
    return xyz_points


def create_polygon_actors(
    color_rgb: tuple[int, int, int],
    polygon_points: np.ndarray,
    z_values: np.ndarray | None = None,
) -> tuple[
    vtk.vtkActor,
    vtk.vtkPolyData,
    vtk.vtkPolyDataMapper,
    vtk.vtkActor,
    vtk.vtkPolyData,
    vtk.vtkPolyDataMapper,
]:
    polygon_xyz = polygon_grid_to_xyz(polygon_points, z_values=z_values)
    if polygon_xyz.shape[0] == 0:
        polygon_xyz = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    line_points = vtk.vtkPoints()
    line_points.SetData(numpy_support.numpy_to_vtk(polygon_xyz, deep=True))
    lines = vtk.vtkCellArray()
    point_count = polygon_xyz.shape[0]
    if point_count >= 2:
        segment_count = point_count if point_count >= 3 else 1
        for index in range(segment_count):
            next_index = (index + 1) % point_count
            if point_count == 2 and index > 0:
                break
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, index)
            line.GetPointIds().SetId(1, next_index)
            lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(line_points)
    polydata.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2.0)
    actor.GetProperty().SetOpacity(0.95)
    actor.GetProperty().SetColor(
        float(color_rgb[0]) / 255.0,
        float(color_rgb[1]) / 255.0,
        float(color_rgb[2]) / 255.0,
    )

    point_points = vtk.vtkPoints()
    point_points.SetData(numpy_support.numpy_to_vtk(polygon_xyz, deep=True))
    verts = vtk.vtkCellArray()
    for index in range(point_count):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(index)

    point_polydata = vtk.vtkPolyData()
    point_polydata.SetPoints(point_points)
    point_polydata.SetVerts(verts)

    point_mapper = vtk.vtkPolyDataMapper()
    point_mapper.SetInputData(point_polydata)
    point_mapper.ScalarVisibilityOff()

    point_actor = vtk.vtkActor()
    point_actor.SetMapper(point_mapper)
    point_actor.GetProperty().SetRepresentationToPoints()
    point_actor.GetProperty().RenderPointsAsSpheresOn()
    point_actor.GetProperty().SetPointSize(11.0)
    point_actor.GetProperty().SetOpacity(1.0)
    point_actor.GetProperty().SetColor(1.0, 0.96, 0.35)
    point_actor.SetVisibility(False)
    return actor, polydata, mapper, point_actor, point_polydata, point_mapper


def create_grid_image(definition: GridDefinition) -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    x_span = float(definition.crossline_end - definition.crossline_start)
    y_span = float(definition.inline_end - definition.inline_start)
    z_span = float(definition.sample_end - definition.sample_start)
    x_dim = 2 if abs(x_span) > 1e-9 else 1
    y_dim = 2 if abs(y_span) > 1e-9 else 1
    z_dim = 2 if abs(z_span) > 1e-9 else 1
    image.SetDimensions(x_dim, y_dim, z_dim)
    x_spacing = x_span if x_dim > 1 else 1.0
    y_spacing = y_span if y_dim > 1 else 1.0
    z_spacing = z_span if z_dim > 1 else 1.0
    image.SetSpacing(x_spacing, y_spacing, z_spacing)
    image.SetOrigin(
        float(definition.crossline_start),
        float(definition.inline_start),
        float(definition.sample_start),
    )
    scalars = numpy_support.numpy_to_vtk(np.zeros(x_dim * y_dim * z_dim, dtype=np.float32), deep=True)
    image.GetPointData().SetScalars(scalars)
    return image


def create_grid_actor(definition: GridDefinition) -> vtk.vtkActor:
    crossline_values = definition.crossline_values
    inline_values = definition.inline_values
    sample_values = definition.sample_values
    x_min = float(crossline_values[0])
    x_max = float(crossline_values[-1])
    y_min = float(inline_values[0])
    y_max = float(inline_values[-1])
    z_min = float(sample_values[0])
    z_max = float(sample_values[-1])

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    def add_line(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> None:
        start_id = points.InsertNextPoint(x0, y0, z0)
        end_id = points.InsertNextPoint(x1, y1, z1)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(start_id)
        lines.InsertCellPoint(end_id)

    # Draw only the outer box edges.
    add_line(x_min, y_min, z_min, x_max, y_min, z_min)
    add_line(x_min, y_max, z_min, x_max, y_max, z_min)
    add_line(x_min, y_min, z_max, x_max, y_min, z_max)
    add_line(x_min, y_max, z_max, x_max, y_max, z_max)

    add_line(x_min, y_min, z_min, x_min, y_max, z_min)
    add_line(x_max, y_min, z_min, x_max, y_max, z_min)
    add_line(x_min, y_min, z_max, x_min, y_max, z_max)
    add_line(x_max, y_min, z_max, x_max, y_max, z_max)

    add_line(x_min, y_min, z_min, x_min, y_min, z_max)
    add_line(x_max, y_min, z_min, x_max, y_min, z_max)
    add_line(x_min, y_max, z_min, x_min, y_max, z_max)
    add_line(x_max, y_max, z_min, x_max, y_max, z_max)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.36, 0.76, 0.96)
    actor.GetProperty().SetOpacity(0.22)
    actor.GetProperty().SetLineWidth(1.0)
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


def polydata_to_mask(
    polydata: vtk.vtkPolyData,
    shape: tuple[int, int, int],
    spacing: RenderSpacing,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dilate_steps: int = 1,
) -> np.ndarray:
    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        return np.zeros(shape, dtype=bool)

    image = vtk.vtkImageData()
    image.SetDimensions(*shape)
    image.SetSpacing(float(spacing.xline), float(spacing.inline), float(spacing.sample))
    image.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(polydata)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()

    image_stencil = vtk.vtkImageStencil()
    image_stencil.SetInputData(image)
    image_stencil.SetStencilConnection(stencil.GetOutputPort())
    image_stencil.ReverseStencilOff()
    image_stencil.SetBackgroundValue(0)
    image_stencil.Update()

    mask = np.asarray(
        numpy_support.vtk_to_numpy(image_stencil.GetOutput().GetPointData().GetScalars()),
        dtype=np.uint8,
    ).reshape(shape, order="F") > 0

    if int(dilate_steps) <= 0:
        return np.asarray(mask, dtype=bool)

    expanded = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(dilate_steps))):
        grown = expanded.copy()
        for dx, dy, dz in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            shifted = np.zeros_like(expanded, dtype=bool)
            src_x = slice(max(0, -dx), expanded.shape[0] - max(0, dx))
            src_y = slice(max(0, -dy), expanded.shape[1] - max(0, dy))
            src_z = slice(max(0, -dz), expanded.shape[2] - max(0, dz))
            dst_x = slice(max(0, dx), expanded.shape[0] - max(0, -dx))
            dst_y = slice(max(0, dy), expanded.shape[1] - max(0, -dy))
            dst_z = slice(max(0, dz), expanded.shape[2] - max(0, -dz))
            shifted[dst_x, dst_y, dst_z] = expanded[src_x, src_y, src_z]
            grown |= shifted
        expanded = grown
    return expanded


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

        target_row = QtWidgets.QHBoxLayout()
        target_row.setSpacing(6)
        target_row.addWidget(QtWidgets.QLabel("Target"))
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("Attribute", "attribute")
        self.target_combo.addItem("Control Point", "control_point")
        target_row.addWidget(self.target_combo, stretch=1)
        layout.addLayout(target_row)

        self.control_point_use_colormap_checkbox = QtWidgets.QCheckBox("Use Value Colormap")
        layout.addWidget(self.control_point_use_colormap_checkbox)

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

    def current_target(self) -> str:
        return str(self.target_combo.currentData() or "attribute")

    def set_target(self, target: str) -> None:
        index = self.target_combo.findData(target)
        if index < 0:
            index = self.target_combo.findData("attribute")
        self.target_combo.blockSignals(True)
        if index >= 0:
            self.target_combo.setCurrentIndex(index)
        self.target_combo.blockSignals(False)

    def set_target_enabled(self, enabled: bool) -> None:
        self.target_combo.setEnabled(enabled)

    def set_control_point_toggle_visible(self, visible: bool) -> None:
        self.control_point_use_colormap_checkbox.setVisible(visible)


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


class InterpolateVolumeDialog(QtWidgets.QDialog):
    def __init__(
        self,
        attribute_names: list[str],
        horizon_names: list[str],
        selected_attribute_name: str | None = None,
        selected_horizon_name: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Interpolate Attribute Volume")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        self.attribute_combo = QtWidgets.QComboBox()
        for name in attribute_names:
            self.attribute_combo.addItem(name, name)
        if selected_attribute_name is not None:
            index = self.attribute_combo.findData(selected_attribute_name)
            if index >= 0:
                self.attribute_combo.setCurrentIndex(index)
        form.addRow("Grid Attribute", self.attribute_combo)

        self.horizon_combo = QtWidgets.QComboBox()
        for name in horizon_names:
            self.horizon_combo.addItem(name, name)
        if selected_horizon_name is not None:
            index = self.horizon_combo.findData(selected_horizon_name)
            if index >= 0:
                self.horizon_combo.setCurrentIndex(index)
        form.addRow("Value Horizon", self.horizon_combo)

        self.output_name_edit = QtWidgets.QLineEdit()
        self.mask_checkbox = QtWidgets.QCheckBox("Mask By Horizon")
        self.mask_checkbox.setChecked(True)
        self.idw_radius_edit = QtWidgets.QLineEdit("0")
        self.idw_radius_edit.setValidator(QtGui.QDoubleValidator(0.0, 1e12, 6, self))
        form.addRow("Output Name", self.output_name_edit)
        form.addRow("IDW Radius", self.idw_radius_edit)
        form.addRow("", self.mask_checkbox)
        layout.addLayout(form)

        self.attribute_combo.currentIndexChanged.connect(self._update_default_name)
        self.horizon_combo.currentIndexChanged.connect(self._update_default_name)
        self._update_default_name()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_default_name(self) -> None:
        attribute_name = self.attribute_combo.currentData()
        horizon_name = self.horizon_combo.currentData()
        if attribute_name is None or horizon_name is None:
            return
        if not self.output_name_edit.text().strip():
            self.output_name_edit.setText(f"{attribute_name}_{horizon_name}_interp")

    def values(self) -> tuple[str, str, str, float, bool] | None:
        attribute_name = self.attribute_combo.currentData()
        horizon_name = self.horizon_combo.currentData()
        if attribute_name is None or horizon_name is None:
            return None
        output_name = self.output_name_edit.text().strip()
        if not output_name:
            output_name = f"{attribute_name}_{horizon_name}_interp"
        radius_text = self.idw_radius_edit.text().strip()
        try:
            radius = 0.0 if not radius_text else max(0.0, float(radius_text))
        except ValueError:
            return None
        return str(attribute_name), str(horizon_name), str(output_name), float(radius), bool(self.mask_checkbox.isChecked())


class ExtractCurrentHorizonMaskDialog(QtWidgets.QDialog):
    def __init__(
        self,
        horizon_name: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract Horizon Mask")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.horizon_label = QtWidgets.QLabel(horizon_name)
        self.output_name_edit = QtWidgets.QLineEdit(f"{horizon_name}_mask")
        form.addRow("Horizon", self.horizon_label)
        form.addRow("Output Name", self.output_name_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> str | None:
        output_name = self.output_name_edit.text().strip()
        if not output_name:
            return None
        return output_name


class ReplaceVolumeByHorizonDialog(QtWidgets.QDialog):
    def __init__(
        self,
        attribute_names: list[str],
        horizon_names: list[str],
        selected_target_attribute_name: str | None = None,
        selected_source_attribute_name: str | None = None,
        selected_horizon_name: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Replace By Horizon Mask")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        self.target_attribute_combo = QtWidgets.QComboBox()
        for name in attribute_names:
            self.target_attribute_combo.addItem(name, name)
        if selected_target_attribute_name is not None:
            index = self.target_attribute_combo.findData(selected_target_attribute_name)
            if index >= 0:
                self.target_attribute_combo.setCurrentIndex(index)
        form.addRow("Target Attribute", self.target_attribute_combo)

        self.source_attribute_combo = QtWidgets.QComboBox()
        for name in attribute_names:
            self.source_attribute_combo.addItem(name, name)
        if selected_source_attribute_name is not None:
            index = self.source_attribute_combo.findData(selected_source_attribute_name)
            if index >= 0:
                self.source_attribute_combo.setCurrentIndex(index)
        elif self.source_attribute_combo.count() > 1:
            self.source_attribute_combo.setCurrentIndex(1)
        form.addRow("Source Attribute", self.source_attribute_combo)

        self.horizon_combo = QtWidgets.QComboBox()
        for name in horizon_names:
            self.horizon_combo.addItem(name, name)
        if selected_horizon_name is not None:
            index = self.horizon_combo.findData(selected_horizon_name)
            if index >= 0:
                self.horizon_combo.setCurrentIndex(index)
        form.addRow("Horizon", self.horizon_combo)

        self.output_name_edit = QtWidgets.QLineEdit()
        form.addRow("Output Name", self.output_name_edit)
        layout.addLayout(form)

        self.target_attribute_combo.currentIndexChanged.connect(self._update_default_name)
        self.source_attribute_combo.currentIndexChanged.connect(self._update_default_name)
        self.horizon_combo.currentIndexChanged.connect(self._update_default_name)
        self._update_default_name()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_default_name(self) -> None:
        if self.output_name_edit.text().strip():
            return
        target_name = self.target_attribute_combo.currentData()
        source_name = self.source_attribute_combo.currentData()
        horizon_name = self.horizon_combo.currentData()
        if target_name is None or source_name is None or horizon_name is None:
            return
        self.output_name_edit.setText(f"{target_name}_replaced_by_{source_name}_{horizon_name}")

    def values(self) -> tuple[str, str, str, str] | None:
        target_name = self.target_attribute_combo.currentData()
        source_name = self.source_attribute_combo.currentData()
        horizon_name = self.horizon_combo.currentData()
        output_name = self.output_name_edit.text().strip()
        if target_name is None or source_name is None or horizon_name is None:
            return None
        if not output_name:
            output_name = f"{target_name}_replaced_by_{source_name}_{horizon_name}"
        return str(target_name), str(source_name), str(horizon_name), str(output_name)


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


class BuildModelWindow(QtWidgets.QDialog):
    define_grid_requested = QtCore.Signal()
    load_dip_direction_requested = QtCore.Signal()
    load_elev_requested = QtCore.Signal()
    load_geomap_requested = QtCore.Signal()
    build_polygon_surface_requested = QtCore.Signal()
    build_model_volume_requested = QtCore.Signal()
    build_selected_model_mask_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Model")
        self.setModal(False)
        self.resize(520, 420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QLabel("Build Model")
        header_font = QtGui.QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        self.settings = QtCore.QSettings("wesi3d", APP_NAME)

        self.grid_group = QtWidgets.QGroupBox("Define Grid")
        grid_layout = QtWidgets.QVBoxLayout(self.grid_group)
        grid_layout.setContentsMargins(8, 6, 8, 6)
        grid_layout.setSpacing(6)
        grid_form = QtWidgets.QGridLayout()
        grid_form.setHorizontalSpacing(6)
        grid_form.setVerticalSpacing(4)
        float_validator = QtGui.QDoubleValidator(-1e12, 1e12, 6, self)

        self.inline_start_edit = QtWidgets.QLineEdit()
        self.inline_end_edit = QtWidgets.QLineEdit()
        self.crossline_start_edit = QtWidgets.QLineEdit()
        self.crossline_end_edit = QtWidgets.QLineEdit()
        self.sample_start_edit = QtWidgets.QLineEdit()
        self.sample_end_edit = QtWidgets.QLineEdit()
        self.inline_size_edit = QtWidgets.QLineEdit()
        self.crossline_size_edit = QtWidgets.QLineEdit()
        self.sample_size_edit = QtWidgets.QLineEdit()
        for widget in (
            self.inline_start_edit,
            self.inline_end_edit,
            self.crossline_start_edit,
            self.crossline_end_edit,
            self.sample_start_edit,
            self.sample_end_edit,
            self.inline_size_edit,
            self.crossline_size_edit,
            self.sample_size_edit,
        ):
            widget.setValidator(float_validator)
        grid_form.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        grid_form.addWidget(QtWidgets.QLabel("Start"), 0, 1)
        grid_form.addWidget(QtWidgets.QLabel("End"), 0, 2)
        grid_form.addWidget(QtWidgets.QLabel("Size"), 0, 3)
        grid_form.addWidget(QtWidgets.QLabel("Inline"), 1, 0)
        grid_form.addWidget(self.inline_start_edit, 1, 1)
        grid_form.addWidget(self.inline_end_edit, 1, 2)
        grid_form.addWidget(self.inline_size_edit, 1, 3)
        grid_form.addWidget(QtWidgets.QLabel("Cxline"), 2, 0)
        grid_form.addWidget(self.crossline_start_edit, 2, 1)
        grid_form.addWidget(self.crossline_end_edit, 2, 2)
        grid_form.addWidget(self.crossline_size_edit, 2, 3)
        grid_form.addWidget(QtWidgets.QLabel("Sample"), 3, 0)
        grid_form.addWidget(self.sample_start_edit, 3, 1)
        grid_form.addWidget(self.sample_end_edit, 3, 2)
        grid_form.addWidget(self.sample_size_edit, 3, 3)
        grid_layout.addLayout(grid_form)
        self.define_grid_button = QtWidgets.QPushButton("Define Grid")
        self.define_grid_button.clicked.connect(self.define_grid_requested.emit)
        grid_layout.addWidget(self.define_grid_button)
        layout.addWidget(self.grid_group)

        self.scatter_group = QtWidgets.QGroupBox("Load Dip/Direction")
        scatter_layout = QtWidgets.QVBoxLayout(self.scatter_group)
        scatter_layout.setContentsMargins(8, 6, 8, 6)
        scatter_layout.setSpacing(6)
        scatter_form = QtWidgets.QFormLayout()
        self.dip_path_edit = QtWidgets.QLineEdit()
        self.direction_path_edit = QtWidgets.QLineEdit()
        dip_row = QtWidgets.QHBoxLayout()
        dip_row.setSpacing(6)
        dip_row.addWidget(self.dip_path_edit, stretch=1)
        self.browse_dip_button = QtWidgets.QPushButton("Browse")
        self.browse_dip_button.setMaximumWidth(72)
        self.browse_dip_button.clicked.connect(self._browse_dip_path)
        dip_row.addWidget(self.browse_dip_button)
        direction_row = QtWidgets.QHBoxLayout()
        direction_row.setSpacing(6)
        direction_row.addWidget(self.direction_path_edit, stretch=1)
        self.browse_direction_button = QtWidgets.QPushButton("Browse")
        self.browse_direction_button.setMaximumWidth(72)
        self.browse_direction_button.clicked.connect(self._browse_direction_path)
        direction_row.addWidget(self.browse_direction_button)
        scatter_form.addRow("Dip File", dip_row)
        scatter_form.addRow("Direction File", direction_row)
        scatter_layout.addLayout(scatter_form)
        self.load_dip_direction_button = QtWidgets.QPushButton("Load Dip/Direction")
        self.load_dip_direction_button.clicked.connect(self.load_dip_direction_requested.emit)
        scatter_layout.addWidget(self.load_dip_direction_button)
        layout.addWidget(self.scatter_group)

        self.elev_group = QtWidgets.QGroupBox("Load Elev")
        elev_layout = QtWidgets.QVBoxLayout(self.elev_group)
        elev_layout.setContentsMargins(8, 6, 8, 6)
        elev_layout.setSpacing(6)
        elev_row = QtWidgets.QHBoxLayout()
        elev_row.setSpacing(6)
        self.elev_path_edit = QtWidgets.QLineEdit()
        elev_row.addWidget(self.elev_path_edit, stretch=1)
        self.browse_elev_button = QtWidgets.QPushButton("Browse")
        self.browse_elev_button.setMaximumWidth(72)
        self.browse_elev_button.clicked.connect(self._browse_elev_path)
        elev_row.addWidget(self.browse_elev_button)
        elev_layout.addLayout(elev_row)
        self.load_elev_button = QtWidgets.QPushButton("Load Elev")
        self.load_elev_button.clicked.connect(self.load_elev_requested.emit)
        elev_layout.addWidget(self.load_elev_button)
        layout.addWidget(self.elev_group)

        self.geomap_group = QtWidgets.QGroupBox("Load Geomap")
        geomap_layout = QtWidgets.QVBoxLayout(self.geomap_group)
        geomap_layout.setContentsMargins(8, 6, 8, 6)
        geomap_layout.setSpacing(6)
        geomap_row = QtWidgets.QHBoxLayout()
        geomap_row.setSpacing(6)
        self.geomap_path_edit = QtWidgets.QLineEdit()
        geomap_row.addWidget(self.geomap_path_edit, stretch=1)
        self.browse_geomap_button = QtWidgets.QPushButton("Browse")
        self.browse_geomap_button.setMaximumWidth(72)
        self.browse_geomap_button.clicked.connect(self._browse_geomap_path)
        geomap_row.addWidget(self.browse_geomap_button)
        geomap_layout.addLayout(geomap_row)
        geomap_elev_row = QtWidgets.QHBoxLayout()
        geomap_elev_row.setSpacing(6)
        self.geomap_elev_path_edit = QtWidgets.QLineEdit()
        geomap_elev_row.addWidget(self.geomap_elev_path_edit, stretch=1)
        self.browse_geomap_elev_button = QtWidgets.QPushButton("Browse")
        self.browse_geomap_elev_button.setMaximumWidth(72)
        self.browse_geomap_elev_button.clicked.connect(self._browse_geomap_elev_path)
        geomap_elev_row.addWidget(self.browse_geomap_elev_button)
        geomap_layout.addWidget(QtWidgets.QLabel("Elev File"))
        geomap_layout.addLayout(geomap_elev_row)
        self.load_geomap_button = QtWidgets.QPushButton("Load Geomap")
        self.load_geomap_button.clicked.connect(self.load_geomap_requested.emit)
        geomap_layout.addWidget(self.load_geomap_button)
        layout.addWidget(self.geomap_group, stretch=1)

        self.surface_group = QtWidgets.QGroupBox("Extend Polygon")
        surface_layout = QtWidgets.QVBoxLayout(self.surface_group)
        surface_layout.setContentsMargins(8, 6, 8, 6)
        surface_layout.setSpacing(6)
        surface_form = QtWidgets.QGridLayout()
        surface_form.setHorizontalSpacing(6)
        surface_form.setVerticalSpacing(4)
        int_validator = QtGui.QIntValidator(1, 10**6, self)
        self.surface_sample_count_edit = QtWidgets.QLineEdit()
        self.surface_layer_step_edit = QtWidgets.QLineEdit()
        self.surface_smooth_iterations_edit = QtWidgets.QLineEdit()
        self.surface_sample_count_edit.setValidator(int_validator)
        self.surface_smooth_iterations_edit.setValidator(int_validator)
        self.surface_layer_step_edit.setValidator(float_validator)
        surface_form.addWidget(QtWidgets.QLabel("Samples"), 0, 0)
        surface_form.addWidget(self.surface_sample_count_edit, 0, 1)
        surface_form.addWidget(QtWidgets.QLabel("Z Step"), 0, 2)
        surface_form.addWidget(self.surface_layer_step_edit, 0, 3)
        surface_form.addWidget(QtWidgets.QLabel("Smooth"), 1, 0)
        surface_form.addWidget(self.surface_smooth_iterations_edit, 1, 1)
        surface_form.addWidget(QtWidgets.QLabel("Depth"), 1, 2)
        surface_form.addWidget(QtWidgets.QLabel("Grid Bottom"), 1, 3)
        surface_layout.addLayout(surface_form)
        self.build_polygon_surface_button = QtWidgets.QPushButton("Build Surface")
        self.build_polygon_surface_button.clicked.connect(self.build_polygon_surface_requested.emit)
        surface_layout.addWidget(self.build_polygon_surface_button)
        layout.addWidget(self.surface_group)

        self.volume_group = QtWidgets.QGroupBox("Build Volume")
        volume_layout = QtWidgets.QVBoxLayout(self.volume_group)
        volume_layout.setContentsMargins(8, 6, 8, 6)
        volume_layout.setSpacing(6)
        volume_form = QtWidgets.QFormLayout()
        model_elev_row = QtWidgets.QHBoxLayout()
        model_elev_row.setSpacing(6)
        self.model_elev_path_edit = QtWidgets.QLineEdit()
        model_elev_row.addWidget(self.model_elev_path_edit, stretch=1)
        self.browse_model_elev_button = QtWidgets.QPushButton("Browse")
        self.browse_model_elev_button.setMaximumWidth(72)
        self.browse_model_elev_button.clicked.connect(self._browse_model_elev_path)
        model_elev_row.addWidget(self.browse_model_elev_button)
        volume_form.addRow("Elev File", model_elev_row)
        self.model_output_name_edit = QtWidgets.QLineEdit()
        volume_form.addRow("Output Name", self.model_output_name_edit)
        volume_layout.addLayout(volume_form)
        self.build_model_volume_button = QtWidgets.QPushButton("Build Model")
        self.build_model_volume_button.clicked.connect(self.build_model_volume_requested.emit)
        volume_layout.addWidget(self.build_model_volume_button)
        self.build_selected_model_mask_button = QtWidgets.QPushButton("Build Selected Model Mask")
        self.build_selected_model_mask_button.clicked.connect(self.build_selected_model_mask_requested.emit)
        volume_layout.addWidget(self.build_selected_model_mask_button)
        layout.addWidget(self.volume_group)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.hide)
        layout.addWidget(self.close_button)

        self._load_history()

    def _load_history(self) -> None:
        self.inline_start_edit.setText(str(self.settings.value("build_model/grid/inline_start", "0")))
        self.inline_end_edit.setText(str(self.settings.value("build_model/grid/inline_end", "100")))
        self.crossline_start_edit.setText(str(self.settings.value("build_model/grid/crossline_start", "0")))
        self.crossline_end_edit.setText(str(self.settings.value("build_model/grid/crossline_end", "100")))
        self.sample_start_edit.setText(str(self.settings.value("build_model/grid/sample_start", "0")))
        self.sample_end_edit.setText(str(self.settings.value("build_model/grid/sample_end", "60")))
        self.inline_size_edit.setText(str(self.settings.value("build_model/grid/inline_size", "10")))
        self.crossline_size_edit.setText(str(self.settings.value("build_model/grid/crossline_size", "10")))
        self.sample_size_edit.setText(str(self.settings.value("build_model/grid/sample_size", "10")))
        self.dip_path_edit.setText(str(self.settings.value("build_model/scatter/dip_path", "")))
        self.direction_path_edit.setText(str(self.settings.value("build_model/scatter/direction_path", "")))
        self.elev_path_edit.setText(str(self.settings.value("build_model/elev/path", "")))
        self.model_elev_path_edit.setText(str(self.settings.value("build_model/elev/path", "")))
        self.geomap_path_edit.setText(str(self.settings.value("build_model/polygon/geomap_path", "")))
        self.geomap_elev_path_edit.setText(str(self.settings.value("build_model/polygon/elev_path", "")))
        self.surface_sample_count_edit.setText(str(self.settings.value("build_model/surface/sample_count", "160")))
        self.surface_layer_step_edit.setText(str(self.settings.value("build_model/surface/layer_step", "4")))
        self.surface_smooth_iterations_edit.setText(str(self.settings.value("build_model/surface/smooth_iterations", "2")))
        self.model_output_name_edit.setText(str(self.settings.value("build_model/volume/output_name", "model_volume")))

    def save_grid_history(self, definition: GridDefinition) -> None:
        self.settings.setValue("build_model/grid/inline_start", definition.inline_start)
        self.settings.setValue("build_model/grid/inline_end", definition.inline_end)
        self.settings.setValue("build_model/grid/crossline_start", definition.crossline_start)
        self.settings.setValue("build_model/grid/crossline_end", definition.crossline_end)
        self.settings.setValue("build_model/grid/sample_start", definition.sample_start)
        self.settings.setValue("build_model/grid/sample_end", definition.sample_end)
        self.settings.setValue("build_model/grid/inline_size", definition.inline_size)
        self.settings.setValue("build_model/grid/crossline_size", definition.crossline_size)
        self.settings.setValue("build_model/grid/sample_size", definition.sample_size)

    def save_scatter_history(self, dip_path: str, direction_path: str) -> None:
        self.settings.setValue("build_model/scatter/dip_path", dip_path)
        self.settings.setValue("build_model/scatter/direction_path", direction_path)

    def save_elev_history(self, elev_path: str) -> None:
        self.settings.setValue("build_model/elev/path", elev_path)
        self.elev_path_edit.setText(elev_path)
        self.model_elev_path_edit.setText(elev_path)

    def save_geomap_history(self, geomap_path: str, elev_path: str) -> None:
        self.settings.setValue("build_model/polygon/geomap_path", geomap_path)
        self.settings.setValue("build_model/polygon/elev_path", elev_path)

    def save_surface_history(
        self,
        *,
        sample_count: int,
        layer_step: float,
        smooth_iterations: int,
    ) -> None:
        self.settings.setValue("build_model/surface/sample_count", int(sample_count))
        self.settings.setValue("build_model/surface/layer_step", float(layer_step))
        self.settings.setValue("build_model/surface/smooth_iterations", int(smooth_iterations))

    def save_model_volume_history(self, output_name: str) -> None:
        self.settings.setValue("build_model/volume/output_name", output_name)

    def current_grid_definition(self) -> GridDefinition | None:
        try:
            return GridDefinition(
                inline_start=float(self.inline_start_edit.text().strip() or "0"),
                inline_end=float(self.inline_end_edit.text().strip() or "0"),
                crossline_start=float(self.crossline_start_edit.text().strip() or "0"),
                crossline_end=float(self.crossline_end_edit.text().strip() or "0"),
                sample_start=float(self.sample_start_edit.text().strip() or "0"),
                sample_end=float(self.sample_end_edit.text().strip() or "0"),
                inline_size=max(1e-6, abs(float(self.inline_size_edit.text().strip() or "1"))),
                crossline_size=max(1e-6, abs(float(self.crossline_size_edit.text().strip() or "1"))),
                sample_size=max(1e-6, abs(float(self.sample_size_edit.text().strip() or "1"))),
            )
        except ValueError:
            return None

    def current_scatter_paths(self) -> tuple[str, str]:
        return self.dip_path_edit.text().strip(), self.direction_path_edit.text().strip()

    def current_geomap_inputs(self) -> tuple[str, str]:
        return self.geomap_path_edit.text().strip(), self.geomap_elev_path_edit.text().strip()

    def current_elev_path(self) -> str:
        return self.elev_path_edit.text().strip()

    def current_model_elev_path(self) -> str:
        path = self.model_elev_path_edit.text().strip()
        return path or self.elev_path_edit.text().strip()

    def current_model_output_name(self) -> str:
        return self.model_output_name_edit.text().strip() or "model_volume"

    def current_surface_options(self) -> dict[str, float] | None:
        try:
            return {
                "sample_count": max(16, int(self.surface_sample_count_edit.text().strip() or "160")),
                "layer_step": max(1e-3, abs(float(self.surface_layer_step_edit.text().strip() or "4"))),
                "smooth_iterations": max(0, int(self.surface_smooth_iterations_edit.text().strip() or "2")),
            }
        except ValueError:
            return None

    def _browse_dip_path(self) -> None:
        start_dir = str(Path(self.dip_path_edit.text().strip()).expanduser().parent) if self.dip_path_edit.text().strip() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Dip File",
            start_dir,
            "GMP Files (*.gmp);;Text Files (*.txt *.dat *.csv);;All Files (*)",
        )
        if path:
            self.dip_path_edit.setText(path)
            self.save_scatter_history(path, self.direction_path_edit.text().strip())

    def _browse_direction_path(self) -> None:
        start_dir = str(Path(self.direction_path_edit.text().strip()).expanduser().parent) if self.direction_path_edit.text().strip() else ""
        if not start_dir and self.dip_path_edit.text().strip():
            start_dir = str(Path(self.dip_path_edit.text().strip()).expanduser().parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Direction File",
            start_dir,
            "GMP Files (*.gmp);;Text Files (*.txt *.dat *.csv);;All Files (*)",
        )
        if path:
            self.direction_path_edit.setText(path)
            self.save_scatter_history(self.dip_path_edit.text().strip(), path)

    def _browse_elev_path(self) -> None:
        start_dir = str(Path(self.elev_path_edit.text().strip()).expanduser().parent) if self.elev_path_edit.text().strip() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Elev File",
            start_dir,
            "GMP Files (*.gmp);;Text Files (*.txt *.dat *.csv);;All Files (*)",
        )
        if path:
            self.save_elev_history(path)

    def _browse_model_elev_path(self) -> None:
        start_dir = (
            str(Path(self.model_elev_path_edit.text().strip()).expanduser().parent)
            if self.model_elev_path_edit.text().strip()
            else ""
        )
        if not start_dir and self.elev_path_edit.text().strip():
            start_dir = str(Path(self.elev_path_edit.text().strip()).expanduser().parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model Elev File",
            start_dir,
            "GMP Files (*.gmp);;Text Files (*.txt *.dat *.csv);;All Files (*)",
        )
        if path:
            self.save_elev_history(path)

    def _browse_geomap_path(self) -> None:
        start_dir = str(Path(self.geomap_path_edit.text().strip()).expanduser().parent) if self.geomap_path_edit.text().strip() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Geomap File",
            start_dir,
            "GMP Files (*.gmp);;Text Files (*.txt *.dat *.csv);;All Files (*)",
        )
        if path:
            self.geomap_path_edit.setText(path)
            self.save_geomap_history(path, self.geomap_elev_path_edit.text().strip())

    def _browse_geomap_elev_path(self) -> None:
        start_dir = (
            str(Path(self.geomap_elev_path_edit.text().strip()).expanduser().parent)
            if self.geomap_elev_path_edit.text().strip()
            else ""
        )
        if not start_dir and self.geomap_path_edit.text().strip():
            start_dir = str(Path(self.geomap_path_edit.text().strip()).expanduser().parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Geomap Elev File",
            start_dir,
            "GMP Files (*.gmp);;Text Files (*.txt *.dat *.csv);;All Files (*)",
        )
        if path:
            self.geomap_elev_path_edit.setText(path)
            self.save_geomap_history(self.geomap_path_edit.text().strip(), path)


class DefineGridDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Define Grid")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        float_validator = QtGui.QDoubleValidator(-1e12, 1e12, 6, self)

        self.inline_start_edit = QtWidgets.QLineEdit("0")
        self.inline_end_edit = QtWidgets.QLineEdit("100")
        self.crossline_start_edit = QtWidgets.QLineEdit("0")
        self.crossline_end_edit = QtWidgets.QLineEdit("100")
        self.sample_start_edit = QtWidgets.QLineEdit("0")
        self.sample_end_edit = QtWidgets.QLineEdit("60")
        self.inline_size_edit = QtWidgets.QLineEdit("10")
        self.crossline_size_edit = QtWidgets.QLineEdit("10")
        self.sample_size_edit = QtWidgets.QLineEdit("10")

        for widget in (
            self.inline_start_edit,
            self.inline_end_edit,
            self.crossline_start_edit,
            self.crossline_end_edit,
            self.sample_start_edit,
            self.sample_end_edit,
            self.inline_size_edit,
            self.crossline_size_edit,
            self.sample_size_edit,
        ):
            widget.setValidator(float_validator)

        form.addRow("Inline Start", self.inline_start_edit)
        form.addRow("Inline End", self.inline_end_edit)
        form.addRow("Cxline Start", self.crossline_start_edit)
        form.addRow("Cxline End", self.crossline_end_edit)
        form.addRow("Sample Start", self.sample_start_edit)
        form.addRow("Sample End", self.sample_end_edit)
        form.addRow("Inline Grid Size", self.inline_size_edit)
        form.addRow("Cxline Grid Size", self.crossline_size_edit)
        form.addRow("Sample Grid Size", self.sample_size_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> GridDefinition | None:
        try:
            return GridDefinition(
                inline_start=float(self.inline_start_edit.text().strip() or "0"),
                inline_end=float(self.inline_end_edit.text().strip() or "0"),
                crossline_start=float(self.crossline_start_edit.text().strip() or "0"),
                crossline_end=float(self.crossline_end_edit.text().strip() or "0"),
                sample_start=float(self.sample_start_edit.text().strip() or "0"),
                sample_end=float(self.sample_end_edit.text().strip() or "0"),
                inline_size=max(1e-6, abs(float(self.inline_size_edit.text().strip() or "1"))),
                crossline_size=max(1e-6, abs(float(self.crossline_size_edit.text().strip() or "1"))),
                sample_size=max(1e-6, abs(float(self.sample_size_edit.text().strip() or "1"))),
            )
        except ValueError:
            return None


class NewProjectDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setModal(True)
        self.settings = QtCore.QSettings("wesi3d", APP_NAME)
        self.last_error: str | None = None

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.project_name_edit = QtWidgets.QLineEdit()
        form.addRow("Project Name", self.project_name_edit)

        self.path_edit = QtWidgets.QLineEdit(str(Path.home()))
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_path)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_button)
        form.addRow("Path", path_row)

        float_validator = QtGui.QDoubleValidator(-1e12, 1e12, 6, self)

        self.inline_start_edit = QtWidgets.QLineEdit("0")
        self.inline_end_edit = QtWidgets.QLineEdit("100")
        self.crossline_start_edit = QtWidgets.QLineEdit("0")
        self.crossline_end_edit = QtWidgets.QLineEdit("100")
        self.sample_start_edit = QtWidgets.QLineEdit("0")
        self.sample_end_edit = QtWidgets.QLineEdit("60")
        self.datum_edit = QtWidgets.QLineEdit("0")
        self.inline_step_edit = QtWidgets.QLineEdit("10")
        self.inline_size_edit = QtWidgets.QLineEdit("10")
        self.crossline_step_edit = QtWidgets.QLineEdit("10")
        self.crossline_size_edit = QtWidgets.QLineEdit("10")
        self.sample_step_edit = QtWidgets.QLineEdit("10")
        self.sample_size_edit = QtWidgets.QLineEdit("10")
        self.inline_num_value = QtWidgets.QLabel("-")
        self.crossline_num_value = QtWidgets.QLabel("-")
        self.sample_num_value = QtWidgets.QLabel("-")
        self._load_grid_history()

        for widget in (
            self.inline_start_edit,
            self.inline_end_edit,
            self.crossline_start_edit,
            self.crossline_end_edit,
            self.sample_start_edit,
            self.sample_end_edit,
            self.datum_edit,
            self.inline_step_edit,
            self.inline_size_edit,
            self.crossline_step_edit,
            self.crossline_size_edit,
            self.sample_step_edit,
            self.sample_size_edit,
        ):
            widget.setValidator(float_validator)

        form.addRow("Datum", self.datum_edit)

        inline_row = QtWidgets.QHBoxLayout()
        inline_row.setSpacing(6)
        inline_row.addWidget(QtWidgets.QLabel("Start"))
        inline_row.addWidget(self.inline_start_edit)
        inline_row.addWidget(QtWidgets.QLabel("End"))
        inline_row.addWidget(self.inline_end_edit)
        inline_row.addWidget(QtWidgets.QLabel("Step"))
        inline_row.addWidget(self.inline_step_edit)
        inline_row.addWidget(QtWidgets.QLabel("Size"))
        inline_row.addWidget(self.inline_size_edit)
        inline_row.addWidget(QtWidgets.QLabel("Num"))
        inline_row.addWidget(self.inline_num_value)
        form.addRow("Inline", inline_row)

        crossline_row = QtWidgets.QHBoxLayout()
        crossline_row.setSpacing(6)
        crossline_row.addWidget(QtWidgets.QLabel("Start"))
        crossline_row.addWidget(self.crossline_start_edit)
        crossline_row.addWidget(QtWidgets.QLabel("End"))
        crossline_row.addWidget(self.crossline_end_edit)
        crossline_row.addWidget(QtWidgets.QLabel("Step"))
        crossline_row.addWidget(self.crossline_step_edit)
        crossline_row.addWidget(QtWidgets.QLabel("Size"))
        crossline_row.addWidget(self.crossline_size_edit)
        crossline_row.addWidget(QtWidgets.QLabel("Num"))
        crossline_row.addWidget(self.crossline_num_value)
        form.addRow("Cxline", crossline_row)

        sample_row = QtWidgets.QHBoxLayout()
        sample_row.setSpacing(6)
        sample_row.addWidget(QtWidgets.QLabel("Start"))
        sample_row.addWidget(self.sample_start_edit)
        sample_row.addWidget(QtWidgets.QLabel("End"))
        sample_row.addWidget(self.sample_end_edit)
        sample_row.addWidget(QtWidgets.QLabel("Step"))
        sample_row.addWidget(self.sample_step_edit)
        sample_row.addWidget(QtWidgets.QLabel("Size"))
        sample_row.addWidget(self.sample_size_edit)
        sample_row.addWidget(QtWidgets.QLabel("Num"))
        sample_row.addWidget(self.sample_num_value)
        form.addRow("Sample", sample_row)
        layout.addLayout(form)

        for widget in (
            self.inline_start_edit,
            self.inline_end_edit,
            self.inline_step_edit,
            self.crossline_start_edit,
            self.crossline_end_edit,
            self.crossline_step_edit,
            self.sample_start_edit,
            self.sample_end_edit,
            self.sample_step_edit,
        ):
            widget.textChanged.connect(self._update_axis_counts)
        self._update_axis_counts()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_grid_history(self) -> None:
        self.inline_start_edit.setText(str(self.settings.value("build_model/grid/inline_start", "0")))
        self.inline_end_edit.setText(str(self.settings.value("build_model/grid/inline_end", "100")))
        self.crossline_start_edit.setText(str(self.settings.value("build_model/grid/crossline_start", "0")))
        self.crossline_end_edit.setText(str(self.settings.value("build_model/grid/crossline_end", "100")))
        self.sample_start_edit.setText(str(self.settings.value("build_model/grid/sample_start", "0")))
        self.sample_end_edit.setText(str(self.settings.value("build_model/grid/sample_end", "60")))
        self.datum_edit.setText(str(self.settings.value("build_model/grid/datum", "0")))
        self.inline_step_edit.setText(str(self.settings.value("build_model/grid/inline_step", self.settings.value("build_model/grid/inline_size", "10"))))
        self.inline_size_edit.setText(str(self.settings.value("build_model/grid/inline_size", "10")))
        self.crossline_step_edit.setText(str(self.settings.value("build_model/grid/crossline_step", self.settings.value("build_model/grid/crossline_size", "10"))))
        self.crossline_size_edit.setText(str(self.settings.value("build_model/grid/crossline_size", "10")))
        self.sample_step_edit.setText(str(self.settings.value("build_model/grid/sample_step", self.settings.value("build_model/grid/sample_size", "10"))))
        self.sample_size_edit.setText(str(self.settings.value("build_model/grid/sample_size", "10")))

    def _browse_path(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Project Base Directory",
            self.path_edit.text().strip() or str(Path.home()),
        )
        if path:
            self.path_edit.setText(path)

    @staticmethod
    def _axis_count(start_text: str, end_text: str, step_text: str) -> int | None:
        try:
            start = float(start_text or "0")
            end = float(end_text or "0")
            step = abs(float(step_text or "0"))
        except ValueError:
            return None
        if step <= 1e-12:
            return None
        span = abs(end - start)
        steps = span / step
        rounded = round(steps)
        if abs(steps - rounded) > 1e-9:
            return None
        return int(rounded) + 1

    def _update_axis_counts(self) -> None:
        inline_count = self._axis_count(
            self.inline_start_edit.text().strip(),
            self.inline_end_edit.text().strip(),
            self.inline_step_edit.text().strip(),
        )
        crossline_count = self._axis_count(
            self.crossline_start_edit.text().strip(),
            self.crossline_end_edit.text().strip(),
            self.crossline_step_edit.text().strip(),
        )
        sample_count = self._axis_count(
            self.sample_start_edit.text().strip(),
            self.sample_end_edit.text().strip(),
            self.sample_step_edit.text().strip(),
        )
        self.inline_num_value.setText("-" if inline_count is None else str(inline_count))
        self.crossline_num_value.setText("-" if crossline_count is None else str(crossline_count))
        self.sample_num_value.setText("-" if sample_count is None else str(sample_count))

    def values(self) -> tuple[str, Path, GridDefinition] | None:
        self.last_error = None
        project_name = self.project_name_edit.text().strip()
        if not project_name or any(char in project_name for char in '<>:"/\\|?*'):
            self.last_error = "Please enter a valid project name."
            return None
        base_path_text = self.path_edit.text().strip()
        if not base_path_text:
            self.last_error = "Please select a valid project path."
            return None
        inline_count = self._axis_count(
            self.inline_start_edit.text().strip(),
            self.inline_end_edit.text().strip(),
            self.inline_step_edit.text().strip(),
        )
        if inline_count is None:
            self.last_error = "Inline count must satisfy (end - start) / step + 1 with exact division."
            return None
        crossline_count = self._axis_count(
            self.crossline_start_edit.text().strip(),
            self.crossline_end_edit.text().strip(),
            self.crossline_step_edit.text().strip(),
        )
        if crossline_count is None:
            self.last_error = "Cxline count must satisfy (end - start) / step + 1 with exact division."
            return None
        sample_count = self._axis_count(
            self.sample_start_edit.text().strip(),
            self.sample_end_edit.text().strip(),
            self.sample_step_edit.text().strip(),
        )
        if sample_count is None:
            self.last_error = "Sample count must satisfy (end - start) / step + 1 with exact division."
            return None
        try:
            definition = GridDefinition(
                inline_start=float(self.inline_start_edit.text().strip() or "0"),
                inline_end=float(self.inline_end_edit.text().strip() or "0"),
                crossline_start=float(self.crossline_start_edit.text().strip() or "0"),
                crossline_end=float(self.crossline_end_edit.text().strip() or "0"),
                sample_start=float(self.sample_start_edit.text().strip() or "0"),
                sample_end=float(self.sample_end_edit.text().strip() or "0"),
                inline_size=max(1e-6, abs(float(self.inline_size_edit.text().strip() or "1"))),
                crossline_size=max(1e-6, abs(float(self.crossline_size_edit.text().strip() or "1"))),
                sample_size=max(1e-6, abs(float(self.sample_size_edit.text().strip() or "1"))),
                datum=float(self.datum_edit.text().strip() or "0"),
                inline_step=max(1e-6, abs(float(self.inline_step_edit.text().strip() or "1"))),
                crossline_step=max(1e-6, abs(float(self.crossline_step_edit.text().strip() or "1"))),
                sample_step=max(1e-6, abs(float(self.sample_step_edit.text().strip() or "1"))),
            )
        except ValueError:
            self.last_error = "Please enter valid grid parameters."
            return None
        return project_name, Path(base_path_text).expanduser().resolve(), definition


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
        self.scatter_sets: dict[str, ScatterDataSet] = {}
        self.polygon_sets: dict[str, PolygonDataSet] = {}
        self.model_surfaces: dict[str, ModelSurfaceDataSet] = {}
        self.grid_definition: GridDefinition | None = None
        self.grid_image: vtk.vtkImageData | None = None
        self.grid_actor: vtk.vtkActor | None = None
        self.current_horizon_name: str | None = None
        self.current_attribute_name: str | None = None
        self.current_scatter_name: str | None = None
        self.current_polygon_name: str | None = None
        self.current_model_name: str | None = None
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
            if self.grid_definition is not None:
                self.xlines = self.grid_definition.crossline_values
                self.inlines = self.grid_definition.inline_values
                self.samples = self.grid_definition.sample_values
            else:
                self.xlines = np.asarray([0.0], dtype=np.float32)
                self.inlines = np.asarray([0.0], dtype=np.float32)
                self.samples = np.asarray([0.0], dtype=np.float32)
            return
        volume_data = attribute.volume_data
        self.xlines = volume_data.xlines
        self.inlines = volume_data.inlines
        self.samples = volume_data.samples

    def scene_image(self) -> vtk.vtkImageData:
        attribute = self.current_attribute()
        if attribute is not None:
            return attribute.image
        if self.grid_image is not None:
            return self.grid_image
        return self.image

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

    def scatter_names(self) -> list[str]:
        return list(self.scatter_sets.keys())

    def polygon_names(self) -> list[str]:
        return list(self.polygon_sets.keys())

    def model_names(self) -> list[str]:
        return list(self.model_surfaces.keys())

    def current_attribute_opacity(self) -> float:
        attribute = self.current_attribute()
        if attribute is None:
            return float(self.opacity)
        return float(attribute.opacity)

    def current_horizon_scalar_range(self) -> tuple[float, float] | None:
        return None

    def current_horizon_opacity(self) -> float | None:
        horizon = self.current_horizon()
        if horizon is None:
            return None
        return float(horizon.opacity)

    def current_horizon(self) -> HorizonSurface | None:
        if self.current_horizon_name is None:
            return None
        return self.horizons.get(self.current_horizon_name)

    def current_scatter(self) -> ScatterDataSet | None:
        if self.current_scatter_name is None:
            return None
        return self.scatter_sets.get(self.current_scatter_name)

    def current_polygon(self) -> PolygonDataSet | None:
        if self.current_polygon_name is None:
            return None
        return self.polygon_sets.get(self.current_polygon_name)

    def current_model_surface(self) -> ModelSurfaceDataSet | None:
        if self.current_model_name is None:
            return None
        return self.model_surfaces.get(self.current_model_name)

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
        self.set_current_scatter(None, render=False)
        self.set_current_polygon(None, render=False)
        self.set_current_model_surface(None, render=False)
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
        if name is not None and name not in self.horizons:
            name = None
        self.current_horizon_name = name
        self.set_current_scatter(None, render=False)
        self.set_current_polygon(None, render=False)
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

    def set_current_scatter(self, name: str | None, render: bool = True) -> None:
        self.current_scatter_name = name if name in self.scatter_sets else None
        for scatter_name, scatter in self.scatter_sets.items():
            is_current = scatter_name == self.current_scatter_name
            scatter.actor.GetProperty().SetPointSize(8.0 if is_current else 6.0)
            scatter.actor.GetProperty().SetOpacity((1.0 if is_current else 0.82) if scatter.visible else 0.0)
            scatter.actor.SetVisibility(scatter.visible)
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_current_polygon(self, name: str | None, render: bool = True) -> None:
        self.current_polygon_name = name if name in self.polygon_sets else None
        for polygon_name, polygon in self.polygon_sets.items():
            is_current = polygon_name == self.current_polygon_name
            polygon.actor.GetProperty().SetLineWidth(3.2 if is_current else 2.0)
            polygon.actor.GetProperty().SetOpacity((1.0 if is_current else 0.90) if polygon.visible else 0.0)
            polygon.actor.SetVisibility(polygon.visible)
            polygon.point_actor.GetProperty().SetPointSize(13.0 if is_current else 10.0)
            polygon.point_actor.SetVisibility(bool(polygon.visible and is_current))
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_current_model_surface(self, name: str | None, render: bool = True) -> None:
        self.current_model_name = name if name in self.model_surfaces else None
        for model_name, model_surface in self.model_surfaces.items():
            is_current = model_name == self.current_model_name
            model_surface.actor.GetProperty().SetOpacity((0.92 if is_current else 0.76) if model_surface.visible else 0.0)
            model_surface.actor.GetProperty().SetEdgeVisibility(is_current)
            model_surface.actor.GetProperty().SetLineWidth(1.6 if is_current else 1.0)
            model_surface.actor.SetVisibility(model_surface.visible)
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

    @staticmethod
    def _control_point_value_stats(points: list[ControlPoint]) -> tuple[float, float]:
        values = np.asarray([float(point.value) for point in points], dtype=np.float32)
        if values.size == 0:
            return (0.0, 1.0)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum == maximum:
            maximum = minimum + 1.0
        return minimum, maximum

    def _build_control_point_value_lut_from_values(
        self,
        points: list[ControlPoint],
        colormap_name: str,
        value_range: tuple[float, float] | None,
    ) -> vtk.vtkLookupTable:
        actual_range = self._control_point_value_stats(points) if value_range is None else value_range
        lut = vtk.vtkLookupTable()
        lut.SetRange(float(actual_range[0]), float(actual_range[1]))
        apply_colormap_preset(lut, colormap_name)
        return lut

    def _control_point_value_lut(self, point_set: ControlPointSet) -> vtk.vtkLookupTable | None:
        if not point_set.use_attribute_colormap:
            return None
        return self._build_control_point_value_lut_from_values(
            point_set.points,
            point_set.value_colormap_name,
            point_set.value_color_range,
        )

    def _refresh_control_point_values(self, point_set: ControlPointSet) -> None:
        value_volume_data = self._control_point_value_volume_data(point_set)
        self._apply_value_volume_to_control_points(point_set, value_volume_data)

    def _apply_value_volume_to_control_points(
        self,
        point_set: ControlPointSet,
        value_volume_data: VolumeData | None,
    ) -> None:
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

    def _rebuild_current_mask_from_control_points(
        self,
        horizon: HorizonSurface,
        point_set: ControlPointSet,
    ) -> np.ndarray | None:
        reference_mask = np.asarray(point_set.original_horizon_mask, dtype=bool)
        if reference_mask.ndim != 3 or reference_mask.size == 0:
            return None
        rebuilt_mask = None
        try:
            rebuilt_mask = rebuild_mask_from_master_points(
                reference_mask.shape,
                point_set.points,
                reference_mask,
            )
        except Exception:
            rebuilt_mask = None

        surface_mask = None
        try:
            surface_mask = polydata_to_mask(horizon.polydata, reference_mask.shape, self.spacing, dilate_steps=1)
        except Exception:
            surface_mask = None

        combined_mask = np.zeros(reference_mask.shape, dtype=bool)
        if rebuilt_mask is not None:
            combined_mask |= np.asarray(rebuilt_mask, dtype=bool)
        if surface_mask is not None:
            combined_mask |= np.asarray(surface_mask, dtype=bool)
        if not np.any(combined_mask):
            return None
        return combined_mask

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
        rebuilt_mask = self._rebuild_current_mask_from_control_points(horizon, point_set)
        if rebuilt_mask is not None:
            horizon.component_mask = rebuilt_mask
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

    def add_scatter_data(
        self,
        name: str,
        *,
        inlines: np.ndarray,
        crosslines: np.ndarray,
        z_values: np.ndarray | None,
        values: np.ndarray,
        source_path: Path,
        select: bool = False,
    ) -> str:
        actor, polydata, mapper, lut, value_range = create_scatter_actor(
            inlines,
            crosslines,
            values,
            z_values=z_values,
        )
        new_name = self._unique_name(self.scatter_sets, name)
        dataset = ScatterDataSet(
            name=new_name,
            actor=actor,
            polydata=polydata,
            mapper=mapper,
            lut=lut,
            value_range=value_range,
            inlines=np.asarray(inlines, dtype=np.float32).ravel(),
            crosslines=np.asarray(crosslines, dtype=np.float32).ravel(),
            z_values=(
                np.zeros_like(np.asarray(values, dtype=np.float32).ravel(), dtype=np.float32)
                if z_values is None
                else np.asarray(z_values, dtype=np.float32).ravel()
            ),
            values=np.asarray(values, dtype=np.float32).ravel(),
            source_path=Path(source_path),
            visible=True,
        )
        self.scatter_sets[new_name] = dataset
        self.renderer.AddActor(actor)
        self.set_current_scatter(new_name if select else self.current_scatter_name, render=False)
        return new_name

    def add_polygon_data(
        self,
        name: str,
        *,
        color_rgb: tuple[int, int, int],
        grid_points: np.ndarray,
        z_values: np.ndarray | None,
        source_path: Path,
        select: bool = False,
    ) -> str:
        normalized_grid_points = normalize_polygon_grid_points(grid_points)
        actor, polydata, mapper, point_actor, point_polydata, point_mapper = create_polygon_actors(
            color_rgb,
            normalized_grid_points,
            z_values=z_values,
        )
        normalized_z_values = (
            np.zeros(normalized_grid_points.shape[0], dtype=np.float32)
            if z_values is None
            else np.asarray(z_values, dtype=np.float32).ravel()
        )
        new_name = self._unique_name(self.polygon_sets, name)
        dataset = PolygonDataSet(
            name=new_name,
            actor=actor,
            polydata=polydata,
            mapper=mapper,
            point_actor=point_actor,
            point_polydata=point_polydata,
            point_mapper=point_mapper,
            color_rgb=tuple(int(v) for v in color_rgb),
            grid_points=np.asarray(normalized_grid_points, dtype=np.float32),
            z_values=np.asarray(normalized_z_values, dtype=np.float32),
            source_path=Path(source_path),
            visible=True,
        )
        self.polygon_sets[new_name] = dataset
        self.renderer.AddActor(actor)
        self.renderer.AddActor(point_actor)
        self.set_current_polygon(new_name if select else self.current_polygon_name, render=False)
        return new_name

    def add_model_surface(
        self,
        name: str,
        *,
        polydata: vtk.vtkPolyData,
        source_polygon_name: str,
        dip_source_path: Path,
        direction_source_path: Path,
        select: bool = False,
    ) -> str:
        actor, mapper = create_model_surface_actor(polydata)
        new_name = self._unique_name(self.model_surfaces, name)
        dataset = ModelSurfaceDataSet(
            name=new_name,
            actor=actor,
            polydata=polydata,
            mapper=mapper,
            source_polygon_name=source_polygon_name,
            dip_source_path=Path(dip_source_path),
            direction_source_path=Path(direction_source_path),
            visible=True,
        )
        self.model_surfaces[new_name] = dataset
        self.renderer.AddActor(actor)
        self.set_current_model_surface(new_name if select else self.current_model_name, render=False)
        return new_name

    def set_grid_definition(self, definition: GridDefinition, render: bool = True) -> None:
        self.grid_definition = definition
        self.spacing = RenderSpacing(
            xline=float(definition.crossline_display_spacing),
            inline=float(definition.inline_display_spacing),
            sample=float(definition.sample_display_spacing),
        )
        self.grid_image = create_grid_image(definition)
        self.image = self.grid_image
        self._sync_axes_from_current_attribute()
        if self.grid_actor is not None:
            self.renderer.RemoveActor(self.grid_actor)
        self.grid_actor = create_grid_actor(definition)
        self.renderer.AddActor(self.grid_actor)
        self.update_overlay()
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

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
        value_colormap_name: str = DEFAULT_COLORMAP_NAME,
        value_color_range: tuple[float, float] | None = None,
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
            value_lut=(
                None
                if not use_attribute_colormap
                else self._build_control_point_value_lut_from_values(
                    points,
                    value_colormap_name,
                    value_color_range,
                )
            ),
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
            value_colormap_name=value_colormap_name,
            value_color_range=(
                self._control_point_value_stats(points)
                if value_color_range is None
                else (float(value_color_range[0]), float(value_color_range[1]))
            ),
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
        value_colormap_name: str = DEFAULT_COLORMAP_NAME,
        value_color_range: tuple[float, float] | None = None,
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
            value_colormap_name=value_colormap_name,
            value_color_range=value_color_range,
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
            self.current_horizon_name = None
            next_name = next(iter(self.horizons), None)
            self.set_current_horizon(next_name, render=False)
        return True

    def remove_scatter(self, name: str) -> bool:
        scatter = self.scatter_sets.pop(name, None)
        if scatter is None:
            return False
        self.renderer.RemoveActor(scatter.actor)
        if self.current_scatter_name == name:
            self.set_current_scatter(next(iter(self.scatter_sets), None), render=False)
        return True

    def remove_polygon(self, name: str) -> bool:
        polygon = self.polygon_sets.pop(name, None)
        if polygon is None:
            return False
        self.renderer.RemoveActor(polygon.actor)
        self.renderer.RemoveActor(polygon.point_actor)
        if self.current_polygon_name == name:
            self.set_current_polygon(next(iter(self.polygon_sets), None), render=False)
        return True

    def remove_model_surface(self, name: str) -> bool:
        model_surface = self.model_surfaces.pop(name, None)
        if model_surface is None:
            return False
        self.renderer.RemoveActor(model_surface.actor)
        if self.current_model_name == name:
            self.set_current_model_surface(next(iter(self.model_surfaces), None), render=False)
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
        if point_set is None:
            return None
        return point_set.value_colormap_name

    def current_control_point_use_attribute_colormap(self) -> bool:
        point_set = self.current_control_point_set()
        if point_set is None:
            return False
        return bool(point_set.use_attribute_colormap)

    def current_control_point_colormap_range(self) -> tuple[float, float] | None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return None
        return point_set.value_color_range or self._control_point_value_stats(point_set.points)

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
        attribute = self.attributes[attribute_name]
        self._apply_value_volume_to_control_points(point_set, attribute.volume_data)
        # Copy values once, then detach from the source attribute.
        point_set.value_attribute_name = None
        point_set.use_attribute_colormap = False
        self._refresh_control_point_set_actor(point_set)
        self.set_current_horizon(self.current_horizon_name, render=False)
        if render:
            self.interactor.GetRenderWindow().Render()
        return True

    def interpolate_attribute_from_control_points(
        self,
        attribute_name: str,
        horizon_name: str,
        output_name: str,
        *,
        idw_radius: float,
        apply_mask: bool,
    ) -> str | None:
        self.last_rebuild_error = None
        attribute = self.attributes.get(attribute_name)
        if attribute is None:
            self.last_rebuild_error = "The selected attribute grid is missing."
            return None
        horizon = self.horizons.get(horizon_name)
        if horizon is None or horizon.control_point_set is None:
            self.last_rebuild_error = "The selected horizon does not contain control points."
            return None
        point_set = horizon.control_point_set
        mask = horizon.component_mask
        if apply_mask:
            current_mask = self._rebuild_current_mask_from_control_points(horizon, point_set)
            if current_mask is not None:
                mask = current_mask
        try:
            output_volume = interpolate_control_point_values_to_volume(
                attribute.volume_data,
                point_set.points,
                apply_mask=bool(apply_mask),
                mask=mask,
                radius=float(idw_radius),
                name=output_name,
            )
        except ValueError as exc:
            self.last_rebuild_error = str(exc)
            return None
        source_opacity = float(attribute.opacity)
        return self.add_attribute_volume(output_volume, name=output_name, opacity=source_opacity, select=True)

    def extract_mask_from_current_horizon(self, output_name: str) -> str | None:
        self.last_rebuild_error = None
        horizon = self.current_horizon()
        if horizon is None:
            self.last_rebuild_error = "No current horizon is selected."
            return None
        if horizon.xlines is None or horizon.inlines is None or horizon.samples is None:
            self.last_rebuild_error = "The current horizon grid is incomplete."
            return None
        mask = horizon.component_mask
        if horizon.control_point_set is not None:
            current_mask = self._rebuild_current_mask_from_control_points(horizon, horizon.control_point_set)
            if current_mask is not None:
                mask = current_mask
        if mask is None:
            self.last_rebuild_error = "The current horizon mask is missing."
            return None
        output_volume = VolumeData(
            data=np.asarray(mask, dtype=np.float32),
            xlines=np.array(horizon.xlines, copy=True),
            inlines=np.array(horizon.inlines, copy=True),
            samples=np.array(horizon.samples, copy=True),
            name=output_name,
            metadata={
                "operation": "extract_horizon_mask",
                "source_horizon_name": horizon.name,
            },
        )
        return self.add_attribute_volume(output_volume, name=output_name, opacity=0.9, select=True)

    def replace_attribute_by_horizon_mask(
        self,
        target_attribute_name: str,
        source_attribute_name: str,
        horizon_name: str,
        output_name: str,
    ) -> str | None:
        self.last_rebuild_error = None
        target_attribute = self.attributes.get(target_attribute_name)
        if target_attribute is None:
            self.last_rebuild_error = "The target attribute is missing."
            return None
        source_attribute = self.attributes.get(source_attribute_name)
        if source_attribute is None:
            self.last_rebuild_error = "The source attribute is missing."
            return None
        if target_attribute.volume_data.shape != source_attribute.volume_data.shape:
            self.last_rebuild_error = "The two attribute volumes do not share the same grid shape."
            return None
        horizon = self.horizons.get(horizon_name)
        if horizon is None:
            self.last_rebuild_error = "The selected horizon is missing."
            return None

        mask = horizon.component_mask
        if horizon.control_point_set is not None:
            current_mask = self._rebuild_current_mask_from_control_points(horizon, horizon.control_point_set)
            if current_mask is not None:
                mask = current_mask
        if mask is None:
            self.last_rebuild_error = "The selected horizon mask is missing."
            return None
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != target_attribute.volume_data.shape:
            self.last_rebuild_error = "The horizon mask shape does not match the selected attributes."
            return None

        output_data = np.array(target_attribute.volume_data.data, copy=True)
        output_data[mask_array] = np.asarray(source_attribute.volume_data.data, dtype=np.float32)[mask_array]
        output_volume = target_attribute.volume_data.with_data(
            output_data,
            name=output_name,
            metadata={
                **target_attribute.volume_data.metadata,
                "operation": "replace_by_horizon_mask",
                "target_attribute_name": target_attribute_name,
                "source_attribute_name": source_attribute_name,
                "source_horizon_name": horizon_name,
            },
        )
        return self.add_attribute_volume(
            output_volume,
            name=output_name,
            opacity=float(target_attribute.opacity),
            select=True,
        )

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
        if point_set is None:
            return
        point_set.value_colormap_name = (
            colormap_name if colormap_name in available_colormap_names() else DEFAULT_COLORMAP_NAME
        )
        self._refresh_control_point_set_actor(point_set)
        self.refresh_scalar_bar()
        if render:
            self.interactor.GetRenderWindow().Render()

    def set_control_point_colormap_range(self, min_value: float, max_value: float, render: bool = True) -> None:
        point_set = self.current_control_point_set()
        if point_set is None:
            return
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        point_set.value_color_range = (float(min_value), float(max_value))
        self._refresh_control_point_set_actor(point_set)
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
        scatter = self.current_scatter()
        polygon = self.current_polygon()
        model_surface = self.current_model_surface()
        point_set = self.current_control_point_set()
        if polygon is not None or model_surface is not None:
            lut = None
        elif scatter is not None:
            lut = scatter.lut
            title = f"Scatter\n{scatter.name}"
        elif (
            point_set is not None
            and point_set.use_attribute_colormap
        ):
            lut = self._control_point_value_lut(point_set)
            title = "Control Points\nValue"
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
        self.project_panel = ProjectPanelWidget(self)
        self._selected_data_item: tuple[str, str] | None = None
        self.build_model_window: BuildModelWindow | None = None
        self.project_name: str | None = None
        self.project_dir: Path | None = None
        self.project_grid_path: Path | None = None
        self._data_panel_collapsed = False
        self._left_panel_mode = "data"

        self.setWindowTitle(APP_NAME)
        self._apply_initial_window_geometry()
        self._apply_soft_theme()
        self._build_top_menu_bar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.left_sidebar = QtWidgets.QWidget()
        self.left_sidebar.setObjectName("leftSidebar")
        self.left_sidebar_layout = QtWidgets.QVBoxLayout(self.left_sidebar)
        self.left_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.left_sidebar_layout.setSpacing(8)
        self.left_sidebar.setMinimumWidth(300)
        self.left_sidebar.setMaximumWidth(380)
        self.data_panel.setMinimumWidth(300)
        self.data_panel.setMaximumWidth(380)
        self.project_panel.setMinimumWidth(300)
        self.project_panel.setMaximumWidth(380)
        self.left_panel_stack = QtWidgets.QStackedWidget()
        self.left_panel_stack.addWidget(self.data_panel)
        self.left_panel_stack.addWidget(self.project_panel)
        self.left_sidebar_layout.addWidget(self.left_panel_stack, stretch=1)

        self.project_info_button = QtWidgets.QPushButton("P")
        self.project_info_button.setObjectName("leftSidebarQuickButton")
        self.project_info_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.project_info_button.setToolTip("Project Information")
        self.project_info_button.clicked.connect(self.open_project_info_dialog)
        self.project_info_button.hide()
        self.left_sidebar_layout.addWidget(self.project_info_button, stretch=0, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        self.data_panel_restore_button = QtWidgets.QPushButton("D")
        self.data_panel_restore_button.setObjectName("leftSidebarQuickButton")
        self.data_panel_restore_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.data_panel_restore_button.setToolTip("Show Data Panel")
        self.data_panel_restore_button.clicked.connect(self.expand_data_panel)
        self.data_panel_restore_button.hide()
        self.left_sidebar_layout.addWidget(self.data_panel_restore_button, stretch=0, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.left_sidebar_layout.addStretch(1)
        layout.addWidget(self.left_sidebar, stretch=0)

        viewer_panel = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        self.vtk_widget.setMinimumSize(1000, 900)
        self.vtk_widget.setStyleSheet("")
        viewer_layout.addWidget(self.vtk_widget, stretch=1)
        layout.addWidget(viewer_panel, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(360)
        layout.addWidget(panel, stretch=0)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        self.extract_button = QtWidgets.QPushButton("Open Range Extraction")
        self.extract_button.clicked.connect(self.open_extract_range_dialog)
        panel_layout.addWidget(self.extract_button)

        self.extract_envelope_button = QtWidgets.QPushButton("Open Horizon Extraction")
        self.extract_envelope_button.clicked.connect(self.open_extract_horizon_dialog)
        panel_layout.addWidget(self.extract_envelope_button)

        self.extract_control_points_button = QtWidgets.QPushButton("Extract Control Point")
        self.extract_control_points_button.clicked.connect(self.open_extract_control_points_dialog)
        panel_layout.addWidget(self.extract_control_points_button)

        self.interpolate_volume_button = QtWidgets.QPushButton("Interpolate Volume")
        self.interpolate_volume_button.clicked.connect(self.open_interpolate_volume_dialog)
        panel_layout.addWidget(self.interpolate_volume_button)

        self.extract_horizon_mask_button = QtWidgets.QPushButton("Extract Horizon Mask")
        self.extract_horizon_mask_button.clicked.connect(self.open_extract_horizon_mask_dialog)
        panel_layout.addWidget(self.extract_horizon_mask_button)

        self.replace_volume_button = QtWidgets.QPushButton("Replace By Horizon")
        self.replace_volume_button.clicked.connect(self.open_replace_volume_dialog)
        panel_layout.addWidget(self.replace_volume_button)

        self.build_model_button = QtWidgets.QPushButton("Build Model")
        self.build_model_button.clicked.connect(self.open_build_model_window)
        panel_layout.addWidget(self.build_model_button)

        attribute_display_group = ColorMapControlWidget("Colormap")
        self.attribute_colormap_widget = attribute_display_group
        self.attribute_colormap_widget.target_combo.currentIndexChanged.connect(self.change_colormap_target)
        self.attribute_colormap_widget.control_point_use_colormap_checkbox.toggled.connect(
            self.toggle_control_point_colormap
        )
        self.attribute_colormap_widget.apply_button.clicked.connect(self.apply_active_colormap_range)
        self.attribute_colormap_widget.preset_combo.currentTextChanged.connect(self.change_active_colormap)
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
        control_point_value_row.addStretch(1)
        control_point_tools_layout.addLayout(control_point_value_row)

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
        self.data_panel.header_clicked.connect(self.toggle_data_panel)
        self.project_panel.header_clicked.connect(self.toggle_data_panel)
        self.data_panel.item_activated.connect(self.activate_data_item)
        self.data_panel.category_load_requested.connect(self.load_data_for_category)
        self.data_panel.item_store_requested.connect(self.store_data_item)
        self.data_panel.item_unload_requested.connect(self.unload_data_item)

        self.refresh_axis_controls()
        self.refresh_data_panel()
        self.refresh_project_panel()
        self.refresh_info()
        self.refresh_display_controls()
        self.vtk_widget.installEventFilter(self)

    def _apply_initial_window_geometry(self) -> None:
        default_width = 1440
        default_height = 920
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(default_width, default_height)
            return

        available = screen.availableGeometry()
        width = min(default_width, max(1100, available.width() - 120))
        height = min(default_height, max(760, available.height() - 120))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)

        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _apply_soft_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #FFFFFF;
                color: #111111;
            }
            QWidget#leftSidebar {
                background: #FFFFFF;
            }
            QMenuBar {
                background: #FFFFFF;
                color: #111111;
                border-bottom: 1px solid #E5E5E5;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                margin: 2px 4px;
                border-radius: 6px;
            }
            QMenuBar::item:selected {
                background: #F5F5F5;
            }
            QMenu {
                background: #FFFFFF;
                color: #111111;
                border: 1px solid #E5E5E5;
            }
            QMenu::item {
                padding: 7px 22px 7px 12px;
            }
            QMenu::item:selected {
                background: #F5F5F5;
            }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E5E5E5;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #111111;
            }
            QPushButton {
                background: #FFFFFF;
                color: #111111;
                border: 1px solid #DADADA;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background: #FAFAFA;
                border-color: #CFCFCF;
            }
            QPushButton:pressed {
                background: #F0F0F0;
            }
            QPushButton:disabled {
                background: #FFFFFF;
                color: #A0A0A0;
                border-color: #ECECEC;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #FFFFFF;
                color: #111111;
                border: 1px solid #DADADA;
                border-radius: 8px;
                padding: 6px 8px;
                selection-background-color: #EDEDED;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #BDBDBD;
            }
            QToolButton {
                background: transparent;
                color: #111111;
                border: none;
                padding: 4px 0;
                text-align: left;
            }
            QToolButton:hover {
                color: #000000;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #E8E8E8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #CFCFCF;
                border: 1px solid #BDBDBD;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QLabel {
                color: #111111;
            }
            QTreeWidget, QListWidget {
                background: #FFFFFF;
                alternate-background-color: #FFFFFF;
                border: 1px solid #E5E5E5;
                border-radius: 8px;
            }
            QHeaderView::section {
                background: #FFFFFF;
                color: #111111;
                border: none;
                border-bottom: 1px solid #E5E5E5;
                padding: 6px 8px;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #FFFFFF;
                border-radius: 6px;
            }
            QPushButton#leftSidebarQuickButton {
                background: #FFFFFF;
                color: #111111;
                border: 1px solid #E5E5E5;
                border-radius: 4px;
                padding: 0px;
                min-width: 24px;
                max-width: 24px;
                min-height: 24px;
                max-height: 24px;
                text-align: center;
                font-weight: 600;
                font-size: 17px;
            }
            QPushButton#leftSidebarQuickButton:hover {
                background: #F7F7F7;
                border-color: #D6D6D6;
            }
            """
        )

    def collapse_data_panel(self) -> None:
        if self._data_panel_collapsed:
            return
        self._data_panel_collapsed = True
        self.left_panel_stack.hide()
        self.project_info_button.show()
        self.data_panel_restore_button.show()
        self.left_sidebar.setMinimumWidth(26)
        self.left_sidebar.setMaximumWidth(26)
        self.left_sidebar.updateGeometry()
        self.schedule_render()

    def expand_data_panel(self) -> None:
        if not self._data_panel_collapsed:
            self._show_left_panel("data")
            return
        self._data_panel_collapsed = False
        self._left_panel_mode = "data"
        self.project_info_button.hide()
        self.data_panel_restore_button.hide()
        self.left_panel_stack.setCurrentWidget(self.data_panel)
        self.left_panel_stack.show()
        self.left_sidebar.setMinimumWidth(300)
        self.left_sidebar.setMaximumWidth(380)
        self.left_sidebar.updateGeometry()
        self.schedule_render()

    def open_project_info_dialog(self) -> None:
        self._show_left_panel("project")

    def toggle_data_panel(self) -> None:
        if self._data_panel_collapsed:
            self.expand_data_panel()
        else:
            self.collapse_data_panel()

    def _show_left_panel(self, mode: str) -> None:
        self._left_panel_mode = "project" if mode == "project" else "data"
        if self._data_panel_collapsed:
            self._data_panel_collapsed = False
            self.project_info_button.hide()
            self.data_panel_restore_button.hide()
            self.left_sidebar.setMinimumWidth(300)
            self.left_sidebar.setMaximumWidth(380)
        self.left_panel_stack.setCurrentWidget(self.project_panel if self._left_panel_mode == "project" else self.data_panel)
        self.left_panel_stack.show()
        self.left_sidebar.updateGeometry()
        self.schedule_render()

    def _build_top_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(True)
        file_menu = menu_bar.addMenu("&File")
        new_project_action = QtGui.QAction("New Project", self)
        new_project_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.open_new_project_dialog)
        file_menu.addAction(new_project_action)
        load_segy_action = QtGui.QAction("Load Segy", self)
        load_segy_action.triggered.connect(self.open_load_seismic_dialog)
        file_menu.addAction(load_segy_action)
        import_seismic_attribute_action = QtGui.QAction("Import Seismic/Attribute Data", self)
        import_seismic_attribute_action.triggered.connect(self.open_seismic_attribute_import_dialog)
        file_menu.addAction(import_seismic_attribute_action)

    def open_new_project_dialog(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            QtWidgets.QMessageBox.information(
                self,
                "Invalid Project",
                dialog.last_error or "Please enter a valid project name, path, and grid parameters.",
            )
            return
        project_name, base_dir, definition = values
        project_dir = base_dir / project_name
        if project_dir.exists() and any(project_dir.iterdir()):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Project Exists",
                f"The project directory already exists and is not empty:\n{project_dir}\n\nContinue and reuse it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.create_new_project(project_name, project_dir, definition)

    def _project_subdir(self, *parts: str) -> Path | None:
        if self.project_dir is None:
            return None
        return self.project_dir.joinpath(*parts)

    def _ensure_project_structure(self, project_name: str, project_dir: Path, definition: GridDefinition) -> Path:
        project_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            project_dir / "config",
            project_dir / "grid",
            project_dir / "raw",
            project_dir / "raw" / "seismic",
            project_dir / "raw" / "attribute",
            project_dir / "raw" / "scatter",
            project_dir / "raw" / "polygon",
            project_dir / "raw" / "model",
            project_dir / "raw" / "well",
            project_dir / "derived",
            project_dir / "derived" / "seismic",
            project_dir / "derived" / "attribute",
            project_dir / "derived" / "horizon",
            project_dir / "derived" / "scatter",
            project_dir / "derived" / "polygon",
            project_dir / "derived" / "model",
            project_dir / "derived" / "well",
            project_dir / "export",
            project_dir / "temp",
        ):
            path.mkdir(parents=True, exist_ok=True)

        grid_path = project_dir / "grid" / "grid_definition.json"
        definition.to_json_file(grid_path)

        project_payload = {
            "project_name": project_name,
            "project_dir": str(project_dir),
            "grid_definition_path": str(grid_path),
            "derived_dir": str(project_dir / "derived"),
            "raw_dir": str(project_dir / "raw"),
            "export_dir": str(project_dir / "export"),
        }
        (project_dir / "config" / "project.json").write_text(
            json.dumps(project_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return grid_path

    @staticmethod
    def _persist_grid_history(definition: GridDefinition) -> None:
        settings = QtCore.QSettings("wesi3d", APP_NAME)
        settings.setValue("build_model/grid/inline_start", definition.inline_start)
        settings.setValue("build_model/grid/inline_end", definition.inline_end)
        settings.setValue("build_model/grid/crossline_start", definition.crossline_start)
        settings.setValue("build_model/grid/crossline_end", definition.crossline_end)
        settings.setValue("build_model/grid/sample_start", definition.sample_start)
        settings.setValue("build_model/grid/sample_end", definition.sample_end)
        settings.setValue("build_model/grid/datum", definition.datum)
        settings.setValue("build_model/grid/inline_step", definition.inline_size if definition.inline_step is None else definition.inline_step)
        settings.setValue("build_model/grid/crossline_step", definition.crossline_size if definition.crossline_step is None else definition.crossline_step)
        settings.setValue("build_model/grid/sample_step", definition.sample_size if definition.sample_step is None else definition.sample_step)
        settings.setValue("build_model/grid/inline_size", definition.inline_size)
        settings.setValue("build_model/grid/crossline_size", definition.crossline_size)
        settings.setValue("build_model/grid/sample_size", definition.sample_size)

    def create_new_project(self, project_name: str, project_dir: Path, definition: GridDefinition) -> None:
        for name in list(self.updater.horizon_names()):
            self.updater.remove_horizon(name)
        for name in list(self.updater.scatter_names()):
            self.updater.remove_scatter(name)
        for name in list(self.updater.polygon_names()):
            self.updater.remove_polygon(name)
        for name in list(self.updater.model_names()):
            self.updater.remove_model_surface(name)
        for name in list(self.updater.attribute_names()):
            self.updater.remove_attribute(name)

        if self.updater.grid_actor is not None:
            self.renderer.RemoveActor(self.updater.grid_actor)
            self.updater.grid_actor = None
        self.updater.grid_image = None
        self.updater.grid_definition = None
        self.updater.segy_path = None
        self.updater.current_attribute_name = None
        self.updater.current_scatter_name = None
        self.updater.current_polygon_name = None
        self.updater.current_model_name = None
        self.updater.current_horizon_name = None

        grid_path = self._ensure_project_structure(project_name, project_dir, definition)
        self.project_name = project_name
        self.project_dir = project_dir
        self.project_grid_path = grid_path
        self.setWindowTitle(f"{APP_NAME} - {project_name}")

        placeholder_image, _ = create_placeholder_image()
        self.updater.image = placeholder_image
        self.image = placeholder_image
        self.updater.indices = {
            "xline": 0,
            "inline": 0,
            "sample": 0,
        }
        self.updater._sync_axes_from_current_attribute()
        self.updater.set_grid_definition(definition, render=False)
        self._persist_grid_history(definition)
        if self.build_model_window is not None:
            self.build_model_window.save_grid_history(definition)
        self.updater.update_overlay()
        self.updater.refresh_scalar_bar()

        self._selected_data_item = None
        self._selected_master_point_index = None
        self._linked_master_point_indices.clear()
        self._refresh_selected_master_actor()
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_data_panel()
        self.refresh_project_panel()
        self.refresh_info()
        configure_default_camera(self.renderer, self.updater.scene_image())
        self.schedule_render()

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
                if self.pick_polygon(
                    int(event.position().x()),
                    int(event.position().y()),
                ):
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
        if not self.updater.has_attribute_data() and self.updater.grid_image is None:
            return
        configure_default_camera(self.renderer, self.updater.scene_image())
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
        scene_image = self.updater.scene_image()
        self.outline_actor = create_outline(scene_image)
        self.axis_texts = create_axis_labels(
            scene_image,
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
        self.attribute_opacity_slider.setEnabled(has_attribute)
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
        colormap_range = self.updater.current_control_point_colormap_range()
        has_colormap_attribute = colormap_range is not None
        smoothness = self.updater.current_control_point_rebuild_smoothness()
        self.control_point_smoothness_slider.blockSignals(True)
        self.control_point_smoothness_slider.setValue(
            int(round((0.55 if smoothness is None else smoothness) * 100.0))
        )
        self.control_point_smoothness_slider.blockSignals(False)

        available_targets = []
        if has_attribute:
            available_targets.append("attribute")
        if has_control_points:
            available_targets.append("control_point")
        active_target = self.attribute_colormap_widget.current_target()
        if active_target not in available_targets:
            fallback_target = available_targets[0] if available_targets else "attribute"
            self.attribute_colormap_widget.set_target(fallback_target)
            active_target = fallback_target
        self.attribute_colormap_widget.set_target_enabled(len(available_targets) > 1)
        show_control_point_toggle = active_target == "control_point"
        self.attribute_colormap_widget.set_control_point_toggle_visible(show_control_point_toggle)
        self.attribute_colormap_widget.control_point_use_colormap_checkbox.blockSignals(True)
        self.attribute_colormap_widget.control_point_use_colormap_checkbox.setChecked(
            self.updater.current_control_point_use_attribute_colormap()
        )
        self.attribute_colormap_widget.control_point_use_colormap_checkbox.blockSignals(False)
        if active_target == "control_point":
            self.attribute_colormap_widget.set_controls_enabled(has_control_points and has_colormap_attribute)
            if colormap_range is not None:
                self.attribute_colormap_widget.set_range(colormap_range)
                self.attribute_colormap_widget.set_current_preset(self.updater.current_control_point_colormap_name())
            else:
                self.attribute_colormap_widget.set_range(None)
        else:
            self.attribute_colormap_widget.set_controls_enabled(has_attribute)
            if has_attribute:
                self.attribute_colormap_widget.set_range(attribute_range)
                self.attribute_colormap_widget.set_current_preset(self.updater.current_attribute_colormap_name())
            else:
                self.attribute_colormap_widget.set_range(None)

        self.extract_button.setEnabled(has_attribute)
        self.extract_envelope_button.setEnabled(has_attribute)
        self.interpolate_volume_button.setEnabled(
            has_attribute and any(horizon.control_point_set is not None for horizon in self.updater.horizons.values())
        )
        self.extract_horizon_mask_button.setEnabled(self.updater.current_horizon() is not None)
        self.replace_volume_button.setEnabled(has_attribute and self.updater.current_horizon() is not None)
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
            "scatter": [
                DataPanelItem(
                    category="scatter",
                    name=name,
                    label=f"{name} ({len(self.updater.scatter_sets[name].values)} pts)",
                )
                for name in self.updater.scatter_names()
            ],
            "polygon": [
                DataPanelItem(
                    category="polygon",
                    name=name,
                    label=f"{name} ({len(self.updater.polygon_sets[name].grid_points)} pts)",
                )
                for name in self.updater.polygon_names()
            ],
            "model": [
                DataPanelItem(
                    category="model",
                    name=name,
                    label=f"{name} ({self.updater.model_surfaces[name].polydata.GetNumberOfPolys()} tris)",
                )
                for name in self.updater.model_names()
            ],
            "well": [],
        }
        self.data_panel.set_items(items)
        if self._selected_data_item is not None:
            selected_category, selected_name = self._selected_data_item
            self.data_panel.select_item(selected_category, selected_name)
        self.refresh_display_controls()

    def refresh_project_panel(self) -> None:
        definition = self.updater.grid_definition
        entries: list[tuple[str, str]] = [
            ("Project Name", "-" if self.project_name is None else self.project_name),
            ("Project Path", "-" if self.project_dir is None else str(self.project_dir)),
            ("Grid File", "-" if self.project_grid_path is None else str(self.project_grid_path)),
        ]
        if definition is not None:
            entries.extend(
                [
                    ("Datum", str(definition.datum)),
                    ("Inline", f"{definition.inline_start} -> {definition.inline_end}"),
                    ("Cxline", f"{definition.crossline_start} -> {definition.crossline_end}"),
                    ("Sample", f"{definition.sample_start} -> {definition.sample_end}"),
                ]
            )
        self.project_panel.set_info(entries)

    def activate_data_item(self, category: str, name: str) -> None:
        self._selected_data_item = (category, name)
        if category in {"seismic", "attribute"}:
            self.attribute_colormap_widget.set_target("attribute")
            self.updater.set_attribute(name, render=False)
            self.image = self.updater.image
            self.refresh_axis_controls()
            self.refresh_scene_guides()
        elif category == "horizon":
            self.attribute_colormap_widget.set_target("control_point")
            self.updater.set_current_horizon(name, render=False)
            self.updater.set_current_model_surface(None, render=False)
            self._selected_master_point_index = None
            self._linked_master_point_indices.clear()
            self._refresh_selected_master_actor()
        elif category == "scatter":
            self.updater.set_current_scatter(name, render=False)
            self.updater.set_current_polygon(None, render=False)
            self.updater.set_current_model_surface(None, render=False)
        elif category == "polygon":
            self.updater.set_current_scatter(None, render=False)
            self.updater.set_current_polygon(name, render=False)
            self.updater.set_current_model_surface(None, render=False)
        elif category == "model":
            self.updater.set_current_scatter(None, render=False)
            self.updater.set_current_polygon(None, render=False)
            self.updater.set_current_model_surface(name, render=False)
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

    def pick_polygon(self, display_x: int, display_y: int) -> bool:
        if not self.updater.polygon_sets:
            return False
        vtk_display_x, vtk_display_y = self._qt_to_vtk_display(display_x, display_y)
        if self._prop_picker.Pick(float(vtk_display_x), float(vtk_display_y), 0.0, self.renderer) == 0:
            return False
        picked_actor = self._prop_picker.GetActor()
        for name, polygon in self.updater.polygon_sets.items():
            if not polygon.visible:
                continue
            if picked_actor in {polygon.actor, polygon.point_actor}:
                self.activate_data_item("polygon", name)
                return True
        return False

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

    def open_seismic_attribute_import_dialog(self, target_category: str = "seismic") -> None:
        dialog = SeismicAttributeImportDialog(self, target_category=target_category)
        dialog.import_requested.connect(self.handle_seismic_attribute_import)
        dialog.exec()

    def handle_seismic_attribute_import(self, values: dict[str, object]) -> None:
        print("[Viewer] handle_seismic_attribute_import", flush=True)
        request = build_import_request(values)
        result = execute_import(request)
        print(
            "[Viewer] import template result:"
            f" file_type={result.request.file_type}"
            f" category={result.request.target_category}"
            f" cache_path={result.cache_path}",
            flush=True,
        )
        # Temporary bridge: keep using the existing SEG-Y load path until the
        # new import service gains real loading implementations.
        self.load_segy_volume(result.values)

    def open_build_model_window(self) -> None:
        if self.build_model_window is None:
            self.build_model_window = BuildModelWindow(self)
            self.build_model_window.define_grid_requested.connect(self.open_define_grid_dialog)
            self.build_model_window.load_dip_direction_requested.connect(self.open_load_dip_direction_dialog)
            self.build_model_window.load_elev_requested.connect(self.open_load_elev_dialog)
            self.build_model_window.load_geomap_requested.connect(self.open_load_geomap_dialog)
            self.build_model_window.build_polygon_surface_requested.connect(self.open_build_polygon_surface_dialog)
            self.build_model_window.build_model_volume_requested.connect(self.open_build_model_volume_dialog)
            self.build_model_window.build_selected_model_mask_requested.connect(self.open_build_selected_model_mask_dialog)
        self.build_model_window.show()
        self.build_model_window.raise_()
        self.build_model_window.activateWindow()

    def open_define_grid_dialog(self) -> None:
        if self.build_model_window is None:
            self.open_build_model_window()
        if self.build_model_window is None:
            return
        definition = self.build_model_window.current_grid_definition()
        if definition is None:
            QtWidgets.QMessageBox.information(self, "Invalid Grid", "Please enter valid grid parameters.")
            return
        self.build_model_window.save_grid_history(definition)
        self.updater.set_grid_definition(definition, render=False)
        self.image = self.updater.scene_image()
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        configure_default_camera(self.renderer, self.updater.scene_image())
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    @staticmethod
    def _read_scatter_gmp(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            data = np.loadtxt(path, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Failed to read scatter file: {path.name}") from exc
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Scatter file must contain three columns: {path.name}")
        return data[:, 0], data[:, 1], data[:, 2]

    def open_build_polygon_surface_dialog(self) -> None:
        if self.build_model_window is None:
            self.open_build_model_window()
        if self.build_model_window is None:
            return
        polygons = list(self.updater.polygon_sets.values())
        if not polygons:
            QtWidgets.QMessageBox.information(self, "Missing Polygon", "Please load polygons first.")
            return
        options = self.build_model_window.current_surface_options()
        if options is None:
            QtWidgets.QMessageBox.information(self, "Invalid Parameters", "Please enter valid surface build parameters.")
            return
        dip_path_text, direction_path_text = self.build_model_window.current_scatter_paths()
        if not dip_path_text or not direction_path_text:
            QtWidgets.QMessageBox.information(self, "Missing Files", "Please select both dip and direction files.")
            return
        self.build_model_window.save_surface_history(
            sample_count=int(options["sample_count"]),
            layer_step=float(options["layer_step"]),
            smooth_iterations=int(options["smooth_iterations"]),
        )
        grid_definition = self.updater.grid_definition
        if grid_definition is None:
            QtWidgets.QMessageBox.information(self, "Missing Grid", "Please define grid before building surface.")
            return
        dip_path = Path(dip_path_text).expanduser().resolve()
        direction_path = Path(direction_path_text).expanduser().resolve()
        try:
            dip_inlines, dip_crosslines, dip_values = self._read_scatter_gmp(dip_path)
            direction_inlines, direction_crosslines, direction_values = self._read_scatter_gmp(direction_path)
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "Build Failed", str(exc))
            return

        first_surface_name: str | None = None
        built_count = 0
        skipped_count = 0
        total_count = len(polygons)
        for polygon in polygons:
            try:
                polydata = build_extruded_polygon_surface(
                    polygon.grid_points,
                    polygon.z_values,
                    dip_inlines,
                    dip_crosslines,
                    dip_values,
                    direction_inlines,
                    direction_crosslines,
                    direction_values,
                    sample_count=int(options["sample_count"]),
                    layer_step=float(options["layer_step"]),
                    target_depth=max(float(grid_definition.sample_start), float(grid_definition.sample_end)),
                    smooth_iterations=int(options["smooth_iterations"]),
                )
            except ValueError:
                skipped_count += 1
                print(
                    f"[Build Surface] skipped polygon {polygon.name} ({built_count}/{total_count} built, {skipped_count} skipped)",
                    flush=True,
                )
                continue

            surface_name = self.updater.add_model_surface(
                f"{polygon.name}_surface",
                polydata=polydata,
                source_polygon_name=polygon.name,
                dip_source_path=dip_path,
                direction_source_path=direction_path,
                select=first_surface_name is None,
            )
            built_count += 1
            print(
                f"[Build Surface] built {built_count}/{total_count}: {surface_name}",
                flush=True,
            )
            if first_surface_name is None:
                first_surface_name = surface_name

        if first_surface_name is None:
            QtWidgets.QMessageBox.information(self, "Build Failed", "No polygon surface could be generated.")
            return

        print(
            f"[Build Surface] completed: {built_count} built, {skipped_count} skipped, {total_count} total",
            flush=True,
        )

        self._selected_data_item = ("model", first_surface_name)
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_build_model_volume_dialog(self) -> None:
        if self.build_model_window is None:
            self.open_build_model_window()
        if self.build_model_window is None:
            return
        grid_definition = self.updater.grid_definition
        if grid_definition is None:
            QtWidgets.QMessageBox.information(self, "Missing Grid", "Please define grid before building model volume.")
            return
        if not self.updater.model_surfaces:
            QtWidgets.QMessageBox.information(self, "Missing Model", "Please build model surfaces first.")
            return

        elev_path_text = self.build_model_window.current_model_elev_path()
        if not elev_path_text:
            QtWidgets.QMessageBox.information(self, "Missing Elev", "Please select an elev file first.")
            return
        output_name = self.build_model_window.current_model_output_name()
        self.build_model_window.save_model_volume_history(output_name)

        elev_path = Path(elev_path_text).expanduser().resolve()
        print(f"[Build Model] start: output={output_name}", flush=True)
        print(
            "[Build Model] grid:"
            f" inl={len(grid_definition.inline_values)}"
            f" cxl={len(grid_definition.crossline_values)}"
            f" samples={len(grid_definition.sample_values)}",
            flush=True,
        )
        print(f"[Build Model] elev file: {elev_path}", flush=True)
        try:
            elev_inlines, elev_crosslines, elev_values = self._read_scatter_gmp(elev_path)
            sample_size = max(1e-6, abs(float(grid_definition.sample_size)))
            elev_surface = build_elev_surface(
                elev_inlines,
                elev_crosslines,
                np.asarray(elev_values, dtype=np.float32) / np.float32(sample_size),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "Build Failed", str(exc))
            return
        print(
            f"[Build Model] elev grid ready: rows={elev_values.size}, sample_size={sample_size}",
            flush=True,
        )

        inline_values = grid_definition.inline_values
        crossline_values = grid_definition.crossline_values
        sample_values = grid_definition.sample_values
        sample_min = float(np.min(sample_values))
        sample_max = float(np.max(sample_values))

        print("[Build Model] sampling elev over grid...", flush=True)
        elev_depths = np.zeros((inline_values.size, crossline_values.size), dtype=np.float32)
        for inline_index, inline_value in enumerate(inline_values):
            for crossline_index, crossline_value in enumerate(crossline_values):
                elev_depths[inline_index, crossline_index] = float(
                    elev_surface.sample(float(inline_value), float(crossline_value))
                )
        print("[Build Model] elev sampling completed", flush=True)

        sorted_models = sorted(
            self.updater.model_surfaces.values(),
            key=lambda dataset: float(np.mean(numpy_support.vtk_to_numpy(dataset.polydata.GetPoints().GetData())[:, 2])),
            reverse=True,
        )
        model_masks: list[tuple[int, np.ndarray]] = []
        total_models = len(sorted_models)
        for index, model_surface in enumerate(sorted_models, start=1):
            print(
                f"[Build Model] rasterizing surface {index}/{total_models}: "
                f"id={index} name={model_surface.name}",
                flush=True,
            )
            model_mask = polydata_to_mask(
                model_surface.polydata,
                shape=(crossline_values.size, inline_values.size, sample_values.size),
                spacing=self.updater.spacing,
                origin=(
                    float(grid_definition.crossline_start),
                    float(grid_definition.inline_start),
                    float(grid_definition.sample_start),
                ),
                dilate_steps=0,
            )
            print(
                f"[Build Model] surface mask ready: id={index} voxels={int(np.count_nonzero(model_mask))}",
                flush=True,
            )
            model_masks.append(
                (
                    index,
                    model_mask,
                )
            )

        print(f"[Build Model] filling volume with {len(sorted_models)} model surfaces...", flush=True)
        volume_data_array = fill_model_volume_from_surfaces(
            sample_values,
            elev_depths,
            model_masks,
        )
        print(
            f"[Build Model] volume filled: shape={volume_data_array.shape}",
            flush=True,
        )
        volume = VolumeData(
            data=np.transpose(volume_data_array, (1, 0, 2)),
            xlines=np.asarray(crossline_values, dtype=np.float32),
            inlines=np.asarray(inline_values, dtype=np.float32),
            samples=np.asarray(sample_values, dtype=np.float32),
            name=output_name,
            metadata={
                "panel_category": "attribute",
                "operation": "build_model_volume",
                "source_elev_path": str(elev_path),
                "source_model_count": len(sorted_models),
            },
        )
        print("[Build Model] loading volume into attribute panel...", flush=True)
        name = self.updater.add_attribute_volume(volume, name=output_name, opacity=0.9, select=True)
        built_attribute = self.updater.attributes.get(name)
        if built_attribute is not None:
            built_attribute.image.SetSpacing(
                float(grid_definition.crossline_size),
                float(grid_definition.inline_size),
                float(grid_definition.sample_size),
            )
            built_attribute.image.SetOrigin(
                float(grid_definition.crossline_start),
                float(grid_definition.inline_start),
                float(grid_definition.sample_start),
            )
            print(
                "[Build Model] image spacing aligned to grid:"
                f" ({float(grid_definition.crossline_size)},"
                f" {float(grid_definition.inline_size)},"
                f" {float(grid_definition.sample_size)})",
                flush=True,
            )
            print(
                "[Build Model] image origin aligned to grid:"
                f" ({float(grid_definition.crossline_start)},"
                f" {float(grid_definition.inline_start)},"
                f" {float(grid_definition.sample_start)})",
                flush=True,
            )
        self.image = self.updater.image
        self._selected_data_item = ("attribute", name)
        print(f"[Build Model] completed: {name}", flush=True)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_build_selected_model_mask_dialog(self) -> None:
        model_surface = self.updater.current_model_surface()
        if model_surface is None:
            QtWidgets.QMessageBox.information(self, "Missing Model", "Please select a model first.")
            return
        grid_definition = self.updater.grid_definition
        if grid_definition is None:
            QtWidgets.QMessageBox.information(self, "Missing Grid", "Please define grid before building model mask.")
            return

        inline_values = grid_definition.inline_values
        crossline_values = grid_definition.crossline_values
        sample_values = grid_definition.sample_values
        output_name = f"{model_surface.name}_mask"
        print(f"[Build Model Mask] start: model={model_surface.name}", flush=True)
        print(
            "[Build Model Mask] grid size:"
            f" ({float(grid_definition.crossline_size)},"
            f" {float(grid_definition.inline_size)},"
            f" {float(grid_definition.sample_size)})",
            flush=True,
        )
        mask_volume_array = build_model_mask_volume(
            sample_values,
            inline_values,
            crossline_values,
            model_surface.polydata,
            debug_label=model_surface.name,
        )
        print(
            f"[Build Model Mask] filled voxels={int(np.count_nonzero(mask_volume_array))}",
            flush=True,
        )
        volume = VolumeData(
            data=np.transpose(mask_volume_array, (1, 0, 2)),
            xlines=np.asarray(crossline_values, dtype=np.float32),
            inlines=np.asarray(inline_values, dtype=np.float32),
            samples=np.asarray(sample_values, dtype=np.float32),
            name=output_name,
            metadata={
                "panel_category": "attribute",
                "operation": "build_model_mask",
                "source_model_name": model_surface.name,
            },
        )
        name = self.updater.add_attribute_volume(volume, name=output_name, opacity=0.9, select=True)
        built_attribute = self.updater.attributes.get(name)
        if built_attribute is not None:
            built_attribute.image.SetSpacing(
                float(grid_definition.crossline_size),
                float(grid_definition.inline_size),
                float(grid_definition.sample_size),
            )
            built_attribute.image.SetOrigin(
                float(grid_definition.crossline_start),
                float(grid_definition.inline_start),
                float(grid_definition.sample_start),
            )
            print(
                "[Build Model Mask] image spacing aligned to grid:"
                f" ({float(grid_definition.crossline_size)},"
                f" {float(grid_definition.inline_size)},"
                f" {float(grid_definition.sample_size)})",
                flush=True,
            )
            print(
                "[Build Model Mask] image origin aligned to grid:"
                f" ({float(grid_definition.crossline_start)},"
                f" {float(grid_definition.inline_start)},"
                f" {float(grid_definition.sample_start)})",
                flush=True,
            )
        self.image = self.updater.image
        self._selected_data_item = ("attribute", name)
        print(f"[Build Model Mask] completed: {name}", flush=True)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_load_elev_dialog(self) -> None:
        if self.build_model_window is None:
            self.open_build_model_window()
        if self.build_model_window is None:
            return
        elev_path_text = self.build_model_window.current_elev_path()
        if not elev_path_text:
            QtWidgets.QMessageBox.information(self, "Missing File", "Please select an elev file.")
            return
        grid_definition = self.updater.grid_definition
        if grid_definition is None:
            QtWidgets.QMessageBox.information(self, "Missing Grid", "Please define grid before loading elev.")
            return
        sample_size = abs(float(grid_definition.sample_size))
        if sample_size <= 1e-6:
            QtWidgets.QMessageBox.information(self, "Invalid Grid", "Grid sample size must be greater than zero.")
            return
        self.build_model_window.save_elev_history(elev_path_text)
        elev_path = Path(elev_path_text).expanduser().resolve()
        try:
            inlines, crosslines, elev_values = self._read_scatter_gmp(elev_path)
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "Load Failed", str(exc))
            return
        z_values = elev_values / np.float32(sample_size)
        elev_name = self.updater.add_scatter_data(
            elev_path.stem,
            inlines=inlines,
            crosslines=crosslines,
            z_values=z_values,
            values=elev_values,
            source_path=elev_path,
            select=True,
        )
        self._selected_data_item = ("scatter", elev_name)
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_load_dip_direction_dialog(self) -> None:
        if self.build_model_window is None:
            self.open_build_model_window()
        if self.build_model_window is None:
            return
        dip_path_text, direction_path_text = self.build_model_window.current_scatter_paths()
        if not dip_path_text or not direction_path_text:
            QtWidgets.QMessageBox.information(self, "Missing Files", "Please select both dip and direction files.")
            return
        self.build_model_window.save_scatter_history(dip_path_text, direction_path_text)
        try:
            dip_inlines, dip_crosslines, dip_values = self._read_scatter_gmp(Path(dip_path_text))
            direction_inlines, direction_crosslines, direction_values = self._read_scatter_gmp(Path(direction_path_text))
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "Load Failed", str(exc))
            return

        dip_name = self.updater.add_scatter_data(
            Path(dip_path_text).stem,
            inlines=dip_inlines,
            crosslines=dip_crosslines,
            z_values=None,
            values=dip_values,
            source_path=Path(dip_path_text),
            select=True,
        )
        self.updater.add_scatter_data(
            Path(direction_path_text).stem,
            inlines=direction_inlines,
            crosslines=direction_crosslines,
            z_values=None,
            values=direction_values,
            source_path=Path(direction_path_text),
            select=False,
        )
        self._selected_data_item = ("scatter", dip_name)
        self.refresh_data_panel()
        self.schedule_render()

    def open_load_geomap_dialog(self) -> None:
        if self.build_model_window is None:
            self.open_build_model_window()
        if self.build_model_window is None:
            return
        geomap_path_text, geomap_elev_path_text = self.build_model_window.current_geomap_inputs()
        if not geomap_path_text:
            QtWidgets.QMessageBox.information(self, "Missing File", "Please select a geomap file.")
            return
        self.build_model_window.save_geomap_history(geomap_path_text, geomap_elev_path_text)
        geomap_path = Path(geomap_path_text).expanduser().resolve()
        if not geomap_path.exists():
            QtWidgets.QMessageBox.information(self, "Missing File", f"Geomap file not found:\n{geomap_path}")
            return
        polygons = load_geomap_polygons(geomap_path)
        if not polygons:
            QtWidgets.QMessageBox.information(self, "Load Failed", "No polygons were found in the geomap file.")
            return
        grid_definition = self.updater.grid_definition
        if grid_definition is None:
            QtWidgets.QMessageBox.information(self, "Missing Grid", "Please define grid before loading geomap.")
            return
        elev_surface: ElevSurface | None = None
        sample_size = abs(float(grid_definition.sample_size))
        if geomap_elev_path_text:
            try:
                elev_inlines, elev_crosslines, elev_values = self._read_scatter_gmp(
                    Path(geomap_elev_path_text).expanduser().resolve()
                )
                elev_surface = build_elev_surface(elev_inlines, elev_crosslines, elev_values)
            except ValueError as exc:
                QtWidgets.QMessageBox.information(self, "Load Failed", str(exc))
                return

        clipped_polygons: list[tuple[tuple[int, int, int], np.ndarray, np.ndarray]] = []
        for color_rgb, points in polygons:
            clipped_points = clip_polygon_to_grid(points, grid_definition)
            clipped_points = normalize_polygon_grid_points(clipped_points)
            if clipped_points.shape[0] < 3:
                continue
            z_values = np.zeros(clipped_points.shape[0], dtype=np.float32)
            if elev_surface is not None:
                for index, point in enumerate(clipped_points):
                    elev_value = elev_surface.sample(float(point[0]), float(point[1]))
                    z_values[index] = float(elev_value / max(sample_size, 1e-6))
            clipped_polygons.append((color_rgb, clipped_points, z_values))

        if not clipped_polygons:
            QtWidgets.QMessageBox.information(
                self,
                "Load Failed",
                "No polygons remain after clipping to the current grid range.",
            )
            self.refresh_data_panel()
            self.schedule_render()
            return
        first_polygon_name: str | None = None
        for index, (color_rgb, points, z_values) in enumerate(clipped_polygons, start=1):
            polygon_name = self.updater.add_polygon_data(
                f"{geomap_path.stem}_{index}",
                color_rgb=color_rgb,
                grid_points=points,
                z_values=z_values,
                source_path=geomap_path,
                select=first_polygon_name is None,
            )
            if first_polygon_name is None:
                first_polygon_name = polygon_name
        if first_polygon_name is None:
            QtWidgets.QMessageBox.information(self, "Load Failed", "No valid polygons were created from the geomap file.")
            return
        self._selected_data_item = ("polygon", first_polygon_name)
        configure_default_camera(self.renderer, self.updater.scene_image())
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

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
        project_derived_dir = self._project_subdir("derived", category)
        path = project_derived_dir if project_derived_dir is not None else DERIVED_DATA_DIR / category
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
                    value_colormap_name=str(
                        self._payload_scalar(payload.get("control_point_value_colormap_name", np.array(DEFAULT_COLORMAP_NAME)))
                    ),
                    value_color_range=(
                        tuple(np.asarray(payload["control_point_value_color_range"], dtype=np.float32).ravel()[:2])
                        if "control_point_value_color_range" in payload
                        else None
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

        if category == "scatter":
            self.open_load_dip_direction_dialog()
            return

        if category == "polygon":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Polygon",
                str(self._default_output_dir(category)),
                "Wesi3D Polygon (*.npz);;Geomap GMP (*.gmp);;All Files (*)",
            )
            if not path:
                return
            selected_path = Path(path).expanduser().resolve()
            if selected_path.suffix.lower() == ".npz":
                payload = self._load_npz_payload(str(selected_path))
                color_rgb = tuple(
                    int(v)
                    for v in np.asarray(
                        payload.get("color_rgb", np.array([240, 210, 120], dtype=np.int32)),
                        dtype=np.int32,
                    ).ravel()[:3]
                )
                z_values = (
                    np.asarray(payload["z_values"], dtype=np.float32)
                    if "z_values" in payload
                    else None
                )
                name = self.updater.add_polygon_data(
                    str(self._payload_scalar(payload["name"])),
                    color_rgb=color_rgb,
                    grid_points=np.asarray(payload["grid_points"], dtype=np.float32),
                    z_values=z_values,
                    source_path=Path(str(self._payload_scalar(payload.get("source_path", np.array(str(selected_path)))))),
                    select=True,
                )
                self._selected_data_item = ("polygon", name)
                self.refresh_data_panel()
                self.schedule_render()
                return
            polygons = load_geomap_polygons(selected_path)
            if not polygons:
                QtWidgets.QMessageBox.information(self, "Load Failed", "No polygons were found in the selected file.")
                return
            first_polygon_name: str | None = None
            for index, (color_rgb, points) in enumerate(polygons, start=1):
                polygon_name = self.updater.add_polygon_data(
                    f"{selected_path.stem}_{index}",
                    color_rgb=color_rgb,
                    grid_points=points,
                    z_values=np.zeros(points.shape[0], dtype=np.float32),
                    source_path=selected_path,
                    select=first_polygon_name is None,
                )
                if first_polygon_name is None:
                    first_polygon_name = polygon_name
            if first_polygon_name is not None:
                self._selected_data_item = ("polygon", first_polygon_name)
                self.refresh_data_panel()
                self.schedule_render()
            return

        if category == "model":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Model Surface",
                str(self._default_output_dir(category)),
                "NumPy Archive (*.npz)",
            )
            if not path:
                return
            payload = self._load_npz_payload(path)
            polydata = polydata_from_payload(
                np.asarray(payload["polydata_points"], dtype=np.float32),
                np.asarray(payload["polydata_polys_offsets"], dtype=np.int64),
                np.asarray(payload["polydata_polys_connectivity"], dtype=np.int64),
            )
            name = self.updater.add_model_surface(
                str(self._payload_scalar(payload["name"])),
                polydata=polydata,
                source_polygon_name=str(self._payload_scalar(payload.get("source_polygon_name", np.array("")))),
                dip_source_path=Path(str(self._payload_scalar(payload.get("dip_source_path", np.array(""))))),
                direction_source_path=Path(str(self._payload_scalar(payload.get("direction_source_path", np.array(""))))),
                select=True,
            )
            self._selected_data_item = ("model", name)
            self.refresh_data_panel()
            self.schedule_render()
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
                            "control_point_value_colormap_name": np.array(point_set.value_colormap_name),
                            "control_point_value_color_range": np.asarray(
                                point_set.value_color_range if point_set.value_color_range is not None else (0.0, 1.0),
                                dtype=np.float32,
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

        if category == "scatter":
            scatter = self.updater.scatter_sets[name]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Store Scatter",
                str(self._default_output_dir(category) / f"{name}.npz"),
                "NumPy Archive (*.npz)",
            )
            if path:
                np.savez_compressed(
                    path,
                    name=np.array(scatter.name),
                    inlines=np.asarray(scatter.inlines, dtype=np.float32),
                    crosslines=np.asarray(scatter.crosslines, dtype=np.float32),
                    z_values=np.asarray(scatter.z_values, dtype=np.float32),
                    values=np.asarray(scatter.values, dtype=np.float32),
                    source_path=np.array(str(scatter.source_path)),
                )
            return

        if category == "polygon":
            polygon = self.updater.polygon_sets[name]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Store Polygon",
                str(self._default_output_dir(category) / f"{name}.npz"),
                "Wesi3D Polygon (*.npz);;Geomap GMP (*.gmp);;All Files (*)",
            )
            if path:
                output_path = Path(path)
                if output_path.suffix.lower() == ".gmp":
                    lines = ["Area", f"##{polygon.color_rgb[0]} {polygon.color_rgb[1]} {polygon.color_rgb[2]}"]
                    for point in np.asarray(polygon.grid_points, dtype=np.float32):
                        lines.append(f"{float(point[0]):.3f} {float(point[1]):.3f}")
                    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                else:
                    np.savez_compressed(
                        output_path,
                        name=np.array(polygon.name),
                        color_rgb=np.asarray(polygon.color_rgb, dtype=np.int32),
                        grid_points=np.asarray(polygon.grid_points, dtype=np.float32),
                        z_values=np.asarray(polygon.z_values, dtype=np.float32),
                        source_path=np.array(str(polygon.source_path)),
                    )
            return

        if category == "model":
            model_surface = self.updater.model_surfaces[name]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Store Model Surface",
                str(self._default_output_dir(category) / f"{name}.npz"),
                "NumPy Archive (*.npz)",
            )
            if path:
                polydata_payload = polydata_to_payload(model_surface.polydata)
                np.savez_compressed(
                    path,
                    name=np.array(model_surface.name),
                    source_polygon_name=np.array(model_surface.source_polygon_name),
                    dip_source_path=np.array(str(model_surface.dip_source_path)),
                    direction_source_path=np.array(str(model_surface.direction_source_path)),
                    polydata_points=polydata_payload["points"],
                    polydata_polys_offsets=polydata_payload["polys_offsets"],
                    polydata_polys_connectivity=polydata_payload["polys_connectivity"],
                )
            return

        QtWidgets.QMessageBox.information(self, "Not Implemented", "井数据存储尚未实现。")

    def unload_data_item(self, category: str, name: str) -> None:
        removed = False
        if category in {"seismic", "attribute"}:
            removed = self.updater.remove_attribute(name)
            self.image = self.updater.image
        elif category == "horizon":
            removed = self.updater.remove_horizon(name)
        elif category == "scatter":
            removed = self.updater.remove_scatter(name)
        elif category == "polygon":
            removed = self.updater.remove_polygon(name)
        elif category == "model":
            removed = self.updater.remove_model_surface(name)
        else:
            QtWidgets.QMessageBox.information(self, "Not Implemented", "井数据卸载尚未实现。")
            return

        if not removed:
            QtWidgets.QMessageBox.information(self, "Unload Failed", "当前数据无法卸载。")
            return

        current_scatter = self.updater.current_scatter()
        if current_scatter is not None:
            self._selected_data_item = ("scatter", current_scatter.name)
        else:
            current_polygon = self.updater.current_polygon()
            if current_polygon is not None:
                self._selected_data_item = ("polygon", current_polygon.name)
            else:
                current_model = self.updater.current_model_surface()
                if current_model is not None:
                    self._selected_data_item = ("model", current_model.name)
                else:
                    current_attribute = self.updater.current_attribute()
                    if current_attribute is not None and self.updater.current_attribute_name is not None:
                        fallback_category = str(
                            current_attribute.volume_data.metadata.get(
                                "panel_category",
                                "seismic" if self.updater.current_attribute_name == "seismic" else "attribute",
                            )
                        )
                        self._selected_data_item = (fallback_category, self.updater.current_attribute_name)
                    else:
                        self._selected_data_item = None
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

    def open_interpolate_volume_dialog(self) -> None:
        horizon_names = [
            name
            for name, horizon in self.updater.horizons.items()
            if horizon.control_point_set is not None
        ]
        attribute_names = self.updater.attribute_names()
        if not attribute_names:
            QtWidgets.QMessageBox.information(
                self,
                "No Attribute",
                "Load or select an attribute to provide the interpolation grid.",
            )
            return
        if not horizon_names:
            QtWidgets.QMessageBox.information(
                self,
                "No Control Points",
                "Select a horizon that already contains control points.",
            )
            return
        current_horizon_name = (
            self.updater.current_horizon_name
            if self.updater.current_horizon_name in horizon_names
            else horizon_names[0]
        )
        dialog = InterpolateVolumeDialog(
            attribute_names,
            horizon_names,
            selected_attribute_name=self.updater.current_attribute_name,
            selected_horizon_name=current_horizon_name,
            parent=self,
        )
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        attribute_name, horizon_name, output_name, idw_radius, apply_mask = values
        new_name = self.updater.interpolate_attribute_from_control_points(
            attribute_name,
            horizon_name,
            output_name,
            idw_radius=idw_radius,
            apply_mask=apply_mask,
        )
        if new_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "Interpolation Failed",
                self.updater.last_rebuild_error or "Failed to interpolate a new attribute volume.",
            )
            return
        self.image = self.updater.image
        self._selected_data_item = ("attribute", new_name)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_extract_horizon_mask_dialog(self) -> None:
        current_horizon = self.updater.current_horizon()
        if current_horizon is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Horizon",
                "Select a current horizon before extracting its mask.",
            )
            return
        dialog = ExtractCurrentHorizonMaskDialog(current_horizon.name, self)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        output_name = dialog.values()
        if output_name is None:
            return
        new_name = self.updater.extract_mask_from_current_horizon(output_name)
        if new_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "Extract Failed",
                self.updater.last_rebuild_error or "Failed to extract the current horizon mask.",
            )
            return
        self.image = self.updater.image
        self._selected_data_item = ("attribute", new_name)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def open_replace_volume_dialog(self) -> None:
        attribute_names = self.updater.attribute_names()
        horizon_names = self.updater.horizon_names()
        if len(attribute_names) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Need Two Attributes",
                "Load at least two attributes before using replace.",
            )
            return
        if not horizon_names:
            QtWidgets.QMessageBox.information(
                self,
                "No Horizon",
                "Load or select a horizon before using replace.",
            )
            return
        current_attribute_name = self.updater.current_attribute_name
        selected_source_name = None
        for name in attribute_names:
            if name != current_attribute_name:
                selected_source_name = name
                break
        dialog = ReplaceVolumeByHorizonDialog(
            attribute_names,
            horizon_names,
            selected_target_attribute_name=current_attribute_name,
            selected_source_attribute_name=selected_source_name,
            selected_horizon_name=self.updater.current_horizon_name,
            parent=self,
        )
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        values = dialog.values()
        if values is None:
            return
        target_name, source_name, horizon_name, output_name = values
        new_name = self.updater.replace_attribute_by_horizon_mask(
            target_name,
            source_name,
            horizon_name,
            output_name,
        )
        if new_name is None:
            QtWidgets.QMessageBox.information(
                self,
                "Replace Failed",
                self.updater.last_rebuild_error or "Failed to replace attribute values by horizon mask.",
            )
            return
        self.image = self.updater.image
        self._selected_data_item = ("attribute", new_name)
        self.refresh_axis_controls()
        self.refresh_scene_guides()
        self.refresh_info()
        self.refresh_data_panel()
        self.schedule_render()

    def change_colormap_target(self) -> None:
        self.refresh_display_controls()

    def toggle_control_point_colormap(self, checked: bool) -> None:
        self.updater.set_control_point_use_attribute_colormap(bool(checked), render=False)
        self._refresh_selected_master_actor()
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

    def apply_active_colormap_range(self) -> None:
        min_text = self.attribute_colormap_widget.min_edit.text().strip()
        max_text = self.attribute_colormap_widget.max_edit.text().strip()
        if not min_text or not max_text:
            return
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return
        if self.attribute_colormap_widget.current_target() == "control_point":
            self.updater.set_control_point_colormap_range(min_value, max_value, render=False)
            self.refresh_display_controls()
        else:
            self.updater.set_attribute_display_range(min_value, max_value, render=False)
        self.schedule_render()

    def change_active_colormap(self, name: str) -> None:
        if not name:
            return
        if self.attribute_colormap_widget.current_target() == "control_point":
            self.updater.set_control_point_colormap(name, render=False)
        else:
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
