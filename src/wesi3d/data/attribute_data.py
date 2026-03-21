#!/usr/bin/env python3
"""
Attribute loading helpers for converting VolumeData into renderable data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volume_data import VolumeData

try:
    import vtk
    from vtk.util import numpy_support
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: vtk\n"
        "Install with: pip install vtk"
    ) from exc


@dataclass(frozen=True)
class RenderSpacing:
    xline: float = 20.0
    inline: float = 20.0
    sample: float = 10.0


@dataclass
class AttributeVolume:
    name: str
    volume_data: VolumeData
    image: vtk.vtkImageData
    lut: vtk.vtkLookupTable
    colormap_name: str = "bright_turbo"
    opacity: float = 0.85


DEFAULT_COLORMAP_NAME = "bright_turbo"
COLORMAP_PRESETS: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "bright_turbo": [
        (0.00, (0.20, 0.26, 0.78)),
        (0.25, (0.08, 0.72, 0.98)),
        (0.50, (0.18, 0.96, 0.72)),
        (0.75, (0.98, 0.90, 0.22)),
        (1.00, (0.92, 0.24, 0.18)),
    ],
    "bright_viridis": [
        (0.00, (0.18, 0.12, 0.42)),
        (0.25, (0.16, 0.42, 0.62)),
        (0.50, (0.14, 0.66, 0.56)),
        (0.75, (0.56, 0.84, 0.28)),
        (1.00, (0.98, 0.96, 0.24)),
    ],
    "bright_magma": [
        (0.00, (0.10, 0.06, 0.20)),
        (0.25, (0.40, 0.12, 0.38)),
        (0.50, (0.74, 0.22, 0.34)),
        (0.75, (0.96, 0.56, 0.18)),
        (1.00, (1.00, 0.96, 0.72)),
    ],
    "bright_cyan_amber": [
        (0.00, (0.08, 0.22, 0.44)),
        (0.30, (0.10, 0.76, 0.92)),
        (0.50, (0.92, 0.94, 0.96)),
        (0.70, (0.98, 0.74, 0.16)),
        (1.00, (0.72, 0.22, 0.08)),
    ],
    "bright_inferno": [
        (0.00, (0.06, 0.04, 0.20)),
        (0.25, (0.34, 0.08, 0.42)),
        (0.50, (0.74, 0.16, 0.24)),
        (0.75, (0.98, 0.62, 0.14)),
        (1.00, (1.00, 0.98, 0.78)),
    ],
    "bright_blue_red": [
        (0.00, (0.10, 0.32, 0.92)),
        (0.25, (0.12, 0.72, 1.00)),
        (0.50, (0.98, 0.98, 0.98)),
        (0.75, (1.00, 0.56, 0.26)),
        (1.00, (0.88, 0.14, 0.14)),
    ],
}


def available_colormap_names() -> list[str]:
    return list(COLORMAP_PRESETS.keys())


def apply_colormap_preset(lut: vtk.vtkLookupTable, name: str) -> vtk.vtkLookupTable:
    preset = COLORMAP_PRESETS.get(name, COLORMAP_PRESETS[DEFAULT_COLORMAP_NAME])
    lut.SetNumberOfTableValues(256)
    lut.Build()
    stops_t = np.asarray([stop[0] for stop in preset], dtype=np.float64)
    stops_rgb = np.asarray([stop[1] for stop in preset], dtype=np.float64)
    for i in range(256):
        t = i / 255.0
        r = float(np.interp(t, stops_t, stops_rgb[:, 0]))
        g = float(np.interp(t, stops_t, stops_rgb[:, 1]))
        b = float(np.interp(t, stops_t, stops_rgb[:, 2]))
        lut.SetTableValue(i, r, g, b, 1.0)
    return lut


def create_vtk_image(
    volume_data: VolumeData,
    spacing: RenderSpacing,
) -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(*volume_data.shape)
    image.SetSpacing(
        float(spacing.xline),
        float(spacing.inline),
        float(spacing.sample),
    )
    image.SetOrigin(0.0, 0.0, 0.0)

    vtk_array = numpy_support.numpy_to_vtk(
        np.asarray(volume_data.data).ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName(volume_data.name)
    image.GetPointData().SetScalars(vtk_array)
    return image


def create_lookup_table_from_scalars(
    scalars: np.ndarray,
    clip_percentile: float,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
) -> vtk.vtkLookupTable:
    scalars = np.asarray(scalars, dtype=np.float32)
    lower = float(np.percentile(scalars, 100.0 - clip_percentile))
    upper = float(np.percentile(scalars, clip_percentile))
    if lower == upper:
        lower = float(np.min(scalars))
        upper = float(np.max(scalars))
    if lower == upper:
        upper = lower + 1.0

    lut = vtk.vtkLookupTable()
    lut.SetRange(lower, upper)
    return apply_colormap_preset(lut, colormap_name)


def create_lookup_table(
    image: vtk.vtkImageData,
    clip_percentile: float,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
) -> vtk.vtkLookupTable:
    scalars = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
    return create_lookup_table_from_scalars(scalars, clip_percentile, colormap_name=colormap_name)


def load_attribute_from_volume(
    volume_data: VolumeData,
    *,
    name: str | None = None,
    spacing: RenderSpacing | None = None,
    clip_percentile: float = 99.0,
    opacity: float = 0.85,
) -> AttributeVolume:
    attribute_name = volume_data.name if name is None else name
    spacing = spacing or RenderSpacing()
    colormap_name = str(volume_data.metadata.get("colormap_name", DEFAULT_COLORMAP_NAME))
    image = create_vtk_image(volume_data, spacing)
    lut = create_lookup_table(image, clip_percentile, colormap_name=colormap_name)
    return AttributeVolume(
        name=attribute_name,
        volume_data=volume_data.renamed(attribute_name),
        image=image,
        lut=lut,
        colormap_name=colormap_name,
        opacity=float(opacity),
    )
