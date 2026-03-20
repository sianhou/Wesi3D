#!/usr/bin/env python3
"""
Attribute loading helpers for converting VolumeData into renderable data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volume_data import VolumeData

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
    opacity: float = 0.85


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


def create_lookup_table(
    image: vtk.vtkImageData,
    clip_percentile: float,
) -> vtk.vtkLookupTable:
    scalars = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
    return create_lookup_table_from_scalars(scalars, clip_percentile)


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
    image = create_vtk_image(volume_data, spacing)
    lut = create_lookup_table(image, clip_percentile)
    return AttributeVolume(
        name=attribute_name,
        volume_data=volume_data.renamed(attribute_name),
        image=image,
        lut=lut,
        opacity=float(opacity),
    )
