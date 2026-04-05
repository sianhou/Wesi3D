#!/usr/bin/env python3
"""
Display-oriented view built from VolumePackage.

This layer is intentionally light: it wraps a VolumePackage and exposes
derived display properties without duplicating the package metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volume_package import AxisRange, VolumePackage


def build_axis_values(axis_range: AxisRange) -> np.ndarray:
    values = np.arange(
        int(axis_range.begin),
        int(axis_range.end) + int(axis_range.step),
        int(axis_range.step),
        dtype=np.int32,
    )
    if values.size == 0:
        raise ValueError("Axis values are empty")
    return values


@dataclass(frozen=True)
class VolumeDisplay:
    package: VolumePackage

    @property
    def type(self) -> str:
        return self.package.type

    @property
    def data(self) -> np.ndarray:
        return self.package.data

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def inline_values(self) -> np.ndarray:
        return build_axis_values(self.package.range.inline)

    @property
    def cxline_values(self) -> np.ndarray:
        return build_axis_values(self.package.range.cxline)

    @property
    def sample_values(self) -> np.ndarray:
        return build_axis_values(self.package.range.sample)

    @property
    def inline_spacing(self) -> float:
        return float(self.package.range.inline.spacing)

    @property
    def cxline_spacing(self) -> float:
        return float(self.package.range.cxline.spacing)

    @property
    def sample_spacing(self) -> float:
        return float(self.package.range.sample.spacing)

    @property
    def p0(self) -> tuple[float, float]:
        return (float(self.package.grid.p0.x), float(self.package.grid.p0.y))

    @property
    def p1(self) -> tuple[float, float]:
        return (float(self.package.grid.p1.x), float(self.package.grid.p1.y))

    @property
    def p3(self) -> tuple[float, float]:
        return (float(self.package.grid.p3.x), float(self.package.grid.p3.y))

    def axis_values(self, axis: str) -> np.ndarray:
        if axis == "inline":
            return self.inline_values
        if axis == "cxline":
            return self.cxline_values
        if axis == "sample":
            return self.sample_values
        raise KeyError(f"Unsupported axis: {axis}")


def volume_display_from_package(package: VolumePackage) -> VolumeDisplay:
    return VolumeDisplay(package=package)
