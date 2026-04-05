#!/usr/bin/env python3
"""
Package-oriented volume data model.

VolumePackage mirrors the current on-disk npz structure:
    - type
    - range
    - grid
    - data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisRange:
    begin: int
    end: int
    step: int
    spacing: float

    def as_dict(self) -> dict[str, object]:
        return {
            "begin": int(self.begin),
            "end": int(self.end),
            "step": int(self.step),
            "spacing": float(self.spacing),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "AxisRange":
        return cls(
            begin=int(values["begin"]),
            end=int(values["end"]),
            step=int(values["step"]),
            spacing=float(values["spacing"]),
        )


@dataclass(frozen=True)
class GridPoint:
    x: float
    y: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "GridPoint":
        return cls(
            x=float(values["x"]),
            y=float(values["y"]),
        )


@dataclass(frozen=True)
class RangeMeta:
    inline: AxisRange
    cxline: AxisRange
    sample: AxisRange

    def as_dict(self) -> dict[str, object]:
        return {
            "inline": self.inline.as_dict(),
            "cxline": self.cxline.as_dict(),
            "sample": self.sample.as_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "RangeMeta":
        return cls(
            inline=AxisRange.from_dict(dict(values["inline"])),
            cxline=AxisRange.from_dict(dict(values["cxline"])),
            sample=AxisRange.from_dict(dict(values["sample"])),
        )


@dataclass(frozen=True)
class GridMeta:
    p0: GridPoint
    p1: GridPoint
    p3: GridPoint

    def as_dict(self) -> dict[str, object]:
        return {
            "p0": self.p0.as_dict(),
            "p1": self.p1.as_dict(),
            "p3": self.p3.as_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "GridMeta":
        return cls(
            p0=GridPoint.from_dict(dict(values["p0"])),
            p1=GridPoint.from_dict(dict(values["p1"])),
            p3=GridPoint.from_dict(dict(values["p3"])),
        )


@dataclass(frozen=True)
class VolumePackage:
    type: str
    range: RangeMeta
    grid: GridMeta
    data: np.ndarray

    def __post_init__(self) -> None:
        if str(self.type) != "volume":
            raise ValueError("VolumePackage.type must be 'volume'")
        array = np.asarray(self.data, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("VolumePackage.data must be 3D")
        object.__setattr__(self, "type", "volume")
        object.__setattr__(self, "data", array)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)
