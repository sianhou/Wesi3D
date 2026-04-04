#!/usr/bin/env python3
"""
NPZ-oriented volume container.

VolumeData2 is a lightweight storage layer that matches the importer output
format exactly:
    - type
    - range
    - grid
    - data
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AxisRange:
    begin: int
    end: int
    step: int
    spacing: int

    def as_dict(self) -> dict[str, int]:
        return {
            "begin": int(self.begin),
            "end": int(self.end),
            "step": int(self.step),
            "spacing": int(self.spacing),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "AxisRange":
        return cls(
            begin=int(values["begin"]),
            end=int(values["end"]),
            step=int(values["step"]),
            spacing=int(values["spacing"]),
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
class VolumeData2:
    type: str
    range: RangeMeta
    grid: GridMeta
    data: np.ndarray

    def __post_init__(self) -> None:
        if str(self.type) != "volume":
            raise ValueError("VolumeData2.type must be 'volume'")
        array = np.asarray(self.data, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("VolumeData2.data must be 3D")
        object.__setattr__(self, "type", "volume")
        object.__setattr__(self, "data", array)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    def to_npz(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser()
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            type=np.asarray(json.dumps(self.type)),
            range=np.asarray(json.dumps(self.range.as_dict())),
            grid=np.asarray(json.dumps(self.grid.as_dict())),
            data=np.asarray(self.data, dtype=np.float32),
        )
        return output_path

    @classmethod
    def from_npz(cls, path: str | Path) -> "VolumeData2":
        npz_path = Path(path).expanduser().resolve()
        with np.load(npz_path, allow_pickle=False) as archive:
            type_name = json.loads(_payload_text(archive["type"]))
            range_meta = RangeMeta.from_dict(json.loads(_payload_text(archive["range"])))
            grid_meta = GridMeta.from_dict(json.loads(_payload_text(archive["grid"])))
            data = np.asarray(archive["data"], dtype=np.float32)
        return cls(
            type=str(type_name),
            range=range_meta,
            grid=grid_meta,
            data=data,
        )


def _payload_text(value: object) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)
