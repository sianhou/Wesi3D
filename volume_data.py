#!/usr/bin/env python3
"""
Memory-mapped volume data structure for large 3D cubes.

Design goals:
- fast orthogonal slice access for very large volumes
- metadata keeps real axis values such as inline/crossline/sample
- lightweight cache for repeated slice browsing
- easy handoff to VTK later

Data layout is fixed as:
    (xline, inline, sample)

This matches the viewer code in this workspace and keeps the
crossline/inline/sample naming explicit.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

AxisName = Literal["xline", "inline", "sample"]


@dataclass(frozen=True)
class AxisDescriptor:
    name: AxisName
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 1:
            raise ValueError(f"{self.name} axis must be 1D")
        if len(values) == 0:
            raise ValueError(f"{self.name} axis must not be empty")
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        return int(len(self.values))

    def clamp_index(self, index: int) -> int:
        return max(0, min(int(index), self.size - 1))

    def nearest_index(self, value: float) -> int:
        idx = int(np.argmin(np.abs(self.values.astype(np.float64) - value)))
        return self.clamp_index(idx)

    def value_at(self, index: int) -> float:
        return float(self.values[self.clamp_index(index)])

    def spacing(self) -> float:
        if self.size < 2:
            return 1.0
        return float(np.median(np.diff(self.values.astype(np.float64))))


@dataclass(frozen=True)
class VolumeSpec:
    shape: tuple[int, int, int]
    dtype: str
    order: Literal["C", "F"]
    xline: AxisDescriptor
    inline: AxisDescriptor
    sample: AxisDescriptor

    def __post_init__(self) -> None:
        sx, si, ss = self.shape
        if self.xline.size != sx:
            raise ValueError("xline axis size does not match shape[0]")
        if self.inline.size != si:
            raise ValueError("inline axis size does not match shape[1]")
        if self.sample.size != ss:
            raise ValueError("sample axis size does not match shape[2]")

    @property
    def ndim(self) -> int:
        return 3

    @property
    def axis_map(self) -> dict[AxisName, AxisDescriptor]:
        return {
            "xline": self.xline,
            "inline": self.inline,
            "sample": self.sample,
        }

    def axis(self, name: AxisName) -> AxisDescriptor:
        return self.axis_map[name]

    def to_json_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "order": self.order,
            "axes": {
                "xline": self.xline.values.tolist(),
                "inline": self.inline.values.tolist(),
                "sample": self.sample.values.tolist(),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "VolumeSpec":
        axes = payload["axes"]
        return cls(
            shape=tuple(int(v) for v in payload["shape"]),
            dtype=str(payload["dtype"]),
            order=str(payload.get("order", "C")),
            xline=AxisDescriptor("xline", np.asarray(axes["xline"])),
            inline=AxisDescriptor("inline", np.asarray(axes["inline"])),
            sample=AxisDescriptor("sample", np.asarray(axes["sample"])),
        )


class SliceCache:
    def __init__(self, capacity: int = 24) -> None:
        self.capacity = max(1, int(capacity))
        self._cache: OrderedDict[tuple[AxisName, int], np.ndarray] = OrderedDict()

    def get(self, key: tuple[AxisName, int]) -> np.ndarray | None:
        value = self._cache.get(key)
        if value is None:
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, key: tuple[AxisName, int], value: np.ndarray) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


class LargeVolumeCube:
    """
    Large 3D volume backed by a memmap file.

    Shape convention:
        data[xline_index, inline_index, sample_index]
    """

    def __init__(
        self,
        data_path: str | Path,
        spec: VolumeSpec,
        mode: Literal["r", "r+", "w+"] = "r",
        slice_cache_size: int = 24,
    ) -> None:
        self.data_path = Path(data_path)
        self.spec = spec
        self.mode = mode
        self._memmap = np.memmap(
            self.data_path,
            dtype=np.dtype(self.spec.dtype),
            mode=mode,
            shape=self.spec.shape,
            order=self.spec.order,
        )
        self.slice_cache = SliceCache(slice_cache_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.spec.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.spec.dtype)

    @property
    def data(self) -> np.memmap:
        return self._memmap

    @classmethod
    def create(
        cls,
        data_path: str | Path,
        spec: VolumeSpec,
        fill_value: float = 0.0,
        metadata_path: str | Path | None = None,
    ) -> "LargeVolumeCube":
        cube = cls(data_path, spec, mode="w+")
        cube.data[:] = fill_value
        cube.flush()
        if metadata_path is not None:
            cube.save_metadata(metadata_path)
        return cube

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        data_path: str | Path,
        xlines: np.ndarray,
        inlines: np.ndarray,
        samples: np.ndarray,
        order: Literal["C", "F"] = "C",
        metadata_path: str | Path | None = None,
    ) -> "LargeVolumeCube":
        arr = np.asarray(array)
        if arr.ndim != 3:
            raise ValueError("array must be 3D")
        spec = VolumeSpec(
            shape=tuple(int(v) for v in arr.shape),
            dtype=str(arr.dtype),
            order=order,
            xline=AxisDescriptor("xline", xlines),
            inline=AxisDescriptor("inline", inlines),
            sample=AxisDescriptor("sample", samples),
        )
        cube = cls.create(data_path, spec, metadata_path=metadata_path)
        cube.data[:] = np.array(arr, copy=False, order=order)
        cube.flush()
        return cube

    @classmethod
    def open_with_metadata(
        cls,
        data_path: str | Path,
        metadata_path: str | Path,
        mode: Literal["r", "r+", "w+"] = "r",
        slice_cache_size: int = 24,
    ) -> "LargeVolumeCube":
        with open(metadata_path, "r", encoding="utf-8") as f:
            spec = VolumeSpec.from_json_dict(json.load(f))
        return cls(data_path, spec, mode=mode, slice_cache_size=slice_cache_size)

    def save_metadata(self, metadata_path: str | Path) -> None:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.spec.to_json_dict(), f, indent=2)

    def flush(self) -> None:
        self.data.flush()

    def axis(self, name: AxisName) -> AxisDescriptor:
        return self.spec.axis(name)

    def axis_index(self, name: AxisName, value: float) -> int:
        return self.axis(name).nearest_index(value)

    def axis_value(self, name: AxisName, index: int) -> float:
        return self.axis(name).value_at(index)

    def slice_by_index(self, axis: AxisName, index: int, copy: bool = False) -> np.ndarray:
        axis_desc = self.axis(axis)
        clamped = axis_desc.clamp_index(index)
        cache_key = (axis, clamped)
        cached = self.slice_cache.get(cache_key)
        if cached is not None:
            return cached.copy() if copy else cached

        if axis == "xline":
            view = self.data[clamped, :, :]
        elif axis == "inline":
            view = self.data[:, clamped, :]
        else:
            view = self.data[:, :, clamped]

        out = np.array(view, copy=copy)
        if not copy:
            self.slice_cache.put(cache_key, out)
        return out

    def slice_by_value(self, axis: AxisName, value: float, copy: bool = False) -> tuple[int, np.ndarray]:
        index = self.axis_index(axis, value)
        return index, self.slice_by_index(axis, index, copy=copy)

    def orthogonal_slices(
        self,
        xline_index: int | None = None,
        inline_index: int | None = None,
        sample_index: int | None = None,
        copy: bool = False,
    ) -> dict[AxisName, np.ndarray]:
        xline_index = self.axis("xline").clamp_index(
            len(self.spec.xline.values) // 2 if xline_index is None else xline_index
        )
        inline_index = self.axis("inline").clamp_index(
            len(self.spec.inline.values) // 2 if inline_index is None else inline_index
        )
        sample_index = self.axis("sample").clamp_index(
            len(self.spec.sample.values) // 2 if sample_index is None else sample_index
        )
        return {
            "xline": self.slice_by_index("xline", xline_index, copy=copy),
            "inline": self.slice_by_index("inline", inline_index, copy=copy),
            "sample": self.slice_by_index("sample", sample_index, copy=copy),
        }

    def subvolume(
        self,
        xline_slice: slice,
        inline_slice: slice,
        sample_slice: slice,
        copy: bool = False,
    ) -> np.ndarray:
        view = self.data[xline_slice, inline_slice, sample_slice]
        return np.array(view, copy=True) if copy else np.asarray(view)

    def preview(
        self,
        stride_xline: int = 4,
        stride_inline: int = 4,
        stride_sample: int = 4,
        copy: bool = False,
    ) -> np.ndarray:
        view = self.data[
            :: max(1, int(stride_xline)),
            :: max(1, int(stride_inline)),
            :: max(1, int(stride_sample)),
        ]
        return np.array(view, copy=True) if copy else np.asarray(view)

    def to_vtk_payload(
        self,
        xline_slice: slice | None = None,
        inline_slice: slice | None = None,
        sample_slice: slice | None = None,
        order: Literal["C", "F"] = "F",
    ) -> dict:
        """
        Package a subvolume for easy VTK handoff.

        Returned keys:
        - volume: contiguous ndarray
        - spacing: tuple[dx, dy, dz]
        - origin: tuple[ox, oy, oz]
        - axes: dict with sliced real axis values
        """
        xs = xline_slice or slice(None)
        ys = inline_slice or slice(None)
        zs = sample_slice or slice(None)
        volume = np.array(self.data[xs, ys, zs], order=order, copy=True)
        x_vals = self.spec.xline.values[xs]
        y_vals = self.spec.inline.values[ys]
        z_vals = self.spec.sample.values[zs]
        return {
            "volume": volume,
            "spacing": (
                self._spacing_from_values(x_vals),
                self._spacing_from_values(y_vals),
                self._spacing_from_values(z_vals),
            ),
            "origin": (
                float(x_vals[0]),
                float(y_vals[0]),
                float(z_vals[0]),
            ),
            "axes": {
                "xline": x_vals,
                "inline": y_vals,
                "sample": z_vals,
            },
        }

    @staticmethod
    def _spacing_from_values(values: np.ndarray) -> float:
        values = np.asarray(values)
        if len(values) < 2:
            return 1.0
        return float(np.median(np.diff(values.astype(np.float64))))


def build_volume_spec(
    xlines: np.ndarray,
    inlines: np.ndarray,
    samples: np.ndarray,
    dtype: str | np.dtype = np.float32,
    order: Literal["C", "F"] = "C",
) -> VolumeSpec:
    return VolumeSpec(
        shape=(len(xlines), len(inlines), len(samples)),
        dtype=str(np.dtype(dtype)),
        order=order,
        xline=AxisDescriptor("xline", np.asarray(xlines)),
        inline=AxisDescriptor("inline", np.asarray(inlines)),
        sample=AxisDescriptor("sample", np.asarray(samples)),
    )
