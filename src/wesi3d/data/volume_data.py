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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from ..utils.constants import INLINE_FIELD, XLINE_FIELD

try:
    import segyio
except ImportError:
    segyio = None


AxisName = Literal["xline", "inline", "sample"]


@dataclass(frozen=True)
class SegyGeometry:
    inlines: np.ndarray
    xlines: np.ndarray
    sample_axis: np.ndarray
    trace_index_grid: np.ndarray


@dataclass(frozen=True)
class VolumeData:
    """
    In-memory seismic cube with explicit physical axes.

    Shape convention:
        data[xline_index, inline_index, sample_index]
    """

    data: np.ndarray
    xlines: np.ndarray
    inlines: np.ndarray
    samples: np.ndarray
    name: str = "volume"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float32)
        xlines = np.asarray(self.xlines)
        inlines = np.asarray(self.inlines)
        samples = np.asarray(self.samples)
        if data.ndim != 3:
            raise ValueError("volume data must be 3D")
        expected_shape = (len(xlines), len(inlines), len(samples))
        if data.shape != expected_shape:
            raise ValueError(
                f"volume shape {data.shape} does not match axes {expected_shape}"
            )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "xlines", xlines)
        object.__setattr__(self, "inlines", inlines)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def axis_values(self, axis: AxisName) -> np.ndarray:
        if axis == "xline":
            return self.xlines
        if axis == "inline":
            return self.inlines
        return self.samples

    def renamed(self, name: str) -> "VolumeData":
        return VolumeData(
            data=self.data,
            xlines=self.xlines,
            inlines=self.inlines,
            samples=self.samples,
            name=name,
            metadata=self.metadata,
        )

    def with_data(
        self,
        data: np.ndarray,
        *,
        name: str | None = None,
        xlines: np.ndarray | None = None,
        inlines: np.ndarray | None = None,
        samples: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "VolumeData":
        return VolumeData(
            data=data,
            xlines=self.xlines if xlines is None else xlines,
            inlines=self.inlines if inlines is None else inlines,
            samples=self.samples if samples is None else samples,
            name=self.name if name is None else name,
            metadata=self.metadata if metadata is None else metadata,
        )


def _require_segyio() -> None:
    if segyio is None:
        raise RuntimeError("Missing dependency: segyio")


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
        raise ValueError("Inline headers do not form a consistent regular grid.")
    if not np.array_equal(sorted_xlines[0, :], unique_xlines):
        raise ValueError("Crossline headers do not form a consistent regular grid.")

    return sort_idx.reshape(ni, nx)


def load_segy_geometry(
    segy_path: str | Path,
    inline_field: int = INLINE_FIELD,
    xline_field: int = XLINE_FIELD,
) -> SegyGeometry:
    _require_segyio()
    segy_path = Path(segy_path)
    with segyio.open(str(segy_path), "r", strict=False, ignore_geometry=True) as segy:
        iline_vals = np.asarray(segy.attributes(inline_field)[:], dtype=np.int64)
        xline_vals = np.asarray(segy.attributes(xline_field)[:], dtype=np.int64)
        sample_axis = np.asarray(segy.samples, dtype=np.float32)

    return SegyGeometry(
        inlines=np.unique(iline_vals),
        xlines=np.unique(xline_vals),
        sample_axis=sample_axis,
        trace_index_grid=detect_regular_grid(iline_vals, xline_vals),
    )


def read_segy_volume(
    segy_path: str | Path,
    *,
    geometry: SegyGeometry | None = None,
    interval_inline: int = 1,
    interval_xline: int = 1,
    interval_sample: int = 1,
    inline_field: int = INLINE_FIELD,
    xline_field: int = XLINE_FIELD,
    dtype: np.dtype | str = np.float32,
    name: str | None = None,
) -> VolumeData:
    _require_segyio()
    segy_path = Path(segy_path)
    geometry = geometry or load_segy_geometry(
        segy_path,
        inline_field=inline_field,
        xline_field=xline_field,
    )

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

    volume = np.empty(
        (len(xl_idx), len(il_idx), len(z_idx)),
        dtype=np.dtype(dtype),
        order="F",
    )

    with segyio.open(str(segy_path), "r", strict=False, ignore_geometry=True) as segy:
        for out_y, il_pos in enumerate(il_idx):
            trace_indices = geometry.trace_index_grid[il_pos, xl_idx]
            for out_x, trace_idx in enumerate(trace_indices):
                trace = np.asarray(segy.trace[int(trace_idx)], dtype=np.float32)
                volume[out_x, out_y, :] = trace[z_idx]

    return VolumeData(
        data=volume,
        xlines=geometry.xlines[xl_idx],
        inlines=geometry.inlines[il_idx],
        samples=geometry.sample_axis[z_idx],
        name=segy_path.stem if name is None else name,
        metadata={
            "source_path": str(segy_path),
            "interval_inline": int(interval_inline),
            "interval_xline": int(interval_xline),
            "interval_sample": int(interval_sample),
        },
    )


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
