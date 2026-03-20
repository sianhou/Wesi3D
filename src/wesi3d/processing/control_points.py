#!/usr/bin/env python3
"""
Control point extraction helpers for closed 3D horizons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.volume_data import VolumeData

_NEIGHBOR_OFFSETS_6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


@dataclass
class ControlPoint:
    xline_index: int
    inline_index: int
    sample_index: int
    xline: float
    inline: float
    sample: float
    value: float
    kind: str
    master_index: int | None = None
    dz: float = 0.0


def _validate_interval(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _shift_mask(mask: np.ndarray, dx: int, dy: int, dz: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    src_x = slice(max(0, -dx), mask.shape[0] - max(0, dx))
    src_y = slice(max(0, -dy), mask.shape[1] - max(0, dy))
    src_z = slice(max(0, -dz), mask.shape[2] - max(0, dz))
    dst_x = slice(max(0, dx), mask.shape[0] - max(0, -dx))
    dst_y = slice(max(0, dy), mask.shape[1] - max(0, -dy))
    dst_z = slice(max(0, dz), mask.shape[2] - max(0, -dz))
    shifted[dst_x, dst_y, dst_z] = mask[src_x, src_y, src_z]
    return shifted


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    for dx, dy, dz in _NEIGHBOR_OFFSETS_6:
        interior &= _shift_mask(mask, dx, dy, dz)
    return mask & ~interior


def _build_point(
    volume_data: VolumeData,
    xline_index: int,
    inline_index: int,
    sample_index: int,
    kind: str,
) -> ControlPoint:
    return ControlPoint(
        xline_index=int(xline_index),
        inline_index=int(inline_index),
        sample_index=int(sample_index),
        xline=float(volume_data.xlines[xline_index]),
        inline=float(volume_data.inlines[inline_index]),
        sample=float(volume_data.samples[sample_index]),
        value=float(volume_data.data[xline_index, inline_index, sample_index]),
        kind=kind,
    )


def extract_control_points(
    volume_data: VolumeData,
    component_mask: np.ndarray,
    *,
    surface_xline_interval: int = 8,
    surface_inline_interval: int = 8,
    interior_xline_interval: int = 8,
    interior_inline_interval: int = 8,
    interior_sample_interval: int = 8,
) -> list[ControlPoint]:
    surface_xline_interval = _validate_interval("surface_xline_interval", surface_xline_interval)
    surface_inline_interval = _validate_interval("surface_inline_interval", surface_inline_interval)
    interior_xline_interval = _validate_interval("interior_xline_interval", interior_xline_interval)
    interior_inline_interval = _validate_interval("interior_inline_interval", interior_inline_interval)
    interior_sample_interval = _validate_interval("interior_sample_interval", interior_sample_interval)

    mask = np.asarray(component_mask, dtype=bool)
    surface = boundary_mask(mask)
    interior = mask & ~surface
    points: list[ControlPoint] = []

    for xline_index, inline_index, sample_index in np.argwhere(surface):
        if xline_index % surface_xline_interval != 0:
            continue
        if inline_index % surface_inline_interval != 0:
            continue
        points.append(
            _build_point(
                volume_data,
                int(xline_index),
                int(inline_index),
                int(sample_index),
                "surface",
            )
        )

    for xline_index, inline_index, sample_index in np.argwhere(interior):
        if xline_index % interior_xline_interval != 0:
            continue
        if inline_index % interior_inline_interval != 0:
            continue
        if sample_index % interior_sample_interval != 0:
            continue
        points.append(
            _build_point(
                volume_data,
                int(xline_index),
                int(inline_index),
                int(sample_index),
                "interior",
            )
        )

    points.sort(key=lambda p: (p.kind, p.xline_index, p.inline_index, p.sample_index))
    master_index = 0
    for point in points:
        if point.kind == "surface":
            point.master_index = master_index
            master_index += 1
    return points


def master_control_points(points: list[ControlPoint]) -> list[ControlPoint]:
    return [
        point
        for point in points
        if point.kind == "surface" and point.master_index is not None
    ]


def apply_master_point_z_move(
    points: list[ControlPoint],
    selected_master_index: int,
    delta_sample: float,
    volume_data: VolumeData,
    *,
    influence_sigma: float = 12.0,
) -> list[ControlPoint]:
    if not points:
        return []
    surface_points = [point for point in points if point.kind == "surface" and point.master_index is not None]
    if not surface_points:
        return [ControlPoint(**vars(point)) for point in points]

    sigma = max(float(influence_sigma), 1.0)
    selected = None
    for point in surface_points:
        if point.master_index == selected_master_index:
            selected = point
            break
    if selected is None:
        raise ValueError(f"Unknown master point index: {selected_master_index}")

    new_points: list[ControlPoint] = []
    min_sample = 0
    max_sample = len(volume_data.samples) - 1
    for point in points:
        dx = float(point.xline_index - selected.xline_index)
        dy = float(point.inline_index - selected.inline_index)
        distance2 = dx * dx + dy * dy
        weight = float(np.exp(-distance2 / (2.0 * sigma * sigma)))
        dz = float(delta_sample) * weight
        new_sample_index = int(np.clip(round(point.sample_index + dz), min_sample, max_sample))
        new_point = ControlPoint(
            xline_index=int(point.xline_index),
            inline_index=int(point.inline_index),
            sample_index=new_sample_index,
            xline=float(point.xline),
            inline=float(point.inline),
            sample=float(volume_data.samples[new_sample_index]),
            value=float(volume_data.data[point.xline_index, point.inline_index, new_sample_index]),
            kind=point.kind,
            master_index=point.master_index,
            dz=float(point.dz + dz),
        )
        new_points.append(new_point)
    return new_points


def rebuild_mask_from_control_points(
    shape: tuple[int, int, int],
    points: list[ControlPoint],
    *,
    fill_radius: int = 1,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not points:
        return mask

    for point in points:
        xi = int(point.xline_index)
        yi = int(point.inline_index)
        zi = int(point.sample_index)
        x0 = max(0, xi - fill_radius)
        x1 = min(shape[0], xi + fill_radius + 1)
        y0 = max(0, yi - fill_radius)
        y1 = min(shape[1], yi + fill_radius + 1)
        z0 = max(0, zi - fill_radius)
        z1 = min(shape[2], zi + fill_radius + 1)
        mask[x0:x1, y0:y1, z0:z1] = True

    surface_points = [point for point in points if point.kind == "surface"]
    if surface_points:
        columns: dict[tuple[int, int], list[int]] = {}
        for point in surface_points:
            columns.setdefault((point.xline_index, point.inline_index), []).append(point.sample_index)
        for (xi, yi), zs in columns.items():
            z_min = max(0, min(zs))
            z_max = min(shape[2] - 1, max(zs))
            mask[int(xi), int(yi), z_min : z_max + 1] = True

    interior_points = [point for point in points if point.kind == "interior"]
    if interior_points:
        for point in interior_points:
            xi = int(point.xline_index)
            yi = int(point.inline_index)
            zi = int(point.sample_index)
            mask[xi, yi, zi] = True
    return mask
