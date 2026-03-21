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
    base_sample_index: int | None = None
    master_index: int | None = None
    dz: float = 0.0


@dataclass(frozen=True)
class MasterMove:
    master_index: int
    delta_sample: float


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
    xlines: np.ndarray,
    inlines: np.ndarray,
    samples: np.ndarray,
    xline_index: int,
    inline_index: int,
    sample_index: int,
    kind: str,
    value_volume_data: VolumeData | None = None,
) -> ControlPoint:
    point_value = (
        0.0
        if value_volume_data is None
        else float(value_volume_data.data[xline_index, inline_index, sample_index])
    )
    return ControlPoint(
        xline_index=int(xline_index),
        inline_index=int(inline_index),
        sample_index=int(sample_index),
        base_sample_index=int(sample_index),
        xline=float(xlines[xline_index]),
        inline=float(inlines[inline_index]),
        sample=float(samples[sample_index]),
        value=point_value,
        kind=kind,
    )


def extract_control_points(
    xlines: np.ndarray,
    inlines: np.ndarray,
    samples: np.ndarray,
    component_mask: np.ndarray,
    value_volume_data: VolumeData | None = None,
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
                xlines,
                inlines,
                samples,
                int(xline_index),
                int(inline_index),
                int(sample_index),
                "surface",
                value_volume_data=value_volume_data,
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
                xlines,
                inlines,
                samples,
                int(xline_index),
                int(inline_index),
                int(sample_index),
                "interior",
                value_volume_data=value_volume_data,
            )
        )

    return reduce_surface_control_points(points)


def reduce_surface_control_points(points: list[ControlPoint]) -> list[ControlPoint]:
    if not points:
        return []

    surface_by_column: dict[tuple[int, int], list[ControlPoint]] = {}
    interior_points: list[ControlPoint] = []
    for point in points:
        cloned = ControlPoint(**vars(point))
        cloned.master_index = None
        if cloned.kind == "surface":
            surface_by_column.setdefault((cloned.xline_index, cloned.inline_index), []).append(cloned)
        else:
            interior_points.append(cloned)

    reduced_surface_points: list[ControlPoint] = []
    for column_points in surface_by_column.values():
        ordered = sorted(column_points, key=lambda point: point.sample_index)
        if len(ordered) == 1:
            reduced_surface_points.append(ordered[0])
            continue
        reduced_surface_points.append(ordered[0])
        if ordered[-1].sample_index != ordered[0].sample_index:
            reduced_surface_points.append(ordered[-1])

    reduced_points = reduced_surface_points + interior_points
    reduced_points.sort(key=lambda p: (p.kind, p.xline_index, p.inline_index, p.sample_index))
    master_index = 0
    for point in reduced_points:
        if point.kind == "surface":
            point.master_index = master_index
            master_index += 1
    return reduced_points


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
    sample_axis: np.ndarray,
    value_volume_data: VolumeData | None = None,
) -> list[ControlPoint]:
    return apply_master_point_z_moves(
        points,
        [MasterMove(master_index=int(selected_master_index), delta_sample=float(delta_sample))],
        sample_axis,
        value_volume_data=value_volume_data,
    )


def apply_master_point_z_moves(
    points: list[ControlPoint],
    moves: list[MasterMove],
    sample_axis: np.ndarray,
    value_volume_data: VolumeData | None = None,
) -> list[ControlPoint]:
    if not points:
        return []
    surface_points = [point for point in points if point.kind == "surface" and point.master_index is not None]
    if not surface_points:
        return [ControlPoint(**vars(point)) for point in points]

    move_map = {int(move.master_index): float(move.delta_sample) for move in moves if abs(float(move.delta_sample)) > 1e-12}
    if not move_map:
        return [ControlPoint(**vars(point)) for point in points]

    surface_by_master = {int(point.master_index): point for point in surface_points if point.master_index is not None}
    missing = [master_index for master_index in move_map if master_index not in surface_by_master]
    if missing:
        raise ValueError(f"Unknown master point index: {missing[0]}")

    column_surface_points: dict[tuple[int, int], list[ControlPoint]] = {}
    for point in surface_points:
        column_surface_points.setdefault((int(point.xline_index), int(point.inline_index)), []).append(point)

    column_surface_points = {
        key: sorted(column_points, key=lambda point: point.sample_index)
        for key, column_points in column_surface_points.items()
    }

    new_points: list[ControlPoint] = []
    min_sample = 0
    samples = np.asarray(sample_axis)
    max_sample = len(samples) - 1
    for point in points:
        dz = _point_delta_from_master_moves(point, move_map, column_surface_points)
        base_sample_index = int(point.sample_index if point.base_sample_index is None else point.base_sample_index)
        total_dz = float(point.dz + dz)
        new_sample_index = int(np.clip(round(base_sample_index + total_dz), min_sample, max_sample))
        new_point = ControlPoint(
            xline_index=int(point.xline_index),
            inline_index=int(point.inline_index),
            sample_index=new_sample_index,
            base_sample_index=base_sample_index,
            xline=float(point.xline),
            inline=float(point.inline),
            sample=float(samples[new_sample_index]),
            value=(
                float(point.value)
                if value_volume_data is None
                else float(value_volume_data.data[point.xline_index, point.inline_index, new_sample_index])
            ),
            kind=point.kind,
            master_index=point.master_index,
            dz=total_dz,
        )
        new_points.append(new_point)
    return new_points


def _point_delta_from_master_moves(
    point: ControlPoint,
    move_map: dict[int, float],
    column_surface_points: dict[tuple[int, int], list[ControlPoint]],
) -> float:
    if point.master_index is not None and int(point.master_index) in move_map:
        return float(move_map[int(point.master_index)])
    if point.kind == "surface" and point.master_index is not None:
        return 0.0

    column_key = (int(point.xline_index), int(point.inline_index))
    column_points = column_surface_points.get(column_key, [])
    moved_column_points = [
        surface_point
        for surface_point in column_points
        if surface_point.master_index is not None and int(surface_point.master_index) in move_map
    ]
    if not moved_column_points:
        return 0.0
    if len(moved_column_points) == 1 or len(column_points) <= 1:
        moved_point = moved_column_points[0]
        return float(move_map[int(moved_point.master_index)])

    lower_surface = column_points[0]
    upper_surface = column_points[-1]
    lower_delta = float(move_map.get(int(lower_surface.master_index), 0.0)) if lower_surface.master_index is not None else 0.0
    upper_delta = float(move_map.get(int(upper_surface.master_index), 0.0)) if upper_surface.master_index is not None else 0.0

    z0 = float(lower_surface.sample_index)
    z1 = float(upper_surface.sample_index)
    if abs(z1 - z0) <= 1e-12:
        if abs(lower_delta) > abs(upper_delta):
            return lower_delta
        return upper_delta

    t = (float(point.sample_index) - z0) / (z1 - z0)
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * lower_delta + t * upper_delta


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


def rebuild_mask_from_master_points(
    shape: tuple[int, int, int],
    points: list[ControlPoint],
    reference_mask: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    master_points = reduce_surface_control_points(master_control_points(points))
    if not master_points:
        return mask

    column_mask = np.asarray(reference_mask, dtype=bool).any(axis=2)
    if not np.any(column_mask):
        return mask

    lower_columns: dict[tuple[int, int], int] = {}
    upper_columns: dict[tuple[int, int], int] = {}
    for point in master_points:
        key = (int(point.xline_index), int(point.inline_index))
        sample_index = int(point.sample_index)
        lower_columns[key] = sample_index if key not in lower_columns else min(lower_columns[key], sample_index)
        upper_columns[key] = sample_index if key not in upper_columns else max(upper_columns[key], sample_index)

    target_columns = np.argwhere(column_mask)
    lower_surface = _interpolate_column_surface(lower_columns, target_columns)
    upper_surface = _interpolate_column_surface(upper_columns, target_columns)
    if lower_surface is None or upper_surface is None:
        return mask

    lower_anchor_lookup = {key: lower_columns[key] for key in lower_columns}
    upper_anchor_lookup = {key: upper_columns[key] for key in upper_columns}
    lower_surface = _smooth_interpolated_surface(lower_surface, target_columns, lower_anchor_lookup)
    upper_surface = _smooth_interpolated_surface(upper_surface, target_columns, upper_anchor_lookup)

    max_sample = shape[2] - 1
    for index, (xi, yi) in enumerate(target_columns):
        z0 = int(np.clip(round(lower_surface[index]), 0, max_sample))
        z1 = int(np.clip(round(upper_surface[index]), 0, max_sample))
        if z0 > z1:
            z0, z1 = z1, z0
        mask[int(xi), int(yi), z0 : z1 + 1] = True

    for (xi, yi), sample_index in lower_columns.items():
        zi = int(np.clip(sample_index, 0, max_sample))
        mask[int(xi), int(yi), zi] = True
    for (xi, yi), sample_index in upper_columns.items():
        zi = int(np.clip(sample_index, 0, max_sample))
        mask[int(xi), int(yi), zi] = True
    return mask


def _interpolate_column_surface(
    anchor_columns: dict[tuple[int, int], int],
    target_columns: np.ndarray,
    *,
    power: float = 2.0,
    chunk_size: int = 2048,
) -> np.ndarray | None:
    if not anchor_columns:
        return None

    anchor_xy = np.asarray(list(anchor_columns.keys()), dtype=np.float64)
    anchor_z = np.asarray(list(anchor_columns.values()), dtype=np.float64)
    if anchor_xy.shape[0] == 1:
        return np.full(len(target_columns), anchor_z[0], dtype=np.float64)

    results = np.empty(len(target_columns), dtype=np.float64)
    for start in range(0, len(target_columns), chunk_size):
        stop = min(len(target_columns), start + chunk_size)
        targets = np.asarray(target_columns[start:stop], dtype=np.float64)
        chunk_results = results[start:stop]
        deltas = targets[:, None, :] - anchor_xy[None, :, :]
        distance2 = np.sum(deltas * deltas, axis=2)
        exact_match = distance2 <= 1e-12
        if np.any(exact_match):
            exact_rows = np.any(exact_match, axis=1)
            exact_indices = np.argmax(exact_match[exact_rows], axis=1)
            chunk_results[exact_rows] = anchor_z[exact_indices]

            inexact_rows = ~exact_rows
            if np.any(inexact_rows):
                local_dist2 = distance2[inexact_rows]
                weights = 1.0 / np.maximum(local_dist2, 1e-12) ** (power * 0.5)
                weight_sums = np.sum(weights, axis=1)
                chunk_results[inexact_rows] = (weights @ anchor_z) / weight_sums
            continue

        weights = 1.0 / np.maximum(distance2, 1e-12) ** (power * 0.5)
        weight_sums = np.sum(weights, axis=1)
        chunk_results[:] = (weights @ anchor_z) / weight_sums
    return results


def _smooth_interpolated_surface(
    values: np.ndarray,
    target_columns: np.ndarray,
    anchor_lookup: dict[tuple[int, int], int],
    *,
    iterations: int = 12,
) -> np.ndarray:
    if len(values) == 0:
        return values

    smoothed = np.asarray(values, dtype=np.float64).copy()
    column_to_index = {
        (int(xi), int(yi)): idx
        for idx, (xi, yi) in enumerate(target_columns)
    }
    anchor_indices = {
        column_to_index[key]: float(anchor_value)
        for key, anchor_value in anchor_lookup.items()
        if key in column_to_index
    }

    for idx, anchor_value in anchor_indices.items():
        smoothed[idx] = anchor_value

    for _ in range(max(0, int(iterations))):
        updated = smoothed.copy()
        for idx, (xi, yi) in enumerate(target_columns):
            if idx in anchor_indices:
                continue
            neighbor_values: list[float] = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_index = column_to_index.get((int(xi) + dx, int(yi) + dy))
                if neighbor_index is not None:
                    neighbor_values.append(float(smoothed[neighbor_index]))
            if not neighbor_values:
                continue
            updated[idx] = 0.55 * float(smoothed[idx]) + 0.45 * float(np.mean(neighbor_values))
        smoothed = updated
        for idx, anchor_value in anchor_indices.items():
            smoothed[idx] = anchor_value
    return smoothed
