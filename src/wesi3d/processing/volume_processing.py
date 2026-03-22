#!/usr/bin/env python3
"""
Operations that take VolumeData and produce new VolumeData.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..data.volume_data import VolumeData, validate_interval

_NEIGHBOR_OFFSETS_6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


@dataclass(frozen=True)
class ConnectedComponent:
    index: int
    mask: np.ndarray
    voxel_count: int


def copy_volume(volume_data: VolumeData, *, name: str | None = None) -> VolumeData:
    return volume_data.with_data(
        np.array(volume_data.data, copy=True),
        name=volume_data.name if name is None else name,
        metadata={**volume_data.metadata, "operation": "copy"},
    )


def extract_range_volume(
    volume_data: VolumeData,
    min_value: float,
    max_value: float,
    *,
    outside_value: float = 0.0,
    name: str | None = None,
) -> VolumeData:
    if min_value > max_value:
        min_value, max_value = max_value, min_value

    data = np.asarray(volume_data.data)
    masked = np.array(data, copy=True)
    outside_mask = (data < min_value) | (data > max_value)
    masked[outside_mask] = np.asarray(outside_value, dtype=masked.dtype)
    output_name = (
        f"{volume_data.name}_range_{min_value:g}_{max_value:g}"
        if name is None
        else name
    )
    return volume_data.with_data(
        masked,
        name=output_name,
        metadata={
            **volume_data.metadata,
            "operation": "extract_range",
            "min_value": float(min_value),
            "max_value": float(max_value),
            "outside_value": float(outside_value),
        },
    )


def downsample_volume(
    volume_data: VolumeData,
    *,
    interval_xline: int = 1,
    interval_inline: int = 1,
    interval_sample: int = 1,
    name: str | None = None,
) -> VolumeData:
    xs = slice(None, None, validate_interval("interval_xline", interval_xline))
    ys = slice(None, None, validate_interval("interval_inline", interval_inline))
    zs = slice(None, None, validate_interval("interval_sample", interval_sample))
    return volume_data.with_data(
        np.asarray(volume_data.data[xs, ys, zs]),
        xlines=volume_data.xlines[xs],
        inlines=volume_data.inlines[ys],
        samples=volume_data.samples[zs],
        name=volume_data.name if name is None else name,
        metadata={
            **volume_data.metadata,
            "operation": "downsample",
            "interval_xline": int(interval_xline),
            "interval_inline": int(interval_inline),
            "interval_sample": int(interval_sample),
        },
    )


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


def _component_envelope_mask(component_mask: np.ndarray) -> np.ndarray:
    interior = component_mask.copy()
    for dx, dy, dz in _NEIGHBOR_OFFSETS_6:
        interior &= _shift_mask(component_mask, dx, dy, dz)
    return component_mask & ~interior


def _connected_component_masks(
    active_mask: np.ndarray,
    *,
    min_voxels: int = 1,
) -> list[ConnectedComponent]:
    min_voxels = max(1, int(min_voxels))
    visited = np.zeros_like(active_mask, dtype=bool)
    components: list[ConnectedComponent] = []
    nx, ny, nz = active_mask.shape

    for start in np.argwhere(active_mask):
        sx, sy, sz = (int(v) for v in start)
        if visited[sx, sy, sz]:
            continue

        queue: deque[tuple[int, int, int]] = deque([(sx, sy, sz)])
        visited[sx, sy, sz] = True
        coords: list[tuple[int, int, int]] = []

        while queue:
            x, y, z = queue.popleft()
            coords.append((x, y, z))
            for dx, dy, dz in _NEIGHBOR_OFFSETS_6:
                xx = x + dx
                yy = y + dy
                zz = z + dz
                if xx < 0 or xx >= nx or yy < 0 or yy >= ny or zz < 0 or zz >= nz:
                    continue
                if visited[xx, yy, zz] or not active_mask[xx, yy, zz]:
                    continue
                visited[xx, yy, zz] = True
                queue.append((xx, yy, zz))

        if len(coords) < min_voxels:
            continue

        component_mask = np.zeros_like(active_mask, dtype=bool)
        xs, ys, zs = zip(*coords)
        component_mask[np.asarray(xs), np.asarray(ys), np.asarray(zs)] = True
        components.append(
            ConnectedComponent(
                index=len(components) + 1,
                mask=component_mask,
                voxel_count=len(coords),
            )
        )

    return components


def extract_connected_components(
    volume_data: VolumeData,
    *,
    active_threshold: float = 0.0,
    min_voxels: int = 1,
) -> list[ConnectedComponent]:
    data = np.asarray(volume_data.data)
    active_mask = np.abs(data) > float(active_threshold)
    return _connected_component_masks(active_mask, min_voxels=min_voxels)


def extract_envelope_volumes(
    volume_data: VolumeData,
    *,
    active_threshold: float = 0.0,
    outside_value: float = 0.0,
    min_voxels: int = 1,
    name_prefix: str | None = None,
) -> list[VolumeData]:
    data = np.asarray(volume_data.data)
    components = extract_connected_components(
        volume_data,
        active_threshold=active_threshold,
        min_voxels=min_voxels,
    )

    output_volumes: list[VolumeData] = []
    prefix = volume_data.name if name_prefix is None else name_prefix
    for component in components:
        component_mask = component.mask
        envelope_mask = _component_envelope_mask(component_mask)
        envelope_data = np.full_like(data, np.asarray(outside_value, dtype=data.dtype))
        envelope_data[envelope_mask] = data[envelope_mask]
        name = f"{prefix}_component_{component.index}_envelope"
        output_volumes.append(
            volume_data.with_data(
                envelope_data,
                name=name,
                metadata={
                    **volume_data.metadata,
                    "operation": "extract_envelope",
                    "component_index": component.index,
                    "min_voxels": int(max(1, min_voxels)),
                    "active_threshold": float(active_threshold),
                    "outside_value": float(outside_value),
                    "component_voxels": int(component.voxel_count),
                    "envelope_voxels": int(np.count_nonzero(envelope_mask)),
                },
            )
        )
    return output_volumes


def interpolate_control_point_values_to_volume(
    template_volume: VolumeData,
    points: list[object],
    *,
    apply_mask: bool = False,
    mask: np.ndarray | None = None,
    radius: float = 0.0,
    power: float = 2.0,
    k_neighbors: int = 16,
    chunk_size: int = 32768,
    name: str | None = None,
) -> VolumeData:
    if not points:
        raise ValueError("No control points are available for interpolation.")

    control_coords: list[tuple[float, float, float]] = []
    control_values: list[float] = []
    for point in points:
        value = float(getattr(point, "value", 0.0))
        if not np.isfinite(value):
            continue
        control_coords.append(
            (
                float(getattr(point, "xline_index")),
                float(getattr(point, "inline_index")),
                float(getattr(point, "sample_index")),
            )
        )
        control_values.append(value)

    if not control_coords:
        raise ValueError("No valid control-point values are available for interpolation.")

    source_coords = np.asarray(control_coords, dtype=np.float32)
    source_values = np.asarray(control_values, dtype=np.float32)
    radius = max(0.0, float(radius))
    radius2 = radius * radius
    power = max(1e-6, float(power))
    k_neighbors = max(1, min(int(k_neighbors), len(source_values)))
    working_mask = np.ones(template_volume.shape, dtype=bool)
    if apply_mask:
        if mask is None:
            raise ValueError("Mask interpolation is enabled, but the horizon mask is missing.")
        working_mask = np.asarray(mask, dtype=bool)
        if working_mask.shape != template_volume.shape:
            raise ValueError("The horizon mask shape does not match the selected attribute grid.")

    target_indices = np.argwhere(working_mask)
    output = np.zeros(template_volume.shape, dtype=np.float32)
    if len(target_indices) == 0:
        output_name = f"{template_volume.name}_interpolated" if name is None else name
        return template_volume.with_data(
            output,
            name=output_name,
            metadata={
                **template_volume.metadata,
                "operation": "interpolate_control_points",
                "apply_mask": bool(apply_mask),
                "source_control_points": int(len(source_values)),
            },
        )

    for start in range(0, len(target_indices), max(1, int(chunk_size))):
        batch = np.asarray(target_indices[start : start + max(1, int(chunk_size))], dtype=np.float32)
        diff = batch[:, None, :] - source_coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        if k_neighbors < source_coords.shape[0]:
            nearest = np.argpartition(dist2, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]
            nearest_dist2 = np.take_along_axis(dist2, nearest, axis=1)
            nearest_values = source_values[nearest]
        else:
            nearest_dist2 = dist2
            nearest_values = np.broadcast_to(source_values[None, :], dist2.shape)

        batch_values = np.empty(len(batch), dtype=np.float32)
        zero_mask = nearest_dist2 <= 1e-12
        exact_rows = np.any(zero_mask, axis=1)
        if np.any(exact_rows):
            row_indices = np.nonzero(exact_rows)[0]
            zero_cols = np.argmax(zero_mask[exact_rows], axis=1)
            batch_values[row_indices] = nearest_values[exact_rows, zero_cols]

        smooth_rows = ~exact_rows
        if np.any(smooth_rows):
            smooth_dist2 = nearest_dist2[smooth_rows]
            smooth_values = nearest_values[smooth_rows]
            if radius > 0.0:
                # Use radius as a soft decay scale instead of a hard cutoff.
                # This avoids abrupt boundary artifacts when a voxel falls just
                # outside the radius and would otherwise collapse to nearest-point fallback.
                idw_weights = 1.0 / np.power(np.maximum(smooth_dist2, 1e-12), power * 0.5)
                radial_weights = np.exp(-smooth_dist2 / max(radius2, 1e-12))
                weights = idw_weights * radial_weights
                weight_sums = np.sum(weights, axis=1)
                valid_rows = weight_sums > 1e-12
                if np.any(valid_rows):
                    weighted_values = np.sum(weights[valid_rows] * smooth_values[valid_rows], axis=1) / weight_sums[valid_rows]
                    smooth_indices = np.nonzero(smooth_rows)[0][valid_rows]
                    batch_values[smooth_indices] = weighted_values.astype(np.float32)
                if np.any(~valid_rows):
                    fallback_dist2 = smooth_dist2[~valid_rows]
                    fallback_values = smooth_values[~valid_rows]
                    nearest_idx = np.argmin(fallback_dist2, axis=1)
                    smooth_indices = np.nonzero(smooth_rows)[0][~valid_rows]
                    batch_values[smooth_indices] = fallback_values[np.arange(len(nearest_idx)), nearest_idx].astype(np.float32)
            else:
                weights = 1.0 / np.power(np.maximum(smooth_dist2, 1e-12), power * 0.5)
                weighted_values = np.sum(weights * smooth_values, axis=1) / np.sum(weights, axis=1)
                batch_values[smooth_rows] = weighted_values.astype(np.float32)

        batch_index = target_indices[start : start + max(1, int(chunk_size))]
        output[batch_index[:, 0], batch_index[:, 1], batch_index[:, 2]] = batch_values

    output_name = f"{template_volume.name}_interpolated" if name is None else name
    return template_volume.with_data(
        output,
        name=output_name,
        metadata={
            **template_volume.metadata,
            "operation": "interpolate_control_points",
            "apply_mask": bool(apply_mask),
            "source_control_points": int(len(source_values)),
            "radius": float(radius),
            "power": float(power),
            "k_neighbors": int(k_neighbors),
        },
    )
