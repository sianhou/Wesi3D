#!/usr/bin/env python3
"""
Service entry for seismic/attribute import.

The dialog collects parameters. This module currently builds the target numpy
volume and saves it as an npz file for workflow testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...data.volume_data2 import AxisRange, GridMeta, GridPoint, RangeMeta, VolumeData2

try:
    import segyio
except ImportError:
    segyio = None


def _require_segyio() -> None:
    if segyio is None:
        raise RuntimeError("Missing dependency: segyio")


def _as_int(values: dict[str, object], key: str) -> int:
    return int(float(str(values.get(key, "")).strip()))


def _as_str(values: dict[str, object], key: str, default: str = "") -> str:
    text = str(values.get(key, default)).strip()
    return text or default


def _inclusive_axis(begin: int, end: int, step: int) -> np.ndarray:
    if step <= 0:
        raise ValueError("axis step must be positive")
    if end < begin:
        raise ValueError("axis end must be greater than or equal to begin")
    return np.arange(begin, end + step, step, dtype=np.int32)


def _sample_indices(begin: int, end: int, step: int) -> np.ndarray:
    if step <= 0:
        raise ValueError("sample step must be positive")
    if end < begin:
        raise ValueError("sample end must be greater than or equal to begin")
    return np.arange(begin, end, step, dtype=np.int64)


def _resolve_output_path(values: dict[str, object], input_path: Path, name: str) -> Path:
    output_text = _as_str(values, "output_file", "")
    if output_text:
        output_path = Path(output_text).expanduser()
    else:
        output_path = input_path.with_name(name)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _import_metadata(
    values: dict[str, object],
    *,
    path: Path,
    file_type: str,
    output_path: Path,
) -> dict[str, object]:
    return {
        "source_path": str(path),
        "file_type": file_type,
        "output_file": str(output_path),
        "target_category": _as_str(values, "target_category", "seismic"),
        "begin_inline": _as_int(values, "begin_inline"),
        "end_inline": _as_int(values, "end_inline"),
        "step_inline": _as_int(values, "step_inline"),
        "spacing_inline": _as_int(values, "spacing_inline"),
        "begin_xline": _as_int(values, "begin_xline"),
        "end_xline": _as_int(values, "end_xline"),
        "step_xline": _as_int(values, "step_xline"),
        "spacing_xline": _as_int(values, "spacing_xline"),
        "begin_sample": _as_int(values, "begin_sample"),
        "end_sample": _as_int(values, "end_sample"),
        "step_sample": _as_int(values, "step_sample"),
        "spacing_sample": _as_int(values, "spacing_sample"),
        "p0_x": float(values.get("p0_x", 0.0)),
        "p0_y": float(values.get("p0_y", 0.0)),
        "p1_x": float(values.get("p1_x", 0.0)),
        "p1_y": float(values.get("p1_y", 0.0)),
        "p3_x": float(values.get("p3_x", 0.0)),
        "p3_y": float(values.get("p3_y", 0.0)),
    }


def _build_volume_data2(
    *,
    data: np.ndarray,
    name: str,
    metadata: dict[str, object],
) -> VolumeData2:
    return VolumeData2(
        type="volume",
        range=RangeMeta(
            inline=AxisRange(
                begin=int(metadata["begin_inline"]),
                end=int(metadata["end_inline"]),
                step=int(metadata["step_inline"]),
                spacing=int(metadata["spacing_inline"]),
            ),
            cxline=AxisRange(
                begin=int(metadata["begin_xline"]),
                end=int(metadata["end_xline"]),
                step=int(metadata["step_xline"]),
                spacing=int(metadata["spacing_xline"]),
            ),
            sample=AxisRange(
                begin=int(metadata["begin_sample"]),
                end=int(metadata["end_sample"]),
                step=int(metadata["step_sample"]),
                spacing=int(metadata["spacing_sample"]),
            ),
        ),
        grid=GridMeta(
            p0=GridPoint(
                x=float(metadata["p0_x"]),
                y=float(metadata["p0_y"]),
            ),
            p1=GridPoint(
                x=float(metadata["p1_x"]),
                y=float(metadata["p1_y"]),
            ),
            p3=GridPoint(
                x=float(metadata["p3_x"]),
                y=float(metadata["p3_y"]),
            ),
        ),
        data=np.asarray(data, dtype=np.float32),
    )


def _open_trace_file(path: Path, file_type: str):
    _require_segyio()
    if file_type == "su":
        su_module = getattr(segyio, "su", None)
        if su_module is None or not hasattr(su_module, "open"):
            raise RuntimeError("Current segyio build does not support SU files")
        return su_module.open(str(path), "r", ignore_geometry=True)
    return segyio.open(str(path), "r", strict=False, ignore_geometry=True)


def _build_binary_volume(
    path: Path,
    *,
    name: str,
    metadata: dict[str, object],
    inlines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    num_inline = int(len(inlines))
    num_xline = int(len(xlines))
    num_sample = int(len(samples))
    expected_count = num_inline * num_xline * num_sample
    expected_bytes = expected_count * np.dtype(np.float32).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            "Binary file size does not match target grid: "
            f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
        )

    flat = np.fromfile(path, dtype=np.float32, count=expected_count)
    # User-oriented layout is (inline, xline, sample). Internal VolumeData
    # layout is (xline, inline, sample).
    inline_major = flat.reshape((num_inline, num_xline, num_sample), order="C")
    volume = np.transpose(inline_major, (1, 0, 2)).copy()

    return volume


def _build_segy_like_volume(
    path: Path,
    *,
    file_type: str,
    name: str,
    metadata: dict[str, object],
    inlines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    sample_indices: np.ndarray,
    inline_field: int,
    xline_field: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if sample_indices.size == 0:
        raise ValueError("sample range is empty")

    num_inline = int(len(inlines))
    num_xline = int(len(xlines))
    num_sample = int(len(samples))
    inline_begin = int(inlines[0])
    xline_begin = int(xlines[0])
    inline_step = int(inlines[1] - inlines[0]) if num_inline > 1 else 1
    xline_step = int(xlines[1] - xlines[0]) if num_xline > 1 else 1
    sample_end_required = int(sample_indices[-1])

    volume = np.zeros((num_xline, num_inline, num_sample), dtype=np.float32)
    loaded_traces = 0
    skipped_out_of_range = 0
    skipped_step_mismatch = 0
    skipped_sample_mismatch = 0

    with _open_trace_file(path, file_type) as segy:
        inline_headers = np.asarray(segy.attributes(inline_field)[:], dtype=np.int64)
        xline_headers = np.asarray(segy.attributes(xline_field)[:], dtype=np.int64)
        trace_count = int(len(inline_headers))

        for trace_index in range(trace_count):
            inline_value = int(inline_headers[trace_index])
            xline_value = int(xline_headers[trace_index])

            if inline_value < int(inlines[0]) or inline_value > int(inlines[-1]):
                skipped_out_of_range += 1
                continue
            if xline_value < int(xlines[0]) or xline_value > int(xlines[-1]):
                skipped_out_of_range += 1
                continue

            inline_offset = inline_value - inline_begin
            xline_offset = xline_value - xline_begin
            if inline_offset % inline_step != 0 or xline_offset % xline_step != 0:
                skipped_step_mismatch += 1
                continue

            inline_index = inline_offset // inline_step
            xline_index = xline_offset // xline_step
            if not (0 <= inline_index < num_inline and 0 <= xline_index < num_xline):
                skipped_out_of_range += 1
                continue

            trace = np.asarray(segy.trace[trace_index], dtype=np.float32)
            if trace.size <= sample_end_required:
                skipped_sample_mismatch += 1
                continue

            selected_trace = trace[sample_indices]
            if selected_trace.size != num_sample:
                skipped_sample_mismatch += 1
                continue

            volume[xline_index, inline_index, :] = selected_trace
            loaded_traces += 1

    print("[SeismicAttributeImporterService] trace stats", flush=True)
    print(f"[SeismicAttributeImporterService] loaded_traces={loaded_traces}", flush=True)
    print(
        f"[SeismicAttributeImporterService] skipped_out_of_range={skipped_out_of_range}",
        flush=True,
    )
    print(
        f"[SeismicAttributeImporterService] skipped_step_mismatch={skipped_step_mismatch}",
        flush=True,
    )
    print(
        f"[SeismicAttributeImporterService] skipped_sample_mismatch={skipped_sample_mismatch}",
        flush=True,
    )

    if loaded_traces == 0:
        raise ValueError("No traces matched the current grid definition")

    return volume, {
        **metadata,
        "loaded_traces": loaded_traces,
        "skipped_out_of_range": skipped_out_of_range,
        "skipped_step_mismatch": skipped_step_mismatch,
        "skipped_sample_mismatch": skipped_sample_mismatch,
    }


def execute_import(values: dict[str, object]) -> dict[str, object]:
    path = Path(_as_str(values, "path")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    file_type = _as_str(values, "file_type", "segy").lower()
    name = _as_str(values, "name", path.stem)
    output_path = _resolve_output_path(values, path, name)

    inline_begin = _as_int(values, "begin_inline")
    inline_end = _as_int(values, "end_inline")
    inline_step = _as_int(values, "step_inline")
    xline_begin = _as_int(values, "begin_xline")
    xline_end = _as_int(values, "end_xline")
    xline_step = _as_int(values, "step_xline")
    sample_begin = _as_int(values, "begin_sample")
    sample_end = _as_int(values, "end_sample")
    sample_step = _as_int(values, "step_sample")

    inlines = _inclusive_axis(inline_begin, inline_end, inline_step)
    xlines = _inclusive_axis(xline_begin, xline_end, xline_step)
    sample_indices = _sample_indices(sample_begin, sample_end, sample_step)
    if sample_indices.size == 0:
        raise ValueError("Sample range is empty")
    samples = sample_indices.astype(np.float32)

    print("[SeismicAttributeImporterService] execute_import", flush=True)
    print(f"[SeismicAttributeImporterService] path={path}", flush=True)
    print(f"[SeismicAttributeImporterService] file_type={file_type}", flush=True)
    print(f"[SeismicAttributeImporterService] output_path={output_path}", flush=True)
    print(
        "[SeismicAttributeImporterService] shape="
        f"({len(inlines)}, {len(xlines)}, {len(samples)})",
        flush=True,
    )

    metadata = _import_metadata(
        values,
        path=path,
        file_type=file_type,
        output_path=output_path,
    )

    if file_type == "binary":
        volume = _build_binary_volume(
            path,
            name=name,
            metadata=metadata,
            inlines=inlines,
            xlines=xlines,
            samples=samples,
        )
    elif file_type in {"segy", "su"}:
        volume, metadata = _build_segy_like_volume(
            path,
            file_type=file_type,
            name=name,
            metadata=metadata,
            inlines=inlines,
            xlines=xlines,
            samples=samples,
            sample_indices=sample_indices,
            inline_field=_as_int(values, "inline_field"),
            xline_field=_as_int(values, "xline_field"),
        )
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    volume_data2 = _build_volume_data2(
        data=volume,
        name=name,
        metadata=metadata,
    )
    volume_data2.to_npz(output_path)
    return {
        "output_path": str(output_path),
        "name": name,
        "shape": tuple(int(v) for v in volume.shape),
    }
