from __future__ import annotations

import argparse
from math import atan2, degrees
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

POINT0 = (517888.79, 4598260.61, 2000.0, 1200.0)
POINT1 = (501208.58, 4636806.30, 2000.0, 3300.0)
POINT3 = (554598.98, 4614146.52, 4000.0, 1200.0)


def _vec(start_xy: tuple[float, float], end_xy: tuple[float, float]) -> tuple[float, float]:
    return (float(end_xy[0]) - float(start_xy[0]), float(end_xy[1]) - float(start_xy[1]))


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(left[0] * right[0] + left[1] * right[1])


def _norm(vec: tuple[float, float]) -> float:
    return float(np.hypot(vec[0], vec[1]))


def build_grid_parameters() -> dict[str, object]:
    point0_xy = (POINT0[0], POINT0[1])
    point1_xy = (POINT1[0], POINT1[1])
    point3_xy = (POINT3[0], POINT3[1])

    vec_inl = _vec(point0_xy, point3_xy)
    vec_cxl = _vec(point0_xy, point1_xy)
    len_inl = _norm(vec_inl)
    len_cxl = _norm(vec_cxl)

    span_inl = POINT3[2] - POINT0[2]
    span_cxl = POINT1[3] - POINT0[3]

    inl_unit = (vec_inl[0] / len_inl, vec_inl[1] / len_inl)
    cxl_unit = (vec_cxl[0] / len_cxl, vec_cxl[1] / len_cxl)

    return {
        "point0_xy": point0_xy,
        "point0_inl": POINT0[2],
        "point0_cxl": POINT0[3],
        "step_inl": len_inl / abs(span_inl),
        "step_cxl": len_cxl / abs(span_cxl),
        "inl_unit": inl_unit,
        "cxl_unit": cxl_unit,
        "inl_angle_deg": degrees(atan2(inl_unit[0], inl_unit[1])) % 360.0,
        "cxl_angle_deg": degrees(atan2(cxl_unit[0], cxl_unit[1])) % 360.0,
    }


def rw_to_grid(grid: dict[str, object], rw_x: float, rw_y: float) -> tuple[float, float]:
    rel = _vec(grid["point0_xy"], (float(rw_x), float(rw_y)))
    dist_inl = _dot(rel, grid["inl_unit"])
    dist_cxl = _dot(rel, grid["cxl_unit"])
    inl = float(grid["point0_inl"]) + dist_inl / float(grid["step_inl"])
    cxl = float(grid["point0_cxl"]) + dist_cxl / float(grid["step_cxl"])
    return float(inl), float(cxl)


def normalize_polygon_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    deduplicated: list[np.ndarray] = []
    for point in points:
        if not deduplicated or not np.allclose(point, deduplicated[-1]):
            deduplicated.append(point)
    normalized = np.asarray(deduplicated, dtype=np.float32)

    while normalized.shape[0] >= 2 and np.allclose(normalized[0], normalized[-1]):
        normalized = np.asarray(normalized[:-1], dtype=np.float32)

    while normalized.shape[0] >= 3:
        point_keys = [tuple(float(value) for value in point) for point in normalized]
        prefix_lengths = [0] * len(point_keys)
        for index in range(1, len(point_keys)):
            prefix_length = prefix_lengths[index - 1]
            while prefix_length > 0 and point_keys[index] != point_keys[prefix_length]:
                prefix_length = prefix_lengths[prefix_length - 1]
            if point_keys[index] == point_keys[prefix_length]:
                prefix_length += 1
            prefix_lengths[index] = prefix_length
        repeated_suffix_length = prefix_lengths[-1]
        if repeated_suffix_length <= 0:
            break
        normalized = np.asarray(normalized[:-repeated_suffix_length], dtype=np.float32)

    return np.asarray(normalized, dtype=np.float32)


def load_geomap(path: Path) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    polygons: list[tuple[tuple[int, int, int], np.ndarray]] = []
    current_points: list[tuple[float, float]] = []
    current_color = (240, 210, 120)
    with path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            if line_index == 0:
                continue
            if line.startswith("##"):
                if len(current_points) >= 2:
                    polygons.append((current_color, np.asarray(current_points, dtype=np.float32)))
                current_points = []
                color_parts = line[2:].strip().split()
                if len(color_parts) >= 3:
                    try:
                        current_color = tuple(int(max(0, min(255, int(part)))) for part in color_parts[:3])  # type: ignore[assignment]
                    except ValueError:
                        current_color = (240, 210, 120)
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                current_points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(current_points) >= 2:
        polygons.append((current_color, np.asarray(current_points, dtype=np.float32)))
    return polygons


def convert_geomap(grid: dict[str, object], polygons: list[tuple[tuple[int, int, int], np.ndarray]]) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    converted: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for color_rgb, polygon_rw in polygons:
        grid_points = np.asarray(
            [rw_to_grid(grid, float(point[0]), float(point[1])) for point in np.asarray(polygon_rw, dtype=np.float32)],
            dtype=np.float32,
        )
        normalized_points = normalize_polygon_points(grid_points)
        if normalized_points.shape[0] >= 2:
            converted.append((color_rgb, normalized_points))
    return converted


def write_geomap(path: Path, polygons: list[tuple[tuple[int, int, int], np.ndarray]]) -> None:
    lines = ["Area"]
    for color_rgb, points in polygons:
        lines.append(f"##{int(color_rgb[0])} {int(color_rgb[1])} {int(color_rgb[2])}")
        for point in np.asarray(points, dtype=np.float32):
            lines.append(f"{float(point[0]):.3f} {float(point[1]):.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_text(grid: dict[str, object]) -> str:
    return "\n".join(
        [
            "SurveyGrid:",
            "  point0: rw_x=517888.790000, rw_y=4598260.610000, inl=2000.000000, cxl=1200.000000",
            "  point1: rw_x=501208.580000, rw_y=4636806.300000, inl=2000.000000, cxl=3300.000000",
            "  point3: rw_x=554598.980000, rw_y=4614146.520000, inl=4000.000000, cxl=1200.000000",
            f"  step_inl: {float(grid['step_inl']):.6f}",
            f"  step_cxl: {float(grid['step_cxl']):.6f}",
            f"  inl_angle_deg: {float(grid['inl_angle_deg']):.6f}",
            f"  cxl_angle_deg: {float(grid['cxl_angle_deg']):.6f}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert geomap.gmp rw_x/rw_y polygons to inline/cxline polygons and remove duplicate points."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "scatter" / "geomap.gmp",
        help="Input geomap.gmp with rw_x/rw_y polygon coordinates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "scatter" / "geomap_inl_cxl.gmp",
        help="Output geomap.gmp with inline/cxline polygon coordinates.",
    )
    args = parser.parse_args()

    grid = build_grid_parameters()
    polygons = load_geomap(args.input)
    converted = convert_geomap(grid, polygons)
    write_geomap(args.output, converted)

    print(summary_text(grid))
    print(f"polygons_read: {len(polygons)}")
    print(f"polygons_written: {len(converted)}")
    print(f"output: {args.output}")
    print("columns: inline crossline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
