from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from math import atan2, degrees
from pathlib import Path
from typing import Any

import numpy as np


JSONDict = dict[str, Any]
RealPoint = tuple[float, float]


@dataclass(frozen=True, slots=True)
class GridControlPoint:
    name: str
    rw: RealPoint
    inl: float
    cxl: float

    @property
    def rw_x(self) -> float:
        return float(self.rw[0])

    @property
    def rw_y(self) -> float:
        return float(self.rw[1])

    def as_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "rw": [self.rw_x, self.rw_y],
            "inl": float(self.inl),
            "cxl": float(self.cxl),
        }

    @classmethod
    def from_dict(cls, payload: JSONDict) -> GridControlPoint:
        rw = payload["rw"]
        return cls(
            name=str(payload["name"]),
            rw=(float(rw[0]), float(rw[1])),
            inl=float(payload["inl"]),
            cxl=float(payload["cxl"]),
        )


@dataclass(slots=True)
class SurveyGrid:
    """Survey work-area grid defined by ordered control points.

    Control point order:
    - Point0: origin point
    - Point1: first inline last point
    - Point2: diagonal point
    - Point3: last inline first point
    """

    point0: GridControlPoint
    point1: GridControlPoint
    point2: GridControlPoint
    point3: GridControlPoint
    step_inl: float = field(init=False)
    step_cxl: float = field(init=False)
    _inl_unit: RealPoint = field(init=False, repr=False)
    _cxl_unit: RealPoint = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vec_inl = _vec(self.point0.rw, self.point3.rw)
        vec_cxl = _vec(self.point0.rw, self.point1.rw)

        span_inl = float(self.point3.inl) - float(self.point0.inl)
        span_cxl = float(self.point1.cxl) - float(self.point0.cxl)
        if span_inl == 0:
            raise ValueError("Point0 and Point3 must have different inl values")
        if span_cxl == 0:
            raise ValueError("Point0 and Point1 must have different cxl values")

        len_inl = _norm(vec_inl)
        len_cxl = _norm(vec_cxl)
        if len_inl == 0 or len_cxl == 0:
            raise ValueError("Control point real coordinates must not collapse to zero-length axes")

        self.step_inl = len_inl / abs(span_inl)
        self.step_cxl = len_cxl / abs(span_cxl)
        self._inl_unit = (vec_inl[0] / len_inl, vec_inl[1] / len_inl)
        self._cxl_unit = (vec_cxl[0] / len_cxl, vec_cxl[1] / len_cxl)

    @classmethod
    def from_three_points(
        cls,
        point0: GridControlPoint,
        point1: GridControlPoint,
        point3: GridControlPoint,
    ) -> SurveyGrid:
        point2 = cls._infer_point2(point0, point1, point3)
        return cls(point0=point0, point1=point1, point2=point2, point3=point3)

    @staticmethod
    def _infer_point2(
        point0: GridControlPoint,
        point1: GridControlPoint,
        point3: GridControlPoint,
    ) -> GridControlPoint:
        rw_x = point1.rw_x + (point3.rw_x - point0.rw_x)
        rw_y = point1.rw_y + (point3.rw_y - point0.rw_y)
        inl = point3.inl
        cxl = point1.cxl
        return GridControlPoint("Point2", (rw_x, rw_y), inl, cxl)

    @property
    def inl_angle_deg(self) -> float:
        ang = degrees(atan2(self._inl_unit[0], self._inl_unit[1]))
        return ang if ang >= 0 else ang + 360.0

    @property
    def cxl_angle_deg(self) -> float:
        ang = degrees(atan2(self._cxl_unit[0], self._cxl_unit[1]))
        return ang if ang >= 0 else ang + 360.0

    def rw_from_grid(self, inl: float, cxl: float) -> RealPoint:
        dist_inl = (float(inl) - self.point0.inl) * self.step_inl
        dist_cxl = (float(cxl) - self.point0.cxl) * self.step_cxl
        return (
            self.point0.rw_x + dist_inl * self._inl_unit[0] + dist_cxl * self._cxl_unit[0],
            self.point0.rw_y + dist_inl * self._inl_unit[1] + dist_cxl * self._cxl_unit[1],
        )

    def grid_from_rw(self, rw: RealPoint) -> tuple[float, float]:
        rel = _vec(self.point0.rw, rw)
        dist_inl = _dot(rel, self._inl_unit)
        dist_cxl = _dot(rel, self._cxl_unit)
        inl = self.point0.inl + dist_inl / self.step_inl
        cxl = self.point0.cxl + dist_cxl / self.step_cxl
        return (float(inl), float(cxl))

    def locate_rw(self, rw: RealPoint) -> tuple[float, float, float, float]:
        inl, cxl = self.grid_from_rw(rw)
        return (float(rw[0]), float(rw[1]), float(inl), float(cxl))

    def summary_text(self) -> str:
        lines = [
            "SurveyGrid:",
            "  control_order: Point0 -> Point1 -> Point2 -> Point3",
            "  point_meaning: Point0=origin, Point1=first inline last point, Point2=diagonal point, Point3=last inline first point",
            "  direction_rule: Point0->Point3 defines inl direction, Point0->Point1 defines cxl direction, angle uses North=0 and clockwise positive",
            f"  step_inl: {self.step_inl} (distance(Point0, Point3) / inl interval)",
            f"  step_cxl: {self.step_cxl} (distance(Point0, Point1) / cxl interval)",
            f"  inl_unit: ({self._inl_unit[0]:.6f}, {self._inl_unit[1]:.6f})",
            f"  cxl_unit: ({self._cxl_unit[0]:.6f}, {self._cxl_unit[1]:.6f})",
            f"  inl_angle_deg: {self.inl_angle_deg:.6f}",
            f"  cxl_angle_deg: {self.cxl_angle_deg:.6f}",
            "  control_points:",
        ]
        for point in (self.point0, self.point1, self.point2, self.point3):
            lines.append(
                f"    {point.name}: rw_x={point.rw_x:.6f}, rw_y={point.rw_y:.6f}, "
                f"inl={point.inl:.6f}, cxl={point.cxl:.6f}"
            )
        return "\n".join(lines)

    def print_info(self) -> None:
        print(self.summary_text())

    def as_dict(self) -> JSONDict:
        return {
            "point0": self.point0.as_dict(),
            "point1": self.point1.as_dict(),
            "point2": self.point2.as_dict(),
            "point3": self.point3.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: JSONDict) -> SurveyGrid:
        return cls(
            point0=GridControlPoint.from_dict(payload["point0"]),
            point1=GridControlPoint.from_dict(payload["point1"]),
            point2=GridControlPoint.from_dict(payload["point2"]),
            point3=GridControlPoint.from_dict(payload["point3"]),
        )

    def to_json_file(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def from_json_file(cls, path: str | Path) -> SurveyGrid:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _parse_point_arg(raw: str, name: str) -> GridControlPoint:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{name} must be 'rw_x,rw_y,inl,cxl'")
    rw_x, rw_y, inl, cxl = (float(value) for value in parts)
    return GridControlPoint(name=name.capitalize(), rw=(rw_x, rw_y), inl=inl, cxl=cxl)


def _vec(start: RealPoint, end: RealPoint) -> RealPoint:
    return (float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def _norm(vec: RealPoint) -> float:
    return float(np.hypot(vec[0], vec[1]))


def _dot(left: RealPoint, right: RealPoint) -> float:
    return float(left[0] * right[0] + left[1] * right[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build survey grid from Point0/Point1/Point3 and convert between rw and grid coordinates.")
    parser.add_argument("--point0", required=True, help="Point0 as rw_x,rw_y,inl,cxl")
    parser.add_argument("--point1", required=True, help="Point1 as rw_x,rw_y,inl,cxl")
    parser.add_argument("--point3", required=True, help="Point3 as rw_x,rw_y,inl,cxl")
    parser.add_argument("--rw_x", type=float, help="Real-world x coordinate")
    parser.add_argument("--rw_y", type=float, help="Real-world y coordinate")
    parser.add_argument("--inl", type=float, help="Inline coordinate")
    parser.add_argument("--cxl", type=float, help="Crossline coordinate")
    parser.add_argument("--print-grid", action="store_true", help="Print grid summary")
    parser.add_argument("--json-out", help="Optional output JSON file path")
    args = parser.parse_args()

    grid = SurveyGrid.from_three_points(
        point0=_parse_point_arg(args.point0, "point0"),
        point1=_parse_point_arg(args.point1, "point1"),
        point3=_parse_point_arg(args.point3, "point3"),
    )

    has_rw = args.rw_x is not None or args.rw_y is not None
    has_grid = args.inl is not None or args.cxl is not None
    if has_rw and has_grid:
        raise ValueError("Use either --rw_x/--rw_y or --inl/--cxl, not both")
    if has_rw and (args.rw_x is None or args.rw_y is None):
        raise ValueError("Both --rw_x and --rw_y are required together")
    if has_grid and (args.inl is None or args.cxl is None):
        raise ValueError("Both --inl and --cxl are required together")

    if args.print_grid or (not has_rw and not has_grid):
        grid.print_info()

    if has_rw:
        inl, cxl = grid.grid_from_rw((args.rw_x, args.rw_y))
        print(f"locate_rw: rw_x={args.rw_x:.6f}, rw_y={args.rw_y:.6f}, inl={inl:.6f}, cxl={cxl:.6f}")
    elif has_grid:
        rw_x, rw_y = grid.rw_from_grid(args.inl, args.cxl)
        print(f"locate_grid: rw_x={rw_x:.6f}, rw_y={rw_y:.6f}, inl={args.inl:.6f}, cxl={args.cxl:.6f}")

    if args.json_out:
        grid.to_json_file(args.json_out)
        print(f"json_written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
