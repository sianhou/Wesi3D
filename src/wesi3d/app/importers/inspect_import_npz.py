#!/usr/bin/env python3
"""
Small helper to inspect generated seismic importer npz files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from wesi3d.data.volume_store import NpzVolumeStore


def inspect_npz(path: str | Path) -> int:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}", flush=True)
        return 1

    package = NpzVolumeStore.load(npz_path)
    data_type = package.type
    range_info = package.range.as_dict()
    grid = package.grid.as_dict()
    data = np.asarray(package.data, dtype=np.float32)

    print(f"path={npz_path}", flush=True)
    print(f"type={json.dumps(data_type, ensure_ascii=False, indent=2)}", flush=True)
    print(f"range={json.dumps(range_info, ensure_ascii=False, indent=2)}", flush=True)
    print(f"grid={json.dumps(grid, ensure_ascii=False, indent=2)}", flush=True)
    print(f"data.shape={data.shape}", flush=True)
    print(f"data.dtype={data.dtype}", flush=True)
    print(f"data.min={float(np.min(data))}", flush=True)
    print(f"data.max={float(np.max(data))}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: inspect_import_npz.py <path-to-npz>", flush=True)
        return 1
    return inspect_npz(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
