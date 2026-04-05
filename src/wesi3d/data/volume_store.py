#!/usr/bin/env python3
"""
Volume storage helpers.

NpzVolumeStore is the current implementation used for development/testing.
ZarrVolumeStore is reserved for future large-volume work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .volume_package import RangeMeta, GridMeta, VolumePackage


def _payload_text(value: object) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


class NpzVolumeStore:
    @staticmethod
    def save(package: VolumePackage, path: str | Path) -> Path:
        output_path = Path(path).expanduser()
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            type=np.asarray(json.dumps(package.type)),
            range=np.asarray(json.dumps(package.range.as_dict())),
            grid=np.asarray(json.dumps(package.grid.as_dict())),
            data=np.asarray(package.data, dtype=np.float32),
        )
        return output_path

    @staticmethod
    def load(path: str | Path) -> VolumePackage:
        npz_path = Path(path).expanduser().resolve()
        with np.load(npz_path, allow_pickle=False) as archive:
            type_name = json.loads(_payload_text(archive["type"]))
            range_meta = RangeMeta.from_dict(json.loads(_payload_text(archive["range"])))
            grid_meta = GridMeta.from_dict(json.loads(_payload_text(archive["grid"])))
            data = np.asarray(archive["data"], dtype=np.float32)
        return VolumePackage(
            type=str(type_name),
            range=range_meta,
            grid=grid_meta,
            data=data,
        )


class ZarrVolumeStore:
    @staticmethod
    def save(package: VolumePackage, path: str | Path) -> Path:
        raise NotImplementedError("ZarrVolumeStore.save is not implemented yet")

    @staticmethod
    def load(path: str | Path) -> VolumePackage:
        raise NotImplementedError("ZarrVolumeStore.load is not implemented yet")
