from __future__ import annotations

import numpy as np
import pytest

from wesi3d.data.volume_package import AxisRange, GridMeta, GridPoint, RangeMeta, VolumePackage
from wesi3d.data.volume_store import NpzVolumeStore, ZarrVolumeStore


def _build_volume_package() -> VolumePackage:
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    return VolumePackage(
        type="volume",
        range=RangeMeta(
            inline=AxisRange(begin=370, end=372, step=1, spacing=20.0),
            cxline=AxisRange(begin=460, end=463, step=1, spacing=20.0),
            sample=AxisRange(begin=0, end=4, step=1, spacing=10.0),
        ),
        grid=GridMeta(
            p0=GridPoint(x=9180.0, y=7380.0),
            p1=GridPoint(x=27980.0, y=7380.0),
            p3=GridPoint(x=9180.0, y=30980.0),
        ),
        data=data,
    )


def test_volume_package_constructs_with_expected_values() -> None:
    volume = _build_volume_package()

    assert volume.type == "volume"
    assert volume.shape == (2, 3, 4)
    assert volume.data.dtype == np.float32

    assert volume.range.inline.begin == 370
    assert volume.range.inline.end == 372
    assert volume.range.inline.step == 1
    assert volume.range.inline.spacing == 20.0

    assert volume.range.cxline.begin == 460
    assert volume.range.cxline.end == 463
    assert volume.range.cxline.step == 1
    assert volume.range.cxline.spacing == 20.0

    assert volume.range.sample.begin == 0
    assert volume.range.sample.end == 4
    assert volume.range.sample.step == 1
    assert volume.range.sample.spacing == 10.0

    assert volume.grid.p0.x == 9180.0
    assert volume.grid.p0.y == 7380.0
    assert volume.grid.p1.x == 27980.0
    assert volume.grid.p1.y == 7380.0
    assert volume.grid.p3.x == 9180.0
    assert volume.grid.p3.y == 30980.0

    np.testing.assert_array_equal(volume.data, np.arange(24, dtype=np.float32).reshape(2, 3, 4))


def test_npz_volume_store_round_trip(tmp_path) -> None:
    volume = _build_volume_package()

    path = tmp_path / "volume-package.npz"
    saved_path = NpzVolumeStore.save(volume, path)
    restored = NpzVolumeStore.load(saved_path)

    assert restored.type == volume.type
    assert restored.range.inline.as_dict() == volume.range.inline.as_dict()
    assert restored.range.cxline.as_dict() == volume.range.cxline.as_dict()
    assert restored.range.sample.as_dict() == volume.range.sample.as_dict()
    assert restored.grid.p0.as_dict() == volume.grid.p0.as_dict()
    assert restored.grid.p1.as_dict() == volume.grid.p1.as_dict()
    assert restored.grid.p3.as_dict() == volume.grid.p3.as_dict()
    assert restored.shape == volume.shape
    assert restored.data.dtype == np.float32
    np.testing.assert_array_equal(restored.data, volume.data)


def test_volume_package_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="VolumePackage.type must be 'volume'"):
        VolumePackage(
            type="attribute",
            range=RangeMeta(
                inline=AxisRange(begin=1, end=1, step=1, spacing=1.0),
                cxline=AxisRange(begin=1, end=1, step=1, spacing=1.0),
                sample=AxisRange(begin=0, end=1, step=1, spacing=1.0),
            ),
            grid=GridMeta(
                p0=GridPoint(x=0.0, y=0.0),
                p1=GridPoint(x=1.0, y=0.0),
                p3=GridPoint(x=0.0, y=1.0),
            ),
            data=np.zeros((1, 1, 1), dtype=np.float32),
        )


def test_volume_package_rejects_non_3d_data() -> None:
    with pytest.raises(ValueError, match="VolumePackage.data must be 3D"):
        VolumePackage(
            type="volume",
            range=RangeMeta(
                inline=AxisRange(begin=1, end=1, step=1, spacing=1.0),
                cxline=AxisRange(begin=1, end=1, step=1, spacing=1.0),
                sample=AxisRange(begin=0, end=1, step=1, spacing=1.0),
            ),
            grid=GridMeta(
                p0=GridPoint(x=0.0, y=0.0),
                p1=GridPoint(x=1.0, y=0.0),
                p3=GridPoint(x=0.0, y=1.0),
            ),
            data=np.zeros((2, 2), dtype=np.float32),
        )


def test_zarr_volume_store_is_reserved_for_future_use(tmp_path) -> None:
    volume = _build_volume_package()

    with pytest.raises(NotImplementedError):
        ZarrVolumeStore.save(volume, tmp_path / "future.zarr")

    with pytest.raises(NotImplementedError):
        ZarrVolumeStore.load(tmp_path / "future.zarr")
