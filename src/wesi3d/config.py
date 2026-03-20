"""
Default project configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DERIVED_DATA_DIR = DATA_DIR / "derived"
SAMPLE_DATA_DIR = DATA_DIR / "samples"


@dataclass(frozen=True)
class ViewerConfig:
    default_segy_path: Path = RAW_DATA_DIR / "vel.sgy"
    interval_inline: int = 4
    interval_xline: int = 4
    interval_sample: int = 4
    step_inline: float = 20.0
    step_xline: float = 20.0
    step_sample: float = 10.0
    clip_percentile: float = 99.0
    opacity: float = 0.85


DEFAULT_VIEWER_CONFIG = ViewerConfig()