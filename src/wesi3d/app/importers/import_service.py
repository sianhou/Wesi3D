#!/usr/bin/env python3
"""
Import service template for seismic/attribute data.

This module owns the import workflow outside the dialog so the UI can stay
focused on collecting parameters. Real loading/caching behavior can be added
incrementally later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImportRequest:
    path: Path
    file_type: str
    output_name: str
    target_category: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportResult:
    request: ImportRequest
    values: dict[str, object]
    cache_path: Path | None = None
    loaded_payload: object | None = None


def build_import_request(values: dict[str, object]) -> ImportRequest:
    print("[ImportService] build_import_request", flush=True)
    print(f"[ImportService] input values keys: {sorted(values.keys())}", flush=True)
    return ImportRequest(
        path=Path(str(values["path"])).expanduser().resolve(),
        file_type=str(values.get("file_type", "segy")),
        output_name=str(values.get("name", "")),
        target_category=str(values.get("target_category", "seismic")),
        options=dict(values),
    )


def scan_import_source(request: ImportRequest) -> dict[str, object]:
    print("[ImportService] scan_import_source", flush=True)
    print(f"[ImportService] path={request.path}", flush=True)
    print(f"[ImportService] file_type={request.file_type}", flush=True)
    return {
        "path_exists": request.path.exists(),
        "file_type": request.file_type,
        "target_category": request.target_category,
    }


def load_import_payload(request: ImportRequest) -> object | None:
    print("[ImportService] load_import_payload", flush=True)
    print(f"[ImportService] output_name={request.output_name}", flush=True)
    return None


def save_import_cache(request: ImportRequest, payload: object | None) -> Path | None:
    print("[ImportService] save_import_cache", flush=True)
    print(f"[ImportService] cache skipped for now: {request.path.name}", flush=True)
    return None


def execute_import(request: ImportRequest) -> ImportResult:
    print("[ImportService] execute_import", flush=True)
    scan_import_source(request)
    payload = load_import_payload(request)
    cache_path = save_import_cache(request, payload)
    return ImportResult(
        request=request,
        values=dict(request.options),
        cache_path=cache_path,
        loaded_payload=payload,
    )
