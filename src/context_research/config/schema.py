from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Any
import uuid


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int | None
    driver_version: str | None


@dataclass(frozen=True)
class HardwareInfo:
    hostname: str
    os_name: str
    os_version: str
    machine: str
    python_version: str
    cpu_count: int | None
    gpus: list[GPUInfo] = field(default_factory=list)


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    created_at_utc: str
    model_name: str
    config_path: str
    config_hash_sha256: str
    hardware: HardwareInfo
    notes: dict[str, Any] = field(default_factory=dict)


def generate_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def compute_file_sha256(config_path: str | Path) -> str:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_gpu_info() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: list[GPUInfo] = []
    for raw_line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            continue

        try:
            index = int(parts[0])
        except ValueError:
            continue

        try:
            memory_total_mb = int(parts[2])
        except ValueError:
            memory_total_mb = None

        driver_version = parts[3] or None
        gpus.append(
            GPUInfo(
                index=index,
                name=parts[1],
                memory_total_mb=memory_total_mb,
                driver_version=driver_version,
            )
        )

    return gpus


def collect_hardware_info() -> HardwareInfo:
    return HardwareInfo(
        hostname=socket.gethostname(),
        os_name=platform.system(),
        os_version=platform.version(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        gpus=_collect_gpu_info(),
    )


def create_run_metadata(
    model_name: str,
    config_path: str | Path,
    run_id: str | None = None,
    notes: dict[str, Any] | None = None,
) -> RunMetadata:
    resolved_config_path = Path(config_path).resolve()
    metadata_run_id = run_id or generate_run_id()

    return RunMetadata(
        run_id=metadata_run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_name=model_name,
        config_path=str(resolved_config_path),
        config_hash_sha256=compute_file_sha256(resolved_config_path),
        hardware=collect_hardware_info(),
        notes=notes or {},
    )


def run_metadata_to_dict(metadata: RunMetadata) -> dict[str, Any]:
    return asdict(metadata)


def write_run_metadata(metadata: RunMetadata, output_root: str | Path) -> Path:
    root = Path(output_root)
    run_dir = root / metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "run_metadata.json"
    output_path.write_text(
        json.dumps(run_metadata_to_dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
