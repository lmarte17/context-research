from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {file_path}")

    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level YAML document must be a mapping in {file_path}")
    return payload
