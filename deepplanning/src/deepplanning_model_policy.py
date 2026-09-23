from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PRIMARY_MODEL = "mimo-v2.5-pro"
PRIMARY_MODEL_CONFIG = "mimo-v2.5-pro"
DEFAULT_MODELS_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "data_downloard"
    / "deepplanning"
    / "Qwen-Agent-main"
    / "benchmark"
    / "deepplanning"
    / "models_config.json"
)


def resolve_model_identity(models_config_path: Path, config_name: str) -> dict[str, Any]:
    payload = json.loads(models_config_path.read_text(encoding="utf-8"))
    config = (payload.get("models") or {}).get(config_name)
    if not isinstance(config, dict):
        raise ValueError(f"Missing model config: {config_name}")
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "model_config": config_name,
        "api_model": str(config.get("model_name", config_name)),
        "model_config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def resolve_primary_model_identity(models_config_path: Path) -> dict[str, Any]:
    identity = resolve_model_identity(models_config_path, PRIMARY_MODEL_CONFIG)
    if identity["api_model"] != PRIMARY_MODEL:
        raise ValueError(
            f"{PRIMARY_MODEL_CONFIG} must resolve to {PRIMARY_MODEL}, got {identity['api_model']}"
        )
    return identity
