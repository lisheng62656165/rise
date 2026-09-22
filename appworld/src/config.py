from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
_ENV_FILE = os.environ.get("AGENT_EVAL_ENV_FILE")
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")
    if _ENV_FILE:
        load_dotenv(Path(_ENV_FILE), override=True)
else:
    env_path = Path(_ENV_FILE) if _ENV_FILE else ROOT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if _ENV_FILE:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
            else:
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.0
    max_completion_tokens: int = 2048
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    request_timeout: int = 120
    request_retries: int = 5  # Increased from 2 for better 429 handling
    seed: int | None = None
    context_overflow_retry: bool = False
    context_window: int = 8192
    context_overflow_min_completion_tokens: int = 1
    # Rate limiting settings
    max_rpm: int = 80  # Conservative limit (100 is hard limit)
    max_tpm: int = 8_000_000  # Conservative limit (10M is hard limit)


def get_settings() -> Settings:
    settings = Settings(
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("MIMO_BASE_URL") or "https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ["MIMO_API_KEY"],
        model_name=os.environ["MODEL_NAME"],
        temperature=float(os.getenv("TEMPERATURE", "0")),
        max_completion_tokens=int(os.getenv("MAX_COMPLETION_TOKENS", "4096")),
        top_p=(float(os.environ["TOP_P"]) if os.getenv("TOP_P") else None),
        frequency_penalty=(
            float(os.environ["FREQUENCY_PENALTY"]) if os.getenv("FREQUENCY_PENALTY") else None
        ),
        presence_penalty=(
            float(os.environ["PRESENCE_PENALTY"]) if os.getenv("PRESENCE_PENALTY") else None
        ),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
        request_retries=int(os.getenv("REQUEST_RETRIES", "5")),
        seed=(int(os.environ["LLM_SEED"]) if os.getenv("LLM_SEED") else None),
        context_overflow_retry=os.getenv("CONTEXT_OVERFLOW_RETRY", "0") in {"1", "true", "True"},
        context_window=int(os.getenv("CONTEXT_WINDOW", "8192")),
        context_overflow_min_completion_tokens=int(
            os.getenv("CONTEXT_OVERFLOW_MIN_COMPLETION_TOKENS", "1")
        ),
        max_rpm=int(os.getenv("MAX_RPM", "80")),
        max_tpm=int(os.getenv("MAX_TPM", "8000000")),
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    placeholder_values = {
        "https://your-xiaomi-base-url",
        "your-xiaomi-api-key",
        "your-model-name",
    }
    if settings.base_url in placeholder_values or "your-xiaomi-base-url" in settings.base_url:
        raise ValueError(
            "MIMO_BASE_URL is still a placeholder. Edit .env and set it to the real Xiaomi API base URL."
        )
    if settings.api_key in placeholder_values or settings.api_key.startswith("your-"):
        raise ValueError(
            "MIMO_API_KEY is still a placeholder. Edit .env and set it to your real Xiaomi API key."
        )
    if settings.model_name in placeholder_values or settings.model_name.startswith("your-"):
        raise ValueError(
            "MODEL_NAME is still a placeholder. Edit .env and set it to a real model name, for example mimo-v2.5-pro."
        )
