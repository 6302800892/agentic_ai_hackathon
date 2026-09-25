"""Central configuration: env vars (python-dotenv) + YAML files under config/. No secrets in code (NFR-01)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LOGS = Path(os.getenv("COPILOT_LOG_DIR") or ROOT / "logs")  # tests redirect this to a temp dir
TRACES = ROOT / "traces"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
STATE_DIR = DATA / "state"


def _yaml(name: str) -> dict:
    with (ROOT / "config" / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_model: str
    gemini_model_light: str
    phi_salt: str
    phoenix_project: str
    phoenix_endpoint: str | None
    phoenix_api_key: str | None
    rag_backend: str
    reference_date: str
    limits: dict = field(default_factory=dict)
    guardrails: dict = field(default_factory=dict)
    pricing: dict = field(default_factory=dict)

    @property
    def has_llm(self) -> bool:
        return bool(self.google_api_key)


def _phoenix_endpoint() -> str | None:
    """Accept only an http(s) URL; a pasted API key or typo is ignored with a warning (never echoed)."""
    raw = (os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "").strip().rstrip("/")
    if not raw:
        return None
    if not raw.startswith(("http://", "https://")):
        import logging
        logging.getLogger(__name__).warning(
            "PHOENIX_COLLECTOR_ENDPOINT is not an http(s) URL (did you paste the API key there? put it in "
            "PHOENIX_API_KEY). Ignoring it and falling back to in-process Phoenix.")
        return None
    raw = raw.removesuffix("/v1/traces")
    # Phoenix Cloud: a URL copied from the browser may include a page path (/s/<space>/projects/...);
    # the API base is exactly https://app.phoenix.arize.com/s/<space>
    import re
    m = re.match(r"^(https://app\.phoenix\.arize\.com/s/[^/?#]+)", raw)
    return m.group(1) if m else raw


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        gemini_model_light=os.getenv("GEMINI_MODEL_LIGHT", "gemini-3.5-flash-lite"),
        phi_salt=os.getenv("PHI_HMAC_SALT", "dev-only-salt-change-me"),
        phoenix_project=os.getenv("PHOENIX_PROJECT_NAME", "patient-intake-copilot"),
        phoenix_endpoint=_phoenix_endpoint(),
        phoenix_api_key=os.getenv("PHOENIX_API_KEY") or os.getenv("ARIZE_PHOENIX_API_KEY") or None,
        rag_backend=os.getenv("COPILOT_RAG_BACKEND", "chroma"),
        # Synthetic "today" so coverage-period checks (COV-R2) are reproducible.
        reference_date=os.getenv("COPILOT_REFERENCE_DATE", "2026-09-25"),
        limits=_yaml("limits.yaml"),
        guardrails=_yaml("guardrails.yaml"),
        pricing=_yaml("pricing.yaml"),
    )
