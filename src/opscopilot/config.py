"""Settings and guardrail limits. Everything can be overridden from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm: str = "rules"                 # rules (offline, deterministic) | gemini
    gemini_model: str = "gemini-2.5-flash"
    use_vertex: bool = False
    gcp_project: str | None = None
    gcp_location: str = "us-central1"
    state_dir: Path = ROOT / ".state"  # checkpoints, traces, escalations
    db_path: Path = ROOT / "data" / "ops.db"
    runbook_dir: Path = ROOT / "data" / "runbooks"
    # guardrails
    max_react_steps: int = 5           # tool-use iterations per worker agent
    max_revisions: int = 2             # self-reflection rounds
    token_budget: int = 20_000         # per run (input + output); supervisor stops delegating beyond it
    # illustrative price sheet, USD per 1M tokens. Check current Vertex AI pricing before trusting cost numbers.
    price_in_per_m: float = 0.30
    price_out_per_m: float = 2.50


def load_settings(**overrides) -> Settings:
    env = os.getenv
    base = Settings(
        llm=env("OPS_LLM", "rules"),
        gemini_model=env("OPS_GEMINI_MODEL", "gemini-2.5-flash"),
        use_vertex=env("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"},
        gcp_project=env("GOOGLE_CLOUD_PROJECT"),
        gcp_location=env("GOOGLE_CLOUD_LOCATION", "us-central1"),
        state_dir=Path(env("OPS_STATE_DIR", str(ROOT / ".state"))),
        db_path=Path(env("OPS_DB_PATH", str(ROOT / "data" / "ops.db"))),
        token_budget=int(env("OPS_TOKEN_BUDGET", "20000")),
    )
    from dataclasses import replace
    return replace(base, **overrides)
