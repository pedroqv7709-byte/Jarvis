"""Runtime configuration loaded from environment variables (prefix ``JARVIS_``) and ``.env``."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis.desktop.catalog import ApplicationConfig
from jarvis.permissions.model import GrantMode

DEFAULT_PERMISSIONS: dict[str, GrantMode] = {
    "fs.read": GrantMode.ALLOW,
    "fs.write": GrantMode.ALLOW,
    "fs.delete": GrantMode.CONFIRM,
    "windows.read": GrantMode.ALLOW,
    "windows.open": GrantMode.ALLOW,
    "windows.focus": GrantMode.ALLOW,
    # Closing may discard unsaved work. The close_application category floor requires
    # confirmation anyway; this grant cannot relax it.
    "windows.close": GrantMode.CONFIRM,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JARVIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path(".jarvis")
    workspace_dir: Path = Path("sandbox")
    user_id: str = Field(default="local-user", min_length=1)
    permissions: dict[str, GrantMode] = Field(default_factory=lambda: dict(DEFAULT_PERMISSIONS))

    max_attempts_per_action: int = Field(default=3, ge=1, le=20)
    max_total_attempts_per_task: int = Field(default=20, ge=1, le=500)
    retry_backoff_seconds: float = Field(default=0.5, ge=0)
    retry_backoff_multiplier: float = Field(default=2.0, ge=1)
    retry_max_backoff_seconds: float = Field(default=5.0, ge=0)

    max_plan_steps: int = Field(default=10, ge=1, le=100)
    max_request_chars: int = Field(default=4000, ge=1)

    # Planning with a language model. "none" = deterministic planner only.
    model_provider: Literal["none", "litellm"] = "none"
    model: str | None = None
    """LiteLLM model string, e.g. ``openai/gpt-4o-mini`` or ``ollama/llama3.1``."""
    model_api_base: str | None = None
    """Optional endpoint override (e.g. a local Ollama server)."""
    model_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    model_structured_output: Literal["auto", "json_schema", "json_object"] = "auto"
    """How the expected JSON shape is requested from the backend. ``auto``: a JSON Schema when
    LiteLLM knows the backend supports it, else JSON mode. The answer is always validated."""
    planner_fallback: bool = True
    """When the model fails or its plan is invalid, try the deterministic planner."""

    # Agent loop hard limits (enforced by Jarvis, never by a planner or model).
    agent_max_iterations: int = Field(default=20, ge=1, le=200)
    """Planner decisions per task."""
    agent_max_actions: int = Field(default=20, ge=1, le=200)
    """Actions that may start executing per task."""
    agent_max_replans: int = Field(default=5, ge=0, le=50)
    """Decisions made in reaction to a failed action or a refused completion."""
    agent_history_steps: int = Field(default=10, ge=1, le=50)
    agent_max_observed_windows: int = Field(default=30, ge=1, le=200)
    agent_max_prompt_chars: int = Field(default=12_000, ge=2_000, le=200_000)

    # Execution time limits.
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    """Hard limit for a single tool attempt."""
    task_timeout_seconds: float = Field(default=300.0, gt=0, le=86400)
    """Global per-task budget: tool execution + retry backoff + planning. Waiting for the user's
    confirmation is not counted."""

    # Local API server (``jarvis server``). Loopback only: see ``_loopback_only``.
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=8765, ge=1024, le=65535)
    shutdown_grace_seconds: float = Field(default=10.0, ge=0, le=300)
    """On shutdown, how long to wait for the task in progress to reach a safe point."""

    # Windows desktop automation (requires Windows and the optional ``windows`` extra).
    windows_automation: Literal["auto", "off"] = "auto"
    windows_applications: dict[str, ApplicationConfig] = Field(default_factory=dict)
    """Extra allowlisted applications, merged with the built-in ones (notepad, calculator)."""
    windows_open_wait_seconds: float = Field(default=10.0, gt=0, le=60)
    """How long open_application waits for the application's window to appear."""
    windows_close_wait_seconds: float = Field(default=5.0, gt=0, le=60)

    @field_validator("server_host")
    @classmethod
    def _loopback_only(cls, host: str) -> str:
        """The API has no authentication, so it must never listen on a network interface."""
        if host == "localhost":
            return host
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError(
                f"server_host must be a loopback address (127.0.0.1, ::1, localhost), got {host!r}"
            )
        return host

    @property
    def server_url(self) -> str:
        host = f"[{self.server_host}]" if ":" in self.server_host else self.server_host
        return f"http://{host}:{self.server_port}"

    @property
    def api_token_path(self) -> Path:
        return self.data_dir / "api_token"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "jarvis.lock"

    @property
    def state_db_path(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def audit_log_path(self) -> Path:
        return self.data_dir / "audit.jsonl"
