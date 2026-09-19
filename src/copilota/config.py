"""Carga y gestión de configuración del proyecto."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "llm": {
        "enabled": False,
        "provider": "ollama",
        "model": "qwen2.5-coder",
        "base_url": "http://localhost",
        "port": 11434,
        "api_path": "/api/generate",
        "chat_api_path": "/api/chat",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 120,
    }
}


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "qwen2.5-coder"
    base_url: str = "http://localhost"
    port: int = 11434
    api_path: str = "/api/generate"
    chat_api_path: str = "/api/chat"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120

    @property
    def full_url(self) -> str:
        return f"{self.base_url.rstrip('/')}:{self.port}"

    @property
    def generate_url(self) -> str:
        return f"{self.full_url}{self.api_path}"

    @property
    def chat_url(self) -> str:
        return f"{self.full_url}{self.chat_api_path}"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    config_source: str = "defaults"


def _find_package_config() -> Path | None:
    """Busca el default.yaml empaquetado con el paquete (si existe)."""
    try:
        from importlib.resources import files

        pkg_files = files("copilota")
        candidate = pkg_files / "config" / "default.yaml"
        if candidate.is_file():
            return Path(candidate.location)
    except (ModuleNotFoundError, AttributeError):
        pass
    return None


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Carga configuración con precedencia: explicit > global > project > package > defaults.

    La primera fuente que exista en ese orden gana (no se mergean múltiples archivos).
    """
    candidates: list[tuple[str, Path]] = []

    if config_path:
        candidates.append(("explicit (-c)", Path(config_path)))

    global_cfg = Path.home() / ".copilota" / "config.yaml"
    candidates.append(("global (~/.copilota)", global_cfg))

    project_cfg = Path.cwd() / "config" / "default.yaml"
    candidates.append(("project (./config)", project_cfg))

    pkg_cfg = _find_package_config()
    if pkg_cfg:
        candidates.append(("package", pkg_cfg))

    for label, path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(DEFAULT_CONFIG, user_cfg)
        logger.debug("Config cargada desde %s (%s)", path, label)
        llm_cfg = merged.get("llm", {})
        return AppConfig(
            llm=_parse_llm_config(llm_cfg),
            config_source=str(path),
        )

    return AppConfig(llm=_parse_llm_config(DEFAULT_CONFIG["llm"]), config_source="defaults")


def _parse_llm_config(cfg: dict) -> LLMConfig:
    try:
        return LLMConfig(
            enabled=bool(cfg.get("enabled", False)),
            provider=str(cfg.get("provider", "ollama")),
            model=str(cfg.get("model", "qwen2.5-coder")),
            base_url=str(cfg.get("base_url", "http://localhost")),
            port=int(cfg.get("port", 11434)),
            api_path=str(cfg.get("api_path", "/api/generate")),
            chat_api_path=str(cfg.get("chat_api_path", "/api/chat")),
            temperature=float(cfg.get("temperature", 0.7)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            timeout=int(cfg.get("timeout", 120)),
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Config LLM inválida: {e}") from e


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
