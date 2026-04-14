"""Carga y gestión de configuración del proyecto."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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


def load_config(config_path: str | Path | None = None) -> AppConfig:
    merged = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, user_cfg)

    llm_cfg = merged.get("llm", {})
    return AppConfig(
        llm=LLMConfig(
            enabled=llm_cfg.get("enabled", False),
            provider=llm_cfg.get("provider", "ollama"),
            model=llm_cfg.get("model", "qwen2.5-coder"),
            base_url=llm_cfg.get("base_url", "http://localhost"),
            port=int(llm_cfg.get("port", 11434)),
            api_path=llm_cfg.get("api_path", "/api/generate"),
            chat_api_path=llm_cfg.get("chat_api_path", "/api/chat"),
            temperature=float(llm_cfg.get("temperature", 0.7)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
            timeout=int(llm_cfg.get("timeout", 120)),
        )
    )


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
