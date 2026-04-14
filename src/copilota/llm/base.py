"""Interfaz base para proveedores de LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Interfaz que todo proveedor de LLM debe implementar."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        ...
