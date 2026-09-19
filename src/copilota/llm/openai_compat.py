"""Proveedor OpenAI-compatible: llama-swap, vLLM, llama.cpp server."""

from __future__ import annotations

import httpx

from copilota.config import LLMConfig
from copilota.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Cliente para cualquier API compatible con OpenAI (/v1/chat/completions)."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client: httpx.AsyncClient | None = None

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=10.0, read=float(self.config.timeout), write=10.0, pool=10.0
        )

    def _payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }

    async def _post(self, payload: dict) -> str:
        client = self._client or httpx.AsyncClient(timeout=self._timeout())
        try:
            resp = await client.post(self.config.chat_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content")
            return content if content is not None else ""
        finally:
            if self._client is None:
                await client.aclose()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._post(self._payload(messages, temperature, max_tokens))

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return await self._post(self._payload(messages, temperature, max_tokens))
