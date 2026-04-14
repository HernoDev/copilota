"""Interfaz base para todos los parsers de lenguaje."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from copilota.storage.models import ASTNode


class BaseParser(ABC):
    """Interfaz que todo parser de lenguaje debe implementar."""

    @property
    @abstractmethod
    def language(self) -> str:
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> tuple[str, ...]:
        ...

    @abstractmethod
    def parse_file(self, filepath: Path, source: str) -> list[ASTNode]:
        ...

    @abstractmethod
    def get_chunk_text(self, node: ASTNode) -> str:
        ...
