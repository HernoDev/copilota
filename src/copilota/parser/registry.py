"""Registry de parsers: gestiona la asociación language/extension -> parser."""

from __future__ import annotations

from pathlib import Path

from copilota.parser.base import BaseParser


class ParserRegistry:
    _parsers: dict[str, BaseParser] = {}
    _ext_map: dict[str, str] = {}

    @classmethod
    def register(cls, parser: type[BaseParser]) -> type[BaseParser]:
        instance = parser()
        cls._parsers[instance.language] = instance
        for ext in instance.file_extensions:
            cls._ext_map[ext] = instance.language
        return parser

    @classmethod
    def get_for_language(cls, language: str) -> BaseParser:
        parser = cls._parsers.get(language)
        if not parser:
            raise ValueError(f"No parser registered for language: {language}")
        return parser

    @classmethod
    def get_for_file(cls, filepath: Path) -> BaseParser:
        ext = filepath.suffix.lower()
        language = cls._ext_map.get(ext)
        if not language:
            raise ValueError(f"No parser registered for extension: {ext}")
        return cls._parsers[language]

    @classmethod
    def has_parser_for_file(cls, filepath: Path) -> bool:
        return filepath.suffix.lower() in cls._ext_map

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return sorted(cls._ext_map.keys())

    @classmethod
    def supported_languages(cls) -> list[str]:
        return sorted(cls._parsers.keys())

    @classmethod
    def reset(cls) -> None:
        cls._parsers.clear()
        cls._ext_map.clear()
