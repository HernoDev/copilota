"""Parser para archivos Markdown: chunk por secciones (encabezados H2)."""

from __future__ import annotations

from pathlib import Path

from copilota.parser.base import BaseParser
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, NodeType

MIN_SECTION_CHARS = 80


@ParserRegistry.register
class MarkdownParser(BaseParser):
    @property
    def language(self) -> str:
        return "markdown"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".md",)

    def parse_file(self, filepath: Path, source: str) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        for title, start_line, end_line, text in self._split_sections(source):
            if len(text.strip()) < MIN_SECTION_CHARS:
                continue
            nodes.append(
                ASTNode(
                    node_type=NodeType.SECTION,
                    name=title,
                    source_code=text,
                    start_line=start_line,
                    end_line=end_line,
                    filepath=str(filepath),
                    language=self.language,
                )
            )
        return nodes

    def _split_sections(self, source: str) -> list[tuple[str, int, int, str]]:
        lines = source.splitlines()
        sections: list[tuple[str, int, int, str]] = []
        title: str | None = None
        start: int | None = None
        buf: list[str] = []

        def flush(end: int) -> None:
            nonlocal title, start, buf
            if title is not None and start is not None and buf:
                sections.append((title, start, end, "\n".join(buf)))
            title, start, buf = None, None, []

        h1 = ""
        for i, line in enumerate(lines, 1):
            if line.startswith("## "):
                flush(i - 1)
                title, start, buf = line[3:].strip(), i, [line]
                continue
            if line.startswith("# ") and not h1:
                h1 = line[2:].strip()
                if title is None:
                    title, start, buf = h1, i, [line]
                    continue
            if title is not None:
                buf.append(line)
        flush(len(lines))
        return sections

    def get_chunk_text(self, node: ASTNode) -> str:
        return node.source_code
