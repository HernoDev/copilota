"""Modelos de datos para chunks de código indexado."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    STRUCT = "struct"
    TRAIT = "trait"
    ENUM = "enum"
    CONSTANT = "constant"
    VARIABLE = "variable"


@dataclass
class ASTNode:
    """Representa un nodo AST extraído de un archivo."""

    node_type: NodeType
    name: str
    source_code: str
    start_line: int
    end_line: int
    filepath: str
    language: str
    children: list[ASTNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f"{self.filepath}:{self.start_line}-{self.end_line}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "name": self.name,
            "source_code": self.source_code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "filepath": self.filepath,
            "language": self.language,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


@dataclass
class CodeChunk:
    """Chunk de código listo para ser embebido y almacenado en la vector DB."""

    id: str
    node: ASTNode
    embedding_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict[str, Any]:
        return {
            "filepath": self.node.filepath,
            "language": self.node.language,
            "node_type": self.node.node_type.value,
            "name": self.node.name,
            "start_line": self.node.start_line,
            "end_line": self.node.end_line,
            **self.metadata,
        }
