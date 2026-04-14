"""Parser para archivos Rust usando tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser

from copilota.parser.base import BaseParser
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, NodeType

TS_LANGUAGE = Language(tsrust.language())


@ParserRegistry.register
class RustParser(BaseParser):
    @property
    def language(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".rs",)

    def parse_file(self, filepath: Path, source: str) -> list[ASTNode]:
        ts_parser = Parser(TS_LANGUAGE)
        tree = ts_parser.parse(source.encode())
        nodes: list[ASTNode] = []
        self._walk(tree.root_node, source, str(filepath), nodes)
        return nodes

    def _walk(self, node, source: str, filepath: str, out: list[ASTNode]) -> None:
        ast_node = self._to_ast_node(node, source, filepath)
        if ast_node:
            out.append(ast_node)
        for child in node.children:
            self._walk(child, source, filepath, out)

    def _to_ast_node(self, node, source: str, filepath: str) -> ASTNode | None:
        mapping: dict[str, NodeType] = {
            "function_item": NodeType.FUNCTION,
            "struct_item": NodeType.STRUCT,
            "enum_item": NodeType.ENUM,
            "trait_item": NodeType.TRAIT,
            "impl_item": NodeType.CLASS,
            "use_declaration": NodeType.IMPORT,
            "mod_item": NodeType.MODULE,
        }
        node_type = mapping.get(node.type)
        if not node_type:
            return None

        name = self._extract_name(node)
        return ASTNode(
            node_type=node_type,
            name=name or "<anonymous>",
            source_code=node.text.decode(),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            filepath=filepath,
            language=self.language,
        )

    def _extract_name(self, node: Node) -> str | None:
        for child in node.children:
            if child.type == "name":
                return child.text.decode()
        return None

    def get_chunk_text(self, node: ASTNode) -> str:
        if node.node_type in (NodeType.FUNCTION,):
            first_line = node.source_code.split("\n")[0]
            return f"// {node.node_type.value}: {node.name}\n{first_line}"
        return node.source_code
