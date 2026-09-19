"""Parser para archivos Go usando tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_go as tsgo
from tree_sitter import Language, Node, Parser

from copilota.parser.base import BaseParser
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, NodeType

TS_LANGUAGE = Language(tsgo.language())


@ParserRegistry.register
class GoParser(BaseParser):
    _ts_parser: "Parser | None" = None

    @property
    def language(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".go",)

    def parse_file(self, filepath: Path, source: str) -> list[ASTNode]:
        if self._ts_parser is None:
            self._ts_parser = Parser(TS_LANGUAGE)
        tree = self._ts_parser.parse(source.encode())
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
            "function_declaration": NodeType.FUNCTION,
            "method_declaration": NodeType.METHOD,
            "import_declaration": NodeType.IMPORT,
            "package_clause": NodeType.MODULE,
        }
        if node.type == "type_declaration":
            return self._handle_type_declaration(node, source, filepath)

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

    def _handle_type_declaration(self, node, source: str, filepath: str) -> ASTNode | None:
        for child in node.children:
            if child.type != "type_spec":
                continue
            is_interface = any(
                sub.type == "interface_type" for sub in child.children
            )
            type_name = self._extract_name(child)
            return ASTNode(
                node_type=NodeType.INTERFACE if is_interface else NodeType.STRUCT,
                name=type_name or "<anonymous>",
                source_code=child.text.decode(),
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                filepath=filepath,
                language=self.language,
            )
        return None

    def _extract_name(self, node: Node) -> str | None:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "package_identifier"):
                return child.text.decode()
        return None

    def get_chunk_text(self, node: ASTNode) -> str:
        if node.node_type in (NodeType.FUNCTION, NodeType.METHOD):
            first_line = node.source_code.split("\n")[0]
            return f"// {node.node_type.value}: {node.name}\n{first_line}"
        return node.source_code
