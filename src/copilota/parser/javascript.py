"""Parser para archivos JavaScript/TypeScript usando tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

from copilota.parser.base import BaseParser
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, NodeType

TS_LANGUAGE = Language(tsjs.language())


@ParserRegistry.register
class JavaScriptParser(BaseParser):
    @property
    def language(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".js", ".jsx", ".ts", ".tsx", ".mjs")

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
            "function_declaration": NodeType.FUNCTION,
            "function_declaration": NodeType.FUNCTION,
            "method_definition": NodeType.METHOD,
            "class_declaration": NodeType.CLASS,
            "import_statement": NodeType.IMPORT,
            "import_declaration": NodeType.IMPORT,
            "lexical_declaration": NodeType.VARIABLE,
            "variable_declaration": NodeType.VARIABLE,
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
            if child.type in ("identifier", "property_identifier"):
                return child.text.decode()
        if node.type in ("lexical_declaration", "variable_declaration"):
            for decl in node.children:
                if decl.type == "variable_declarator":
                    for sub in decl.children:
                        if sub.type == "identifier":
                            return sub.text.decode()
        return None

    def get_chunk_text(self, node: ASTNode) -> str:
        if node.node_type in (NodeType.FUNCTION, NodeType.METHOD):
            first_line = node.source_code.split("\n")[0]
            return f"// {node.node_type.value}: {node.name}\n{first_line}"
        if node.node_type == NodeType.CLASS:
            first_line = node.source_code.split("\n")[0]
            return f"// Class: {node.name}\n{first_line}"
        return node.source_code
