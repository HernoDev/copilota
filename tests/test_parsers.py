"""Tests para los parsers de código."""

from pathlib import Path
from textwrap import dedent

import pytest

from copilota.parser.python import PythonParser
from copilota.parser.javascript import JavaScriptParser
from copilota.parser.php import PHPParser
from copilota.parser.go import GoParser
from copilota.parser.rust import RustParser
from copilota.storage.models import NodeType


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(dedent(content))
    return p


class TestPythonParser:
    def test_parse_function(self, tmp_path):
        f = _write(tmp_path, "app.py", """
def hello(name):
    return f"Hello {name}"
""")
        parser = PythonParser()
        nodes = parser.parse_file(f, f.read_text())
        funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "hello"
        assert funcs[0].start_line == 2

    def test_parse_class(self, tmp_path):
        f = _write(tmp_path, "models.py", """
class User:
    def __init__(self, name):
        self.name = name
""")
        parser = PythonParser()
        nodes = parser.parse_file(f, f.read_text())
        classes = [n for n in nodes if n.node_type == NodeType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "User"

    def test_get_chunk_text_function(self):
        from copilota.storage.models import ASTNode
        node = ASTNode(
            node_type=NodeType.FUNCTION, name="hello",
            source_code="def hello():\n    pass",
            start_line=1, end_line=2,
            filepath="test.py", language="python",
        )
        parser = PythonParser()
        text = parser.get_chunk_text(node)
        assert "hello" in text

    def test_get_chunk_text_class(self):
        from copilota.storage.models import ASTNode
        node = ASTNode(
            node_type=NodeType.CLASS, name="User",
            source_code="class User:\n    pass",
            start_line=1, end_line=2,
            filepath="test.py", language="python",
        )
        parser = PythonParser()
        text = parser.get_chunk_text(node)
        assert "User" in text

    def test_file_extensions(self):
        assert PythonParser().file_extensions == (".py",)

    def test_language(self):
        assert PythonParser().language == "python"


class TestJavaScriptParser:
    def test_parse_function(self, tmp_path):
        f = _write(tmp_path, "index.js", """
function greet(name) {
    return `Hello ${name}`;
}
""")
        parser = JavaScriptParser()
        nodes = parser.parse_file(f, f.read_text())
        funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
        assert len(funcs) >= 1

    def test_parse_class(self, tmp_path):
        f = _write(tmp_path, "app.js", """
class App {
    constructor() {
        this.name = "test";
    }
}
""")
        parser = JavaScriptParser()
        nodes = parser.parse_file(f, f.read_text())
        classes = [n for n in nodes if n.node_type == NodeType.CLASS]
        assert len(classes) >= 1


class TestPHPParser:
    def test_parse_function(self, tmp_path):
        f = _write(tmp_path, "funcs.php", """
<?php
function hello($name) {
    return "Hello $name";
}
""")
        parser = PHPParser()
        nodes = parser.parse_file(f, f.read_text())
        funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
        assert len(funcs) >= 1

    def test_parse_class(self, tmp_path):
        f = _write(tmp_path, "User.php", """
<?php
class User {
    public function getName() {
        return $this->name;
    }
}
""")
        parser = PHPParser()
        nodes = parser.parse_file(f, f.read_text())
        classes = [n for n in nodes if n.node_type == NodeType.CLASS]
        assert len(classes) >= 1


class TestGoParser:
    def test_parse_function(self, tmp_path):
        f = _write(tmp_path, "main.go", """
package main
func Hello(name string) string {
    return "Hello " + name
}
""")
        parser = GoParser()
        nodes = parser.parse_file(f, f.read_text())
        funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
        assert len(funcs) >= 1


class TestRustParser:
    def test_parse_function(self, tmp_path):
        f = _write(tmp_path, "main.rs", """
fn main() {
    println!("Hello");
}
""")
        parser = RustParser()
        nodes = parser.parse_file(f, f.read_text())
        funcs = [n for n in nodes if n.node_type == NodeType.FUNCTION]
        assert len(funcs) >= 1

    def test_parse_struct(self, tmp_path):
        f = _write(tmp_path, "lib.rs", """
struct User {
    name: String,
}
""")
        parser = RustParser()
        nodes = parser.parse_file(f, f.read_text())
        structs = [n for n in nodes if n.node_type == NodeType.STRUCT]
        assert len(structs) >= 1


class TestParserRegistry:
    def test_supported_languages(self):
        from copilota.parser import python, javascript, php, go, rust
        from copilota.parser.registry import ParserRegistry
        langs = ParserRegistry.supported_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "php" in langs
        assert "go" in langs
        assert "rust" in langs

    def test_supported_extensions(self):
        from copilota.parser import python, javascript, php, go, rust
        from copilota.parser.registry import ParserRegistry
        exts = ParserRegistry.supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".php" in exts
        assert ".go" in exts
        assert ".rs" in exts
