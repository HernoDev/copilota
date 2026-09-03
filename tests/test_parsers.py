"""Tests para los parsers de código."""

from pathlib import Path
from textwrap import dedent

from copilota.parser.go import GoParser
from copilota.parser.javascript import JavaScriptParser
from copilota.parser.markdown import MarkdownParser
from copilota.parser.php import PHPParser
from copilota.parser.python import PythonParser
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


class TestMarkdownParser:
    def test_parse_sections_by_h2(self, tmp_path):
        f = _write(tmp_path, "doc.md", """\
# Titulo
Intro del documento.

## Seccion uno
Contenido de la primera seccion con suficiente texto para superar el minimo.

## Seccion dos
Contenido de la segunda seccion con suficiente texto para superar el minimo.
""")
        parser = MarkdownParser()
        nodes = parser.parse_file(f, f.read_text())
        assert [n.name for n in nodes] == ["Seccion uno", "Seccion dos"]
        assert all(n.node_type == NodeType.SECTION for n in nodes)
        assert nodes[0].start_line == 4
        assert "primera seccion" in nodes[0].source_code

    def test_small_sections_filtered(self, tmp_path):
        f = _write(tmp_path, "doc.md", """\
# Titulo

## Chica
corta

## Grande
Seccion con contenido lo suficientemente largo como para superar el minimo de caracteres.
""")
        parser = MarkdownParser()
        nodes = parser.parse_file(f, f.read_text())
        assert [n.name for n in nodes] == ["Grande"]

    def test_file_without_h2_is_single_chunk(self, tmp_path):
        f = _write(tmp_path, "doc.md", "# Solo titulo\n" + "Texto. " * 30)
        parser = MarkdownParser()
        nodes = parser.parse_file(f, f.read_text())
        assert len(nodes) == 1
        assert nodes[0].name == "Solo titulo"

    def test_get_chunk_text_is_full_section(self, tmp_path):
        f = _write(tmp_path, "doc.md", """\
# Titulo

## Seccion
Contenido de la seccion lo suficientemente largo como para superar el minimo.
""")
        parser = MarkdownParser()
        nodes = parser.parse_file(f, f.read_text())
        assert parser.get_chunk_text(nodes[0]) == nodes[0].source_code

    def test_file_extensions(self):
        assert MarkdownParser().file_extensions == (".md",)

    def test_language(self):
        assert MarkdownParser().language == "markdown"


class TestParserRegistry:
    def test_supported_languages(self):
        from copilota.parser import go, javascript, markdown, php, python, rust  # noqa: F401
        from copilota.parser.registry import ParserRegistry
        langs = ParserRegistry.supported_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "php" in langs
        assert "go" in langs
        assert "rust" in langs
        assert "markdown" in langs

    def test_supported_extensions(self):
        from copilota.parser import go, javascript, php, python, rust  # noqa: F401
        from copilota.parser.registry import ParserRegistry
        exts = ParserRegistry.supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".php" in exts
        assert ".go" in exts
        assert ".rs" in exts
