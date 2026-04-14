"""CLI de Copilota: interfaz de línea de comandos."""

from __future__ import annotations

from pathlib import Path

import asyncio
import click
from rich.console import Console
from rich.table import Table

from copilota.core.embedder import EmbeddingModel
from copilota.core.indexer import Indexer
from copilota.core.rag import RAGPipeline
from copilota.core.retriever import Retriever
from copilota.llm.ollama import OllamaLLM
from copilota.parser.registry import ParserRegistry
from copilota.storage.vector_db import VectorStore

console = Console()


def _import_parsers():
    from copilota.parser import python
    from copilota.parser import javascript
    from copilota.parser import php
    from copilota.parser import go
    from copilota.parser import rust


def _get_components(mock_embeddings: bool = False):
    embedder = EmbeddingModel(use_mock=mock_embeddings)
    store = VectorStore()
    return embedder, store


@click.group()
@click.version_option("0.1.0")
def main():
    """Copilota - Asistente de código local con RAG."""
    pass


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--mock-embeddings", is_flag=True, help="Usar embeddings mock (sin sentence-transformers)")
def index(repo_path: str, mock_embeddings: bool):
    """Indexa un repositorio Git en la base de vectores."""
    _import_parsers()
    console.print(f"Indexando repo: [bold cyan]{repo_path}[/bold cyan]")

    embedder, store = _get_components(mock_embeddings)
    indexer = Indexer(store, embedder)

    total = indexer.index_repo(repo_path)
    console.print(f"[green]✓[/green] Indexados [bold]{total}[/bold] chunks de código.")


@main.command()
@click.argument("query")
@click.option("--language", "-l", default=None, help="Filtrar por lenguaje")
@click.option("--top-k", "-k", default=5, help="Número de resultados")
@click.option("--mock-embeddings", is_flag=True, help="Usar embeddings mock")
def search(query: str, language: str | None, top_k: int, mock_embeddings: bool):
    """Busca código relevante para una consulta."""
    embedder, store = _get_components(mock_embeddings)
    retriever = Retriever(store, embedder)

    results = retriever.search(query, top_k=top_k, language=language)

    table = Table(title=f"Resultados para: {query}")
    table.add_column("Score", style="cyan")
    table.add_column("Archivo", style="green")
    table.add_column("Tipo", style="yellow")
    table.add_column("Nombre", style="magenta")

    for r in results:
        table.add_row(
            f"{r.score:.3f}",
            r.filepath,
            r.node_type,
            r.name,
        )

    console.print(table)


@main.command()
@click.argument("question")
@click.option("--language", "-l", default=None)
@click.option("--mock-embeddings", is_flag=True, help="Usar embeddings mock")
def ask(question: str, language: str | None, mock_embeddings: bool):
    """Haz una pregunta sobre el código indexado (RAG)."""
    embedder, store = _get_components(mock_embeddings)
    retriever = Retriever(store, embedder)
    llm = OllamaLLM()
    rag = RAGPipeline(retriever, llm)

    result = asyncio.run(rag.query(question, language=language))

    console.print(f"\n[bold]Respuesta:[/bold]\n{result['answer']}\n")

    if result["sources"]:
        table = Table(title="Fuentes")
        table.add_column("Archivo", style="green")
        table.add_column("Nombre", style="magenta")
        table.add_column("Tipo", style="yellow")
        table.add_column("Score", style="cyan")
        for s in result["sources"]:
            table.add_row(s["filepath"], s["name"], s["node_type"], str(s["score"]))
        console.print(table)


@main.command()
@click.option("--mock-embeddings", is_flag=True, help="Usar embeddings mock")
def info(mock_embeddings: bool):
    """Muestra información sobre el índice y parsers disponibles."""
    _import_parsers()
    _, store = _get_components(mock_embeddings)

    console.print(f"[bold]Chunks indexados:[/bold] {store.count()}")
    console.print(f"[bold]Lenguajes soportados:[/bold] {', '.join(ParserRegistry.supported_languages())}")
    console.print(f"[bold]Extensiones:[/bold] {', '.join(ParserRegistry.supported_extensions())}")


if __name__ == "__main__":
    main()
