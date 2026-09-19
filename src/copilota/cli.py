"""CLI de Copilota: interfaz de línea de comandos."""

from __future__ import annotations

import asyncio
from importlib.metadata import version as pkg_version

import click
import httpx
from rich.console import Console
from rich.table import Table

from copilota.config import LLMConfig, load_config
from copilota.core.embedder import EmbeddingModel
from copilota.core.indexer import Indexer
from copilota.core.rag import RAGPipeline
from copilota.core.retriever import Retriever, resolve_repo
from copilota.llm.factory import create_llm
from copilota.parser.registry import ParserRegistry
from copilota.storage.vector_db import VectorStore

console = Console()


def _fetch_models(config: LLMConfig) -> list[str]:
    if config.provider == "ollama":
        url = f"{config.full_url}/api/tags"
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
    else:
        url = f"{config.full_url}/v1/models"
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]


def _import_parsers():
    from copilota.parser import (  # noqa: F401
        go,
        javascript,
        markdown,
        php,
        python,
        rust,
        typescript,
    )


def _get_components(mock_embeddings: bool = False):
    embedder = EmbeddingModel(use_mock=mock_embeddings)
    store = VectorStore()
    return embedder, store


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import anyio.from_thread

    return anyio.from_thread.run(coro)


@click.group()
@click.version_option(pkg_version("copilota"), prog_name="copilota")
@click.option("--mock-embeddings", is_flag=True, default=False, help="Usar embeddings mock")
@click.pass_context
def main(ctx: click.Context, mock_embeddings: bool):
    """Copilota - Asistente de código local con RAG."""
    ctx.ensure_object(dict)
    ctx.obj["mock_embeddings"] = mock_embeddings


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--exclude", "-e", multiple=True, help="Patrón para excluir (ej: reports, docs/*.md)")
@click.pass_context
def index(ctx: click.Context, repo_path: str, exclude: tuple[str, ...]):
    """Indexa un repositorio Git en la base de vectores (reindex completo del repo)."""
    _import_parsers()
    console.print(f"Indexando repo: [bold cyan]{repo_path}[/bold cyan]")

    mock = ctx.obj.get("mock_embeddings", False)
    embedder, store = _get_components(mock)
    indexer = Indexer(store, embedder)

    result = indexer.index_repo(repo_path, exclude=list(exclude))
    console.print(
        f"[green]✓[/green] Indexados [bold]{result.chunks}[/bold] chunks "
        f"de [bold]{result.files}[/bold] archivos.\n"
        f"Repo: [dim]{result.repo_id}[/dim]"
    )


@main.command()
@click.argument("query")
@click.option("--language", "-l", default=None, help="Filtrar por lenguaje")
@click.option("--repo", "-r", default=None, help="Limitar a un repo (ruta del repositorio)")
@click.option("--top-k", "-k", default=5, help="Número de resultados")
@click.pass_context
def search(ctx: click.Context, query: str, language: str | None, repo: str | None, top_k: int):
    """Busca código relevante para una consulta."""
    mock = ctx.obj.get("mock_embeddings", False)
    embedder, store = _get_components(mock)
    retriever = Retriever(store, embedder)

    results = retriever.search(query, top_k=top_k, language=language, repo=repo)

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
@click.option("--repo", "-r", default=None, help="Limitar a un repo (ruta del repositorio)")
@click.option("--top-k", "-k", default=5, help="Número de fragmentos de contexto")
@click.option("--config", "-c", default=None, help="Ruta a archivo de configuración YAML")
@click.option(
    "--model-select",
    "model_select",
    type=int,
    default=None,
    help="Seleccionar modelo por índice (ver: copilota models)",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Seleccionar modelo por nombre exacto",
)
@click.pass_context
def ask(
    ctx: click.Context,
    question: str,
    language: str | None,
    repo: str | None,
    top_k: int,
    config: str | None,
    model_select: int | None,
    model_name: str | None,
):
    """Haz una pregunta sobre el código indexado (RAG)."""
    _import_parsers()
    mock = ctx.obj.get("mock_embeddings", False)
    embedder, store = _get_components(mock)
    retriever = Retriever(store, embedder)

    app_config = load_config(config) if config else load_config()

    if model_select is not None or model_name is not None:
        if model_select is not None:
            try:
                ids = _fetch_models(app_config.llm)
            except Exception as e:
                console.print(f"[red]No se pudo obtener la lista de modelos:[/red] {e}")
                raise SystemExit(1)
            if model_select < 1 or model_select > len(ids):
                console.print(
                    f"[red]Índice inválido:[/red] {model_select} "
                    f"(rango 1-{len(ids)})"
                )
                raise SystemExit(1)
            app_config.llm.model = ids[model_select - 1]
            console.print(f"[dim]Modelo seleccionado: {app_config.llm.model}[/dim]")
        elif model_name is not None:
            app_config.llm.model = model_name
            console.print(f"[dim]Modelo seleccionado: {model_name}[/dim]")

    llm = create_llm(app_config)

    rag = RAGPipeline(retriever, llm)
    result = _run_async(rag.query(question, top_k=top_k, language=language, repo=repo))

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
@click.argument("query")
@click.option("--repo", "-r", default=None, help="Limitar a un repo (ruta del repositorio)")
@click.option("--language", "-l", default=None, help="Filtrar por lenguaje")
@click.option("--top-k", "-k", default=5, help="Número de fragmentos")
@click.pass_context
def context(ctx: click.Context, query: str, repo: str | None, language: str | None, top_k: int):
    """Imprime contexto semántico listo para inyectar a un LLM (salida pipeable).

    Ej: copilota context "como se envia el email" -k 8 -r /ruta/repo > contexto.md
    """
    mock = ctx.obj.get("mock_embeddings", False)
    embedder, store = _get_components(mock)
    retriever = Retriever(store, embedder)

    results = retriever.search(query, top_k=top_k, language=language, repo=repo)

    lines = [f'# Contexto de código — consulta: "{query}"']
    if repo:
        lines.append(f"# Repo: {resolve_repo(repo)}")
    lines.append(f'# Fragmentos: {len(results)}')
    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"## [{i}] {r.name} ({r.node_type}) — {r.filepath} [{r.language}]")
        fence = "js" if r.language in ("javascript", "typescript") else r.language
        lines.append(f"```{fence}")
        lines.append(r.document.rstrip())
        lines.append("```")
    print("\n".join(lines))


@main.command()
def models():
    """Lista los modelos disponibles en el servidor LLM."""
    app_config = load_config()
    if not app_config.llm.enabled:
        console.print("[yellow]LLM deshabilitado en la configuración.[/yellow]")
        return
    try:
        ids = _fetch_models(app_config.llm)
    except Exception as e:
        console.print(f"[red]No se pudo conectar al servidor:[/red] {e}")
        return
    table = Table(title="Modelos disponibles")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Modelo")
    for i, mid in enumerate(ids, 1):
        marker = " ← actual" if mid == app_config.llm.model else ""
        table.add_row(str(i), f"{mid}{marker}")
    console.print(table)


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
def delete(repo_path: str):
    """Elimina todos los chunks de un repo de la base de vectores."""
    store = VectorStore()
    repo_id = resolve_repo(repo_path)
    store.delete_by_repo(repo_id)
    console.print(f"[green]✓[/green] Repo eliminado: [dim]{repo_id}[/dim]")


@main.command()
def info():
    """Muestra información sobre el índice, parsers y configuración LLM."""
    _import_parsers()
    store = VectorStore()

    console.print(f"[bold]Chunks indexados:[/bold] {store.count()}")
    repos = store.list_repos()
    if repos:
        table = Table(title="Repos indexados")
        table.add_column("Repo", style="green")
        table.add_column("Chunks", style="cyan")
        for repo_id, count in sorted(repos.items()):
            table.add_row(repo_id, str(count))
        console.print(table)
    langs = ", ".join(ParserRegistry.supported_languages())
    console.print(f"[bold]Lenguajes soportados:[/bold] {langs}")
    exts = ", ".join(ParserRegistry.supported_extensions())
    console.print(f"[bold]Extensiones:[/bold] {exts}")

    app_config = load_config()
    if app_config.llm.enabled:
        llm_status = f"[green]{app_config.llm.provider} ({app_config.llm.model})[/green]"
    else:
        llm_status = "[yellow]mock (test)[/yellow]"
    console.print(f"[bold]LLM:[/bold] {llm_status}")


if __name__ == "__main__":
    main()
