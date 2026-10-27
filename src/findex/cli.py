import json
import logging
import sys
import time
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from findex.corpus import iter_documents
from findex.index import build_index_from
from findex.query import QueryError
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import Searcher
from findex.snippets import fetch_texts, make_snippet
from findex.store import open_index, save_index

log = logging.getLogger(__name__)

app = typer.Typer(help="findex: a small full-text search engine.", no_args_is_help=True)
console = Console()


class ScorerName(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


def make_scorer(name: ScorerName) -> Scorer:
    return BM25() if name is ScorerName.bm25 else TfIdf()


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def require_file(path: Path, what: str) -> Path:
    if not path.exists():
        fail(f"{what} not found: {path}")
    return path


def highlight(snippet: str) -> Text:
    text = Text()
    for i, part in enumerate(snippet.split("**")):
        text.append(part, style="bold yellow" if i % 2 else None)
    return text


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v for info, -vv for debug")
    ] = 0,
) -> None:
    configure_logging(verbose)


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="a .jsonl file or a directory of them")],
    out: Annotated[Path, typer.Option("--out", "-o", help="index file (.pkl or .json)")],
    positions: Annotated[bool, typer.Option(help="store token positions (phrase queries)")] = False,
    limit: Annotated[int | None, typer.Option(help="index only the first N documents")] = None,
) -> None:
    """Build an inverted index from a corpus and save it."""
    require_file(corpus, "corpus")
    start = time.perf_counter()
    with Progress(console=Console(stderr=True), transient=True) as progress:
        docs = progress.track(islice(iter_documents(corpus), limit), description="Indexing")
        built = build_index_from(docs, positions=positions)
    save_index(built, out)
    elapsed = time.perf_counter() - start
    console.print(
        f"Indexed {built.num_docs} documents, {len(built)} terms in {elapsed:.2f}s -> {out}"
    )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="index file made by `findex index`")],
    query: Annotated[str, typer.Argument(help='e.g. "white rabbit" OR (cat NOT dog)')],
    k: Annotated[int, typer.Option("--k", help="number of results")] = 10,
    scorer: Annotated[ScorerName, typer.Option(help="ranking function")] = ScorerName.bm25,
    json_output: Annotated[bool, typer.Option("--json", help="JSON lines on stdout")] = False,
) -> None:
    """Search an index and print ranked results."""
    require_file(index_path, "index file")
    with open_index(index_path) as idx:
        searcher = Searcher(idx)
        try:
            results = searcher.search(query, scorer=make_scorer(scorer), k=k)
        except QueryError as e:
            fail(f"bad query: {e}")

        node, matched = searcher.match(query)
        terms = set(node.positive_terms())
        texts = fetch_texts(idx, [r.doc_id for r in results])

        rows: list[tuple[str, float, str]] = []
        for result in results:
            text = texts.get(result.doc_id)
            snippet = make_snippet(text, terms) if text else result.title
            rows.append((idx.doc_meta[result.doc_id].doc_id, result.score, snippet))

    if json_output:
        for rank, (doc_id, score, snippet) in enumerate(rows, start=1):
            record = {"rank": rank, "id": doc_id, "score": round(score, 4), "snippet": snippet}
            typer.echo(json.dumps(record, ensure_ascii=False))
        return

    if not rows:
        typer.echo("no matches", err=True)
        return

    table = Table(caption=f"{len(matched)} matches, showing {len(rows)}")
    table.add_column("#", justify="right")
    table.add_column("id")
    table.add_column("score", justify="right")
    table.add_column("snippet", overflow="fold")
    for rank, (doc_id, score, snippet) in enumerate(rows, start=1):
        table.add_row(str(rank), doc_id, f"{score:.3f}", highlight(snippet))
    console.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(help="index file made by `findex index`")],
    top: Annotated[int, typer.Option(help="how many top terms to show")] = 50,
) -> None:
    """Show corpus statistics taken from an index."""
    require_file(index_path, "index file")
    with open_index(index_path) as idx:
        summary = Table("metric", "value", title="Index")
        summary.add_row("documents", str(idx.num_docs))
        summary.add_row("tokens", str(idx.total_tokens))
        summary.add_row("vocabulary", str(len(idx)))
        summary.add_row("avg document length", f"{idx.avg_doc_length:.1f}")
        summary.add_row("positions stored", "yes" if idx.has_positions else "no")

        terms = Table("#", "term", "count", title=f"Top {top} terms")
        for rank, (term, count) in enumerate(idx.top_terms(top), start=1):
            terms.add_row(str(rank), term, str(count))

    console.print(summary)
    console.print(terms)
