import argparse
import heapq
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from functools import lru_cache
from operator import itemgetter
from pathlib import Path

from findex.models import Index
from findex.query import Node, QueryError, parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import fetch_texts, make_snippet
from findex.store import open_index
from findex.timing import timed
from findex.tokenize import tokenize

_KEYWORDS = {"AND", "OR", "NOT"}


# ---- Lab 2: boolean engines (kept for the merge-vs-set benchmark) ----


def parse_query(query: str) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    op = "START"
    for raw in query.split():
        if raw in _KEYWORDS:
            op = raw
            continue
        term = next(tokenize(raw), None)
        if term:
            steps.append((op, term))
        op = "AND"
    return steps


def get_ids(index: Index, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, [])]


def merge_and(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def merge_not(a: list[int], b: list[int]) -> list[int]:
    b_ids = set(b)
    return [doc_id for doc_id in a if doc_id not in b_ids]


def evaluate(index: Index, query: str, engine: str = "merge") -> list[int]:
    steps = parse_query(query)
    if not steps:
        return []

    _, first_term = steps[0]
    current = set(get_ids(index, first_term)) if engine == "set" else get_ids(index, first_term)

    for op, term in steps[1:]:
        other = get_ids(index, term)
        if engine == "set":
            other = set(other)
            if op == "AND":
                current &= other
            elif op == "OR":
                current |= other
            elif op == "NOT":
                current -= other
        else:
            if op == "AND":
                current = merge_and(current, other)
            elif op == "OR":
                current = merge_or(current, other)
            elif op == "NOT":
                current = merge_not(current, other)

    return sorted(current) if engine == "set" else current


# ---- Lab 3: ranked search ----


@dataclass(order=True)
class SearchResult:
    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)


class Searcher:
    def __init__(self, index: Index):
        self.index = index
        self.match = lru_cache(maxsize=256)(self._match)

    def _match(self, query: str) -> tuple[Node, tuple[int, ...]]:
        node = parse(query)
        return node, tuple(sorted(node.evaluate(self.index)))

    @timed
    def search(
        self, query: str, scorer: Scorer | None = None, k: int = 10
    ) -> list[SearchResult]:
        scorer = scorer or BM25()
        node, doc_ids = self.match(query)
        scores = dict.fromkeys(doc_ids, 0.0)
        for term in set(node.positive_terms()):
            for posting in self.index.get(term, ()):
                if posting.doc_id in scores:
                    scores[posting.doc_id] += scorer(term, posting, self.index)

        top = heapq.nlargest(k, scores.items(), key=itemgetter(1))
        return [
            SearchResult(doc_id, score, self.index.doc_meta[doc_id].preview)
            for doc_id, score in top
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="search the saved inverted index")
    parser.add_argument("index_path", type=Path)
    parser.add_argument("query")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="run the query N times")
    parser.add_argument("-v", "--verbose", action="store_true", help="show timing log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(message)s"
    )
    scorer = BM25() if args.scorer == "bm25" else TfIdf()

    tracemalloc.start()
    start = time.perf_counter()
    with open_index(args.index_path) as index:
        searcher = Searcher(index)
        try:
            for _ in range(args.repeat):
                results = searcher.search(args.query, scorer=scorer, k=args.limit)
        except QueryError as e:
            raise SystemExit(f"bad query: {e}") from e
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        node, doc_ids = searcher.match(args.query)
        terms = set(node.positive_terms())
        texts = fetch_texts(index, [r.doc_id for r in results])

        print(f"query:    {args.query}")
        print(f"scorer:   {scorer}")
        print(f"matches:  {len(doc_ids)}")
        print(f"elapsed:  {elapsed:.3f}s")
        print(f"peak mem: {peak / 1024 / 1024:.2f} MB")
        if args.verbose:
            print(f"cache:    {searcher.match.cache_info()}")
        print()
        for result in results:
            meta = index.doc_meta[result.doc_id]
            text = texts.get(result.doc_id)
            snippet = make_snippet(text, terms) if text else result.title
            print(f"[{meta.doc_id}] {result.score:.3f}")
            print(f"    {snippet}")


if __name__ == "__main__":
    main()
