import heapq
from dataclasses import dataclass, field
from functools import lru_cache
from operator import itemgetter
from typing import Literal

from findex.models import Index
from findex.query import Node, parse
from findex.scoring import BM25, Scorer
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
    result: list[int] = []
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
    result: list[int] = []
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


def _evaluate_merge(index: Index, steps: list[tuple[str, str]]) -> list[int]:
    current = get_ids(index, steps[0][1])
    for op, term in steps[1:]:
        other = get_ids(index, term)
        if op == "AND":
            current = merge_and(current, other)
        elif op == "OR":
            current = merge_or(current, other)
        elif op == "NOT":
            current = merge_not(current, other)
    return current


def _evaluate_set(index: Index, steps: list[tuple[str, str]]) -> list[int]:
    current = set(get_ids(index, steps[0][1]))
    for op, term in steps[1:]:
        other = set(get_ids(index, term))
        if op == "AND":
            current &= other
        elif op == "OR":
            current |= other
        elif op == "NOT":
            current -= other
    return sorted(current)


def evaluate(index: Index, query: str, engine: Literal["merge", "set"] = "merge") -> list[int]:
    steps = parse_query(query)
    if not steps:
        return []
    if engine == "set":
        return _evaluate_set(index, steps)
    return _evaluate_merge(index, steps)


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
    def search(self, query: str, scorer: Scorer | None = None, k: int = 10) -> list[SearchResult]:
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
