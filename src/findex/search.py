import argparse
import time
import tracemalloc
from pathlib import Path

from findex.models import Index
from findex.store import load_json, load_pickle
from findex.tokenize import tokenize

_KEYWORDS = {"AND", "OR", "NOT"}


def parse_query(query: str) -> list[tuple[str, str]]:
    """Turn "cats OR dogs NOT sleepy" into [(START, cats), (OR, dogs), (NOT, sleepy)].

    A term with no keyword before it is treated as AND. Left-to-right,
    no operator precedence or parentheses - that's Lab 3's job.
    """
    steps: list[tuple[str, str]] = []
    op = "START"
    for raw in query.split():
        if raw.upper() in _KEYWORDS:
            op = raw.upper()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="search the saved inverted index")
    parser.add_argument("index_path", type=Path)
    parser.add_argument("query")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    load = load_json if args.index_path.suffix == ".json" else load_pickle

    tracemalloc.start()
    start = time.perf_counter()
    index = load(args.index_path)
    doc_ids = evaluate(index, args.query, engine=args.engine)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"query:   {args.query}")
    print(f"engine:  {args.engine}")
    print(f"matches: {len(doc_ids)}")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"peak mem:{peak / 1024 / 1024:.2f} MB")
    print()
    for doc_id in doc_ids[: args.limit]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id}] {meta.doc_id}  {meta.preview}")


if __name__ == "__main__":
    main()
