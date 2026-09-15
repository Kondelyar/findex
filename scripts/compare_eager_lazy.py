"""M4 measurement: eager (lists) vs lazy (generators) on the same corpus slice.

Usage:
    python scripts/compare_eager_lazy.py data/sample.jsonl --limit 2000

Paste the printed table into README.md.
"""

import argparse
import itertools
import time
import tracemalloc
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def eager(root: Path, limit: int | None) -> tuple[int, int]:
    docs = list(iter_documents(root))
    if limit is not None:
        docs = docs[:limit]
    token_lists = [list(tokenize(doc.text)) for doc in docs]
    return len(docs), sum(len(t) for t in token_lists)


def lazy(root: Path, limit: int | None) -> tuple[int, int]:
    docs = iter_documents(root)
    if limit is not None:
        docs = itertools.islice(docs, limit)
    doc_count = 0
    token_count = 0
    for doc in docs:
        doc_count += 1
        for _ in tokenize(doc.text):
            token_count += 1
    return doc_count, token_count


def measure(fn, root: Path, limit: int | None):
    tracemalloc.start()
    start = time.perf_counter()
    doc_count, token_count = fn(root, limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return doc_count, token_count, elapsed, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"| {'version':<17} | {'documents':<9} | {'peak memory':<11} | {'elapsed':<8} |")
    print(f"|:{'-'*17}|:{'-'*9}:|:{'-'*11}:|:{'-'*8}:|")

    for name, fn in (("eager (lists)", eager), ("lazy (generators)", lazy)):
        docs, _tokens, elapsed, peak = measure(fn, args.root, args.limit)
        mem_mb = f"{peak / 1024 / 1024:.2f} MB"
        time_sec = f"{elapsed:.3f}s"
        
        # Виводимо рядки з жорстким вирівнюванням по ширині
        print(f"| {name:<17} | {str(docs):<9} | {mem_mb:<11} | {time_sec:<8} |")

if __name__ == "__main__":
    main()
