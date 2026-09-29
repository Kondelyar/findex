"""M2 measurement: merge engine vs set engine on common and rare terms.

Usage:
    python scripts/compare_engines.py data/arxiv.jsonl
"""

import argparse
import time
from pathlib import Path

from findex.index import build_index
from findex.search import evaluate


def pick_terms(index, n: int) -> tuple[list[str], list[str]]:
    by_length = sorted(index.postings.items(), key=lambda kv: len(kv[1]))
    rare = [term for term, _ in by_length[:n]]
    common = [term for term, _ in by_length[-n:]]
    return common, rare


def timed_query(index, query: str, engine: str, repeats: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        evaluate(index, query, engine=engine)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--terms", type=int, default=2)
    args = parser.parse_args()

    index = build_index(args.root)
    common, rare = pick_terms(index, args.terms)

    print(f"common terms: {common}")
    print(f"rare terms:   {rare}")
    print()
    print("| query | engine | avg time |")
    print("|---|---|---|")
    for label, pair in (("common AND", common), ("rare AND", rare)):
        if len(pair) < 2:
            continue
        query = f"{pair[0]} AND {pair[1]}"
        for engine in ("merge", "set"):
            avg = timed_query(index, query, engine)
            print(f"| {label} ({query}) | {engine} | {avg * 1000:.4f} ms |")


if __name__ == "__main__":
    main()
