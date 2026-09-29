"""M4 measurement: three ways to store postings, same corpus.

Usage:
    python scripts/posting_memory_study.py data/arxiv.jsonl
"""

import argparse
import pickle
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class FatPosting:
    doc_id: int
    tf: int


@dataclass(slots=True)
class ThinPosting:
    doc_id: int
    tf: int


def term_counts_per_doc(root: Path):
    for doc_id, doc in enumerate(iter_documents(root)):
        yield doc_id, Counter(tokenize(doc.text))


def build_fat(root: Path) -> dict[str, list[FatPosting]]:
    postings: dict[str, list[FatPosting]] = defaultdict(list)
    for doc_id, counts in term_counts_per_doc(root):
        for term, tf in counts.items():
            postings[term].append(FatPosting(doc_id, tf))
    return dict(postings)


def build_thin(root: Path) -> dict[str, list[ThinPosting]]:
    postings: dict[str, list[ThinPosting]] = defaultdict(list)
    for doc_id, counts in term_counts_per_doc(root):
        for term, tf in counts.items():
            postings[term].append(ThinPosting(doc_id, tf))
    return dict(postings)


def build_arrays(root: Path) -> dict[str, tuple[array, array]]:
    doc_ids: dict[str, array] = defaultdict(lambda: array("I"))
    tfs: dict[str, array] = defaultdict(lambda: array("I"))
    for doc_id, counts in term_counts_per_doc(root):
        for term, tf in counts.items():
            doc_ids[term].append(doc_id)
            tfs[term].append(tf)
    return {term: (doc_ids[term], tfs[term]) for term in doc_ids}


def measure(build_fn, root: Path, tmp_path: Path) -> tuple[float, int, float]:
    tracemalloc.start()
    data = build_fn(root)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with tmp_path.open("wb") as f:
        pickle.dump(data, f)
    size = tmp_path.stat().st_size

    start = time.perf_counter()
    with tmp_path.open("rb") as f:
        pickle.load(f)
    load_time = time.perf_counter() - start

    return peak, size, load_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--tmp", type=Path, default=Path("/tmp/posting_variant.pkl"))
    args = parser.parse_args()

    variants = [
        ("list[Posting], plain dataclass", build_fat),
        ("list[Posting], slots=True", build_thin),
        ("array('I') pairs", build_arrays),
    ]

    print("| representation | peak memory (build) | index file size | load time |")
    print("|---|---|---|---|")
    for name, fn in variants:
        peak, size, load_time = measure(fn, args.root, args.tmp)
        print(
            f"| {name} | {peak / 1024 / 1024:.2f} MB | {size / 1024:.1f} KB "
            f"| {load_time * 1000:.2f} ms |"
        )


if __name__ == "__main__":
    main()
