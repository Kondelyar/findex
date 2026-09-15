import argparse
import itertools
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run(root: Path, limit: int | None = None) -> None:
    docs = iter_documents(root)
    if limit is not None:
        docs = itertools.islice(docs, limit)

    tracemalloc.start()
    start = time.perf_counter()

    doc_count = 0
    token_count = 0
    counts: Counter[str] = Counter()

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            token_count += 1
            counts[token] += 1

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"documents:   {doc_count}")
    print(f"tokens:      {token_count}")
    print(f"vocabulary:  {len(counts)}")
    print(f"elapsed:     {elapsed:.3f}s")
    print(f"peak memory: {peak / 1024 / 1024:.2f} MB")
    print()
    print("top 50 terms:")
    for term, n in counts.most_common(50):
        print(f"  {term:<20} {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="findex corpus statistics")
    parser.add_argument("root", type=Path, help="corpus directory or .jsonl file")
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N documents"
    )
    args = parser.parse_args()
    run(args.root, args.limit)


if __name__ == "__main__":
    main()
