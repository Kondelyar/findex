"""Lab 3 M2: three ranking sanity checks on a real index.

Usage:
    python scripts/ranking_sanity.py data/index.pkl
"""

import argparse
from pathlib import Path

from findex.models import Posting
from findex.scoring import BM25, TfIdf
from findex.store import open_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", type=Path)
    args = parser.parse_args()

    scorers = {"tfidf": TfIdf(), "bm25": BM25()}

    with open_index(args.index_path) as index:
        by_df = sorted(index, key=index.df)
        rare, common = by_df[0], by_df[-1]
        by_length = sorted(index.doc_lengths, key=index.doc_length)
        shortest, middle, longest = by_length[0], by_length[len(by_length) // 2], by_length[-1]

        print(f"1. rare vs common term, same document (doc {middle}, tf=1)")
        for name, scorer in scorers.items():
            r = scorer(rare, Posting(middle, 1), index)
            c = scorer(common, Posting(middle, 1), index)
            print(f"   {name:<6} {rare!r} (df={index.df(rare)}) = {r:.3f}")
            print(f"   {name:<6} {common!r} (df={index.df(common)}) = {c:.3f}")

        print(f"\n2. repeating {common!r} in doc {middle}")
        for tf in (1, 2, 5, 10, 20, 21):
            row = [
                f"{name}={scorer(common, Posting(middle, tf), index):.3f}"
                for name, scorer in scorers.items()
            ]
            print(f"   tf={tf:<3} " + "  ".join(row))

        print(f"\n3. one match of {common!r}: shortest vs longest document")
        for name, scorer in scorers.items():
            short = scorer(common, Posting(shortest, 1), index)
            lng = scorer(common, Posting(longest, 1), index)
            print(
                f"   {name:<6} doc {shortest} ({index.doc_length(shortest)} tokens) = {short:.3f}"
                f"  |  doc {longest} ({index.doc_length(longest)} tokens) = {lng:.3f}"
            )


if __name__ == "__main__":
    main()
