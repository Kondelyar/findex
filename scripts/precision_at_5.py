"""Lab 3 M4: precision@5 for TF-IDF vs BM25 on hand-labelled queries.

Usage:
    python scripts/precision_at_5.py data/index.pkl eval/queries.json --show   # look at top 5, then label
    python scripts/precision_at_5.py data/index.pkl eval/queries.json          # print the table
"""

import argparse
import json
from pathlib import Path

from findex.scoring import BM25, TfIdf
from findex.search import Searcher
from findex.store import open_index

K = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    scorers = {"TF-IDF": TfIdf(), "BM25": BM25()}

    with open_index(args.index_path) as index:
        searcher = Searcher(index)

        if args.show:
            for item in queries:
                print(f"\n{item['query']}")
                for result in searcher.search(item["query"], k=K):
                    print(f"  {index.doc_meta[result.doc_id].doc_id}  {result.title}")
            return

        totals = dict.fromkeys(scorers, 0.0)
        print("| query | TF-IDF P@5 | BM25 P@5 |")
        print("|---|---|---|")
        for item in queries:
            relevant = set(item["relevant"])
            row = []
            for name, scorer in scorers.items():
                results = searcher.search(item["query"], scorer=scorer, k=K)
                hits = sum(index.doc_meta[r.doc_id].doc_id in relevant for r in results)
                totals[name] += hits / K
                row.append(f"{hits / K:.1f}")
            print(f"| {item['query']} | {row[0]} | {row[1]} |")
        means = [f"{totals[name] / len(queries):.2f}" for name in scorers]
        print(f"| **mean** | {means[0]} | {means[1]} |")


if __name__ == "__main__":
    main()
