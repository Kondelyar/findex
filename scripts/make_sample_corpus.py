"""Generate a small synthetic .jsonl corpus for local testing.

This is NOT the real corpus for the lab - it only exists so the pipeline
can be tested before the real arXiv dump is downloaded (see README.md,
section "Getting the real corpus"). Usage:

    python scripts/make_sample_corpus.py data/sample.jsonl --docs 2000
"""

import argparse
import json
import random
from pathlib import Path

WORDS = [
    "graph", "neural", "network", "attention", "transformer", "optimization",
    "gradient", "convex", "sparse", "embedding", "kernel", "manifold",
    "Bayesian", "inference", "dataset", "algorithm", "convergence",
    "regularization", "sampling", "entropy", "distribution",
]


def make_abstract(rng: random.Random) -> str:
    length = rng.randint(30, 80)
    return " ".join(rng.choice(WORDS) for _ in range(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.docs):
            record = {"id": f"arxiv.{i:06d}", "abstract": make_abstract(rng)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {args.docs} documents to {args.out}")


if __name__ == "__main__":
    main()
