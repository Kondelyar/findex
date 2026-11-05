"""Make a bigger corpus by repeating an existing one.

The vocabulary stays the same (it is the same text), so the postings lists get
longer but the term table does not grow. Good enough to measure indexing speed.

Usage:
    python scripts/scale_corpus.py data/arxiv.jsonl data/big.jsonl --copies 100
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--copies", type=int, default=100)
    args = parser.parse_args()

    lines = args.source.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]

    with args.out.open("w", encoding="utf-8") as f:
        for copy in range(args.copies):
            for record in records:
                record = {**record, "id": f"{record['id']}-{copy}"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    size_mb = args.out.stat().st_size / 2**20
    print(f"wrote {len(records) * args.copies} documents, {size_mb:.1f} MB, to {args.out}")


if __name__ == "__main__":
    main()
