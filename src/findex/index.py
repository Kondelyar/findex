import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import DocMeta, Index, Posting
from findex.store import save_json, save_pickle
from findex.tokenize import tokenize


def build_index(root: Path) -> Index:
    """Stream the corpus once and build the inverted index.

    Documents get a sequential internal doc_id (0, 1, 2, ...) in stream
    order, so postings end up sorted by doc_id for free - no separate
    sort step needed.
    """
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents(root)):
        counts = Counter(tokenize(doc.text))
        doc_lengths[doc_id] = sum(counts.values())
        doc_meta[doc_id] = DocMeta(
            doc_id=doc.doc_id,
            path=doc.path,
            preview=doc.text[:80].replace("\n", " "),
        )
        for term, tf in counts.items():
            postings[term].append(Posting(doc_id, tf))

    return Index(postings=dict(postings), doc_lengths=doc_lengths, doc_meta=doc_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="build and save the inverted index")
    parser.add_argument("root", type=Path, help="corpus directory or .jsonl file")
    parser.add_argument("--out", type=Path, required=True, help="output path, e.g. data/index.pkl")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()
    index = build_index(args.root)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    save = save_json if args.out.suffix == ".json" else save_pickle
    save(index, args.out)

    print(f"documents: {len(index.doc_meta)}")
    print(f"terms:     {len(index.postings)}")
    print(f"elapsed:   {elapsed:.3f}s")
    print(f"peak mem:  {peak / 1024 / 1024:.2f} MB")
    print(f"saved to:  {args.out}")


if __name__ == "__main__":
    main()
