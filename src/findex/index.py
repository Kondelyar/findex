import argparse
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import DocMeta, Index, Posting
from findex.store import save_json, save_pickle
from findex.timing import timed
from findex.tokenize import tokenize


@timed
def build_index(root: Path, positions: bool = False) -> Index:
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents(root)):
        offsets: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokenize(doc.text)):
            offsets[token].append(pos)

        doc_lengths[doc_id] = sum(len(where) for where in offsets.values())
        doc_meta[doc_id] = DocMeta(
            doc_id=doc.doc_id,
            path=doc.path,
            preview=doc.text[:80].replace("\n", " "),
        )
        for term, where in offsets.items():
            postings[term].append(Posting(doc_id, len(where), tuple(where) if positions else ()))

    return Index(postings=dict(postings), doc_lengths=doc_lengths, doc_meta=doc_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="build and save the inverted index")
    parser.add_argument("root", type=Path, help="corpus directory or .jsonl file")
    parser.add_argument("--out", type=Path, required=True, help="output path, e.g. data/index.pkl")
    parser.add_argument("--positions", action="store_true", help="store token positions")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()
    index = build_index(args.root, positions=args.positions)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    save = save_json if args.out.suffix == ".json" else save_pickle
    save(index, args.out)

    print(f"documents: {index.num_docs}")
    print(f"terms:     {len(index)}")
    print(f"positions: {index.has_positions}")
    print(f"elapsed:   {elapsed:.3f}s")
    print(f"peak mem:  {peak / 1024 / 1024:.2f} MB")
    print(f"saved to:  {args.out}")


if __name__ == "__main__":
    main()
