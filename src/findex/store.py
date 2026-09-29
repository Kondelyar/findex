import json
import pickle
from dataclasses import asdict
from pathlib import Path

from findex.models import DocMeta, Index, Posting


def save_pickle(index: Index, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_pickle(path: Path) -> Index:
    # pickle.load runs arbitrary code embedded in the file, so this is only
    # safe to call on a file findex itself produced. Never load a pickle
    # you downloaded from somewhere else.
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(index: Index, path: Path) -> None:
    data = {
        "postings": {
            term: [asdict(p) for p in plist] for term, plist in index.postings.items()
        },
        "doc_lengths": index.doc_lengths,
        "doc_meta": {str(doc_id): asdict(meta) for doc_id, meta in index.doc_meta.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(path: Path) -> Index:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    postings = {
        term: [Posting(**p) for p in plist] for term, plist in data["postings"].items()
    }
    doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
    doc_meta = {int(k): DocMeta(**v) for k, v in data["doc_meta"].items()}
    return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)
