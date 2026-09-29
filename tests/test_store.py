import json
from pathlib import Path

from findex.index import build_index
from findex.store import load_json, load_pickle, save_json, save_pickle

_DOCS = [
    {"id": "d1", "abstract": "hello world"},
    {"id": "d2", "abstract": "hello there"},
]


def make_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for doc in _DOCS:
            f.write(json.dumps(doc) + "\n")
    return path


def test_pickle_roundtrip(tmp_path):
    index = build_index(make_corpus(tmp_path))
    out = tmp_path / "index.pkl"
    save_pickle(index, out)
    loaded = load_pickle(out)
    assert loaded.postings.keys() == index.postings.keys()
    assert loaded.doc_meta == index.doc_meta
    assert loaded.doc_lengths == index.doc_lengths


def test_json_roundtrip(tmp_path):
    index = build_index(make_corpus(tmp_path))
    out = tmp_path / "index.json"
    save_json(index, out)
    loaded = load_json(out)
    assert loaded.postings.keys() == index.postings.keys()
    assert loaded.doc_meta == index.doc_meta
    assert loaded.doc_lengths == index.doc_lengths
