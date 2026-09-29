import json
from pathlib import Path

from findex.index import build_index

_DOCS = [
    {"id": "d1", "abstract": "the cat sat on the mat"},
    {"id": "d2", "abstract": "the dog sat on the log"},
    {"id": "d3", "abstract": "cats and dogs are friends"},
]


def make_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for doc in _DOCS:
            f.write(json.dumps(doc) + "\n")
    return path


def test_postings_sorted_by_doc_id(tmp_path):
    index = build_index(make_corpus(tmp_path))
    for postings in index.postings.values():
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(doc_ids)


def test_term_frequency_is_counted(tmp_path):
    index = build_index(make_corpus(tmp_path))
    tf_by_doc = {p.doc_id: p.tf for p in index.postings["the"]}
    assert tf_by_doc == {0: 2, 1: 2}


def test_doc_lengths(tmp_path):
    index = build_index(make_corpus(tmp_path))
    assert index.doc_lengths[0] == 6


def test_doc_meta_keeps_original_id(tmp_path):
    index = build_index(make_corpus(tmp_path))
    assert index.doc_meta[0].doc_id == "d1"
    assert index.doc_meta[2].doc_id == "d3"


def test_postings_are_hashable():
    from findex.models import Posting

    p = Posting(doc_id=1, tf=2)
    assert {p, Posting(doc_id=1, tf=2)} == {p}
