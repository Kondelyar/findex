import json

import pytest

from findex.index import build_index
from findex.store import open_index, save_pickle


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "corpus.jsonl"
    docs = [
        {"id": "d1", "abstract": "red apple"},
        {"id": "d2", "abstract": "green apple pie"},
        {"id": "d3", "abstract": "blue sky"},
    ]
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    return build_index(path)


def test_container_protocol(index):
    assert len(index) == 6
    assert "apple" in index
    assert "banana" not in index
    assert sorted(index) == ["apple", "blue", "green", "pie", "red", "sky"]
    assert [p.doc_id for p in index["apple"]] == [0, 1]


def test_missing_term_raises_key_error(index):
    with pytest.raises(KeyError):
        index["banana"]


def test_repr(index):
    assert repr(index) == "Index(terms=6, docs=3)"


def test_stats(index):
    assert index.num_docs == 3
    assert index.avg_doc_length == pytest.approx(7 / 3)
    assert index.doc_length(1) == 3
    assert index.df("apple") == 2
    assert index.df("banana") == 0


def test_open_index_closes_after_block(index, tmp_path):
    out = tmp_path / "index.pkl"
    save_pickle(index, out)
    with open_index(out) as ix:
        assert len(ix) == 6
    assert len(ix) == 0


def test_open_index_closes_on_error(index, tmp_path):
    out = tmp_path / "index.pkl"
    save_pickle(index, out)
    with pytest.raises(RuntimeError), open_index(out) as ix:
        raise RuntimeError("boom")
    assert len(ix) == 0
