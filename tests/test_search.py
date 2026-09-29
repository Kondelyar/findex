import json

import pytest

from findex.index import build_index
from findex.search import evaluate

_DOCS = [
    {"id": "d1", "abstract": "cats sleep all day"},
    {"id": "d2", "abstract": "dogs bark at cats"},
    {"id": "d3", "abstract": "dogs sleep too"},
]


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for doc in _DOCS:
            f.write(json.dumps(doc) + "\n")
    return build_index(path)


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_implicit_and(index, engine):
    assert evaluate(index, "cats sleep", engine=engine) == [0]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_or(index, engine):
    assert evaluate(index, "cats OR dogs", engine=engine) == [0, 1, 2]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_not(index, engine):
    assert evaluate(index, "dogs NOT cats", engine=engine) == [2]


def test_unknown_term_gives_no_results(index):
    assert evaluate(index, "elephants") == []


def test_empty_query_gives_no_results(index):
    assert evaluate(index, "") == []
