import json

import pytest

from findex.index import build_index
from findex.search import Searcher
from findex.snippets import make_snippet

_DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day",
    "a quick brown cat",
]


@pytest.fixture
def searcher(tmp_path):
    path = tmp_path / "corpus.jsonl"
    lines = [json.dumps({"id": f"d{i}", "abstract": t}) for i, t in enumerate(_DOCS)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return Searcher(build_index(path, positions=True))


def test_shorter_document_ranks_first(searcher):
    results = searcher.search("dog")
    assert [r.doc_id for r in results] == [1, 0]


def test_results_sort_by_score(searcher):
    results = searcher.search("quick OR dog")
    assert [r.score for r in sorted(results, reverse=True)] == [r.score for r in results]


def test_limit(searcher):
    assert len(searcher.search("quick OR dog", k=1)) == 1


def test_no_match(searcher):
    assert searcher.search("elephant") == []


def test_repeated_query_hits_cache(searcher):
    before = searcher.match.cache_info().hits
    searcher.search("lazy dog")
    searcher.search("lazy dog")
    assert searcher.match.cache_info().hits - before == 1


def test_snippet_highlights_matches():
    text = "word " * 40 + "the lazy dog sleeps" + " word" * 40
    snippet = make_snippet(text, {"lazy", "dog"})
    assert "**lazy** **dog**" in snippet
    assert snippet.startswith("...") and snippet.endswith("...")


def test_snippet_without_hits_returns_start_of_text():
    assert make_snippet("hello world", {"missing"}) == "hello world"
