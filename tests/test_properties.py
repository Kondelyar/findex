import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import build_index
from findex.query import And, Node, Not, Or, Phrase, Term, parse, render
from findex.search import merge_and, merge_not, merge_or
from findex.store import load_json, load_pickle, save_json, save_pickle
from findex.tokenize import tokenize

sorted_ids = st.lists(st.integers(0, 200), unique=True).map(sorted)
documents = st.lists(st.text(max_size=200), max_size=8)
words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6)


@given(st.text())
def test_tokens_are_never_empty(text: str) -> None:
    assert all(tokenize(text))


@given(st.text())
def test_tokenize_is_idempotent(text: str) -> None:
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


@given(sorted_ids, sorted_ids)
def test_merge_agrees_with_sets(a: list[int], b: list[int]) -> None:
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


def write_corpus(directory: Path, texts: list[str]) -> Path:
    path = directory / "corpus.jsonl"
    lines = [json.dumps({"id": f"d{i}", "abstract": t}) for i, t in enumerate(texts)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@given(documents)
def test_postings_are_sorted(texts: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index = build_index(write_corpus(Path(tmp), texts))
    for postings in index.postings.values():
        ids = [p.doc_id for p in postings]
        assert ids == sorted(ids)


@settings(max_examples=50, deadline=None)
@given(documents, st.booleans())
def test_save_load_round_trip(texts: list[str], positions: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        index = build_index(write_corpus(directory, texts), positions=positions)
        for save, load, name in [
            (save_pickle, load_pickle, "index.pkl"),
            (save_json, load_json, "index.json"),
        ]:
            save(index, directory / name)
            loaded = load(directory / name)
            assert loaded.postings == index.postings
            assert loaded.doc_lengths == index.doc_lengths
            assert loaded.doc_meta == index.doc_meta


atoms = st.one_of(
    st.builds(Term, words),
    st.builds(Phrase, st.lists(words, min_size=1, max_size=3).map(tuple)),
)
trees = st.recursive(
    atoms,
    lambda children: st.one_of(
        st.builds(And, children, children),
        st.builds(Or, children, children),
        st.builds(Not, children),
    ),
    max_leaves=8,
)


@given(trees)
def test_parse_inverts_render(tree: Node) -> None:
    assert parse(render(tree)) == tree


@pytest.mark.parametrize(
    ("tree", "text"),
    [
        (Term("a"), "a"),
        (Phrase(("event", "loop")), '"event loop"'),
        (And(Term("a"), Or(Term("b"), Term("c"))), "(a AND (b OR c))"),
        (Not(Term("a")), "NOT a"),
    ],
)
def test_render_examples(tree: Node, text: str) -> None:
    assert render(tree) == text
