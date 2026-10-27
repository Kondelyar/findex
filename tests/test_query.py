import json

import pytest

from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QueryError, Term, parse

_DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day",
    "a quick brown cat",
    "foxes and dogs",
]


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "corpus.jsonl"
    lines = [json.dumps({"id": f"d{i}", "abstract": t}) for i, t in enumerate(_DOCS)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return build_index(path, positions=True)


def test_implicit_and():
    assert parse("a b") == And(Term("a"), Term("b"))


def test_and_binds_tighter_than_or():
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_parentheses_override_precedence():
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))


def test_explicit_and_and_not():
    assert parse("a AND NOT b") == And(Term("a"), Not(Term("b")))


def test_phrase():
    assert parse('"Event Loop"') == Phrase(("event", "loop"))


def test_terms_are_normalised():
    assert parse("Python") == Term("python")


def test_operators_build_the_same_tree():
    built = Term("a") & (Term("b") | Term("c")) & ~Term("d")
    assert built == parse("a AND (b OR c) AND NOT d")


@pytest.mark.parametrize("bad", ["", "   ", "(a", "a)", "a OR", "AND a", "()"])
def test_bad_queries_raise(bad):
    with pytest.raises(QueryError):
        parse(bad)


def test_or(index):
    assert parse("fox OR cat").evaluate(index) == {0, 2}


def test_not(index):
    assert parse("quick NOT fox").evaluate(index) == {2}
    assert parse("NOT quick").evaluate(index) == {1, 3}


def test_parentheses(index):
    assert parse("(fox OR cat) brown").evaluate(index) == {0, 2}


def test_phrase_needs_adjacent_words_in_order(index):
    assert parse('"lazy dog"').evaluate(index) == {0, 1}
    assert parse('"dog lazy"').evaluate(index) == set()
    assert parse('"quick brown"').evaluate(index) == {0, 2}
    assert parse('"quick brown fox"').evaluate(index) == {0}


def test_phrase_without_positions_raises(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"id": "d", "abstract": "lazy dog"}), encoding="utf-8")
    with pytest.raises(QueryError):
        parse('"lazy dog"').evaluate(build_index(path))


def test_hyphenated_word_becomes_a_phrase():
    assert parse("well-known") == Phrase(("well", "known"))


def test_punctuation_only_words_are_dropped():
    assert parse("a ... b") == And(Term("a"), Term("b"))
