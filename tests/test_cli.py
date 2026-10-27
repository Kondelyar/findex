import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def index_file(corpus_path: Path, tmp_path: Path) -> Path:
    out = tmp_path / "index.pkl"
    result = runner.invoke(app, ["index", str(corpus_path), "--out", str(out), "--positions"])
    assert result.exit_code == 0
    return out


def test_help_lists_the_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_index_reports_counts(corpus_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.pkl"
    result = runner.invoke(app, ["index", str(corpus_path), "--out", str(out)])
    assert result.exit_code == 0
    assert "Indexed 5 documents" in result.stdout
    assert out.exists()


def test_index_limit(corpus_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.json"
    result = runner.invoke(app, ["index", str(corpus_path), "--out", str(out), "--limit", "2"])
    assert result.exit_code == 0
    assert "Indexed 2 documents" in result.stdout
    assert out.exists()


def test_index_missing_corpus(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "nope.jsonl"), "--out", "x.pkl"])
    assert result.exit_code == 1
    assert result.stderr.startswith("error: corpus not found")


def test_search_prints_a_table(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "rabbit"])
    assert result.exit_code == 0
    assert "doc_1" in result.stdout
    assert "2 matches" in result.stdout


def test_search_json_lines(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "rabbit", "--json"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert {row["id"] for row in rows} == {"doc_1", "doc_5"}
    assert [row["rank"] for row in rows] == [1, 2]
    assert all({"score", "snippet"} <= row.keys() for row in rows)


def test_search_k_limits_results(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "the", "--k", "1", "--json"])
    assert len(result.stdout.splitlines()) == 1


def test_search_phrase(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), '"white rabbit"', "--json"])
    assert {json.loads(line)["id"] for line in result.stdout.splitlines()} == {"doc_1", "doc_5"}


def test_search_tfidf_scorer(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "alice", "--scorer", "tfidf"])
    assert result.exit_code == 0


def test_search_no_matches_is_not_an_error(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "elephant"])
    assert result.exit_code == 0
    assert "no matches" in result.stderr


def test_search_missing_index(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "nope.pkl"), "x"])
    assert result.exit_code == 1
    assert result.stderr.strip() == f"error: index file not found: {tmp_path / 'nope.pkl'}"


@pytest.mark.parametrize("bad", ["(rabbit", "rabbit OR", ")"])
def test_search_bad_query_is_one_clear_line(index_file: Path, bad: str) -> None:
    result = runner.invoke(app, ["search", str(index_file), bad])
    assert result.exit_code == 1
    assert result.stderr.startswith("error: bad query")
    assert len(result.stderr.strip().splitlines()) == 1


def test_phrase_needs_positions(corpus_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "plain.pkl"
    runner.invoke(app, ["index", str(corpus_path), "--out", str(out)])
    result = runner.invoke(app, ["search", str(out), '"white rabbit"'])
    assert result.exit_code == 1
    assert "--positions" in result.stderr


def test_stats(index_file: Path) -> None:
    result = runner.invoke(app, ["stats", str(index_file), "--top", "3"])
    assert result.exit_code == 0
    for text in ("documents", "vocabulary", "Top 3 terms", "the"):
        assert text in result.stdout


def test_verbose_logs_go_to_stderr_and_json_stays_clean(index_file: Path) -> None:
    result = runner.invoke(app, ["-vv", "search", str(index_file), "rabbit", "--json"])
    assert "DEBUG" in result.stderr
    assert "INFO" in result.stderr
    assert all(json.loads(line) for line in result.stdout.splitlines())


def test_quiet_by_default(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "rabbit", "--json"])
    assert result.stderr == ""


@pytest.mark.slow
def test_large_corpus_end_to_end(tmp_path: Path) -> None:
    rng = random.Random(1)
    vocab = [f"word{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}" for i in range(2000)]
    corpus = tmp_path / "big.jsonl"
    with corpus.open("w", encoding="utf-8") as f:
        for i in range(4000):
            text = " ".join(rng.choice(vocab) for _ in range(60))
            f.write(json.dumps({"id": f"d{i}", "abstract": text}) + "\n")

    out = tmp_path / "big.pkl"
    assert runner.invoke(app, ["index", str(corpus), "--out", str(out)]).exit_code == 0
    result = runner.invoke(app, ["search", str(out), f"{vocab[0]} OR {vocab[1]}", "--json"])
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 10
