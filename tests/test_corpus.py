import inspect
import json
import logging
from pathlib import Path

import pytest

from findex import stats
from findex.corpus import iter_documents
from findex.models import DocMeta, Index
from findex.snippets import fetch_texts


def test_iter_documents_is_a_generator(corpus_path: Path) -> None:
    assert inspect.isgeneratorfunction(iter_documents)
    first = next(iter_documents(corpus_path))
    assert first.doc_id == "doc_1"
    assert first.path == str(corpus_path)


def test_bad_lines_are_skipped_and_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"id": "a", "abstract": "good"}\nnot json\n\n{"id": "b"}\n{"id": "c", "text": "also good"}\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(path))
    assert [d.doc_id for d in docs] == ["a", "c"]
    assert "bad json" in caplog.text
    assert "no text field" in caplog.text


def test_directory_is_read_in_sorted_order(tmp_path: Path) -> None:
    for name, doc_id in [("b.jsonl", "from_b"), ("a.jsonl", "from_a")]:
        (tmp_path / name).write_text(json.dumps({"id": doc_id, "text": "x"}), encoding="utf-8")
    assert [d.doc_id for d in iter_documents(tmp_path)] == ["from_a", "from_b"]


def test_missing_id_falls_back_to_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "noid.jsonl"
    path.write_text(json.dumps({"abstract": "hello"}), encoding="utf-8")
    assert next(iter_documents(path)).doc_id == "noid.jsonl:1"


def test_stats_run_prints_counts(corpus_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stats.run(corpus_path, limit=2)
    out = capsys.readouterr().out
    assert "documents:   2" in out
    assert "vocabulary:" in out


def test_missing_corpus_file_means_no_snippets(
    index: Index, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    index.doc_meta[0] = DocMeta("doc_1", str(tmp_path / "gone.jsonl"), "preview")
    with caplog.at_level(logging.WARNING):
        assert fetch_texts(index, [0]) == {}
    assert "not found" in caplog.text
