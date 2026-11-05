import gc
import json
import os
import pickle
import random
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex import parallel
from findex.corpus import Chunk, iter_chunk, iter_documents, plan_chunks
from findex.index import build_index
from findex.models import DocMeta, Index, Posting
from findex.parallel import (
    ExecutorKind,
    auto_executor,
    build_index_parallel,
    build_partial,
    gc_paused,
    merge,
    run_chunks,
)
from findex.store import save_json

WORDS = [
    "alice",
    "rabbit",
    "queen",
    "cat",
    "hatter",
    "tea",
    "party",
    "croquet",
    "garden",
    "key",
    "door",
    "pool",
    "tears",
]


def make_lines(count: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    lines = []
    for i in range(count):
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40)))
        lines.append(json.dumps({"id": f"d{i}", "abstract": text}))
    lines.insert(7, "")
    lines.insert(11, "this is not json")
    return lines


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("parallel") / "corpus.jsonl"
    path.write_text("\n".join(make_lines(300)) + "\n", encoding="utf-8")
    return path


def saved_bytes(index: Index, directory: Path) -> bytes:
    path = directory / "index.json"
    save_json(index, path)
    return path.read_bytes()


def same_index(a: Index, b: Index) -> bool:
    return a.postings == b.postings and a.doc_lengths == b.doc_lengths and a.doc_meta == b.doc_meta


@pytest.mark.parametrize("parts", [1, 2, 3, 5, 8, 50])
def test_chunks_cover_every_document_exactly_once(corpus: Path, parts: int) -> None:
    from_chunks = [doc for chunk in plan_chunks(corpus, parts) for doc in iter_chunk(chunk)]
    assert from_chunks == list(iter_documents(corpus))


def test_chunks_start_at_line_starts(corpus: Path) -> None:
    data = corpus.read_bytes()
    for chunk in plan_chunks(corpus, 7):
        assert chunk.start == 0 or data[chunk.start - 1 : chunk.start] == b"\n"


def test_chunk_line_numbers_do_not_overlap(corpus: Path) -> None:
    chunks = plan_chunks(corpus, 6)
    assert chunks[0].first_line == 0
    lines = corpus.read_text(encoding="utf-8").splitlines()
    for chunk in chunks:
        first = next(iter_chunk(chunk))
        assert first.line >= chunk.first_line
        assert json.loads(lines[first.line])["id"] == first.doc_id


def test_chunks_over_several_files(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("\n".join(make_lines(40, seed=1)) + "\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("\n".join(make_lines(25, seed=2)), encoding="utf-8")
    from_chunks = [doc for chunk in plan_chunks(tmp_path, 5) for doc in iter_chunk(chunk)]
    assert from_chunks == list(iter_documents(tmp_path))
    assert len({doc.line for doc in from_chunks}) == len(from_chunks)


def test_empty_corpus(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert plan_chunks(path, 4) == []
    assert merge([]).num_docs == 0
    assert build_index_parallel(path, 2, ExecutorKind.threads).num_docs == 0


@given(st.lists(st.text(max_size=120), max_size=12), st.integers(1, 6), st.booleans())
@settings(max_examples=60, deadline=None)
def test_merged_partials_equal_the_serial_index(texts: list[str], parts: int, positions: bool):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.jsonl"
        lines = [json.dumps({"id": f"d{i}", "abstract": t}) for i, t in enumerate(texts)]
        path.write_text("\n".join(lines), encoding="utf-8")
        partials = (build_partial(c, positions) for c in plan_chunks(path, parts))
        assert same_index(merge(partials), build_index(path, positions))


@pytest.mark.parametrize(
    ("kind", "workers"),
    [
        (ExecutorKind.serial, 1),
        (ExecutorKind.serial, 3),
        (ExecutorKind.threads, 2),
        (ExecutorKind.threads, 4),
    ],
)
@pytest.mark.parametrize("positions", [False, True])
def test_executors_build_identical_indexes(
    corpus: Path, tmp_path: Path, kind: ExecutorKind, workers: int, positions: bool
) -> None:
    expected = saved_bytes(build_index(corpus, positions), tmp_path)
    built = build_index_parallel(corpus, workers, kind, positions)
    assert saved_bytes(built, tmp_path) == expected


@pytest.mark.usefixtures("spawn")
def test_processes_build_the_identical_index(corpus: Path, tmp_path: Path) -> None:
    expected = saved_bytes(build_index(corpus, positions=True), tmp_path)
    built = build_index_parallel(corpus, 2, ExecutorKind.processes, positions=True)
    assert saved_bytes(built, tmp_path) == expected


@pytest.mark.usefixtures("spawn")
def test_processes_report_their_pid_and_cpu(corpus: Path) -> None:
    results = list(run_chunks(plan_chunks(corpus, 2), 2, ExecutorKind.processes))
    assert len(results) == 2
    assert all(r.pid != os.getpid() for r in results)
    assert all(r.cpu > 0 for r in results)


@pytest.mark.parametrize("kind", list(ExecutorKind))
@pytest.mark.usefixtures("spawn")
def test_worker_errors_are_raised_not_swallowed(tmp_path: Path, kind: ExecutorKind) -> None:
    chunks = [Chunk(tmp_path / "missing.jsonl", 0, 10, 0)]
    with pytest.raises(FileNotFoundError):
        list(run_chunks(chunks, 2, kind))


def test_workers_must_be_positive(corpus: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(run_chunks(plan_chunks(corpus, 1), 0, ExecutorKind.serial))


def test_auto_executor_keeps_small_corpora_serial(corpus: Path) -> None:
    assert auto_executor(corpus, workers=8) is ExecutorKind.serial


def test_auto_executor_picks_processes_for_big_corpora(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "MIN_PARALLEL_BYTES", 1)
    assert auto_executor(corpus, workers=8) is ExecutorKind.processes
    assert auto_executor(corpus, workers=1) is ExecutorKind.serial


def test_postings_and_metadata_survive_pickling() -> None:
    posting = Posting(3, 2, (1, 5))
    meta = DocMeta("doc_1", "data/corpus.jsonl", "preview")
    assert pickle.loads(pickle.dumps(posting)) == posting
    assert pickle.loads(pickle.dumps(meta)) == meta
    assert pickle.loads(pickle.dumps(Posting(1, 1))).positions == ()


def test_gc_paused_restores_the_previous_state() -> None:
    before = gc.isenabled()
    with gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled() == before


def test_gc_paused_restores_state_after_an_error() -> None:
    before = gc.isenabled()
    with pytest.raises(RuntimeError), gc_paused():
        raise RuntimeError("boom")
    assert gc.isenabled() == before


def test_nested_gc_pauses_end_with_the_outer_one() -> None:
    with gc_paused():
        with gc_paused():
            pass
        assert not gc.isenabled()
