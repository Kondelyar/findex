import functools
import gc
import logging
import os
import time
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from findex.corpus import Chunk, corpus_size, iter_chunk, plan_chunks
from findex.index import build_index_from
from findex.models import DocMeta, Index, Posting
from findex.timing import timed

log = logging.getLogger(__name__)

# Starting worker processes costs more than indexing a small corpus.
MIN_PARALLEL_BYTES = 4 * 1024 * 1024

type PartialIndex = Index


class ExecutorKind(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


class WorkerResult(NamedTuple):
    partial: PartialIndex
    pid: int
    cpu: float


@contextmanager
def gc_paused() -> Generator[None]:
    """Building an index allocates millions of objects that never become garbage.

    The cycle collector keeps rescanning them, which costs more than the work
    itself, so it is switched off while the index is built (in the parent here,
    and in every worker process through the pool's initializer).
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def build_partial(chunk: Chunk, positions: bool = False) -> PartialIndex:
    return build_index_from(iter_chunk(chunk), positions)


def _work(chunk: Chunk, positions: bool) -> WorkerResult:
    partial = build_partial(chunk, positions)
    return WorkerResult(partial, os.getpid(), time.process_time())


def run_chunks(
    chunks: list[Chunk], workers: int, executor: ExecutorKind, positions: bool = False
) -> Iterator[WorkerResult]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    work = functools.partial(_work, positions=positions)

    if executor is ExecutorKind.serial:
        yield from map(work, chunks)
    elif executor is ExecutorKind.threads:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(work, chunks)
    else:
        # A worker lives only for this build, and it also pickles its result
        # at the end: the collector stays off for all of it.
        with ProcessPoolExecutor(max_workers=workers, initializer=gc.disable) as pool:
            yield from pool.map(work, chunks)


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Join partial indexes built over consecutive chunks, given in chunk order.

    doc_ids were assigned up front, so the postings of a term only have to be
    appended: they stay sorted because the chunks are in order.
    """
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for part in partials:
        for term, plist in part.postings.items():
            existing = postings.get(term)
            if existing is None:
                postings[term] = plist
            else:
                existing.extend(plist)
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
    return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)


def auto_executor(root: Path, workers: int) -> ExecutorKind:
    if workers > 1 and corpus_size(root) >= MIN_PARALLEL_BYTES:
        return ExecutorKind.processes
    return ExecutorKind.serial


@timed
def build_index_parallel(
    root: Path,
    workers: int = 1,
    executor: ExecutorKind = ExecutorKind.serial,
    positions: bool = False,
) -> Index:
    chunks = plan_chunks(root, workers)
    log.info("indexing %d chunks with %s (%d workers)", len(chunks), executor, workers)
    with gc_paused():
        results = run_chunks(chunks, workers, executor, positions)
        return merge(result.partial for result in results)
