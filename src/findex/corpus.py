import json
import logging
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO, NamedTuple

logger = logging.getLogger(__name__)

_BLOCK = 1 << 20


class Document(NamedTuple):
    doc_id: str
    path: str
    text: str
    line: int


class Chunk(NamedTuple):
    path: Path
    start: int
    end: int
    first_line: int


def find_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]


def corpus_size(root: Path) -> int:
    return sum(path.stat().st_size for path in find_files(root))


def iter_documents(root: Path) -> Iterator[Document]:
    """Stream Document objects from an arXiv-style JSON-lines corpus.

    root can be a single .jsonl file or a directory containing several
    .jsonl files (they are read one after another, in sorted order).
    Each line must be a JSON object. The text is taken from the first
    field found among "text", "abstract", "summary". Lines that are not
    valid JSON, or have none of those fields, are logged and skipped
    instead of crashing the whole run.

    Every document carries `line`, its line number in the whole corpus
    (0-based, counting skipped lines too). The index uses it as the doc_id.
    """
    first_line = 0
    for path in find_files(root):
        chunk = Chunk(path, 0, path.stat().st_size, first_line)
        for line_no, offset, text in _read_lines(chunk):
            doc = _parse(text, path, line_no, offset)
            if doc is not None:
                yield doc
            first_line = line_no + 1


def iter_chunk(chunk: Chunk) -> Iterator[Document]:
    for line_no, offset, text in _read_lines(chunk):
        doc = _parse(text, chunk.path, line_no, offset)
        if doc is not None:
            yield doc


def plan_chunks(root: Path, parts: int = 1) -> list[Chunk]:
    """Split the corpus into about `parts` slices of whole lines.

    Each chunk knows the number of its first line, so doc_ids of different
    chunks never overlap and nothing has to be renumbered afterwards.
    """
    files = [path for path in find_files(root) if path.stat().st_size > 0]
    total = sum(path.stat().st_size for path in files)
    chunks: list[Chunk] = []
    first_line = 0
    for path in files:
        size = path.stat().st_size
        pieces = max(1, round(parts * size / total))
        with path.open("rb") as f:
            starts = _line_starts(f, size, pieces)
            for start, end in pairwise(starts):
                chunks.append(Chunk(path, start, end, first_line))
                first_line += _count_lines(f, start, end)
    return chunks


def _line_starts(f: BinaryIO, size: int, pieces: int) -> list[int]:
    starts = [0]
    for i in range(1, pieces):
        f.seek(size * i // pieces)
        f.readline()
        pos = f.tell()
        if starts[-1] < pos < size:
            starts.append(pos)
    starts.append(size)
    return starts


def _count_lines(f: BinaryIO, start: int, end: int) -> int:
    f.seek(start)
    remaining = end - start
    count = 0
    last = b""
    while remaining > 0:
        block = f.read(min(remaining, _BLOCK))
        if not block:
            break
        count += block.count(b"\n")
        remaining -= len(block)
        last = block[-1:]
    if last and last != b"\n":
        count += 1
    return count


def _read_lines(chunk: Chunk) -> Iterator[tuple[int, int, str]]:
    offset = chunk.start
    line_no = chunk.first_line
    with chunk.path.open("rb") as f:
        f.seek(chunk.start)
        for raw in f:
            if offset >= chunk.end:
                break
            yield line_no, offset, raw.decode("utf-8", errors="replace")
            offset += len(raw)
            line_no += 1


def _parse(text: str, path: Path, line_no: int, offset: int) -> Document | None:
    text = text.strip()
    if not text:
        return None

    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("bad json in %s at byte %d, skipping", path, offset)
        return None

    body = record.get("text") or record.get("abstract") or record.get("summary")
    if not body:
        logger.warning("no text field in %s at byte %d, skipping", path, offset)
        return None

    doc_id = str(record.get("id", f"{path.name}@{offset}"))
    return Document(doc_id=doc_id, path=str(path), text=body, line=line_no)
