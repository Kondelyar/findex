import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: str
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Stream Document objects from an arXiv-style JSON-lines corpus.

    root can be a single .jsonl file or a directory containing several
    .jsonl files (they are read one after another, in sorted order).
    Each line must be a JSON object. The text is taken from the first
    field found among "text", "abstract", "summary". Lines that are not
    valid JSON, or have none of those fields, are logged and skipped
    instead of crashing the whole run.
    """
    root = Path(root)
    paths = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]

    for path in paths:
        yield from _iter_file(path)


def _iter_file(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("bad json at %s:%d, skipping", path, line_no)
                continue

            text = record.get("text") or record.get("abstract") or record.get("summary")
            if not text:
                logger.warning("no text field at %s:%d, skipping", path, line_no)
                continue

            doc_id = str(record.get("id", f"{path.name}:{line_no}"))
            yield Document(doc_id=doc_id, path=str(path), text=text)
