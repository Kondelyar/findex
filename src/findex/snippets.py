import logging
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import Index
from findex.tokenize import WORD_RE, tokenize

log = logging.getLogger(__name__)


def fetch_texts(index: Index, doc_ids: list[int]) -> dict[int, str]:
    wanted: dict[tuple[str, str], int] = {}
    for doc_id in doc_ids:
        meta = index.doc_meta[doc_id]
        wanted[(meta.path, meta.doc_id)] = doc_id

    texts: dict[int, str] = {}
    for path in {path for path, _ in wanted}:
        if not Path(path).exists():
            log.warning("corpus file %s not found, no snippets for it", path)
            continue
        for doc in iter_documents(Path(path)):
            key = (path, doc.doc_id)
            if key in wanted:
                texts[wanted[key]] = doc.text
            if len(texts) == len(wanted):
                return texts
    return texts


def make_snippet(text: str, terms: set[str], width: int = 80) -> str:
    text = " ".join(text.split())
    hits = [m for m in WORD_RE.finditer(text) if next(tokenize(m.group()), "") in terms]
    if not hits:
        return text[: 2 * width]

    best = max(hits, key=lambda h: sum(abs(o.start() - h.start()) <= width for o in hits))
    start = max(0, best.start() - width)
    end = min(len(text), best.end() + width)
    if start > 0:
        start = text.rfind(" ", 0, start) + 1
    if end < len(text):
        end = text.find(" ", end)
        end = len(text) if end == -1 else end

    parts: list[str] = []
    pos = start
    for m in hits:
        if m.start() >= start and m.end() <= end:
            parts.append(text[pos : m.start()])
            parts.append(f"**{m.group()}**")
            pos = m.end()
    parts.append(text[pos:end])

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + "".join(parts) + suffix
