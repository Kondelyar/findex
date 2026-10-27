import logging
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import DocMeta, Index, Posting
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@timed
def build_index_from(docs: Iterable[Document], positions: bool = False) -> Index:
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):
        offsets: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokenize(doc.text)):
            offsets[token].append(pos)

        doc_lengths[doc_id] = sum(len(where) for where in offsets.values())
        doc_meta[doc_id] = DocMeta(
            doc_id=doc.doc_id,
            path=doc.path,
            preview=doc.text[:80].replace("\n", " "),
        )
        for term, where in offsets.items():
            postings[term].append(Posting(doc_id, len(where), tuple(where) if positions else ()))

    log.info("indexed %d documents, %d terms", len(doc_lengths), len(postings))
    return Index(postings=dict(postings), doc_lengths=doc_lengths, doc_meta=doc_meta)


def build_index(root: Path, positions: bool = False, limit: int | None = None) -> Index:
    return build_index_from(islice(iter_documents(root), limit), positions)
