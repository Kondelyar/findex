import heapq
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from operator import itemgetter


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: str
    path: str
    preview: str


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ):
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, docs={self.num_docs})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def total_tokens(self) -> int:
        return sum(self.doc_lengths.values())

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs

    @cached_property
    def has_positions(self) -> bool:
        first = next(iter(self.postings.values()), None)
        return bool(first and first[0].positions)

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def top_terms(self, n: int = 50) -> list[tuple[str, int]]:
        totals = ((term, sum(p.tf for p in plist)) for term, plist in self.postings.items())
        return heapq.nlargest(n, totals, key=itemgetter(1))

    def close(self) -> None:
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_meta.clear()
