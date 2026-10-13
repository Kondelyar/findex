import math
from dataclasses import dataclass
from typing import Protocol

from findex.models import Index, Posting


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True)
class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        return (1 + math.log(posting.tf)) * idf


@dataclass(frozen=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        idf = math.log(1 + (index.num_docs - df + 0.5) / (df + 0.5))
        length_norm = 1 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length
        return idf * posting.tf * (self.k1 + 1) / (posting.tf + self.k1 * length_norm)
