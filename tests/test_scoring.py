import pytest

from findex.models import Index, Posting
from findex.scoring import BM25, TfIdf


def make_index(doc_lengths, postings):
    return Index(postings=postings, doc_lengths=doc_lengths, doc_meta={})


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_scores_higher_than_common_term(scorer):
    index = make_index(
        {i: 10 for i in range(10)},
        {
            "rare": [Posting(0, 1)],
            "common": [Posting(i, 1) for i in range(10)],
        },
    )
    assert scorer("rare", Posting(0, 1), index) > scorer("common", Posting(0, 1), index)


def test_bm25_saturates():
    index = make_index({0: 10, 1: 10}, {"x": [Posting(0, 1)]})
    bm25 = BM25()
    score = {tf: bm25("x", Posting(0, tf), index) for tf in (1, 2, 20, 21)}
    first_gain = score[2] - score[1]
    late_gain = score[21] - score[20]
    assert late_gain < 0.05 * first_gain


def test_bm25_prefers_short_document():
    index = make_index({0: 5, 1: 500}, {"x": [Posting(0, 1), Posting(1, 1)]})
    bm25 = BM25()
    assert bm25("x", Posting(0, 1), index) > bm25("x", Posting(1, 1), index)


def test_bm25_without_length_normalisation_ignores_length():
    index = make_index({0: 5, 1: 500}, {"x": [Posting(0, 1), Posting(1, 1)]})
    bm25 = BM25(b=0)
    assert bm25("x", Posting(0, 1), index) == pytest.approx(bm25("x", Posting(1, 1), index))
