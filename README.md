# findex

Full-text search engine, built incrementally over the Python course labs.
This is Lab 1: a lazy corpus intake (streaming, no full-corpus load into memory).

## Corpus

The engine processes documents in `.jsonl` format (one JSON object per line). Each line has at least an `id` and an `abstract` field:

```json
{"id": "2101.00001", "abstract": "We propose a new method for ..."}
```

Note: While the primary target structure is the Cornell arXiv dataset, the pipeline works with any valid .jsonl file. For the benchmarks below, a real text corpus (Alice's Adventures in Wonderland) was converted into 2595 JSONL documents and saved as data/arxiv.jsonl.

While developing without the full dump, scripts/make_sample_corpus.py generates a small synthetic .jsonl corpus with the same shape, so the pipeline can be tested end to end:

```bash
uv run python scripts/make_sample_corpus.py data/sample.jsonl --docs 5000
```

## Setup

```bash
uv sync
```

## Running

```bash
uv run python -m findex.stats data/arxiv.jsonl
uv run python -m findex.stats data/arxiv.jsonl --limit 1000   # first N docs only
```

Prints document count, token count, vocabulary size, top-50 terms, elapsed time
and peak memory (measured with `tracemalloc`).

## Tokenizer policy

- Unicode-aware: `\w`-based, so Cyrillic and accented Latin letters count as
  letters, not just ASCII.
- `unicodedata.normalize("NFC", ...)` runs before anything else, so a
  precomposed character (`é`) and a decomposed one (`e` + combining accent)
  produce the same token.
- `str.casefold()`, not `.lower()` — the stricter form intended for
  case-insensitive comparison.
- Digits break a token: `python3` -> `python`.
- A hyphen breaks a token: `well-known` -> `well`, `known`.
- An apostrophe (`'` or `’`) inside a word keeps it together: `don't`, `п'ять`.

See `tests/test_tokenize.py` for the exact cases this is checked against.

## Eager vs. lazy (M4)

Measured with scripts/compare_eager_lazy.py on a real 2595-document corpus (Alice's Adventures in Wonderland):

| version           | documents | peak memory | elapsed  |
|:-----------------|:---------:|:-----------:|:--------:|
| eager (lists)     | 2595      | 2.38 MB     | 0.058s   |
| lazy (generators) | 2595      | 0.18 MB     | 0.047s   |

The eager version reads every document into a list, then builds a list of
token lists for all of them, before counting anything — at its peak it is
holding the full corpus text and the full tokenized corpus in memory at the
same time. The lazy version never holds more than the current document and
its tokens; the only thing that legitimately grows is the `Counter` that
holds the final term counts, which is unavoidable — that's the answer we
actually need.

## Project layout

```
src/findex/
  corpus.py     iter_documents(root) -> Iterator[Document]
  tokenize.py   tokenize(text) -> Iterator[str]
  stats.py      the pipeline, entry point: python -m findex.stats
scripts/
  make_sample_corpus.py   synthetic test corpus generator
  compare_eager_lazy.py   M4 measurement script
tests/
  test_tokenize.py
```
