# findex

Full-text search engine, built incrementally over the Python course labs.

## Corpus

The engine processes documents in `.jsonl` format (one JSON object per line). Each line has at least an `id` and an `abstract` field:

```json
{"id": "2101.00001", "abstract": "We propose a new method for ..."}
```

Note: while the primary target structure is the Cornell arXiv dataset, the pipeline works with any valid .jsonl file. For the benchmarks below, a real text corpus (Alice's Adventures in Wonderland) was converted into 2595 JSONL documents and saved as `data/arxiv.jsonl` (see `convert_to_jsonl.py`).

While developing without a real corpus, `scripts/make_sample_corpus.py` generates a small synthetic `.jsonl` corpus with the same shape, so the pipeline can be tested end to end:

```bash
uv run python scripts/make_sample_corpus.py data/sample.jsonl --docs 5000
```

## Setup

```bash
uv sync
```

## Lab 1 — corpus intake

```bash
uv run python -m findex.stats data/arxiv.jsonl
uv run python -m findex.stats data/arxiv.jsonl --limit 1000   # first N docs only
```

Prints document count, token count, vocabulary size, top-50 terms, elapsed time
and peak memory (measured with `tracemalloc`).

### Tokenizer policy

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

### Eager vs. lazy (M4)

Measured with `scripts/compare_eager_lazy.py` on the real 2595-document corpus (Alice's Adventures in Wonderland):

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

## Lab 2 — inverted index

### Build and save

```bash
uv run python -m findex.index data/arxiv.jsonl --out data/index.pkl
uv run python -m findex.index data/arxiv.jsonl --out data/index.json   # second format, by extension
```

Builds `postings: dict[str, list[Posting]]`, `doc_lengths` and `doc_meta` in one
streaming pass over the Lab 1 pipeline, and saves the result. Prints document
count, term count, elapsed time and peak memory.

### Search

```bash
uv run python -m findex.search data/index.pkl "cat AND mouse"
uv run python -m findex.search data/index.pkl "cat OR mouse NOT hat" --engine set --limit 5
```

Query language: terms separated by spaces are implicit `AND`; `AND`, `OR`, `NOT`
can be written explicitly, evaluated strictly left to right (no precedence, no
parentheses — that needs a real parser, which is Lab 3). `--engine merge` (default)
walks the sorted postings lists with two pointers; `--engine set` uses Python's
built-in set operations on the same doc_ids, for comparison.

### merge vs. set benchmark (M2)

Measured with `scripts/compare_engines.py` on `data/arxiv.jsonl`:

```bash
uv run python scripts/compare_engines.py data/arxiv.jsonl
```

| query | engine | avg time |
|---|---|---|
| common AND (and AND the) | merge | 0.0245 ms |
| common AND (and AND the) | set | 0.0510 ms |
| rare AND (title AND author) | merge | 0.0023 ms |
| rare AND (title AND author) | set | 0.0024 ms |


### Persistence

`store.py` supports two formats:

- **pickle** (`.pkl`) — serializes the `Posting`/`DocMeta`/`Index` objects directly.
  Fast, but `pickle.load` runs arbitrary code embedded in the file — only ever
  load a pickle findex itself produced, never one from an untrusted source.
- **json** (`.json`) — safe, human-readable, slightly bigger and slower, since
  every dataclass instance becomes a plain `dict` on the way out and back.

### Postings memory study (M4)

Measured with `scripts/posting_memory_study.py` on `data/arxiv.jsonl` — same
postings, three different representations, each saved and reloaded with pickle:

```bash
uv run python scripts/posting_memory_study.py data/arxiv.jsonl --tmp data/posting_variant.pkl```

| representation | peak memory (build) | index file size | load time |
|---|---|---|---|
| list[Posting], plain dataclass | 3.23 MB | 551.7 KB | 22.26 ms |
| list[Posting], slots=True | 2.16 MB | 633.1 KB | 20.54 ms |
| array('I') pairs | 1.28 MB | 346.7 KB | 12.15 ms |

## Project layout

```
src/findex/
  corpus.py     iter_documents(root) -> Iterator[Document]        (Lab 1)
  tokenize.py   tokenize(text) -> Iterator[str]                   (Lab 1)
  stats.py      corpus statistics pipeline, python -m findex.stats (Lab 1)
  models.py     Posting, DocMeta, Index dataclasses                (Lab 2)
  index.py      build_index(root) -> Index, python -m findex.index (Lab 2)
  search.py     evaluate(index, query) -> doc_ids, python -m findex.search (Lab 2)
  store.py      save/load the index as pickle or json               (Lab 2)
scripts/
  make_sample_corpus.py    synthetic test corpus generator
  compare_eager_lazy.py    Lab 1 M4 measurement script
  compare_engines.py       Lab 2 M2 merge-vs-set benchmark
  posting_memory_study.py  Lab 2 M4 measurement script
tests/
  test_tokenize.py   Lab 1
  test_index.py      Lab 2
  test_search.py     Lab 2
  test_store.py       Lab 2
```
