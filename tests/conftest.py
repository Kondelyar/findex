import json
from pathlib import Path

import pytest

from findex.index import build_index
from findex.models import Index

DOCS = [
    "Alice saw the White Rabbit run past with a watch",
    "The Queen shouted off with her head at the croquet game",
    "The Cheshire Cat grinned at Alice and vanished",
    "a mad tea party with the Hatter and the Dormouse",
    "the White Rabbit was late",
]


@pytest.fixture(scope="session")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    lines = [json.dumps({"id": f"doc_{i}", "abstract": text}) for i, text in enumerate(DOCS, 1)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def index(corpus_path: Path) -> Index:
    return build_index(corpus_path, positions=True)
