import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", ["hello", "world"]),
        ("PyThOn RoCks", ["python", "rocks"]),
        ("Привіт Світ", ["привіт", "світ"]),
        ("don't п'ять", ["don't", "п'ять"]),
        ("café", ["café"]),
        ("cafe\u0301", ["café"]),
        ("Python3 is fun2day", ["python", "is", "fun", "day"]),
        ("Hello, world! Really?", ["hello", "world", "really"]),
        ("well-known state-of-the-art", ["well", "known", "state", "of", "the", "art"]),
        ("STRASSE Straße", ["strasse", "strasse"]),
        ("", []),
        ("   ... 123 ---", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_is_lazy() -> None:
    tokens = tokenize("one two three")
    assert next(tokens) == "one"
