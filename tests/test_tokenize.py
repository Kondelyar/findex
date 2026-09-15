from findex.tokenize import tokenize


def test_basic_words():
    assert list(tokenize("Hello world")) == ["hello", "world"]


def test_mixed_case():
    assert list(tokenize("PyThOn RoCks")) == ["python", "rocks"]


def test_cyrillic():
    assert list(tokenize("Привіт Світ")) == ["привіт", "світ"]


def test_apostrophe_kept_together():
    assert list(tokenize("don't п'ять")) == ["don't", "п'ять"]


def test_combining_mark_normalized():
    precomposed = "café"
    decomposed = "cafe\u0301"
    assert list(tokenize(precomposed)) == list(tokenize(decomposed))


def test_digits_break_a_token():
    assert list(tokenize("Python3 is fun2day")) == ["python", "is", "fun", "day"]


def test_punctuation_is_ignored():
    assert list(tokenize("Hello, world! Really?")) == ["hello", "world", "really"]


def test_hyphen_splits_words():
    assert list(tokenize("well-known state-of-the-art")) == [
        "well",
        "known",
        "state",
        "of",
        "the",
        "art",
    ]


def test_empty_string_gives_nothing():
    assert list(tokenize("")) == []
