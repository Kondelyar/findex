import re
import unicodedata
from collections.abc import Iterator

# A token is a run of letters (any Unicode letter, so Cyrillic and accented
# Latin count too), with an optional apostrophe in the middle for cases like
# "don't" or "п'ять". Digits break a token. A hyphen breaks a token, so
# "well-known" becomes two tokens: "well", "known".
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?")


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized word tokens from text, one at a time.

    Text is NFC-normalized first so that visually identical characters
    written with different Unicode code points (e.g. precomposed "é" vs
    "e" + combining accent) turn into the same token, then casefolded
    for case-insensitive matching.
    """
    normalized = unicodedata.normalize("NFC", text)
    folded = normalized.casefold()
    for match in _WORD_RE.finditer(folded):
        yield match.group()
