import pytest

from findex.search import merge_and, merge_not, merge_or


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 3, 5], [3, 4, 5, 6], [3, 5]),
        ([1, 2], [3, 4], []),
        ([], [1, 2], []),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3]),
    ],
)
def test_merge_and(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_and(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 3, 5], [3, 4, 5, 6], [1, 3, 4, 5, 6]),
        ([], [1, 2], [1, 2]),
        ([1, 2], [], [1, 2]),
        ([], [], []),
    ],
)
def test_merge_or(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_or(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 2, 3, 4], [2, 4], [1, 3]),
        ([1, 2], [], [1, 2]),
        ([], [1], []),
    ],
)
def test_merge_not(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_not(a, b) == expected
