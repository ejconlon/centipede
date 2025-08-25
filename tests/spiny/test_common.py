"""Tests for common utility types and functions."""

from typing import Iterator, List, Tuple

from spiny.common import (
    Entry,
    Flip,
    Ordering,
    compare,
    compare_lex,
    group_runs,
)


def test_entry_comparison():
    """Test Entry comparison based on keys only."""
    entry1 = Entry("apple", 10)
    entry2 = Entry("banana", 5)
    entry3 = Entry("apple", 20)  # Same key, different value

    # Entry comparison should only consider keys
    assert compare(entry1, entry2) == Ordering.Lt
    assert compare(entry2, entry1) == Ordering.Gt
    assert compare(entry1, entry3) == Ordering.Eq  # Same key, different values

    # Test Python comparison operators
    assert entry1 < entry2
    assert entry2 > entry1
    assert entry1 == entry3  # Same key
    assert entry1 != entry2


def test_entry_comparison_numeric_keys():
    """Test Entry comparison with numeric keys."""
    entry1 = Entry(1, "first")
    entry2 = Entry(2, "second")
    entry3 = Entry(1, "different_value")

    assert compare(entry1, entry2) == Ordering.Lt
    assert compare(entry2, entry1) == Ordering.Gt
    assert compare(entry1, entry3) == Ordering.Eq

    assert entry1 < entry2
    assert entry2 > entry1
    assert entry1 == entry3


def test_flip_comparison():
    """Test Flip wrapper reverses comparison results."""
    # Normal comparison
    assert compare(1, 2) == Ordering.Lt
    assert compare(2, 1) == Ordering.Gt
    assert compare(1, 1) == Ordering.Eq

    # Flipped comparison
    assert compare(Flip(1), Flip(2)) == Ordering.Gt
    assert compare(Flip(2), Flip(1)) == Ordering.Lt
    assert compare(Flip(1), Flip(1)) == Ordering.Eq


def test_flip_with_strings():
    """Test Flip with string values."""
    # Normal string comparison
    assert compare("apple", "banana") == Ordering.Lt
    assert compare("zebra", "apple") == Ordering.Gt

    # Flipped string comparison
    assert compare(Flip("apple"), Flip("banana")) == Ordering.Gt
    assert compare(Flip("zebra"), Flip("apple")) == Ordering.Lt


def test_flip_ordering_operators():
    """Test that Flip works with Python comparison operators."""
    # Normal ordering
    assert 1 < 2
    assert 2 > 1
    assert 1 == 1

    # Flipped ordering
    assert Flip(1) > Flip(2)  # Flipped!
    assert Flip(2) < Flip(1)  # Flipped!
    assert Flip(1) == Flip(1)


def test_flip_with_entry():
    """Test Flip with Entry objects."""
    entry1 = Entry("apple", 1)
    entry2 = Entry("banana", 2)

    # Normal comparison
    assert compare(entry1, entry2) == Ordering.Lt

    # Flipped comparison
    assert compare(Flip(entry1), Flip(entry2)) == Ordering.Gt


def test_compare_function():
    """Test the compare function with various types."""
    # Integers
    assert compare(1, 2) == Ordering.Lt
    assert compare(2, 1) == Ordering.Gt
    assert compare(1, 1) == Ordering.Eq

    # Strings
    assert compare("a", "b") == Ordering.Lt
    assert compare("b", "a") == Ordering.Gt
    assert compare("a", "a") == Ordering.Eq

    # Floats
    assert compare(1.5, 2.5) == Ordering.Lt
    assert compare(2.5, 1.5) == Ordering.Gt
    assert compare(1.5, 1.5) == Ordering.Eq


def test_compare_lex_empty_iterators():
    """Test compare_lex with empty iterators."""
    empty_list: List[int] = []
    gen1: Iterator[int] = (x for x in empty_list)
    gen2: Iterator[int] = (x for x in empty_list)
    assert compare_lex(gen1, gen2) == Ordering.Eq


def test_compare_lex_one_empty():
    """Test compare_lex when one iterator is empty."""
    empty_list: List[int] = []
    gen1: Iterator[int] = (x for x in empty_list)
    gen2: Iterator[int] = (x for x in [1, 2, 3])
    assert compare_lex(gen1, gen2) == Ordering.Lt

    gen1 = (x for x in [1, 2, 3])
    gen2 = (x for x in empty_list)
    assert compare_lex(gen1, gen2) == Ordering.Gt


def test_compare_lex_equal_sequences():
    """Test compare_lex with equal sequences."""
    gen1 = (x for x in [1, 2, 3])
    gen2 = (x for x in [1, 2, 3])
    assert compare_lex(gen1, gen2) == Ordering.Eq


def test_compare_lex_different_sequences():
    """Test compare_lex with different sequences."""
    # First element different
    gen1 = (x for x in [1, 2, 3])
    gen2 = (x for x in [2, 2, 3])
    assert compare_lex(gen1, gen2) == Ordering.Lt

    gen1 = (x for x in [2, 2, 3])
    gen2 = (x for x in [1, 2, 3])
    assert compare_lex(gen1, gen2) == Ordering.Gt

    # Later element different
    gen1 = (x for x in [1, 2, 3])
    gen2 = (x for x in [1, 2, 4])
    assert compare_lex(gen1, gen2) == Ordering.Lt


def test_compare_lex_different_lengths():
    """Test compare_lex with sequences of different lengths."""
    # Shorter sequence is a prefix of longer
    gen1 = (x for x in [1, 2])
    gen2 = (x for x in [1, 2, 3])
    assert compare_lex(gen1, gen2) == Ordering.Lt

    gen1 = (x for x in [1, 2, 3])
    gen2 = (x for x in [1, 2])
    assert compare_lex(gen1, gen2) == Ordering.Gt


def test_compare_lex_strings():
    """Test compare_lex with string sequences."""
    gen1 = (x for x in "abc")
    gen2 = (x for x in "abd")
    assert compare_lex(gen1, gen2) == Ordering.Lt

    gen1 = (x for x in "abd")
    gen2 = (x for x in "abc")
    assert compare_lex(gen1, gen2) == Ordering.Gt

    gen1 = (x for x in "abc")
    gen2 = (x for x in "abc")
    assert compare_lex(gen1, gen2) == Ordering.Eq


def test_group_runs_empty():
    """Test group_runs with empty iterator."""
    empty_list: List[Tuple[str, int]] = []
    gen: Iterator[Tuple[str, int]] = (x for x in empty_list)
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == []


def test_group_runs_single_element():
    """Test group_runs with single element."""
    gen = (x for x in [("a", 1)])
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == [("a", [1])]


def test_group_runs_same_key():
    """Test group_runs with all same keys."""
    gen = (x for x in [("a", 1), ("a", 2), ("a", 3)])
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == [("a", [1, 2, 3])]


def test_group_runs_different_keys():
    """Test group_runs with all different keys."""
    gen = (x for x in [("a", 1), ("b", 2), ("c", 3)])
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == [("a", [1]), ("b", [2]), ("c", [3])]


def test_group_runs_mixed():
    """Test group_runs with mixed key patterns."""
    gen = (x for x in [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("a", 5)])
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == [("a", [1, 2]), ("b", [3, 4]), ("a", [5])]


def test_group_runs_complex_values():
    """Test group_runs with complex values."""
    data: List[Tuple[str, Tuple[int, str]]] = [
        ("group1", (1, "first")),
        ("group1", (2, "second")),
        ("group2", (3, "third")),
        ("group1", (4, "fourth")),
    ]
    gen = (x for x in data)
    result: List[Tuple[str, List[Tuple[int, str]]]] = list(group_runs(gen))
    expected = [
        ("group1", [(1, "first"), (2, "second")]),
        ("group2", [(3, "third")]),
        ("group1", [(4, "fourth")]),
    ]
    assert result == expected


def test_group_runs_numeric_keys():
    """Test group_runs with numeric keys."""
    gen = (x for x in [(1, "a"), (1, "b"), (2, "c"), (2, "d"), (3, "e")])
    result: List[Tuple[int, List[str]]] = list(group_runs(gen))
    assert result == [(1, ["a", "b"]), (2, ["c", "d"]), (3, ["e"])]


def test_group_runs_preserves_order():
    """Test that group_runs preserves value order within groups."""
    gen = (x for x in [("x", 3), ("x", 1), ("x", 4), ("y", 2), ("y", 5)])
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == [("x", [3, 1, 4]), ("y", [2, 5])]


def test_group_runs_single_key_multiple_runs():
    """Test group_runs where same key appears in separate runs."""
    gen = (x for x in [("a", 1), ("b", 2), ("a", 3), ("c", 4), ("a", 5)])
    result: List[Tuple[str, List[int]]] = list(group_runs(gen))
    assert result == [("a", [1]), ("b", [2]), ("a", [3]), ("c", [4]), ("a", [5])]
