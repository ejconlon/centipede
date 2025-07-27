"""Property-based tests for Heap using Hypothesis."""

from typing import List, Tuple

from hypothesis import given
from hypothesis import strategies as st

from centipede.spiny.heap import Heap
from tests.centipede.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def heap_strategy(draw, key_strategy=st.integers(), value_strategy=st.text()):
    """Generate a Heap with random key-value pairs."""
    entries = draw(
        st.lists(st.tuples(key_strategy, value_strategy), min_size=0, max_size=20)
    )
    return Heap.mk(entries)


@st.composite
def heap_with_entries_strategy(
    draw, key_strategy=st.integers(), value_strategy=st.text()
):
    """Generate a Heap along with the entries used to create it."""
    entries = draw(
        st.lists(st.tuples(key_strategy, value_strategy), min_size=0, max_size=20)
    )
    return Heap.mk(entries), entries


@given(st.lists(st.tuples(st.integers(), st.text()), min_size=0, max_size=50))
def test_heap_mk_size_consistency(entries: List[Tuple[int, str]]):
    """Creating a Heap from entries should have correct size."""
    heap = Heap.mk(entries)
    assert heap.size() == len(entries)
    assert heap.null() == (len(entries) == 0)


@given(heap_strategy())
def test_exhaustive_heap_operations(heap):
    """Test that all elements can be extracted from the heap."""
    original_size = heap.size()
    extracted = []
    current = heap

    while not current.null():
        min_result = current.find_min()
        assert min_result is not None
        key, value, remaining = min_result

        # Check delete_min consistency
        delete_result = current.delete_min()
        assert delete_result is not None
        assert delete_result.size() == remaining.size()

        extracted.append((key, value))
        current = delete_result

    # Should have extracted all elements
    assert len(extracted) == original_size
    assert current.null()
    assert current.find_min() is None
    assert current.delete_min() is None


@given(heap_strategy(), st.integers(), st.text())
def test_insert_increases_size(heap, key, value):
    """Inserting an element should increase size by 1."""
    original_size = heap.size()
    new_heap = heap.insert(key, value)

    assert new_heap.size() == original_size + 1
    assert not new_heap.null()


@given(heap_strategy(), st.integers(), st.text())
def test_insert_preserves_min_heap_property(heap, key, value):
    """After insertion, find_min should return a valid minimum."""
    new_heap = heap.insert(key, value)

    if heap.null():
        # If original was empty, new minimum should be the inserted element
        min_result = new_heap.find_min()
        assert min_result is not None
        min_key, min_value, _ = min_result
        assert min_key == key
        assert min_value == value
    else:
        # New minimum should be <= inserted key and <= original minimum
        original_min = heap.find_min()
        assert original_min is not None
        original_min_key, _, _ = original_min

        new_min = new_heap.find_min()
        assert new_min is not None
        new_min_key, _, _ = new_min

        assert new_min_key <= key
        assert new_min_key <= original_min_key


@given(heap_with_entries_strategy())
def test_find_min_returns_actual_minimum(heap_and_entries):
    """find_min should return the actual minimum key from all entries."""
    heap, entries = heap_and_entries

    if not entries:
        assert heap.find_min() is None
    else:
        expected_min_key = min(key for key, _ in entries)
        min_result = heap.find_min()
        assert min_result is not None
        actual_min_key, _, _ = min_result
        assert actual_min_key == expected_min_key


@given(heap_strategy())
def test_find_min_delete_min_consistency(heap):
    """find_min and delete_min should be consistent."""
    if heap.null():
        assert heap.find_min() is None
        assert heap.delete_min() is None
    else:
        find_result = heap.find_min()
        delete_result = heap.delete_min()

        assert find_result is not None
        assert delete_result is not None

        _, _, remaining_from_find = find_result

        assert remaining_from_find.size() == delete_result.size()
        assert remaining_from_find.size() == heap.size() - 1


@given(heap_strategy())
def test_delete_min_decreases_size(heap):
    """delete_min should decrease size by 1 for non-empty heaps."""
    if heap.null():
        assert heap.delete_min() is None
    else:
        original_size = heap.size()
        result = heap.delete_min()
        assert result is not None
        assert result.size() == original_size - 1


@given(heap_strategy(), heap_strategy())
def test_meld_size_additive(heap1, heap2):
    """Melded heap size should equal sum of individual sizes."""
    melded = heap1.meld(heap2)
    assert melded.size() == heap1.size() + heap2.size()


@given(heap_strategy())
def test_meld_empty_identity(heap1):
    """Melding with empty heap should be identity."""
    empty = Heap.empty(int, str)

    assert heap1.meld(empty).size() == heap1.size()
    assert empty.meld(heap1).size() == heap1.size()


@given(heap_strategy(), heap_strategy())
def test_meld_preserves_min_heap_property(heap1, heap2):
    """Melded heap should maintain min-heap property."""
    melded = heap1.meld(heap2)

    if heap1.null() and heap2.null():
        assert melded.null()
    else:
        # Find minimums of individual heaps
        min1 = heap1.find_min()
        min2 = heap2.find_min()

        expected_overall_min = None
        if min1 is not None and min2 is not None:
            expected_overall_min = min(min1[0], min2[0])
        elif min1 is not None:
            expected_overall_min = min1[0]
        elif min2 is not None:
            expected_overall_min = min2[0]

        melded_min = melded.find_min()
        assert melded_min is not None
        assert melded_min[0] == expected_overall_min


@given(heap_strategy(), heap_strategy())
def test_meld_associative(heap1, heap2):
    """Meld should be associative: (a + b) + c == a + (b + c)."""
    heap3 = Heap.mk([(100, "hundred"), (200, "two_hundred")])

    left_assoc = heap1.meld(heap2).meld(heap3)
    right_assoc = heap1.meld(heap2.meld(heap3))

    # Both should have same size
    assert left_assoc.size() == right_assoc.size()

    # Both should have same minimum
    left_min = left_assoc.find_min()
    right_min = right_assoc.find_min()

    if left_min is None:
        assert right_min is None
    else:
        assert right_min is not None
        assert left_min[0] == right_min[0]


@given(heap_strategy())
def test_iter_sorted_order(heap):
    """Iterating through heap should yield elements in sorted order."""
    items = list(heap.iter())
    keys = [key for key, _ in items]

    assert len(items) == heap.size()
    assert keys == sorted(keys)


@given(heap_strategy())
def test_iter_exhausts_heap_elements(heap):
    """Iterating should yield all elements that were in the heap."""
    items = list(heap.iter())

    # Check that we can extract the same elements by repeated delete_min
    extracted = []
    current = heap
    while not current.null():
        min_result = current.find_min()
        assert min_result is not None
        key, value, remaining = min_result
        extracted.append((key, value))
        current = remaining

    assert len(items) == len(extracted)
    assert items == extracted


@given(heap_with_entries_strategy())
def test_iter_contains_all_inserted_elements(heap_and_entries):
    """Iteration should contain all originally inserted elements."""
    heap, entries = heap_and_entries

    items = list(heap.iter())
    items_sorted = sorted(items)
    entries_sorted = sorted(entries)

    assert items_sorted == entries_sorted


@given(st.integers(), st.text())
def test_singleton_properties(key, value):
    """Singleton heap should have expected properties."""
    heap = Heap.singleton(key, value)

    assert not heap.null()
    assert heap.size() == 1

    min_result = heap.find_min()
    assert min_result is not None
    min_key, min_value, remaining = min_result
    assert min_key == key
    assert min_value == value
    assert remaining.null()

    delete_result = heap.delete_min()
    assert delete_result is not None
    assert delete_result.null()

    items = list(heap.iter())
    assert items == [(key, value)]


@given(heap_strategy())
def test_multiple_delete_min_maintains_order(heap):
    """Repeated delete_min should maintain sorted order."""
    extracted_keys = []
    current = heap

    while not current.null():
        min_result = current.find_min()
        assert min_result is not None
        key, _, remaining = min_result
        extracted_keys.append(key)

        delete_result = current.delete_min()
        assert delete_result is not None
        assert delete_result.size() == remaining.size()

        current = remaining

    # Extracted keys should be in non-decreasing order
    assert extracted_keys == sorted(extracted_keys)


@given(heap_strategy(), st.integers(), st.text())
def test_insert_then_delete_min_with_duplicates(heap, key, value):
    """Insert then delete_min should handle duplicates correctly."""
    # Insert the same key multiple times
    heap_with_dups = (
        heap.insert(key, value).insert(key, f"{value}_2").insert(key, f"{value}_3")
    )

    original_size = heap.size()
    assert heap_with_dups.size() == original_size + 3

    # The minimum should be no greater than our inserted key
    min_result = heap_with_dups.find_min()
    assert min_result is not None
    min_key, _, _ = min_result
    assert min_key <= key


@given(heap_strategy())
def test_heap_immutability(heap):
    """Operations should not modify the original heap."""
    original_size = heap.size()
    original_null = heap.null()

    # Perform various operations
    heap.insert(999, "test")
    heap.meld(Heap.singleton(1000, "another"))
    heap.delete_min()
    heap.find_min()
    list(heap.iter())

    # Original heap should be unchanged
    assert heap.size() == original_size
    assert heap.null() == original_null


@given(heap_strategy(), heap_strategy())
def test_addition_operator_equals_meld(heap1, heap2):
    """+ operator should be equivalent to meld method."""
    meld_result = heap1.meld(heap2)
    add_result = heap1 + heap2

    assert meld_result.size() == add_result.size()

    # Both should have same minimum
    meld_min = meld_result.find_min()
    add_min = add_result.find_min()

    if meld_min is None:
        assert add_min is None
    else:
        assert add_min is not None
        assert meld_min[0] == add_min[0]
        assert meld_min[1] == add_min[1]


@given(st.lists(st.tuples(st.integers(), st.text()), min_size=1, max_size=10))
def test_heap_sort_property(entries: List[Tuple[int, str]]):
    """Using heap as priority queue should sort elements by key."""
    heap = Heap.mk(entries)

    sorted_by_heap = list(heap.iter())

    # Should extract all elements
    assert len(sorted_by_heap) == len(entries)

    # Keys should be in non-decreasing order
    keys = [key for key, _ in sorted_by_heap]
    assert keys == sorted(keys)

    # Should contain the same elements (order may vary for equal keys)
    assert sorted(sorted_by_heap) == sorted(entries)


@given(heap_strategy())
def test_find_min_idempotent(heap):
    """Multiple calls to find_min should return consistent key-value pairs."""
    result1 = heap.find_min()
    result2 = heap.find_min()
    result3 = heap.find_min()

    if result1 is None:
        assert result2 is None and result3 is None
    else:
        assert result2 is not None and result3 is not None
        # Keys and values should be the same, though heap objects may differ
        assert result1[0] == result2[0] == result3[0]
        assert result1[1] == result2[1] == result3[1]


@given(heap_with_entries_strategy())
def test_size_matches_entry_count(heap_and_entries):
    """Heap size should match the number of entries inserted."""
    heap, entries = heap_and_entries
    assert heap.size() == len(entries)


@given(st.lists(st.tuples(st.integers(), st.text()), min_size=0, max_size=15))
def test_sequential_operations_maintain_invariants(entries: List[Tuple[int, str]]):
    """Sequential insert/delete operations should maintain heap invariants."""
    heap = Heap.empty(int, str)

    # Insert all entries
    for key, value in entries:
        heap = heap.insert(key, value)
        assert not heap.null()
        min_result = heap.find_min()
        assert min_result is not None

    # Delete half the entries
    for _ in range(len(entries) // 2):
        if not heap.null():
            original_size = heap.size()
            delete_result = heap.delete_min()
            assert delete_result is not None
            assert delete_result.size() == original_size - 1
            heap = delete_result


@given(heap_strategy())
def test_meld_with_self(heap):
    """Melding a heap with itself should double the size."""
    melded = heap.meld(heap)
    assert melded.size() == 2 * heap.size()

    if not heap.null():
        # Minimum should be the same
        original_min = heap.find_min()
        melded_min = melded.find_min()
        assert original_min is not None
        assert melded_min is not None
        assert original_min[0] == melded_min[0]
