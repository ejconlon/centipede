"""Property-based tests for PHeap using Hypothesis."""

from typing import List

from hypothesis import given
from hypothesis import strategies as st

from spiny.heap import PHeap
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def heap_strategy(
    draw: st.DrawFn, value_strategy: st.SearchStrategy[int] = st.integers()
) -> PHeap[int]:
    """Generate a PHeap with random values."""
    values = draw(st.lists(value_strategy, min_size=0, max_size=20))
    return PHeap.mk(values)


@st.composite
def heap_with_values_strategy(
    draw: st.DrawFn, value_strategy: st.SearchStrategy[int] = st.integers()
) -> tuple[PHeap[int], List[int]]:
    """Generate a PHeap along with the values used to create it."""
    values = draw(st.lists(value_strategy, min_size=0, max_size=20))
    return PHeap.mk(values), values


@given(st.lists(st.integers(), min_size=0, max_size=50))
def test_heap_mk_size_consistency(values: List[int]) -> None:
    """Creating a PHeap from values should have correct size."""
    heap = PHeap.mk(values)
    assert heap.size() == len(values)
    assert heap.null() == (len(values) == 0)


@given(heap_strategy())
def test_exhaustive_heap_operations(heap: PHeap[int]) -> None:
    """Test that all elements can be extracted from the heap."""
    original_size = heap.size()
    extracted = []
    current = heap

    while not current.null():
        min_result = current.find_min()
        assert min_result is not None
        value, remaining = min_result

        # Check delete_min consistency
        delete_result = current.delete_min()
        assert delete_result is not None
        assert delete_result.size() == remaining.size()

        extracted.append(value)
        current = delete_result

    # Should have extracted all elements
    assert len(extracted) == original_size
    assert current.null()
    assert current.find_min() is None
    assert current.delete_min() is None


@given(heap_strategy(), st.integers())
def test_insert_increases_size(heap: PHeap[int], value: int) -> None:
    """Inserting an element should increase size by 1."""
    original_size = heap.size()
    new_heap = heap.insert(value)

    assert new_heap.size() == original_size + 1
    assert not new_heap.null()


@given(heap_strategy(), st.integers())
def test_insert_preserves_min_heap_property(heap: PHeap[int], value: int) -> None:
    """After insertion, find_min should return a valid minimum."""
    new_heap = heap.insert(value)

    if heap.null():
        # If original was empty, new minimum should be the inserted element
        min_result = new_heap.find_min()
        assert min_result is not None
        min_value, _ = min_result
        assert min_value == value
    else:
        # New minimum should be <= inserted value and <= original minimum
        original_min = heap.find_min()
        assert original_min is not None
        original_min_value, _ = original_min

        new_min = new_heap.find_min()
        assert new_min is not None
        new_min_value, _ = new_min

        assert new_min_value <= value
        assert new_min_value <= original_min_value


@given(heap_with_values_strategy())
def test_find_min_returns_actual_minimum(
    heap_and_values: tuple[PHeap[int], List[int]],
) -> None:
    """find_min should return the actual minimum value from all values."""
    heap, values = heap_and_values

    if not values:
        assert heap.find_min() is None
    else:
        expected_min_value = min(values)
        min_result = heap.find_min()
        assert min_result is not None
        actual_min_value, _ = min_result
        assert actual_min_value == expected_min_value


@given(heap_strategy())
def test_find_min_delete_min_consistency(heap: PHeap[int]) -> None:
    """find_min and delete_min should be consistent."""
    if heap.null():
        assert heap.find_min() is None
        assert heap.delete_min() is None
    else:
        find_result = heap.find_min()
        delete_result = heap.delete_min()

        assert find_result is not None
        assert delete_result is not None

        _, remaining_from_find = find_result

        assert remaining_from_find.size() == delete_result.size()
        assert remaining_from_find.size() == heap.size() - 1


@given(heap_strategy())
def test_delete_min_decreases_size(heap: PHeap[int]) -> None:
    """delete_min should decrease size by 1 for non-empty heaps."""
    if heap.null():
        assert heap.delete_min() is None
    else:
        original_size = heap.size()
        result = heap.delete_min()
        assert result is not None
        assert result.size() == original_size - 1


@given(heap_strategy(), heap_strategy())
def test_merge_size_additive(heap1: PHeap[int], heap2: PHeap[int]) -> None:
    """mergeed heap size should equal sum of individual sizes."""
    mergeed = heap1.merge(heap2)
    assert mergeed.size() == heap1.size() + heap2.size()


@given(heap_strategy())
def test_merge_empty_identity(heap1: PHeap[int]) -> None:
    """merging with empty heap should be identity."""
    empty = PHeap.empty(int)

    assert heap1.merge(empty).size() == heap1.size()
    assert empty.merge(heap1).size() == heap1.size()


@given(heap_strategy(), heap_strategy())
def test_merge_preserves_min_heap_property(
    heap1: PHeap[int], heap2: PHeap[int]
) -> None:
    """mergeed heap should maintain min-heap property."""
    mergeed = heap1.merge(heap2)

    if heap1.null() and heap2.null():
        assert mergeed.null()
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

        mergeed_min = mergeed.find_min()
        assert mergeed_min is not None
        assert mergeed_min[0] == expected_overall_min


@given(heap_strategy(), heap_strategy())
def test_merge_associative(heap1: PHeap[int], heap2: PHeap[int]) -> None:
    """merge should be associative: (a + b) + c == a + (b + c)."""
    heap3 = PHeap.mk([100, 200])

    left_assoc = heap1.merge(heap2).merge(heap3)
    right_assoc = heap1.merge(heap2.merge(heap3))

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
def test_iter_sorted_order(heap: PHeap[int]) -> None:
    """Iterating through heap should yield elements in sorted order."""
    values = list(heap.iter())

    assert len(values) == heap.size()
    assert values == sorted(values)


@given(heap_strategy())
def test_iter_exhausts_heap_elements(heap: PHeap[int]) -> None:
    """Iterating should yield all elements that were in the heap."""
    values = list(heap.iter())

    # Check that we can extract the same elements by repeated delete_min
    extracted = []
    current = heap
    while not current.null():
        min_result = current.find_min()
        assert min_result is not None
        value, remaining = min_result
        extracted.append(value)
        current = remaining

    assert len(values) == len(extracted)
    assert values == extracted


@given(heap_with_values_strategy())
def test_iter_contains_all_inserted_elements(
    heap_and_values: tuple[PHeap[int], List[int]],
) -> None:
    """Iteration should contain all originally inserted values."""
    heap, values = heap_and_values

    heap_values = list(heap.iter())
    heap_values_sorted = sorted(heap_values)
    values_sorted = sorted(values)

    assert heap_values_sorted == values_sorted


@given(st.integers())
def test_singleton_properties(value: int) -> None:
    """Singleton heap should have expected properties."""
    heap = PHeap.singleton(value)

    assert not heap.null()
    assert heap.size() == 1

    min_result = heap.find_min()
    assert min_result is not None
    min_value, remaining = min_result
    assert min_value == value
    assert remaining.null()

    delete_result = heap.delete_min()
    assert delete_result is not None
    assert delete_result.null()

    values = list(heap.iter())
    assert values == [value]


@given(heap_strategy())
def test_multiple_delete_min_maintains_order(heap: PHeap[int]) -> None:
    """Repeated delete_min should maintain sorted order."""
    extracted_values = []
    current = heap

    while not current.null():
        min_result = current.find_min()
        assert min_result is not None
        value, remaining = min_result
        extracted_values.append(value)

        delete_result = current.delete_min()
        assert delete_result is not None
        assert delete_result.size() == remaining.size()

        current = remaining

    # Extracted values should be in non-decreasing order
    assert extracted_values == sorted(extracted_values)


@given(heap_strategy(), st.integers())
def test_insert_then_delete_min_with_duplicates(heap: PHeap[int], value: int) -> None:
    """Insert then delete_min should handle duplicates correctly."""
    # Insert the same value multiple times
    heap_with_dups = heap.insert(value).insert(value).insert(value)

    original_size = heap.size()
    assert heap_with_dups.size() == original_size + 3

    # The minimum should be no greater than our inserted value
    min_result = heap_with_dups.find_min()
    assert min_result is not None
    min_value, _ = min_result
    assert min_value <= value


@given(heap_strategy())
def test_heap_immutability(heap: PHeap[int]) -> None:
    """Operations should not modify the original heap."""
    original_size = heap.size()
    original_null = heap.null()

    # Perform various operations
    heap.insert(999)
    heap.merge(PHeap.singleton(1000))
    heap.delete_min()
    heap.find_min()
    list(heap.iter())

    # Original heap should be unchanged
    assert heap.size() == original_size
    assert heap.null() == original_null


@given(heap_strategy(), heap_strategy())
def test_addition_operator_equals_merge(heap1: PHeap[int], heap2: PHeap[int]) -> None:
    """+ operator should be equivalent to merge method."""
    merge_result = heap1.merge(heap2)
    add_result = heap1 + heap2

    assert merge_result.size() == add_result.size()

    # Both should have same minimum
    merge_min = merge_result.find_min()
    add_min = add_result.find_min()

    if merge_min is None:
        assert add_min is None
    else:
        assert add_min is not None
        assert merge_min[0] == add_min[0]


@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_heap_sort_property(values: List[int]) -> None:
    """Using heap as priority queue should sort elements by value."""
    heap = PHeap.mk(values)

    sorted_by_heap = list(heap.iter())

    # Should extract all elements
    assert len(sorted_by_heap) == len(values)

    # values should be in non-decreasing order
    assert sorted_by_heap == sorted(sorted_by_heap)

    # Should contain the same elements
    assert sorted(sorted_by_heap) == sorted(values)


@given(heap_strategy())
def test_find_min_idempotent(heap: PHeap[int]) -> None:
    """Multiple calls to find_min should return consistent values."""
    result1 = heap.find_min()
    result2 = heap.find_min()
    result3 = heap.find_min()

    if result1 is None:
        assert result2 is None and result3 is None
    else:
        assert result2 is not None and result3 is not None
        # values should be the same, though heap objects may differ
        assert result1[0] == result2[0] == result3[0]


@given(heap_with_values_strategy())
def test_size_matches_value_count(
    heap_and_values: tuple[PHeap[int], List[int]],
) -> None:
    """PHeap size should match the number of values inserted."""
    heap, values = heap_and_values
    assert heap.size() == len(values)


@given(st.lists(st.integers(), min_size=0, max_size=15))
def test_sequential_operations_maintain_invariants(values: List[int]) -> None:
    """Sequential insert/delete operations should maintain heap invariants."""
    heap = PHeap.empty(int)

    # Insert all values
    for value in values:
        heap = heap.insert(value)
        assert not heap.null()
        min_result = heap.find_min()
        assert min_result is not None

    # Delete half the values
    for _ in range(len(values) // 2):
        if not heap.null():
            original_size = heap.size()
            delete_result = heap.delete_min()
            assert delete_result is not None
            assert delete_result.size() == original_size - 1
            heap = delete_result


@given(heap_strategy())
def test_merge_with_self(heap: PHeap[int]) -> None:
    """merging a heap with itself should double the size."""
    mergeed = heap.merge(heap)
    assert mergeed.size() == 2 * heap.size()

    if not heap.null():
        # Minimum should be the same
        original_min = heap.find_min()
        mergeed_min = mergeed.find_min()
        assert original_min is not None
        assert mergeed_min is not None
        assert original_min[0] == mergeed_min[0]
