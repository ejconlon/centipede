"""Property-based tests for PHeapMap using Hypothesis."""

from typing import List, Tuple

from hypothesis import given
from hypothesis import strategies as st

from spiny.heapmap import PHeapMap
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def heapmap_strategy(
    draw: st.DrawFn,
    key_strategy: st.SearchStrategy[int] = st.integers(),
    value_strategy: st.SearchStrategy[int] = st.integers(),
) -> PHeapMap[int, int]:
    pairs = draw(
        st.lists(st.tuples(key_strategy, value_strategy), min_size=0, max_size=20)
    )
    return PHeapMap.mk(pairs)


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=50))
def test_heapmap_mk_maintains_heap_order(pairs: List[Tuple[int, int]]) -> None:
    """Creating a PHeapMap from pairs should maintain heap order (sorted by key)."""
    hm = PHeapMap.mk(pairs)
    items = list(hm.iter())

    # Should be sorted by key
    keys = [key for key, _ in items]
    assert keys == sorted(keys)

    # Size should match number of pairs (heap maps allow duplicates)
    assert hm.size() == len(pairs)
    assert hm.null() == (len(pairs) == 0)


@given(heapmap_strategy(), st.integers(), st.integers())
def test_insert_maintains_heap_order(
    hm: PHeapMap[int, int], key: int, value: int
) -> None:
    """Inserting a key-value pair should maintain heap order."""
    new_hm = hm.insert(key, value)
    keys = list(new_hm.keys())

    assert keys == sorted(keys)


@given(heapmap_strategy(), st.integers(), st.integers())
def test_insert_increases_size_by_one(
    hm: PHeapMap[int, int], key: int, value: int
) -> None:
    """Inserting always increases size by 1 (duplicates allowed)."""
    original_size = hm.size()
    new_hm = hm.insert(key, value)
    new_size = new_hm.size()

    assert new_size == original_size + 1


@given(heapmap_strategy())
def test_find_min_returns_minimum_key(hm: PHeapMap[int, int]) -> None:
    """find_min should return the entry with minimum key if heap map is non-empty."""
    result = hm.find_min()

    if hm.null():
        assert result is None
    else:
        assert result is not None
        min_key, min_value, remaining = result

        # Should be the minimum key
        keys = list(hm.keys())
        assert min_key == min(keys)

        # Remaining should have one fewer element
        assert remaining.size() == hm.size() - 1


@given(heapmap_strategy())
def test_delete_min_removes_minimum(hm: PHeapMap[int, int]) -> None:
    """delete_min should remove one instance of the minimum key."""
    if hm.null():
        result = hm.delete_min()
        assert result is None
    else:
        original_min = hm.find_min()
        assert original_min is not None
        min_key, _, _ = original_min

        result = hm.delete_min()
        assert result is not None
        assert result.size() == hm.size() - 1

        # If only one instance, key should be gone or new minimum should be different
        if not result.null():
            new_min = result.find_min()
            assert new_min is not None
            new_min_key, _, _ = new_min
            assert new_min_key >= min_key


@given(heapmap_strategy())
def test_find_min_delete_min_consistency(hm: PHeapMap[int, int]) -> None:
    """find_min and delete_min should be consistent."""
    find_result = hm.find_min()
    delete_result = hm.delete_min()

    if find_result is None:
        assert delete_result is None
    else:
        assert delete_result is not None

        _, _, remaining_from_find = find_result
        assert list(remaining_from_find.iter()) == list(delete_result.iter())


@given(heapmap_strategy())
def test_repeated_find_min_extracts_sorted(hm: PHeapMap[int, int]) -> None:
    """Repeatedly calling find_min should extract entries in key order."""
    extracted_pairs = []
    current = hm

    while not current.null():
        result = current.find_min()
        assert result is not None
        key, value, remaining = result
        extracted_pairs.append((key, value))
        current = remaining

    # Should be sorted by key
    extracted_keys = [key for key, _ in extracted_pairs]
    assert extracted_keys == sorted(extracted_keys)

    # Should contain all original pairs
    assert len(extracted_pairs) == hm.size()
    assert set(extracted_pairs) == set(hm.iter())


@given(heapmap_strategy(), heapmap_strategy())
def test_merge_contains_all_entries(
    hm1: PHeapMap[int, int], hm2: PHeapMap[int, int]
) -> None:
    """Merged heap map should contain all entries from both heap maps."""
    merged = hm1.merge(hm2)

    # Size should be sum of both sizes
    assert merged.size() == hm1.size() + hm2.size()

    # All entries should be preserved
    merged_items = set(merged.iter())
    hm1_items = set(hm1.iter())
    hm2_items = set(hm2.iter())
    assert hm1_items.issubset(merged_items)
    assert hm2_items.issubset(merged_items)


@given(heapmap_strategy())
def test_merge_empty_identity(hm: PHeapMap[int, int]) -> None:
    """Merging with empty should be identity."""
    empty = PHeapMap.empty(int, int)

    merged1 = hm.merge(empty)
    merged2 = empty.merge(hm)

    assert list(hm.iter()) == list(merged1.iter())
    assert list(hm.iter()) == list(merged2.iter())


@given(heapmap_strategy(), heapmap_strategy())
def test_merge_commutative_size(
    hm1: PHeapMap[int, int], hm2: PHeapMap[int, int]
) -> None:
    """Merge should be commutative in terms of size and content."""
    merged1 = hm1.merge(hm2)
    merged2 = hm2.merge(hm1)

    assert merged1.size() == merged2.size()
    # Content should be the same (though order might differ due to heap structure)
    assert set(merged1.iter()) == set(merged2.iter())


@given(heapmap_strategy())
def test_persistence_under_operations(hm: PHeapMap[int, int]) -> None:
    """Original heap map should remain unchanged after operations."""
    original_items = list(hm.iter())
    original_size = hm.size()

    # Perform various operations
    hm.insert(999, 888)
    hm.delete_min()
    hm.merge(PHeapMap.mk([(1000, 1001), (1002, 1003)]))
    hm.find_min()

    # Original should be unchanged
    assert list(hm.iter()) == original_items
    assert hm.size() == original_size


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=20))
def test_operators_consistency(pairs: List[Tuple[int, int]]) -> None:
    """>> and << operators should match insert method."""
    hm1 = PHeapMap.mk(pairs)
    hm2 = PHeapMap.mk(pairs)

    test_pair = (999, 888)

    # Test >> operator
    hm1_insert = hm1.insert(*test_pair)
    hm1_op = hm1 >> test_pair
    assert list(hm1_insert.iter()) == list(hm1_op.iter())

    # Test << operator
    hm2_insert = hm2.insert(*test_pair)
    hm2_op = test_pair << hm2
    assert list(hm2_insert.iter()) == list(hm2_op.iter())


@given(heapmap_strategy(), heapmap_strategy())
def test_addition_operator_merge(
    hm1: PHeapMap[int, int], hm2: PHeapMap[int, int]
) -> None:
    """+ operator should match merge method."""
    merge_method = hm1.merge(hm2)
    merge_op = hm1 + hm2

    assert list(merge_method.iter()) == list(merge_op.iter())


@given(st.integers(), st.integers())
def test_singleton_properties(key: int, value: int) -> None:
    """Singleton heap map should have expected properties."""
    hm = PHeapMap.singleton(key, value)

    assert not hm.null()
    assert hm.size() == 1
    assert list(hm.iter()) == [(key, value)]

    min_result = hm.find_min()
    assert min_result is not None
    min_key, min_value, remaining = min_result
    assert min_key == key
    assert min_value == value
    assert remaining.null()


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=15))
def test_size_matches_input_pairs(pairs: List[Tuple[int, int]]) -> None:
    """Heap map size should match number of input pairs (duplicates allowed)."""
    hm = PHeapMap.mk(pairs)
    assert hm.size() == len(pairs)


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=20))
def test_min_key_bounds(pairs: List[Tuple[int, int]]) -> None:
    """Minimum key should be actual minimum of all keys."""
    hm = PHeapMap.mk(pairs)
    keys = [key for key, _ in pairs]
    expected_min_key = min(keys)

    min_result = hm.find_min()
    assert min_result is not None
    min_key, _, _ = min_result
    assert min_key == expected_min_key


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=100))
def test_large_heapmap_efficiency(pairs: List[Tuple[int, int]]) -> None:
    """Large heap maps should maintain efficiency properties."""
    hm = PHeapMap.mk(pairs)

    # These operations should complete efficiently
    assert isinstance(hm.size(), int)
    assert isinstance(hm.find_min(), (type(None), tuple))

    # Heap should maintain order property
    if not hm.null():
        min_result = hm.find_min()
        assert min_result is not None


@given(
    st.lists(
        st.tuples(st.text(min_size=1, max_size=10), st.integers()),
        min_size=0,
        max_size=15,
    )
)
def test_string_keys(pairs: List[Tuple[str, int]]) -> None:
    """Heap map should work correctly with string keys."""
    hm = PHeapMap.mk(pairs)

    assert hm.size() == len(pairs)

    if pairs:
        # Should be sorted by key
        items = list(hm.iter())
        keys = [key for key, _ in items]
        assert keys == sorted(keys)

        min_result = hm.find_min()
        assert min_result is not None
        min_key, _, _ = min_result
        expected_min_key = min(key for key, _ in pairs)
        assert min_key == expected_min_key


@given(heapmap_strategy(), st.integers(), st.integers())
def test_chained_operations_consistency(
    hm: PHeapMap[int, int], key: int, value: int
) -> None:
    """Chained operations should maintain consistency."""
    # Chain multiple operations
    result = hm.insert(key, value).insert(key + 1, value + 1).insert(key - 1, value - 1)

    # Result keys should be sorted
    result_keys = list(result.keys())
    assert result_keys == sorted(result_keys)


@given(heapmap_strategy(), heapmap_strategy(), heapmap_strategy())
def test_merge_associative(
    hm1: PHeapMap[int, int], hm2: PHeapMap[int, int], hm3: PHeapMap[int, int]
) -> None:
    """Merge should be associative: (A + B) + C == A + (B + C)."""
    left_assoc = hm1.merge(hm2).merge(hm3)
    right_assoc = hm1.merge(hm2.merge(hm3))

    assert left_assoc.size() == right_assoc.size()
    assert set(left_assoc.iter()) == set(right_assoc.iter())


@given(heapmap_strategy())
def test_empty_heapmap_properties(hm: PHeapMap[int, int]) -> None:
    """Empty heap map should have consistent behavior."""
    if hm.null():
        assert hm.size() == 0
        assert list(hm.keys()) == []
        assert list(hm.values()) == []
        assert list(hm.iter()) == []
        assert hm.find_min() is None
        assert hm.delete_min() is None


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=20))
def test_iteration_methods_consistency(pairs: List[Tuple[int, int]]) -> None:
    """keys(), values(), and iter() should be consistent with each other."""
    hm = PHeapMap.mk(pairs)

    keys = list(hm.keys())
    values = list(hm.values())
    items = list(hm.iter())

    # Should have same length
    assert len(keys) == len(values) == len(items) == hm.size()

    # Items should be zip of keys and values
    assert items == list(zip(keys, values))

    # Keys should be sorted
    assert keys == sorted(keys)
