"""Property-based tests for PSet using Hypothesis."""

from typing import List, Set

from hypothesis import assume, given
from hypothesis import strategies as st

from spiny.set import PSet
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def set_strategy(
    draw: st.DrawFn, element_strategy: st.SearchStrategy[int] = st.integers()
) -> PSet[int]:
    elements = draw(st.lists(element_strategy, min_size=0, max_size=20))
    return PSet.mk(elements)


@given(st.lists(st.integers(), min_size=0, max_size=50))
def test_set_mk_deduplicates_and_sorts(elements: List[int]) -> None:
    """Creating a PSet from elements should deduplicate and maintain sorted order."""
    pset = PSet.mk(elements)
    unique_elements = sorted(set(elements))

    assert pset.list() == unique_elements
    assert pset.size() == len(unique_elements)
    assert pset.null() == (len(unique_elements) == 0)


@given(set_strategy(), st.integers())
def test_insert_idempotent(pset: PSet[int], element: int) -> None:
    """Inserting the same element multiple times should be idempotent."""
    # Insert element once
    pset1 = pset.insert(element)

    # Insert same element again
    pset2 = pset1.insert(element)

    # Should be identical
    assert pset1.list() == pset2.list()
    assert pset1.size() == pset2.size()

    # Check that element is present
    assert element in pset1.list()


@given(set_strategy(), st.integers())
def test_insert_maintains_sorted_order(pset: PSet[int], element: int) -> None:
    """Inserting an element should maintain sorted order."""
    new_pset = pset.insert(element)
    result = new_pset.list()

    assert result == sorted(result)
    assert element in result


@given(set_strategy(), st.integers())
def test_insert_increases_size_at_most_one(pset: PSet[int], element: int) -> None:
    """Inserting an element should increase size by at most 1."""
    original_size = pset.size()
    new_pset = pset.insert(element)
    new_size = new_pset.size()

    assert new_size <= original_size + 1
    if element in pset.list():
        assert new_size == original_size
    else:
        assert new_size == original_size + 1


@given(set_strategy(), set_strategy())
def test_union_commutative(pset1: PSet[int], pset2: PSet[int]) -> None:
    """Union should be commutative: A + B == B + A."""
    merged1 = pset1.union(pset2)
    merged2 = pset2.union(pset1)

    assert merged1.list() == merged2.list()
    assert merged1.size() == merged2.size()


@given(set_strategy(), set_strategy())
def test_union_associative(pset1: PSet[int], pset2: PSet[int]) -> None:
    """Union should be associative: (A + B) + C == A + (B + C)."""
    pset3 = PSet.mk([100, 200, 300])

    left_assoc = pset1.union(pset2).union(pset3)
    right_assoc = pset1.union(pset2.union(pset3))

    assert left_assoc.list() == right_assoc.list()


@given(set_strategy())
def test_union_empty_identity(pset: PSet[int]) -> None:
    """Union with empty should be identity."""
    empty = PSet.empty(int)

    assert pset.union(empty).list() == pset.list()
    assert empty.union(pset).list() == pset.list()


@given(set_strategy(), set_strategy())
def test_union_contains_all_elements(pset1: PSet[int], pset2: PSet[int]) -> None:
    """Union set should contain all elements from both sets."""
    merged = pset1.union(pset2)

    elements1 = set(pset1.list())
    elements2 = set(pset2.list())
    merged_elements = set(merged.list())

    assert merged_elements == elements1.union(elements2)


@given(set_strategy())
def test_find_min_returns_minimum(pset: PSet[int]) -> None:
    """find_min should return the minimum element if set is non-empty."""
    elements = pset.list()
    result = pset.find_min()

    if pset.null():
        assert result is None
    else:
        assert result is not None
        min_val, remaining = result
        assert min_val == min(elements)
        assert remaining.size() == pset.size() - 1
        assert min_val not in remaining.list()


@given(set_strategy())
def test_find_max_returns_maximum(pset: PSet[int]) -> None:
    """find_max should return the maximum element if set is non-empty."""
    elements = pset.list()
    result = pset.find_max()

    if pset.null():
        assert result is None
    else:
        assert result is not None
        remaining, max_val = result
        assert max_val == max(elements)
        assert remaining.size() == pset.size() - 1
        assert max_val not in remaining.list()


@given(set_strategy())
def test_find_min_max_consistency(pset: PSet[int]) -> None:
    """find_min and find_max should be consistent on single-element sets."""
    if pset.size() == 1:
        min_result = pset.find_min()
        max_result = pset.find_max()

        assert min_result is not None
        assert max_result is not None

        min_val, min_remaining = min_result
        max_remaining, max_val = max_result

        assert min_val == max_val
        assert min_remaining.null()
        assert max_remaining.null()


@given(set_strategy())
def test_delete_min_consistency(pset: PSet[int]) -> None:
    """delete_min should be consistent with find_min."""
    find_result = pset.find_min()
    delete_result = pset.delete_min()

    if find_result is None:
        assert delete_result is None
    else:
        assert delete_result is not None
        _, remaining_from_find = find_result
        assert delete_result.list() == remaining_from_find.list()


@given(set_strategy())
def test_delete_max_consistency(pset: PSet[int]) -> None:
    """delete_max should be consistent with find_max."""
    find_result = pset.find_max()
    delete_result = pset.delete_max()

    if find_result is None:
        assert delete_result is None
    else:
        assert delete_result is not None
        remaining_from_find, _ = find_result
        assert delete_result.list() == remaining_from_find.list()


@given(set_strategy())
def test_repeated_find_min_extracts_sorted(pset: PSet[int]) -> None:
    """Repeatedly calling find_min should extract elements in sorted order."""
    elements = pset.list()
    extracted = []
    current = pset

    while not current.null():
        result = current.find_min()
        assert result is not None
        min_val, remaining = result
        extracted.append(min_val)
        current = remaining

    assert extracted == elements
    assert current.null()


@given(set_strategy())
def test_repeated_find_max_extracts_reverse_sorted(pset: PSet[int]) -> None:
    """Repeatedly calling find_max should extract elements in reverse sorted order."""
    elements = pset.list()
    extracted = []
    current = pset

    while not current.null():
        result = current.find_max()
        assert result is not None
        remaining, max_val = result
        extracted.append(max_val)
        current = remaining

    assert extracted == list(reversed(elements))
    assert current.null()


@given(set_strategy())
def test_persistence_under_operations(pset: PSet[int]) -> None:
    """Original set should remain unchanged after operations."""
    original_list = pset.list()
    original_size = pset.size()

    # Perform various operations
    pset.insert(999)
    pset.union(PSet.mk([1000, 1001]))
    pset.find_min()
    pset.find_max()
    pset.delete_min()
    pset.delete_max()

    # Original should be unchanged
    assert pset.list() == original_list
    assert pset.size() == original_size


@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_operators_consistency(elements: List[int]) -> None:
    """>> and << operators should match insert method."""
    pset1 = PSet.mk(elements)
    pset2 = PSet.mk(elements)

    test_elem = 999

    # Test >> operator
    pset1_insert = pset1.insert(test_elem)
    pset1_op = pset1 >> test_elem
    assert pset1_insert.list() == pset1_op.list()

    # Test << operator
    pset2_insert = pset2.insert(test_elem)
    pset2_op = test_elem << pset2
    assert pset2_insert.list() == pset2_op.list()


@given(set_strategy(), set_strategy())
def test_or_operator_union(pset1: PSet[int], pset2: PSet[int]) -> None:
    """| operator should match union method."""
    union_method = pset1.union(pset2)
    union_op = pset1 | pset2

    assert union_method.list() == union_op.list()


@given(st.integers())
def test_singleton_properties(value: int) -> None:
    """Singleton set should have expected properties."""
    pset = PSet.singleton(value)

    assert not pset.null()
    assert pset.size() == 1
    assert pset.list() == [value]

    min_result = pset.find_min()
    assert min_result is not None
    min_val, min_remaining = min_result
    assert min_val == value
    assert min_remaining.null()

    max_result = pset.find_max()
    assert max_result is not None
    max_remaining, max_val = max_result
    assert max_val == value
    assert max_remaining.null()


@given(st.lists(st.integers(), min_size=0, max_size=15))
def test_size_matches_unique_elements(elements: List[int]) -> None:
    """Set size should match number of unique elements."""
    pset = PSet.mk(elements)
    unique_count = len(set(elements))

    assert pset.size() == unique_count


@given(st.lists(st.integers(), min_size=1, max_size=20))
def test_min_max_bounds(elements: List[int]) -> None:
    """Min and max elements should be actual bounds of the set."""
    pset = PSet.mk(elements)
    expected_min = min(elements)
    expected_max = max(elements)

    min_result = pset.find_min()
    max_result = pset.find_max()

    assert min_result is not None
    assert max_result is not None

    min_val, _ = min_result
    _, max_val = max_result

    assert min_val == expected_min
    assert max_val == expected_max


@given(st.lists(st.integers(), min_size=2, max_size=20))
def test_min_max_removal_bounds(elements: List[int]) -> None:
    """After removing min/max, remaining bounds should be correct."""
    assume(len(set(elements)) >= 2)  # Need at least 2 unique elements

    pset = PSet.mk(elements)
    unique_elements = sorted(set(elements))

    # Remove minimum
    min_removed = pset.delete_min()
    assert min_removed is not None
    if min_removed.size() > 0:
        new_min_result = min_removed.find_min()
        assert new_min_result is not None
        new_min, _ = new_min_result
        assert new_min == unique_elements[1]

    # Remove maximum
    max_removed = pset.delete_max()
    assert max_removed is not None
    if max_removed.size() > 0:
        new_max_result = max_removed.find_max()
        assert new_max_result is not None
        _, new_max = new_max_result
        assert new_max == unique_elements[-2]


@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_list_iter_consistency(elements: List[int]) -> None:
    """list() and iter() should produce the same elements in the same order."""
    pset = PSet.mk(elements)

    list_result = pset.list()
    iter_result = list(pset.iter())

    assert list_result == iter_result


@given(set_strategy())
def test_empty_set_properties(pset: PSet[int]) -> None:
    """Empty set should have consistent behavior."""
    if pset.null():
        assert pset.size() == 0
        assert pset.list() == []
        assert pset.find_min() is None
        assert pset.find_max() is None
        assert pset.delete_min() is None
        assert pset.delete_max() is None


@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_large_set_efficiency(elements: List[int]) -> None:
    """Large sets should maintain efficiency properties."""
    pset = PSet.mk(elements)

    # These operations should complete efficiently
    assert isinstance(pset.size(), int)
    assert isinstance(pset.find_min(), (type(None), tuple))
    assert isinstance(pset.find_max(), (type(None), tuple))

    # Tree should remain reasonably balanced (indirect test)
    # If operations complete without timeout, tree is likely balanced


@given(st.sets(st.integers(), min_size=0, max_size=20))
def test_set_equivalence_with_python_set(elements: Set[int]) -> None:
    """PSet behavior should match Python set for basic operations."""
    pset = PSet.mk(elements)

    assert pset.size() == len(elements)
    assert set(pset.list()) == elements
    assert pset.null() == (len(elements) == 0)

    if elements:
        min_result = pset.find_min()
        assert min_result is not None
        min_val, _ = min_result
        max_result = pset.find_max()
        assert max_result is not None
        _, max_val = max_result
        assert min_val == min(elements)
        assert max_val == max(elements)


@given(st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=15))
def test_string_elements(elements: List[str]) -> None:
    """Set should work correctly with string elements."""
    pset = PSet.mk(elements)
    unique_elements = sorted(set(elements))

    assert pset.list() == unique_elements
    assert pset.size() == len(unique_elements)

    if unique_elements:
        min_result = pset.find_min()
        max_result = pset.find_max()

        assert min_result is not None
        assert max_result is not None

        min_val, _ = min_result
        _, max_val = max_result

        assert min_val == min(unique_elements)
        assert max_val == max(unique_elements)


@given(set_strategy(), st.integers())
def test_chained_operations_consistency(pset: PSet[int], element: int) -> None:
    """Chained operations should maintain consistency."""
    # Chain multiple operations
    result = pset.insert(element).insert(element + 1).insert(element - 1)

    # Result should be sorted and contain all elements
    result_list = result.list()
    assert result_list == sorted(result_list)

    # Original elements should be preserved
    for orig_elem in pset.list():
        assert orig_elem in result_list

    # New elements should be present
    assert element in result_list
    assert (element + 1) in result_list
    assert (element - 1) in result_list
