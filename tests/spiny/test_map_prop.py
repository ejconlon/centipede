"""Property-based tests for PMap using Hypothesis."""

from typing import Dict, List, Tuple

from hypothesis import assume, given
from hypothesis import strategies as st

from spiny.map import PMap
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def map_strategy(
    draw, key_strategy=st.integers(), value_strategy=st.integers()
) -> PMap[int, int]:
    pairs = draw(
        st.lists(st.tuples(key_strategy, value_strategy), min_size=0, max_size=20)
    )
    return PMap.mk(pairs)


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=50))
def test_map_mk_maintains_order_and_deduplicates(pairs: List[Tuple[int, int]]):
    """Creating a PMap from pairs should deduplicate keys and maintain sorted order."""
    pmap = PMap.mk(pairs)

    # Convert to dict to deduplicate (later values overwrite earlier ones)
    expected_dict = {}
    for key, value in pairs:
        expected_dict[key] = value

    expected_pairs = sorted(expected_dict.items())
    actual_pairs = list(pmap.items())

    assert actual_pairs == expected_pairs
    assert pmap.size() == len(expected_dict)
    assert pmap.null() == (len(expected_dict) == 0)


@given(map_strategy(), st.integers(), st.integers())
def test_put_idempotent_with_same_value(pmap, key, value):
    """Putting the same key-value pair multiple times should be idempotent."""
    # Put key-value once
    pmap1 = pmap.put(key, value)

    # Put same key-value again
    pmap2 = pmap1.put(key, value)

    # Should be identical
    assert list(pmap1.items()) == list(pmap2.items())
    assert pmap1.size() == pmap2.size()

    # Check that key-value is present
    assert pmap1.get(key) == value
    assert pmap1.contains(key)


@given(map_strategy(), st.integers(), st.integers())
def test_put_maintains_sorted_order(pmap, key, value):
    """Putting a key-value pair should maintain sorted order by key."""
    new_pmap = pmap.put(key, value)
    keys = list(new_pmap.keys())

    assert keys == sorted(keys)
    assert new_pmap.get(key) == value
    assert new_pmap.contains(key)


@given(map_strategy(), st.integers(), st.integers())
def test_put_increases_size_at_most_one(pmap, key, value):
    """Putting a key-value pair should increase size by at most 1."""
    original_size = pmap.size()
    new_pmap = pmap.put(key, value)
    new_size = new_pmap.size()

    assert new_size <= original_size + 1
    if pmap.contains(key):
        assert new_size == original_size  # Overwrite existing
    else:
        assert new_size == original_size + 1  # Add new


@given(map_strategy(), st.integers(), st.integers(), st.integers())
def test_put_overwrites_existing_key(pmap, key, value1, value2):
    """Putting a key that already exists should overwrite the value."""
    pmap1 = pmap.put(key, value1)
    pmap2 = pmap1.put(key, value2)

    assert pmap2.get(key) == value2
    assert pmap1.size() == pmap2.size()  # Size shouldn't change


@given(map_strategy(), st.integers())
def test_get_nonexistent_key_returns_none(pmap, key):
    """Getting a nonexistent key should return None."""
    assume(not pmap.contains(key))
    assert pmap.lookup(key) is None


@given(map_strategy(), st.integers())
def test_contains_consistency_with_get(pmap, key):
    """contains() should be consistent with lookup() returning non-None."""
    has_key = pmap.contains(key)
    value = pmap.lookup(key)

    assert has_key == (value is not None)


@given(map_strategy(), st.integers())
def test_remove_existing_key(pmap, key):
    """Removing an existing key should remove it from the map."""
    if pmap.contains(key):
        original_size = pmap.size()
        new_pmap = pmap.remove(key)

        assert not new_pmap.contains(key)
        assert new_pmap.lookup(key) is None
        assert new_pmap.size() == original_size - 1


@given(map_strategy(), st.integers())
def test_remove_nonexistent_key_unchanged(pmap, key):
    """Removing a nonexistent key should leave the map unchanged."""
    assume(not pmap.contains(key))

    new_pmap = pmap.remove(key)

    assert list(pmap.items()) == list(new_pmap.items())
    assert pmap.size() == new_pmap.size()


@given(map_strategy(), map_strategy())
def test_merge_contains_all_entries(pmap1, pmap2):
    """Merged map should contain all entries from both maps."""
    merged = pmap1.merge(pmap2)

    # All keys from both maps should be present
    for key in pmap1.keys():
        assert merged.contains(key)
    for key in pmap2.keys():
        assert merged.contains(key)

    # Values from pmap1 should take precedence in case of conflicts
    for key, value in pmap1.items():
        assert merged.get(key) == value


@given(map_strategy())
def test_merge_empty_identity(pmap):
    """Merging with empty should be identity."""
    empty = PMap.empty(int, int)

    merged1 = pmap.merge(empty)
    merged2 = empty.merge(pmap)

    assert list(pmap.items()) == list(merged1.items())
    assert list(pmap.items()) == list(merged2.items())


@given(map_strategy(), map_strategy())
def test_merge_size_bounds(pmap1, pmap2):
    """Merged map size should be bounded correctly."""
    merged = pmap1.merge(pmap2)

    # Size should be at least the size of the larger map
    assert merged.size() >= max(pmap1.size(), pmap2.size())

    # Size should be at most the sum of both sizes
    assert merged.size() <= pmap1.size() + pmap2.size()


@given(map_strategy())
def test_find_min_returns_minimum_key(pmap):
    """find_min should return the entry with minimum key if map is non-empty."""
    result = pmap.find_min()

    if pmap.null():
        assert result is None
    else:
        assert result is not None
        min_key, min_value, remaining = result

        # Should be the minimum key
        keys = list(pmap.keys())
        assert min_key == min(keys)
        assert pmap.get(min_key) == min_value

        # Remaining map should not contain the minimum key
        assert not remaining.contains(min_key)
        assert remaining.size() == pmap.size() - 1


@given(map_strategy())
def test_find_max_returns_maximum_key(pmap):
    """find_max should return the entry with maximum key if map is non-empty."""
    result = pmap.find_max()

    if pmap.null():
        assert result is None
    else:
        assert result is not None
        remaining, max_key, max_value = result

        # Should be the maximum key
        keys = list(pmap.keys())
        assert max_key == max(keys)
        assert pmap.get(max_key) == max_value

        # Remaining map should not contain the maximum key
        assert not remaining.contains(max_key)
        assert remaining.size() == pmap.size() - 1


@given(map_strategy())
def test_find_min_max_consistency_single_entry(pmap):
    """find_min and find_max should be consistent on single-entry maps."""
    if pmap.size() == 1:
        min_result = pmap.find_min()
        max_result = pmap.find_max()

        assert min_result is not None
        assert max_result is not None

        min_key, min_value, min_remaining = min_result
        max_remaining, max_key, max_value = max_result

        assert min_key == max_key
        assert min_value == max_value
        assert min_remaining.null()
        assert max_remaining.null()


@given(map_strategy())
def test_delete_min_consistency(pmap):
    """delete_min should be consistent with find_min."""
    find_result = pmap.find_min()
    delete_result = pmap.delete_min()

    if find_result is None:
        assert delete_result is None
    else:
        assert delete_result is not None
        _, _, remaining_from_find = find_result
        assert list(delete_result.items()) == list(remaining_from_find.items())


@given(map_strategy())
def test_delete_max_consistency(pmap):
    """delete_max should be consistent with find_max."""
    find_result = pmap.find_max()
    delete_result = pmap.delete_max()

    if find_result is None:
        assert delete_result is None
    else:
        assert delete_result is not None
        remaining_from_find, _, _ = find_result
        assert list(delete_result.items()) == list(remaining_from_find.items())


@given(map_strategy())
def test_repeated_find_min_extracts_sorted(pmap):
    """Repeatedly calling find_min should extract entries in key order."""
    extracted_pairs = []
    current = pmap

    while not current.null():
        result = current.find_min()
        assert result is not None
        min_key, min_value, remaining = result
        extracted_pairs.append((min_key, min_value))
        current = remaining

    expected_pairs = list(pmap.items())
    assert extracted_pairs == expected_pairs
    assert current.null()


@given(map_strategy())
def test_repeated_find_max_extracts_reverse_sorted(pmap):
    """Repeatedly calling find_max should extract entries in reverse key order."""
    extracted_pairs = []
    current = pmap

    while not current.null():
        result = current.find_max()
        assert result is not None
        remaining, max_key, max_value = result
        extracted_pairs.append((max_key, max_value))
        current = remaining

    expected_pairs = list(reversed(list(pmap.items())))
    assert extracted_pairs == expected_pairs
    assert current.null()


@given(map_strategy())
def test_persistence_under_operations(pmap):
    """Original map should remain unchanged after operations."""
    original_items = list(pmap.items())
    original_size = pmap.size()

    # Perform various operations
    pmap.put(999, 888)
    pmap.remove(999)
    pmap.merge(PMap.mk([(1000, 1001), (1002, 1003)]))
    pmap.find_min()
    pmap.find_max()
    pmap.delete_min()
    pmap.delete_max()

    # Original should be unchanged
    assert list(pmap.items()) == original_items
    assert pmap.size() == original_size


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=20))
def test_operators_consistency(pairs: List[Tuple[int, int]]):
    """>> and << operators should match put method."""
    pmap1 = PMap.mk(pairs)
    pmap2 = PMap.mk(pairs)

    test_pair = (999, 888)

    # Test >> operator
    pmap1_put = pmap1.put(*test_pair)
    pmap1_op = pmap1 >> test_pair
    assert list(pmap1_put.items()) == list(pmap1_op.items())

    # Test << operator
    pmap2_put = pmap2.put(*test_pair)
    pmap2_op = test_pair << pmap2
    assert list(pmap2_put.items()) == list(pmap2_op.items())


@given(map_strategy(), map_strategy())
def test_addition_operator_merge(pmap1, pmap2):
    """+ operator should match merge method."""
    merge_method = pmap1.merge(pmap2)
    merge_op = pmap1 + pmap2

    assert list(merge_method.items()) == list(merge_op.items())


@given(st.integers(), st.integers())
def test_singleton_properties(key: int, value: int):
    """Singleton map should have expected properties."""
    pmap = PMap.singleton(key, value)

    assert not pmap.null()
    assert pmap.size() == 1
    assert pmap.get(key) == value
    assert pmap.contains(key)
    assert list(pmap.items()) == [(key, value)]

    min_result = pmap.find_min()
    assert min_result is not None
    min_key, min_value, min_remaining = min_result
    assert min_key == key
    assert min_value == value
    assert min_remaining.null()

    max_result = pmap.find_max()
    assert max_result is not None
    max_remaining, max_key, max_value = max_result
    assert max_key == key
    assert max_value == value
    assert max_remaining.null()


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=15))
def test_size_matches_unique_keys(pairs: List[Tuple[int, int]]):
    """Map size should match number of unique keys."""
    pmap = PMap.mk(pairs)
    unique_keys = set(key for key, _ in pairs)

    assert pmap.size() == len(unique_keys)


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=20))
def test_min_max_key_bounds(pairs: List[Tuple[int, int]]):
    """Min and max keys should be actual bounds of the map."""
    pmap = PMap.mk(pairs)
    keys = [key for key, _ in pairs]
    expected_min_key = min(keys)
    expected_max_key = max(keys)

    min_result = pmap.find_min()
    max_result = pmap.find_max()

    assert min_result is not None
    assert max_result is not None

    min_key, _, _ = min_result
    _, max_key, _ = max_result

    assert min_key == expected_min_key
    assert max_key == expected_max_key


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=2, max_size=20))
def test_min_max_removal_bounds(pairs: List[Tuple[int, int]]):
    """After removing min/max, remaining bounds should be correct."""
    unique_pairs = {}
    for key, value in pairs:
        unique_pairs[key] = value
    assume(len(unique_pairs) >= 2)  # Need at least 2 unique keys

    pmap = PMap.mk(pairs)
    sorted_keys = sorted(unique_pairs.keys())

    # Remove minimum
    min_removed = pmap.delete_min()
    assert min_removed is not None
    if min_removed.size() > 0:
        new_min_result = min_removed.find_min()
        assert new_min_result is not None
        new_min_key, _, _ = new_min_result
        assert new_min_key == sorted_keys[1]

    # Remove maximum
    max_removed = pmap.delete_max()
    assert max_removed is not None
    if max_removed.size() > 0:
        new_max_result = max_removed.find_max()
        assert new_max_result is not None
        _, new_max_key, _ = new_max_result
        assert new_max_key == sorted_keys[-2]


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=20))
def test_iteration_methods_consistency(pairs: List[Tuple[int, int]]):
    """keys(), values(), and items() should be consistent with each other."""
    pmap = PMap.mk(pairs)

    keys = list(pmap.keys())
    values = list(pmap.values())
    items = list(pmap.items())

    # Should have same length
    assert len(keys) == len(values) == len(items) == pmap.size()

    # Items should be zip of keys and values
    assert items == list(zip(keys, values))

    # Keys should be sorted
    assert keys == sorted(keys)


@given(map_strategy())
def test_empty_map_properties(pmap):
    """Empty map should have consistent behavior."""
    if pmap.null():
        assert pmap.size() == 0
        assert list(pmap.keys()) == []
        assert list(pmap.values()) == []
        assert list(pmap.items()) == []
        assert pmap.find_min() is None
        assert pmap.find_max() is None
        assert pmap.delete_min() is None
        assert pmap.delete_max() is None


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=100))
def test_large_map_efficiency(pairs: List[Tuple[int, int]]):
    """Large maps should maintain efficiency properties."""
    pmap = PMap.mk(pairs)

    # These operations should complete efficiently
    assert isinstance(pmap.size(), int)
    assert isinstance(pmap.find_min(), (type(None), tuple))
    assert isinstance(pmap.find_max(), (type(None), tuple))

    # Tree should remain reasonably balanced (indirect test)
    # If operations complete without timeout, tree is likely balanced


@given(st.dictionaries(st.integers(), st.integers(), min_size=0, max_size=20))
def test_map_equivalence_with_python_dict(pairs_dict: Dict[int, int]):
    """PMap behavior should match Python dict for basic operations."""
    pairs = list(pairs_dict.items())
    pmap = PMap.mk(pairs)

    assert pmap.size() == len(pairs_dict)
    assert dict(pmap.items()) == pairs_dict
    assert pmap.null() == (len(pairs_dict) == 0)

    for key, value in pairs_dict.items():
        assert pmap.get(key) == value
        assert pmap.contains(key)

    if pairs_dict:
        min_result = pmap.find_min()
        assert min_result is not None
        min_key, _, _ = min_result
        max_result = pmap.find_max()
        assert max_result is not None
        _, max_key, _ = max_result
        assert min_key == min(pairs_dict.keys())
        assert max_key == max(pairs_dict.keys())


@given(
    st.lists(
        st.tuples(st.text(min_size=1, max_size=10), st.integers()),
        min_size=0,
        max_size=15,
    )
)
def test_string_keys(pairs: List[Tuple[str, int]]):
    """Map should work correctly with string keys."""
    pmap = PMap.mk(pairs)

    # Convert to dict to deduplicate
    expected_dict = {}
    for key, value in pairs:
        expected_dict[key] = value
    expected_pairs = sorted(expected_dict.items())

    assert list(pmap.items()) == expected_pairs
    assert pmap.size() == len(expected_dict)

    if expected_pairs:
        min_result = pmap.find_min()
        max_result = pmap.find_max()

        assert min_result is not None
        assert max_result is not None

        min_key, _, _ = min_result
        _, max_key, _ = max_result

        assert min_key == min(expected_dict.keys())
        assert max_key == max(expected_dict.keys())


@given(map_strategy(), st.integers(), st.integers())
def test_chained_operations_consistency(pmap, key, value):
    """Chained operations should maintain consistency."""
    # Chain multiple operations
    result = pmap.put(key, value).put(key + 1, value + 1).put(key - 1, value - 1)

    # Result keys should be sorted
    result_keys = list(result.keys())
    assert result_keys == sorted(result_keys)

    # Original entries should be preserved (unless overwritten)
    for orig_key, orig_value in pmap.items():
        if orig_key not in [key, key + 1, key - 1]:
            assert result.get(orig_key) == orig_value

    # New entries should be present
    assert result.get(key) == value
    assert result.get(key + 1) == value + 1
    assert result.get(key - 1) == value - 1


@given(map_strategy(), map_strategy(), map_strategy())
def test_merge_associative(pmap1, pmap2, pmap3):
    """Merge should be associative: (A + B) + C == A + (B + C)."""
    left_assoc = pmap1.merge(pmap2).merge(pmap3)
    right_assoc = pmap1.merge(pmap2.merge(pmap3))

    assert list(left_assoc.items()) == list(right_assoc.items())


@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=20))
def test_put_remove_round_trip(pairs: List[Tuple[int, int]]):
    """Putting and then removing should be consistent."""
    pmap = PMap.mk(pairs)

    # Add a new key-value pair
    test_key, test_value = 9999, 8888
    assume(not pmap.contains(test_key))

    # Put then remove
    pmap_with = pmap.put(test_key, test_value)
    pmap_without = pmap_with.remove(test_key)

    # Should be back to original
    assert list(pmap.items()) == list(pmap_without.items())
    assert pmap.size() == pmap_without.size()
