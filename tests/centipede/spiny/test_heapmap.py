"""Tests for PHeapMap implementation."""

from typing import List

from centipede.spiny.heapmap import PHeapMap


class TestPHeapMapBasics:
    def test_empty_heapmap(self):
        """Test empty heap map creation and properties."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        assert hm.null()
        assert hm.size() == 0
        assert list(hm.iter()) == []
        assert list(hm.keys()) == []
        assert list(hm.values()) == []

    def test_singleton_insert(self):
        """Test inserting a single element."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        hm1 = hm.insert(1, "a")

        assert not hm1.null()
        assert hm1.size() == 1
        assert list(hm1.iter()) == [(1, "a")]
        assert list(hm1.keys()) == [1]
        assert list(hm1.values()) == ["a"]

    def test_singleton_method(self):
        """Test singleton creation method."""
        hm = PHeapMap.singleton(5, "hello")

        assert not hm.null()
        assert hm.size() == 1
        assert list(hm.iter()) == [(5, "hello")]

    def test_multiple_inserts(self):
        """Test inserting multiple elements."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        hm1 = hm.insert(3, "c").insert(1, "a").insert(2, "b")

        assert hm1.size() == 3
        # Heap order: should be sorted by key
        assert list(hm1.iter()) == [(1, "a"), (2, "b"), (3, "c")]

    def test_duplicate_keys(self):
        """Test that heap maps can contain duplicate keys."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        hm1 = hm.insert(1, "first").insert(1, "second").insert(1, "third")

        assert hm1.size() == 3
        items = list(hm1.iter())
        # All should have key 1, but maintain heap order
        assert all(key == 1 for key, _ in items)
        assert len(items) == 3

    def test_mk_from_iterable(self):
        """Test creating heap map from iterable."""
        pairs = [(3, "c"), (1, "a"), (2, "b")]
        hm = PHeapMap.mk(pairs)

        assert hm.size() == 3
        assert list(hm.iter()) == [(1, "a"), (2, "b"), (3, "c")]

    def test_mk_empty_iterable(self):
        """Test creating heap map from empty iterable."""
        hm: PHeapMap[int, str] = PHeapMap.mk([])
        assert hm.null()
        assert hm.size() == 0


class TestPHeapMapHeapOperations:
    def test_find_min_empty(self):
        """Test find_min on empty heap map."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        assert hm.find_min() is None

    def test_find_min_single(self):
        """Test find_min on single-element heap map."""
        hm = PHeapMap.singleton(5, "value")
        result = hm.find_min()
        assert result is not None
        min_key, min_value, remaining = result
        assert min_key == 5
        assert min_value == "value"
        assert remaining.null()

    def test_find_min_multiple(self):
        """Test find_min on multiple-element heap map."""
        hm = PHeapMap.mk([(3, "c"), (1, "a"), (5, "e"), (2, "b")])
        result = hm.find_min()
        assert result is not None
        min_key, min_value, remaining = result
        assert min_key == 1
        assert min_value == "a"
        assert remaining.size() == 3

    def test_delete_min_empty(self):
        """Test delete_min on empty heap map."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        result = hm.delete_min()
        assert result is None

    def test_delete_min_single(self):
        """Test delete_min on single-element heap map."""
        hm = PHeapMap.singleton(5, "value")
        result = hm.delete_min()
        assert result is not None
        assert result.null()

    def test_delete_min_multiple(self):
        """Test delete_min on multiple-element heap map."""
        hm = PHeapMap.mk([(3, "c"), (1, "a"), (5, "e"), (2, "b")])
        result = hm.delete_min()

        assert result is not None
        assert result.size() == 3
        next_min = result.find_min()
        assert next_min is not None
        assert next_min[0] == 2  # Next minimum key

    def test_find_min_consistency_with_delete_min(self):
        """Test that find_min and delete_min are consistent."""
        hm = PHeapMap.mk([(3, "c"), (1, "a"), (5, "e"), (2, "b")])

        find_result = hm.find_min()
        delete_result = hm.delete_min()

        assert find_result is not None
        assert delete_result is not None

        _, _, remaining_from_find = find_result
        assert list(remaining_from_find.iter()) == list(delete_result.iter())

    def test_repeated_find_min(self):
        """Test repeatedly extracting minimum elements."""
        hm = PHeapMap.mk([(3, "c"), (1, "a"), (5, "e"), (2, "b")])
        extracted = []
        current = hm

        while not current.null():
            result = current.find_min()
            assert result is not None
            key, value, remaining = result
            extracted.append((key, value))
            current = remaining

        assert extracted == [(1, "a"), (2, "b"), (3, "c"), (5, "e")]


class TestPHeapMapMergeOperations:
    def test_merge_empty_maps(self):
        """Test merging two empty heap maps."""
        hm1: PHeapMap[int, str] = PHeapMap.empty()
        hm2: PHeapMap[int, str] = PHeapMap.empty()
        result = hm1.merge(hm2)

        assert result.null()

    def test_merge_empty_with_non_empty(self):
        """Test merging empty with non-empty heap map."""
        hm1: PHeapMap[int, str] = PHeapMap.empty()
        hm2 = PHeapMap.mk([(1, "a"), (2, "b")])

        result1 = hm1.merge(hm2)
        result2 = hm2.merge(hm1)

        assert list(result1.iter()) == [(1, "a"), (2, "b")]
        assert list(result2.iter()) == [(1, "a"), (2, "b")]

    def test_merge_disjoint_maps(self):
        """Test merging heap maps with disjoint key sets."""
        hm1 = PHeapMap.mk([(1, "a"), (3, "c")])
        hm2 = PHeapMap.mk([(2, "b"), (4, "d")])
        result = hm1.merge(hm2)

        assert result.size() == 4
        assert list(result.iter()) == [(1, "a"), (2, "b"), (3, "c"), (4, "d")]

    def test_merge_overlapping_maps(self):
        """Test merging heap maps with overlapping keys."""
        hm1 = PHeapMap.mk([(1, "a1"), (2, "b1")])
        hm2 = PHeapMap.mk([(2, "b2"), (3, "c2")])
        result = hm1.merge(hm2)

        assert result.size() == 4  # Duplicates are preserved
        # Check that all entries are present
        items = list(result.iter())
        assert (1, "a1") in items
        assert (2, "b1") in items or (2, "b2") in items  # One of the key 2 entries
        assert (3, "c2") in items


class TestPHeapMapOperators:
    def test_insert_operator_right(self):
        """Test >> operator for inserting key-value pairs."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        result = hm >> (1, "a") >> (2, "b")

        assert result.size() == 2
        assert list(result.iter()) == [(1, "a"), (2, "b")]

    def test_insert_operator_left(self):
        """Test << operator for inserting key-value pairs."""
        hm: PHeapMap[int, str] = PHeapMap.empty()
        result = (2, "b") << ((1, "a") << hm)

        assert result.size() == 2
        assert list(result.iter()) == [(1, "a"), (2, "b")]

    def test_merge_operator_plus(self):
        """Test + operator for merging heap maps."""
        hm1 = PHeapMap.mk([(1, "a"), (2, "b")])
        hm2 = PHeapMap.mk([(3, "c"), (4, "d")])
        result = hm1 + hm2

        assert result.size() == 4
        assert list(result.iter()) == [(1, "a"), (2, "b"), (3, "c"), (4, "d")]


class TestPHeapMapUtilityMethods:
    def test_iteration_methods_consistency(self):
        """Test that keys(), values(), and iter() are consistent."""
        hm = PHeapMap.mk([(3, "c"), (1, "a"), (2, "b")])

        keys = list(hm.keys())
        values = list(hm.values())
        items = list(hm.iter())

        assert len(keys) == len(values) == len(items) == hm.size()
        assert items == list(zip(keys, values))
        assert keys == [1, 2, 3]  # Should be sorted
        assert values == ["a", "b", "c"]


class TestPHeapMapPersistence:
    def test_persistence(self):
        """Test that original heap map remains unchanged after operations."""
        original = PHeapMap.mk([(2, "b"), (1, "a")])
        original_items = list(original.iter())
        original_size = original.size()

        # Perform various operations
        original.insert(3, "c")
        original.delete_min()
        original.merge(PHeapMap.singleton(4, "d"))
        original.find_min()

        # Original should be unchanged
        assert list(original.iter()) == original_items
        assert original.size() == original_size


class TestPHeapMapEdgeCases:
    def test_large_heapmap_insertion(self):
        """Test inserting many elements."""
        pairs = [(i, f"value_{i}") for i in range(100, 0, -1)]  # Reverse order
        hm = PHeapMap.mk(pairs)

        assert hm.size() == 100
        result = hm.find_min()
        assert result is not None
        assert result[0] == 1  # Min key
        assert result[1] == "value_1"  # Min value

        # Should be sorted by key
        items = list(hm.iter())
        assert items[0] == (1, "value_1")
        assert items[-1] == (100, "value_100")
        assert all(items[i][0] <= items[i + 1][0] for i in range(len(items) - 1))

    def test_string_keys(self):
        """Test heap map with string keys."""
        hm = PHeapMap.mk([("c", 3), ("a", 1), ("b", 2)])

        assert list(hm.iter()) == [("a", 1), ("b", 2), ("c", 3)]
        result = hm.find_min()
        assert result is not None
        assert result[0] == "a"
        assert result[1] == 1

    def test_negative_number_keys(self):
        """Test heap map with negative number keys."""
        hm = PHeapMap.mk([(1, "pos"), (-1, "neg"), (0, "zero")])

        assert list(hm.iter()) == [(-1, "neg"), (0, "zero"), (1, "pos")]
        result = hm.find_min()
        assert result is not None
        assert result[0] == -1
        assert result[1] == "neg"

    def test_duplicate_values_different_keys(self):
        """Test heap map with same values for different keys."""
        hm = PHeapMap.mk([(3, "same"), (1, "same"), (2, "same")])

        assert hm.size() == 3
        assert list(hm.iter()) == [(1, "same"), (2, "same"), (3, "same")]
        result = hm.find_min()
        assert result is not None
        assert result[0] == 1
        assert result[1] == "same"


def test_filter_keys_empty():
    """Test filter_keys on an empty heap map"""
    empty = PHeapMap.empty(str, int)
    filtered = empty.filter_keys(lambda k: len(k) > 3)
    assert filtered.null()
    assert list(filtered.iter()) == []


def test_filter_keys_single():
    """Test filter_keys on a single element heap map"""
    hm = PHeapMap.singleton("test", 42)

    # Key matches predicate
    filtered_match = hm.filter_keys(lambda k: len(k) == 4)
    assert filtered_match.size() == 1
    assert list(filtered_match.iter()) == [("test", 42)]

    # Key doesn't match predicate
    filtered_no_match = hm.filter_keys(lambda k: len(k) > 10)
    assert filtered_no_match.null()
    assert list(filtered_no_match.iter()) == []


def test_filter_keys_multiple():
    """Test filter_keys on heap maps with multiple elements"""
    hm = PHeapMap.mk([("apple", 1), ("banana", 2), ("apricot", 3), ("cherry", 4)])

    # Filter keys starting with 'a'
    filtered_a = hm.filter_keys(lambda k: k.startswith("a"))
    result_a: List = list(filtered_a.iter())
    assert result_a == [("apple", 1), ("apricot", 3)]

    # Filter keys with length > 5
    filtered_long = hm.filter_keys(lambda k: len(k) > 5)
    result_long: List = list(filtered_long.iter())
    assert result_long == [("apricot", 3), ("banana", 2), ("cherry", 4)]

    # Original heap map unchanged
    assert hm.size() == 4
    assert list(hm.iter()) == [
        ("apple", 1),
        ("apricot", 3),
        ("banana", 2),
        ("cherry", 4),
    ]


def test_filter_keys_none_match():
    """Test filter_keys where no keys match"""
    hm = PHeapMap.mk([("a", 1), ("b", 2), ("c", 3)])
    filtered = hm.filter_keys(lambda k: k.startswith("z"))
    assert filtered.null()
    assert list(filtered.iter()) == []


def test_filter_keys_all_match():
    """Test filter_keys where all keys match"""
    hm = PHeapMap.mk([("a1", 1), ("a2", 2), ("a3", 3)])
    filtered = hm.filter_keys(lambda k: k.startswith("a"))
    assert filtered.size() == 3
    assert list(filtered.iter()) == [("a1", 1), ("a2", 2), ("a3", 3)]


def test_map_values_empty():
    """Test map_values on an empty heap map"""
    empty = PHeapMap.empty(str, int)
    mapped = empty.map_values(lambda v: v * 2)
    assert mapped.null()
    assert list(mapped.iter()) == []


def test_map_values_single():
    """Test map_values on a single element heap map"""
    hm = PHeapMap.singleton("test", 5)
    mapped = hm.map_values(lambda v: v * 10)
    assert mapped.size() == 1
    assert list(mapped.iter()) == [("test", 50)]


def test_map_values_multiple():
    """Test map_values on heap maps with multiple elements"""
    hm = PHeapMap.mk([("c", 3), ("a", 1), ("b", 2)])

    # Double all values
    doubled = hm.map_values(lambda v: v * 2)
    assert doubled.size() == 3
    result: List = list(doubled.iter())
    assert result == [("a", 2), ("b", 4), ("c", 6)]

    # Convert values to strings
    str_values = hm.map_values(lambda v: f"value_{v}")
    str_result: List = list(str_values.iter())
    assert str_result == [("a", "value_1"), ("b", "value_2"), ("c", "value_3")]

    # Original heap map unchanged
    assert hm.size() == 3
    original_result: List = list(hm.iter())
    assert original_result == [("a", 1), ("b", 2), ("c", 3)]


def test_map_values_type_change():
    """Test map_values with type change"""
    hm = PHeapMap.mk([("x", 1), ("y", 2), ("z", 3)])

    # Transform int values to strings
    str_mapped = hm.map_values(lambda v: str(v))
    result: List = list(str_mapped.iter())
    assert result == [("x", "1"), ("y", "2"), ("z", "3")]

    # Transform to lists
    list_mapped = hm.map_values(lambda v: [v, v])
    list_result: List = list(list_mapped.iter())
    assert list_result == [("x", [1, 1]), ("y", [2, 2]), ("z", [3, 3])]


def test_map_values_heap_order_preserved():
    """Test that map_values preserves heap order"""
    # Create heap map with specific key order
    hmap = PHeapMap.mk([("zebra", 26), ("apple", 1), ("banana", 2)])

    # Transform values
    mapped = hmap.map_values(lambda v: v + 100)

    # Order should be preserved (sorted by key)
    result: List = list(mapped.iter())
    assert result == [("apple", 101), ("banana", 102), ("zebra", 126)]
