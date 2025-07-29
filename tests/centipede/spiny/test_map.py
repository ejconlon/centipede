from centipede.spiny.map import PMap
from centipede.spiny.set import PSet


def test_empty_map():
    """Test creating an empty PMap and asserting it is empty"""
    map_obj = PMap.empty(int, str)
    assert map_obj.null()
    assert map_obj.size() == 0
    assert list(map_obj.items()) == []
    assert list(map_obj.keys()) == []
    assert list(map_obj.values()) == []


def test_singleton_put():
    """Test putting a single key-value pair"""
    empty_map = PMap.empty(int, str)
    single_map = empty_map.put(42, "forty-two")

    assert not single_map.null()
    assert single_map.size() == 1
    assert single_map.get(42) == "forty-two"
    assert single_map.contains(42)
    assert list(single_map.items()) == [(42, "forty-two")]


def test_singleton_method():
    """Test creating a singleton map directly"""
    single_map = PMap.singleton(42, "forty-two")

    assert not single_map.null()
    assert single_map.size() == 1
    assert single_map.get(42) == "forty-two"
    assert list(single_map.items()) == [(42, "forty-two")]


def test_multiple_puts():
    """Test putting multiple key-value pairs"""
    map_obj = PMap.empty(int, str)

    # Put multiple elements
    map_obj = map_obj.put(3, "three").put(1, "one").put(4, "four").put(2, "two")

    assert map_obj.size() == 4
    assert not map_obj.null()

    # Keys should be in sorted order
    result_keys = list(map_obj.keys())
    assert result_keys == sorted(result_keys)
    assert result_keys == [1, 2, 3, 4]

    # Values should correspond to sorted keys
    result_values = list(map_obj.values())
    assert result_values == ["one", "two", "three", "four"]


def test_duplicate_puts():
    """Test that putting the same key multiple times overwrites the value"""
    map_obj = PMap.empty(int, str)

    # Put same key multiple times
    map_obj = map_obj.put(42, "first").put(42, "second").put(42, "third")

    assert map_obj.size() == 1
    assert map_obj.get(42) == "third"


def test_put_ordering():
    """Test that keys maintain sorted order after puts"""
    map_obj = PMap.empty(int, str)
    pairs = [
        (5, "five"),
        (2, "two"),
        (8, "eight"),
        (1, "one"),
        (9, "nine"),
        (3, "three"),
    ]

    for key, value in pairs:
        map_obj = map_obj.put(key, value)

    result_keys = list(map_obj.keys())
    assert result_keys == sorted([key for key, _ in pairs])
    assert map_obj.size() == len(pairs)


def test_mk_from_iterable():
    """Test creating a map from an iterable"""
    pairs = [(3, "three"), (1, "one"), (4, "four"), (1, "ONE"), (5, "five")]
    map_obj = PMap.mk(pairs)

    # Should contain unique keys with last value for duplicates
    expected_items = [(1, "ONE"), (3, "three"), (4, "four"), (5, "five")]
    assert list(map_obj.items()) == expected_items
    assert map_obj.size() == 4


def test_mk_empty_iterable():
    """Test creating a map from an empty iterable"""
    map_obj: PMap[int, str] = PMap.mk([])
    assert map_obj.null()
    assert map_obj.size() == 0
    assert list(map_obj.items()) == []


def test_put_operator_right():
    """Test >> operator for putting"""
    map_obj = PMap.empty(int, str)
    map_obj = map_obj >> (42, "forty-two") >> (24, "twenty-four") >> (13, "thirteen")

    assert map_obj.size() == 3
    assert map_obj.get(42) == "forty-two"
    assert map_obj.get(24) == "twenty-four"
    assert map_obj.get(13) == "thirteen"


def test_put_operator_left():
    """Test << operator for putting"""
    map_obj = PMap.empty(int, str)
    map_obj = (42, "forty-two") << (
        (24, "twenty-four") << ((13, "thirteen") << map_obj)
    )

    assert map_obj.size() == 3
    assert map_obj.get(42) == "forty-two"
    assert map_obj.get(24) == "twenty-four"
    assert map_obj.get(13) == "thirteen"


def test_string_keys():
    """Test map with string keys"""
    map_obj = PMap.empty(str, int)
    map_obj = map_obj.put("hello", 5).put("world", 5).put("abc", 3)

    assert map_obj.size() == 3
    result_keys = list(map_obj.keys())
    assert sorted(result_keys) == result_keys  # Should be sorted
    assert set(result_keys) == {"hello", "world", "abc"}


def test_large_map_insertion():
    """Test putting many key-value pairs to verify tree balancing"""
    map_obj = PMap.empty(int, str)
    pairs = [(i, f"value_{i}") for i in range(100)]

    # Insert in random order to test balancing
    import random

    random.shuffle(pairs)

    for key, value in pairs:
        map_obj = map_obj.put(key, value)

    assert map_obj.size() == 100
    assert list(map_obj.keys()) == list(range(100))


def test_put_with_overwrites_mixed():
    """Test putting with mixed overwrites"""
    map_obj = PMap.empty(int, str)

    # Put pattern: overwrite some values
    insertions = [
        (1, "one"),
        (2, "two"),
        (1, "ONE"),
        (3, "three"),
        (2, "TWO"),
        (4, "four"),
        (1, "first"),
    ]
    for key, value in insertions:
        map_obj = map_obj.put(key, value)

    assert map_obj.size() == 4  # Only unique keys
    assert map_obj.get(1) == "first"  # Last value wins
    assert map_obj.get(2) == "TWO"
    assert map_obj.get(3) == "three"
    assert map_obj.get(4) == "four"


def test_put_negative_numbers():
    """Test putting with negative number keys"""
    map_obj = PMap.empty(int, str)
    pairs = [
        (-5, "neg_five"),
        (-1, "neg_one"),
        (0, "zero"),
        (3, "three"),
        (-10, "neg_ten"),
        (7, "seven"),
    ]

    for key, value in pairs:
        map_obj = map_obj.put(key, value)

    assert map_obj.size() == len(pairs)
    expected_keys = sorted([key for key, _ in pairs])
    assert list(map_obj.keys()) == expected_keys


def test_persistence():
    """Test that puts create new maps without modifying originals"""
    original = PMap.empty(int, str).put(1, "one").put(2, "two")
    modified = original.put(3, "three")

    # Original should be unchanged
    assert list(original.items()) == [(1, "one"), (2, "two")]
    assert original.size() == 2

    # Modified should have new element
    assert list(modified.items()) == [(1, "one"), (2, "two"), (3, "three")]
    assert modified.size() == 3


def test_chained_puts():
    """Test chaining multiple puts"""
    result = (
        PMap.empty(int, str)
        .put(5, "five")
        .put(2, "two")
        .put(8, "eight")
        .put(1, "one")
        .put(9, "nine")
    )

    assert result.size() == 5
    assert list(result.items()) == [
        (1, "one"),
        (2, "two"),
        (5, "five"),
        (8, "eight"),
        (9, "nine"),
    ]


def test_put_same_key_returns_new_instance():
    """Test that putting a new value for existing key returns new map"""
    map_obj = PMap.empty(int, str).put(42, "first")
    new_map = map_obj.put(42, "second")

    # Should be different objects
    assert map_obj is not new_map
    assert map_obj.get(42) == "first"
    assert new_map.get(42) == "second"


def test_get_nonexistent_key():
    """Test getting a key that doesn't exist"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])

    assert map_obj.lookup(4) is None
    assert map_obj.lookup(0) is None
    assert not map_obj.contains(4)
    assert not map_obj.contains(0)


def test_contains():
    """Test contains method"""
    map_obj = PMap.mk([(1, "one"), (3, "three"), (5, "five")])

    assert map_obj.contains(1)
    assert map_obj.contains(3)
    assert map_obj.contains(5)
    assert not map_obj.contains(2)
    assert not map_obj.contains(4)
    assert not map_obj.contains(0)


def test_remove_existing_key():
    """Test removing existing keys"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three"), (4, "four")])

    # Remove middle key
    result = map_obj.remove(2)
    assert result.size() == 3
    assert result.lookup(2) is None
    assert result.get(1) == "one"
    assert result.get(3) == "three"
    assert result.get(4) == "four"

    # Remove first key
    result2 = result.remove(1)
    assert result2.size() == 2
    assert result2.lookup(1) is None
    assert result2.get(3) == "three"
    assert result2.get(4) == "four"


def test_remove_nonexistent_key():
    """Test removing a key that doesn't exist"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    result = map_obj.remove(4)

    # Should be unchanged
    assert result.size() == 3
    assert list(result.items()) == list(map_obj.items())


def test_remove_all_keys():
    """Test removing all keys one by one"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])

    result = map_obj.remove(1).remove(2).remove(3)
    assert result.null()
    assert result.size() == 0


def test_merge_empty_maps():
    """Test merging empty maps"""
    empty1 = PMap.empty(int, str)
    empty2 = PMap.empty(int, str)
    result = empty1.merge(empty2)

    assert result.null()
    assert result.size() == 0
    assert list(result.items()) == []


def test_merge_empty_with_non_empty():
    """Test merging empty map with non-empty map"""
    empty_map = PMap.empty(int, str)
    non_empty = PMap.mk([(1, "one"), (2, "two"), (3, "three")])

    result1 = empty_map.merge(non_empty)
    result2 = non_empty.merge(empty_map)

    assert list(result1.items()) == [(1, "one"), (2, "two"), (3, "three")]
    assert list(result2.items()) == [(1, "one"), (2, "two"), (3, "three")]
    assert result1.size() == 3
    assert result2.size() == 3


def test_merge_disjoint_maps():
    """Test merging maps with no common keys"""
    map1 = PMap.mk([(1, "one"), (3, "three"), (5, "five")])
    map2 = PMap.mk([(2, "two"), (4, "four"), (6, "six")])

    result = map1.merge(map2)

    assert result.size() == 6
    assert list(result.items()) == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
    ]


def test_merge_overlapping_maps():
    """Test merging maps with some common keys"""
    map1 = PMap.mk([(1, "one"), (2, "two"), (3, "three"), (4, "four")])
    map2 = PMap.mk([(3, "THREE"), (4, "FOUR"), (5, "five"), (6, "six")])

    result = map1.merge(map2)

    # Values from map1 should win for common keys
    assert result.size() == 6
    expected_items = [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
    ]
    assert list(result.items()) == expected_items


def test_split_method():
    """Test the enhanced split method with key lookup"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")])

    # Test with key in map
    smaller, found_value, larger = map_obj.split(3)
    assert found_value == "three"
    assert list(smaller.items()) == [(1, "one"), (2, "two")]
    assert list(larger.items()) == [(4, "four"), (5, "five")]

    # Test with key not in map (using a sparse map)
    sparse_map = PMap.mk([(1, "one"), (3, "three"), (5, "five"), (7, "seven")])
    smaller2, found_value2, larger2 = sparse_map.split(4)
    assert found_value2 is None
    assert list(smaller2.items()) == [(1, "one"), (3, "three")]
    assert list(larger2.items()) == [(5, "five"), (7, "seven")]

    # Test with key smaller than all elements
    smaller3, found_value3, larger3 = map_obj.split(0)
    assert found_value3 is None
    assert smaller3.null()
    assert list(larger3.items()) == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
    ]

    # Test with key larger than all elements
    smaller4, found_value4, larger4 = map_obj.split(10)
    assert found_value4 is None
    assert list(smaller4.items()) == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
    ]
    assert larger4.null()


def test_split_empty_map():
    """Test split on empty map"""
    empty = PMap.empty(int, str)
    smaller, found_value, larger = empty.split(5)

    assert found_value is None
    assert smaller.null()
    assert larger.null()


def test_split_single_entry():
    """Test split on single-entry map"""
    single = PMap.singleton(5, "five")

    # Split on existing key
    smaller1, found_value1, larger1 = single.split(5)
    assert found_value1 == "five"
    assert smaller1.null()
    assert larger1.null()

    # Split on smaller key
    smaller2, found_value2, larger2 = single.split(3)
    assert found_value2 is None
    assert smaller2.null()
    assert list(larger2.items()) == [(5, "five")]

    # Split on larger key
    smaller3, found_value3, larger3 = single.split(7)
    assert found_value3 is None
    assert list(smaller3.items()) == [(5, "five")]
    assert larger3.null()


def test_split_with_none_values():
    """Test split method with None values"""
    map_obj = PMap.mk([(1, None), (2, "two"), (3, None)])

    # Split on key with None value
    smaller, found_value, larger = map_obj.split(1)
    assert found_value is None  # This is the actual value, not "not found"
    assert smaller.null()
    assert list(larger.items()) == [(2, "two"), (3, None)]

    # Split on key not in map should also return None, but for different reason
    sparse_map_with_none = PMap.mk([(1, None), (3, None), (5, "five")])
    smaller2, found_value2, larger2 = sparse_map_with_none.split(2)
    assert found_value2 is None  # This means "not found"
    assert list(smaller2.items()) == [(1, None)]
    assert list(larger2.items()) == [(3, None), (5, "five")]


def test_split_string_keys():
    """Test split method with string keys"""
    map_obj = PMap.mk([("apple", 1), ("banana", 2), ("cherry", 3), ("date", 4)])

    smaller, found_value, larger = map_obj.split("banana")
    assert found_value == 2
    assert list(smaller.items()) == [("apple", 1)]
    assert list(larger.items()) == [("cherry", 3), ("date", 4)]


def test_merge_identical_maps():
    """Test merging identical maps"""
    map1 = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    map2 = PMap.mk([(1, "one"), (2, "two"), (3, "three")])

    result = map1.merge(map2)

    assert result.size() == 3
    assert list(result.items()) == [(1, "one"), (2, "two"), (3, "three")]


def test_merge_subset_maps():
    """Test merging when one map is a subset of another"""
    map1 = PMap.mk([(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")])
    map2 = PMap.mk([(2, "TWO"), (4, "FOUR")])

    result1 = map1.merge(map2)
    result2 = map2.merge(map1)

    # map1 values should win in result1 (left precedence)
    expected1 = [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")]
    assert list(result1.items()) == expected1
    assert result1.size() == 5

    # map2 values should win in result2 (left precedence)
    expected2 = [(1, "one"), (2, "TWO"), (3, "three"), (4, "FOUR"), (5, "five")]
    assert list(result2.items()) == expected2
    assert result2.size() == 5


def test_merge_single_entry_maps():
    """Test merging single entry maps"""
    map1 = PMap.singleton(1, "one")
    map2 = PMap.singleton(2, "two")
    map3 = PMap.singleton(1, "ONE")  # Same key

    result1 = map1.merge(map2)
    result2 = map1.merge(map3)

    assert list(result1.items()) == [(1, "one"), (2, "two")]
    assert result1.size() == 2

    assert list(result2.items()) == [(1, "one")]  # map1 wins with left precedence
    assert result2.size() == 1


def test_merge_large_maps():
    """Test merging large maps"""
    map1 = PMap.mk([(i, f"even_{i}") for i in range(0, 100, 2)])  # Even numbers 0-98
    map2 = PMap.mk([(i, f"odd_{i}") for i in range(1, 100, 2)])  # Odd numbers 1-99

    result = map1.merge(map2)

    assert result.size() == 100
    assert list(result.keys()) == list(range(100))


def test_merge_string_maps():
    """Test merging maps with string keys"""
    map1 = PMap.mk([("apple", 1), ("banana", 2), ("cherry", 3)])
    map2 = PMap.mk([("banana", 20), ("date", 4), ("elderberry", 5)])

    result = map1.merge(map2)

    expected = [
        ("apple", 1),
        ("banana", 2),  # map1 wins with left precedence
        ("cherry", 3),
        ("date", 4),
        ("elderberry", 5),
    ]
    assert list(result.items()) == expected
    assert result.size() == 5


def test_merge_operator_plus():
    """Test merge using + operator"""
    map1 = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    map2 = PMap.mk([(3, "THREE"), (4, "four"), (5, "five")])

    result = map1 + map2

    expected = [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
    ]  # map1 wins with left precedence
    assert list(result.items()) == expected
    assert result.size() == 5


def test_merge_chaining():
    """Test chaining multiple merge operations"""
    map1 = PMap.mk([(1, "one"), (2, "two")])
    map2 = PMap.mk([(3, "three"), (4, "four")])
    map3 = PMap.mk([(5, "five"), (6, "six")])

    result = map1.merge(map2).merge(map3)

    expected = [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
    ]
    assert list(result.items()) == expected
    assert result.size() == 6


def test_merge_persistence():
    """Test that merge operations don't modify original maps"""
    map1 = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    map2 = PMap.mk([(4, "four"), (5, "five"), (6, "six")])

    original1_items = list(map1.items())
    original2_items = list(map2.items())

    result = map1.merge(map2)

    # Original maps should be unchanged
    assert list(map1.items()) == original1_items
    assert list(map2.items()) == original2_items
    assert map1.size() == 3
    assert map2.size() == 3

    # Result should contain all elements
    expected = [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
    ]
    assert list(result.items()) == expected
    assert result.size() == 6


def test_merge_negative_numbers():
    """Test merging maps with negative number keys"""
    map1 = PMap.mk([(-3, "neg_three"), (-1, "neg_one"), (1, "one"), (3, "three")])
    map2 = PMap.mk([(-2, "neg_two"), (0, "zero"), (2, "two")])

    result = map1.merge(map2)

    expected = [
        (-3, "neg_three"),
        (-2, "neg_two"),
        (-1, "neg_one"),
        (0, "zero"),
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ]
    assert list(result.items()) == expected
    assert result.size() == 7


def test_find_min_empty_map():
    """Test find_min on empty map returns None"""
    empty_map = PMap.empty(int, str)
    result = empty_map.find_min()
    assert result is None


def test_find_min_singleton():
    """Test find_min on singleton map"""
    single_map = PMap.singleton(42, "forty-two")
    result = single_map.find_min()

    assert result is not None
    min_key, min_value, remaining = result
    assert min_key == 42
    assert min_value == "forty-two"
    assert remaining.null()
    assert remaining.size() == 0


def test_find_min_multiple_entries():
    """Test find_min on map with multiple entries"""
    map_obj = PMap.mk([(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine")])
    result = map_obj.find_min()

    assert result is not None
    min_key, min_value, remaining = result
    assert min_key == 1
    assert min_value == "one"
    assert remaining.size() == 4
    assert list(remaining.items()) == [
        (2, "two"),
        (5, "five"),
        (8, "eight"),
        (9, "nine"),
    ]


def test_find_min_negative_numbers():
    """Test find_min with negative number keys"""
    map_obj = PMap.mk(
        [
            (-5, "neg_five"),
            (-1, "neg_one"),
            (0, "zero"),
            (3, "three"),
            (-10, "neg_ten"),
            (7, "seven"),
        ]
    )
    result = map_obj.find_min()

    assert result is not None
    min_key, min_value, remaining = result
    assert min_key == -10
    assert min_value == "neg_ten"
    assert remaining.size() == 5
    expected_remaining = [
        (-5, "neg_five"),
        (-1, "neg_one"),
        (0, "zero"),
        (3, "three"),
        (7, "seven"),
    ]
    assert list(remaining.items()) == expected_remaining


def test_find_min_strings():
    """Test find_min with string keys"""
    map_obj = PMap.mk([("zebra", 1), ("apple", 2), ("banana", 3), ("cherry", 4)])
    result = map_obj.find_min()

    assert result is not None
    min_key, min_value, remaining = result
    assert min_key == "apple"
    assert min_value == 2
    assert remaining.size() == 3
    expected_remaining = [("banana", 3), ("cherry", 4), ("zebra", 1)]
    assert list(remaining.items()) == expected_remaining


def test_find_min_persistence():
    """Test that find_min doesn't modify original map"""
    original = PMap.mk([(3, "three"), (1, "one"), (4, "four"), (2, "two")])
    original_items = list(original.items())
    original_size = original.size()

    result = original.find_min()

    # Original should be unchanged
    assert list(original.items()) == original_items
    assert original.size() == original_size

    # Result should be correct
    assert result is not None
    min_key, min_value, remaining = result
    assert min_key == 1
    assert min_value == "one"
    expected_remaining = [(2, "two"), (3, "three"), (4, "four")]
    assert list(remaining.items()) == expected_remaining


def test_find_min_repeated_calls():
    """Test repeated calls to find_min to extract all entries"""
    original = PMap.mk(
        [(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine"), (3, "three")]
    )
    extracted = []
    current = original

    while True:
        result = current.find_min()
        if result is None:
            break
        min_key, min_value, remaining = result
        extracted.append((min_key, min_value))
        current = remaining

    expected = [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (5, "five"),
        (8, "eight"),
        (9, "nine"),
    ]
    assert extracted == expected
    assert current.null()


def test_find_max_empty_map():
    """Test find_max on empty map returns None"""
    empty_map = PMap.empty(int, str)
    result = empty_map.find_max()
    assert result is None


def test_find_max_singleton():
    """Test find_max on singleton map"""
    single_map = PMap.singleton(42, "forty-two")
    result = single_map.find_max()

    assert result is not None
    remaining, max_key, max_value = result
    assert max_key == 42
    assert max_value == "forty-two"
    assert remaining.null()
    assert remaining.size() == 0


def test_find_max_multiple_entries():
    """Test find_max on map with multiple entries"""
    map_obj = PMap.mk([(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine")])
    result = map_obj.find_max()

    assert result is not None
    remaining, max_key, max_value = result
    assert max_key == 9
    assert max_value == "nine"
    assert remaining.size() == 4
    expected_remaining = [(1, "one"), (2, "two"), (5, "five"), (8, "eight")]
    assert list(remaining.items()) == expected_remaining


def test_find_max_negative_numbers():
    """Test find_max with negative number keys"""
    map_obj = PMap.mk(
        [
            (-5, "neg_five"),
            (-1, "neg_one"),
            (0, "zero"),
            (3, "three"),
            (-10, "neg_ten"),
            (7, "seven"),
        ]
    )
    result = map_obj.find_max()

    assert result is not None
    remaining, max_key, max_value = result
    assert max_key == 7
    assert max_value == "seven"
    assert remaining.size() == 5
    expected_remaining = [
        (-10, "neg_ten"),
        (-5, "neg_five"),
        (-1, "neg_one"),
        (0, "zero"),
        (3, "three"),
    ]
    assert list(remaining.items()) == expected_remaining


def test_find_max_strings():
    """Test find_max with string keys"""
    map_obj = PMap.mk([("zebra", 1), ("apple", 2), ("banana", 3), ("cherry", 4)])
    result = map_obj.find_max()

    assert result is not None
    remaining, max_key, max_value = result
    assert max_key == "zebra"
    assert max_value == 1
    assert remaining.size() == 3
    expected_remaining = [("apple", 2), ("banana", 3), ("cherry", 4)]
    assert list(remaining.items()) == expected_remaining


def test_find_max_persistence():
    """Test that find_max doesn't modify original map"""
    original = PMap.mk([(3, "three"), (1, "one"), (4, "four"), (2, "two")])
    original_items = list(original.items())
    original_size = original.size()

    result = original.find_max()

    # Original should be unchanged
    assert list(original.items()) == original_items
    assert original.size() == original_size

    # Result should be correct
    assert result is not None
    remaining, max_key, max_value = result
    assert max_key == 4
    assert max_value == "four"
    expected_remaining = [(1, "one"), (2, "two"), (3, "three")]
    assert list(remaining.items()) == expected_remaining


def test_find_max_repeated_calls():
    """Test repeated calls to find_max to extract all entries"""
    original = PMap.mk(
        [(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine"), (3, "three")]
    )
    extracted = []
    current = original

    while True:
        result = current.find_max()
        if result is None:
            break
        remaining, max_key, max_value = result
        extracted.append((max_key, max_value))
        current = remaining

    expected = [
        (9, "nine"),
        (8, "eight"),
        (5, "five"),
        (3, "three"),
        (2, "two"),
        (1, "one"),
    ]
    assert extracted == expected
    assert current.null()


def test_delete_min_empty_map():
    """Test delete_min on empty map returns None"""
    empty_map = PMap.empty(int, str)
    result = empty_map.delete_min()
    assert result is None


def test_delete_min_singleton():
    """Test delete_min on singleton map"""
    single_map = PMap.singleton(42, "forty-two")
    result = single_map.delete_min()

    assert result is not None
    assert result.null()
    assert result.size() == 0


def test_delete_min_multiple_entries():
    """Test delete_min on map with multiple entries"""
    map_obj = PMap.mk([(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine")])
    result = map_obj.delete_min()

    assert result is not None
    assert result.size() == 4
    expected = [(2, "two"), (5, "five"), (8, "eight"), (9, "nine")]
    assert list(result.items()) == expected


def test_delete_min_consistency_with_find_min():
    """Test that delete_min is consistent with find_min"""
    map_obj = PMap.mk(
        [
            (7, "seven"),
            (3, "three"),
            (11, "eleven"),
            (1, "one"),
            (9, "nine"),
            (5, "five"),
        ]
    )

    find_result = map_obj.find_min()
    delete_result = map_obj.delete_min()

    assert find_result is not None
    assert delete_result is not None

    _, _, remaining_from_find = find_result

    # delete_min result should be same as remaining map from find_min
    assert list(delete_result.items()) == list(remaining_from_find.items())
    assert delete_result.size() == remaining_from_find.size()


def test_delete_max_empty_map():
    """Test delete_max on empty map returns None"""
    empty_map = PMap.empty(int, str)
    result = empty_map.delete_max()
    assert result is None


def test_delete_max_singleton():
    """Test delete_max on singleton map"""
    single_map = PMap.singleton(42, "forty-two")
    result = single_map.delete_max()

    assert result is not None
    assert result.null()
    assert result.size() == 0


def test_delete_max_multiple_entries():
    """Test delete_max on map with multiple entries"""
    map_obj = PMap.mk([(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine")])
    result = map_obj.delete_max()

    assert result is not None
    assert result.size() == 4
    expected = [(1, "one"), (2, "two"), (5, "five"), (8, "eight")]
    assert list(result.items()) == expected


def test_delete_max_consistency_with_find_max():
    """Test that delete_max is consistent with find_max"""
    map_obj = PMap.mk(
        [
            (7, "seven"),
            (3, "three"),
            (11, "eleven"),
            (1, "one"),
            (9, "nine"),
            (5, "five"),
        ]
    )

    find_result = map_obj.find_max()
    delete_result = map_obj.delete_max()

    assert find_result is not None
    assert delete_result is not None

    remaining_from_find, _, _ = find_result

    # delete_max result should be same as remaining map from find_max
    assert list(delete_result.items()) == list(remaining_from_find.items())
    assert delete_result.size() == remaining_from_find.size()


def test_find_min_max_symmetry():
    """Test that find_min and find_max work correctly together"""
    map_obj = PMap.mk(
        [(5, "five"), (2, "two"), (8, "eight"), (1, "one"), (9, "nine"), (3, "three")]
    )

    min_result = map_obj.find_min()
    max_result = map_obj.find_max()

    assert min_result is not None
    assert max_result is not None

    min_key, _, _ = min_result
    _, max_key, _ = max_result

    assert min_key == 1
    assert max_key == 9

    # Test that removing min and max from original gives same result as
    # removing max then min (or min then max)
    after_min_removal = map_obj.delete_min()
    after_max_removal = map_obj.delete_max()

    assert after_min_removal is not None
    assert after_max_removal is not None

    min_then_max = after_min_removal.delete_max()
    max_then_min = after_max_removal.delete_min()

    assert min_then_max is not None
    assert max_then_min is not None
    assert list(min_then_max.items()) == list(max_then_min.items())


def test_find_max_single_entry_after_operations():
    """Test find_max behavior after various operations"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])

    # Remove minimum twice
    after_first_min = map_obj.delete_min()
    assert after_first_min is not None
    after_second_min = after_first_min.delete_min()
    assert after_second_min is not None

    # Should have only entry (3, "three") left
    result = after_second_min.find_max()
    assert result is not None
    remaining, max_key, max_value = result
    assert max_key == 3
    assert max_value == "three"
    assert remaining.null()


def test_iteration_methods():
    """Test keys(), values(), and items() iteration methods"""
    map_obj = PMap.mk([(3, "three"), (1, "one"), (4, "four"), (2, "two")])

    # Test keys()
    keys_list = list(map_obj.keys())
    assert keys_list == [1, 2, 3, 4]

    # Test values()
    values_list = list(map_obj.values())
    assert values_list == ["one", "two", "three", "four"]

    # Test items()
    items_list = list(map_obj.items())
    assert items_list == [(1, "one"), (2, "two"), (3, "three"), (4, "four")]

    # Test that all yield same number of elements
    assert len(keys_list) == len(values_list) == len(items_list) == map_obj.size()


def test_edge_cases():
    """Test various edge cases"""
    # Empty map operations
    empty = PMap.empty(int, str)
    assert empty.lookup(1) is None
    assert not empty.contains(1)
    assert empty.remove(1).null()

    # Single entry operations
    single = PMap.singleton(1, "one")
    assert single.remove(2) == single  # Remove non-existent key
    assert single.remove(1).null()  # Remove existing key

    # Test with None values
    map_with_none = PMap.mk([(1, None), (2, "two")])
    assert map_with_none.lookup(1) is None
    assert map_with_none.contains(1)  # Should still contain the key
    assert map_with_none.lookup(3) is None
    assert not map_with_none.contains(3)  # Should not contain non-existent key


def test_merge_favors_left_values_on_collision():
    """Test that merging maps favors values from the left map when keys collide"""
    left_map = PMap.mk([(1, "left_one"), (2, "left_two"), (3, "left_three")])
    right_map = PMap.mk([(2, "right_two"), (3, "right_three"), (4, "right_four")])

    result = left_map.merge(right_map)

    # Values from left_map should win for common keys (2 and 3)
    assert result.size() == 4
    expected_items = [
        (1, "left_one"),
        (2, "left_two"),  # Should keep left value, not "right_two"
        (3, "left_three"),  # Should keep left value, not "right_three"
        (4, "right_four"),
    ]
    assert list(result.items()) == expected_items


def test_keys_set_empty_map():
    """Test keys_set on empty map returns empty set"""
    empty_map = PMap.empty(int, str)
    keys_set = empty_map.keys_set()

    assert keys_set.size() == 0
    assert keys_set.null()
    assert list(keys_set.iter()) == []


def test_keys_set_singleton():
    """Test keys_set on singleton map"""
    single_map = PMap.singleton(42, "forty-two")
    keys_set = single_map.keys_set()

    assert keys_set.size() == 1
    assert not keys_set.null()
    assert list(keys_set.iter()) == [42]
    assert keys_set.contains(42)
    assert not keys_set.contains(43)


def test_keys_set_multiple_entries():
    """Test keys_set on map with multiple entries"""
    map_obj = PMap.mk([(3, "three"), (1, "one"), (4, "four"), (2, "two")])
    keys_set = map_obj.keys_set()

    assert keys_set.size() == 4
    assert not keys_set.null()

    # Keys should be in sorted order (same as map iteration)
    keys_list = list(keys_set.iter())
    assert keys_list == [1, 2, 3, 4]

    # Test contains for all keys
    for key in [1, 2, 3, 4]:
        assert keys_set.contains(key)

    # Test contains for non-existent keys
    assert not keys_set.contains(0)
    assert not keys_set.contains(5)


def test_keys_set_preserves_order():
    """Test that keys_set preserves the sorted order from the map"""
    pairs = [(7, "seven"), (3, "three"), (11, "eleven"), (1, "one"), (9, "nine")]
    map_obj = PMap.mk(pairs)
    keys_set = map_obj.keys_set()

    map_keys = list(map_obj.keys())
    set_keys = list(keys_set.iter())

    assert map_keys == set_keys
    assert set_keys == sorted([key for key, _ in pairs])


def test_keys_set_with_string_keys():
    """Test keys_set with string keys"""
    map_obj = PMap.mk([("zebra", 1), ("apple", 2), ("banana", 3), ("cherry", 4)])
    keys_set = map_obj.keys_set()

    assert keys_set.size() == 4
    expected_keys = ["apple", "banana", "cherry", "zebra"]
    assert list(keys_set.iter()) == expected_keys

    for key in expected_keys:
        assert keys_set.contains(key)


def test_keys_set_large_map():
    """Test keys_set on a large map"""
    pairs = [(i, f"value_{i}") for i in range(100)]
    map_obj = PMap.mk(pairs)
    keys_set = map_obj.keys_set()

    assert keys_set.size() == 100
    assert list(keys_set.iter()) == list(range(100))

    # Test random sampling of contains
    for key in [0, 25, 50, 75, 99]:
        assert keys_set.contains(key)
    assert not keys_set.contains(100)
    assert not keys_set.contains(-1)


def test_keys_set_type_consistency():
    """Test that keys_set returns proper PSet type"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    keys_set = map_obj.keys_set()

    # Should be a PSet instance
    assert isinstance(keys_set, PSet)

    # Should support PSet operations
    new_set = keys_set.insert(4)
    assert new_set.size() == 4
    assert new_set.contains(4)
    assert keys_set.size() == 3  # Original unchanged


def test_keys_set_equivalence_with_manual_creation():
    """Test that keys_set produces equivalent result to manual PSet creation"""
    map_obj = PMap.mk([(5, "five"), (2, "two"), (8, "eight"), (1, "one")])

    # Create using keys_set
    keys_set = map_obj.keys_set()

    # Create manually from keys iterator
    manual_set = PSet.mk(map_obj.keys())

    # Should have same elements
    assert keys_set.size() == manual_set.size()
    assert list(keys_set.iter()) == list(manual_set.iter())

    # Test equivalence by checking contains for all elements
    for key in map_obj.keys():
        assert keys_set.contains(key) == manual_set.contains(key)


def test_keys_set_with_negative_keys():
    """Test keys_set with negative number keys"""
    map_obj = PMap.mk([(-3, "a"), (-1, "b"), (0, "c"), (2, "d"), (-5, "e")])
    keys_set = map_obj.keys_set()

    expected_keys = [-5, -3, -1, 0, 2]
    assert list(keys_set.iter()) == expected_keys
    assert keys_set.size() == 5

    for key in expected_keys:
        assert keys_set.contains(key)


def test_assoc_empty_set():
    """Test assoc with empty set"""
    empty_set = PSet.empty(int)
    result_map = PMap.assoc(empty_set, "default")

    assert result_map.null()
    assert result_map.size() == 0
    assert list(result_map.items()) == []


def test_assoc_singleton_set():
    """Test assoc with singleton set"""
    single_set = PSet.singleton(42)
    result_map = PMap.assoc(single_set, "value")

    assert result_map.size() == 1
    assert not result_map.null()
    assert result_map.get(42) == "value"
    assert list(result_map.items()) == [(42, "value")]


def test_assoc_multiple_elements():
    """Test assoc with set containing multiple elements"""
    multi_set = PSet.mk([3, 1, 4, 2])
    result_map = PMap.assoc(multi_set, "constant")

    assert result_map.size() == 4
    assert not result_map.null()

    # All keys should have the same value
    expected_items = [
        (1, "constant"),
        (2, "constant"),
        (3, "constant"),
        (4, "constant"),
    ]
    assert list(result_map.items()) == expected_items

    # Test individual gets
    for key in [1, 2, 3, 4]:
        assert result_map.get(key) == "constant"
        assert result_map.contains(key)


def test_assoc_different_value_types():
    """Test assoc with different value types"""
    keys_set = PSet.mk(["x", "y", "z"])

    # Test with int values
    int_map = PMap.assoc(keys_set, 100)
    assert list(int_map.items()) == [("x", 100), ("y", 100), ("z", 100)]

    # Test with float values
    float_map = PMap.assoc(keys_set, 3.14)
    assert list(float_map.items()) == [("x", 3.14), ("y", 3.14), ("z", 3.14)]

    # Test with list values
    list_value = [1, 2, 3]
    list_map = PMap.assoc(keys_set, list_value)
    expected = [("x", list_value), ("y", list_value), ("z", list_value)]
    assert list(list_map.items()) == expected

    # Test with None values
    none_map = PMap.assoc(keys_set, None)
    expected_none = [("x", None), ("y", None), ("z", None)]
    assert list(none_map.items()) == expected_none


def test_assoc_preserves_order():
    """Test that assoc preserves the order from the original set"""
    original_set = PSet.mk([7, 3, 11, 1, 9])
    result_map = PMap.assoc(original_set, "value")

    # Order should match set iteration order
    set_keys = list(original_set.iter())
    map_keys = list(result_map.keys())

    assert set_keys == map_keys
    assert set_keys == [1, 3, 7, 9, 11]  # Should be sorted


def test_assoc_large_set():
    """Test assoc with a large set"""
    large_keys = [str(i) for i in range(50)]
    large_set = PSet.mk(large_keys)
    result_map = PMap.assoc(large_set, "constant")

    assert result_map.size() == 50

    # Test first few items
    items = list(result_map.items())
    for i in range(5):
        key, value = items[i]
        assert value == "constant"
        assert result_map.contains(key)

    # Test that all values are the same
    all_values = list(result_map.values())
    assert all(v == "constant" for v in all_values)


def test_assoc_with_string_keys():
    """Test assoc with string keys"""
    string_set = PSet.mk(["apple", "banana", "cherry"])
    result_map = PMap.assoc(string_set, 42)

    expected_items = [("apple", 42), ("banana", 42), ("cherry", 42)]
    assert list(result_map.items()) == expected_items
    assert result_map.size() == 3


def test_assoc_round_trip_with_keys_set():
    """Test round-trip conversion: PSet -> PMap via assoc -> PSet via keys_set"""
    original_set = PSet.mk(["foo", "bar", "baz", "qux"])

    # Convert to map using assoc
    map_from_set = PMap.assoc(original_set, "test_value")

    # Convert back to set using keys_set
    recovered_set = map_from_set.keys_set()

    # Should preserve structure and order
    original_keys = list(original_set.iter())
    recovered_keys = list(recovered_set.iter())

    assert original_keys == recovered_keys
    assert original_set.size() == recovered_set.size()

    # Test element-wise equivalence
    for key in original_keys:
        assert original_set.contains(key)
        assert recovered_set.contains(key)


def test_assoc_type_consistency():
    """Test that assoc returns proper PMap type"""
    test_set = PSet.mk([1, 2, 3])
    result_map = PMap.assoc(test_set, "value")

    # Should be a PMap instance
    assert isinstance(result_map, PMap)

    # Should support PMap operations
    new_map = result_map.put(4, "new_value")
    assert new_map.size() == 4
    assert new_map.get(4) == "new_value"
    assert result_map.size() == 3  # Original unchanged


def test_assoc_with_negative_keys():
    """Test assoc with negative number keys"""
    negative_set = PSet.mk([-5, -1, 0, 3, -10])
    result_map = PMap.assoc(negative_set, "negative_test")

    expected_keys = [-10, -5, -1, 0, 3]  # Should be sorted
    assert list(result_map.keys()) == expected_keys
    assert result_map.size() == 5

    for key in expected_keys:
        assert result_map.get(key) == "negative_test"


def test_assoc_efficiency_structure_preservation():
    """Test that assoc preserves the tree structure efficiently"""
    # Create a set with known structure
    keys = [i for i in range(1, 16)]  # 1 to 15, should create balanced tree
    original_set = PSet.mk(keys)
    result_map = PMap.assoc(original_set, "test")

    # Size should be preserved
    assert original_set.size() == result_map.size()

    # Order should be preserved
    assert list(original_set.iter()) == list(result_map.keys())

    # All values should be the same
    for key in keys:
        assert result_map.get(key) == "test"


def test_filter_keys_empty():
    """Test filtering keys on an empty map"""
    empty = PMap.empty(int, str)
    filtered = empty.filter_keys(lambda x: x > 0)
    assert filtered.null()
    assert list(filtered.items()) == []


def test_filter_keys_single_match():
    """Test filtering keys on a single entry that matches"""
    single_map = PMap.singleton(5, "five")
    filtered = single_map.filter_keys(lambda x: x > 0)
    assert list(filtered.items()) == [(5, "five")]
    assert filtered.size() == 1


def test_filter_keys_single_no_match():
    """Test filtering keys on a single entry that doesn't match"""
    single_map = PMap.singleton(-5, "negative")
    filtered = single_map.filter_keys(lambda x: x > 0)
    assert filtered.null()
    assert list(filtered.items()) == []


def test_filter_keys_multiple():
    """Test filtering keys on maps with multiple entries"""
    map_obj = PMap.mk(
        [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five"), (6, "six")]
    )
    filtered = map_obj.filter_keys(lambda x: x % 2 == 0)  # Even keys
    assert list(filtered.items()) == [(2, "two"), (4, "four"), (6, "six")]
    assert filtered.size() == 3

    # Original map unchanged
    assert list(map_obj.items()) == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
    ]


def test_filter_keys_all_match():
    """Test filtering keys where all keys match"""
    map_obj = PMap.mk([(2, "two"), (4, "four"), (6, "six"), (8, "eight")])
    filtered = map_obj.filter_keys(lambda x: x % 2 == 0)
    assert list(filtered.items()) == [(2, "two"), (4, "four"), (6, "six"), (8, "eight")]
    assert filtered.size() == 4


def test_filter_keys_none_match():
    """Test filtering keys where no keys match"""
    map_obj = PMap.mk([(1, "one"), (3, "three"), (5, "five"), (7, "seven")])
    filtered = map_obj.filter_keys(lambda x: x % 2 == 0)
    assert filtered.null()
    assert list(filtered.items()) == []


def test_filter_keys_string_keys():
    """Test filtering string keys"""
    map_obj = PMap.mk(
        [("apple", 1), ("banana", 2), ("cherry", 3), ("apricot", 4), ("blueberry", 5)]
    )
    filtered = map_obj.filter_keys(lambda s: s.startswith("a"))
    assert list(filtered.items()) == [("apple", 1), ("apricot", 4)]
    assert filtered.size() == 2


def test_filter_keys_negative_numbers():
    """Test filtering keys with negative numbers"""
    map_obj = PMap.mk(
        [
            (-5, "neg_five"),
            (-2, "neg_two"),
            (0, "zero"),
            (3, "three"),
            (-1, "neg_one"),
            (7, "seven"),
        ]
    )
    filtered = map_obj.filter_keys(lambda x: x < 0)
    assert list(filtered.items()) == [
        (-5, "neg_five"),
        (-2, "neg_two"),
        (-1, "neg_one"),
    ]
    assert filtered.size() == 3


def test_filter_keys_persistence():
    """Test that filter_keys creates new maps without modifying originals"""
    original = PMap.mk([(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")])
    filtered = original.filter_keys(lambda x: x > 3)

    # Original should be unchanged
    assert list(original.items()) == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
    ]
    assert original.size() == 5

    # Filtered should contain only matching entries
    assert list(filtered.items()) == [(4, "four"), (5, "five")]
    assert filtered.size() == 2


def test_map_values_empty():
    """Test mapping values on an empty map"""
    empty = PMap.empty(int, str)
    mapped = empty.map_values(lambda x: x.upper())
    assert mapped.null()
    assert list(mapped.items()) == []


def test_map_values_single():
    """Test mapping values on a single entry"""
    single_map = PMap.singleton(5, "five")
    mapped = single_map.map_values(lambda x: x.upper())
    assert list(mapped.items()) == [(5, "FIVE")]
    assert mapped.size() == 1


def test_map_values_multiple():
    """Test mapping values on maps with multiple entries"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    mapped = map_obj.map_values(lambda x: x.upper())
    assert list(mapped.items()) == [(1, "ONE"), (2, "TWO"), (3, "THREE")]
    assert mapped.size() == 3

    # Original map unchanged
    assert list(map_obj.items()) == [(1, "one"), (2, "two"), (3, "three")]


def test_map_values_type_change():
    """Test mapping values that changes the value type"""
    map_obj = PMap.mk([(1, "one"), (2, "two"), (3, "three")])
    mapped = map_obj.map_values(lambda x: len(x))  # String -> int
    assert list(mapped.items()) == [(1, 3), (2, 3), (3, 5)]
    assert mapped.size() == 3


def test_map_values_preserves_structure():
    """Test that map_values preserves the tree structure"""
    # Create a large map to test structure preservation
    map_obj = PMap.mk([(i, f"value_{i}") for i in range(20)])
    mapped = map_obj.map_values(lambda x: x.upper())

    # Should have same keys in same order
    assert list(map_obj.keys()) == list(mapped.keys())
    assert map_obj.size() == mapped.size()

    # Values should be transformed
    for i in range(20):
        assert mapped.get(i) == f"VALUE_{i}"


def test_map_values_numeric_transformation():
    """Test mapping numeric values"""
    map_obj = PMap.mk([("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    mapped = map_obj.map_values(lambda x: x * 10)
    assert list(mapped.items()) == [("a", 10), ("b", 20), ("c", 30), ("d", 40)]
    assert mapped.size() == 4


def test_map_values_complex_transformation():
    """Test mapping values with complex transformations"""
    map_obj = PMap.mk([(1, ["a", "b"]), (2, ["c", "d", "e"]), (3, ["f"])])
    mapped = map_obj.map_values(lambda x: len(x))  # List -> length
    assert list(mapped.items()) == [(1, 2), (2, 3), (3, 1)]
    assert mapped.size() == 3


def test_map_values_persistence():
    """Test that map_values creates new maps without modifying originals"""
    original = PMap.mk([(1, "hello"), (2, "world"), (3, "test")])
    mapped = original.map_values(lambda x: x.upper())

    # Original should be unchanged
    assert list(original.items()) == [(1, "hello"), (2, "world"), (3, "test")]
    assert original.size() == 3

    # Mapped should have transformed values
    assert list(mapped.items()) == [(1, "HELLO"), (2, "WORLD"), (3, "TEST")]
    assert mapped.size() == 3


def test_method_chaining_filter_and_map():
    """Test chaining filter_keys and map_values operations"""
    map_obj = PMap.mk(
        [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five"), (6, "six")]
    )

    # Chain: filter even keys, then uppercase values
    result = map_obj.filter_keys(
        lambda x: x % 2 == 0
    ).map_values(  # [(2, "two"), (4, "four"), (6, "six")]
        lambda x: x.upper()
    )  # [(2, "TWO"), (4, "FOUR"), (6, "SIX")]

    assert list(result.items()) == [(2, "TWO"), (4, "FOUR"), (6, "SIX")]
    assert result.size() == 3

    # Original map unchanged
    assert list(map_obj.items()) == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
    ]


def test_map_values_with_none_values():
    """Test map_values behavior with None values"""
    map_obj = PMap.mk([(1, None), (2, "hello"), (3, None)])
    mapped = map_obj.map_values(lambda x: "empty" if x is None else x.upper())
    assert list(mapped.items()) == [(1, "empty"), (2, "HELLO"), (3, "empty")]
    assert mapped.size() == 3


def test_filter_keys_large_map():
    """Test filter_keys on large map"""
    map_obj = PMap.mk([(i, f"value_{i}") for i in range(100)])
    filtered = map_obj.filter_keys(lambda x: x % 10 == 0)  # Multiples of 10
    expected = [(i, f"value_{i}") for i in range(100) if i % 10 == 0]
    assert list(filtered.items()) == expected
    assert filtered.size() == len(expected)


def test_map_values_large_map():
    """Test map_values on large map"""
    map_obj = PMap.mk([(i, i) for i in range(50)])
    mapped = map_obj.map_values(lambda x: x * 2)
    expected = [(i, i * 2) for i in range(50)]
    assert list(mapped.items()) == expected
    assert mapped.size() == 50
