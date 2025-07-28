from centipede.spiny.map import PMap


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

    assert map_obj.get(4) is None
    assert map_obj.get(0) is None
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
    assert result.get(2) is None
    assert result.get(1) == "one"
    assert result.get(3) == "three"
    assert result.get(4) == "four"

    # Remove first key
    result2 = result.remove(1)
    assert result2.size() == 2
    assert result2.get(1) is None
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
    assert empty.get(1) is None
    assert not empty.contains(1)
    assert empty.remove(1).null()

    # Single entry operations
    single = PMap.singleton(1, "one")
    assert single.remove(2) == single  # Remove non-existent key
    assert single.remove(1).null()  # Remove existing key

    # Test with None values
    map_with_none = PMap.mk([(1, None), (2, "two")])
    assert map_with_none.get(1) is None
    assert map_with_none.contains(1)  # Should still contain the key
    assert map_with_none.get(3) is None
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
