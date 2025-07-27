from centipede.spiny.set import PSet


def test_empty_set():
    """Test creating an empty PSet and asserting it is empty"""
    set_obj = PSet.empty(int)
    assert set_obj.null()
    assert set_obj.size() == 0
    assert set_obj.list() == []


def test_singleton_insert():
    """Test inserting a single element"""
    empty_set = PSet.empty(int)
    single_set = empty_set.insert(42)

    assert not single_set.null()
    assert single_set.size() == 1
    assert 42 in single_set.list()
    assert single_set.list() == [42]


def test_singleton_method():
    """Test creating a singleton set directly"""
    single_set = PSet.singleton(42)

    assert not single_set.null()
    assert single_set.size() == 1
    assert single_set.list() == [42]


def test_multiple_inserts():
    """Test inserting multiple elements"""
    set_obj = PSet.empty(int)

    # Insert multiple elements
    set_obj = set_obj.insert(3).insert(1).insert(4).insert(2)

    assert set_obj.size() == 4
    assert not set_obj.null()

    # Elements should be in sorted order due to in-order traversal
    result = set_obj.list()
    assert sorted(result) == result  # Should be sorted
    assert set(result) == {1, 2, 3, 4}


def test_duplicate_inserts():
    """Test that inserting duplicate elements doesn't increase size"""
    set_obj = PSet.empty(int)

    # Insert same element multiple times
    set_obj = set_obj.insert(42).insert(42).insert(42)

    assert set_obj.size() == 1
    assert set_obj.list() == [42]


def test_insert_ordering():
    """Test that elements maintain sorted order after insertion"""
    set_obj = PSet.empty(int)
    values = [5, 2, 8, 1, 9, 3, 7, 4, 6]

    for val in values:
        set_obj = set_obj.insert(val)

    result = set_obj.list()
    assert result == sorted(values)
    assert set_obj.size() == len(values)


def test_mk_from_iterable():
    """Test creating a set from an iterable"""
    values = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    set_obj = PSet.mk(values)

    # Should contain unique values in sorted order
    expected = sorted(set(values))
    assert set_obj.list() == expected
    assert set_obj.size() == len(expected)


def test_mk_empty_iterable():
    """Test creating a set from an empty iterable"""
    set_obj: PSet[int] = PSet.mk([])
    assert set_obj.null()
    assert set_obj.size() == 0
    assert set_obj.list() == []


def test_insert_operator_right():
    """Test >> operator for insertion"""
    set_obj = PSet.empty(int)
    set_obj = set_obj >> 42 >> 24 >> 13

    assert set_obj.size() == 3
    assert set(set_obj.list()) == {42, 24, 13}


def test_insert_operator_left():
    """Test << operator for insertion"""
    set_obj = PSet.empty(int)
    set_obj = 42 << (24 << (13 << set_obj))

    assert set_obj.size() == 3
    assert set(set_obj.list()) == {42, 24, 13}


def test_string_elements():
    """Test set with string elements"""
    set_obj = PSet.empty(str)
    set_obj = set_obj.insert("hello").insert("world").insert("abc")

    assert set_obj.size() == 3
    result = set_obj.list()
    assert sorted(result) == result  # Should be sorted
    assert set(result) == {"hello", "world", "abc"}


def test_large_set_insertion():
    """Test inserting many elements to verify tree balancing"""
    set_obj = PSet.empty(int)
    values = list(range(100))

    # Insert in random order to test balancing
    import random

    random.shuffle(values)

    for val in values:
        set_obj = set_obj.insert(val)

    assert set_obj.size() == 100
    assert set_obj.list() == list(range(100))


def test_insert_with_duplicates_mixed():
    """Test inserting with mixed duplicates"""
    set_obj = PSet.empty(int)

    # Insert pattern: 1, 2, 1, 3, 2, 4, 1
    insertions = [1, 2, 1, 3, 2, 4, 1]
    for val in insertions:
        set_obj = set_obj.insert(val)

    assert set_obj.size() == 4  # Only unique values
    assert set_obj.list() == [1, 2, 3, 4]


def test_insert_negative_numbers():
    """Test inserting negative numbers"""
    set_obj = PSet.empty(int)
    values = [-5, -1, 0, 3, -10, 7]

    for val in values:
        set_obj = set_obj.insert(val)

    assert set_obj.size() == len(values)
    assert set_obj.list() == sorted(values)


def test_persistence():
    """Test that insertions create new sets without modifying originals"""
    original = PSet.empty(int).insert(1).insert(2)
    modified = original.insert(3)

    # Original should be unchanged
    assert original.list() == [1, 2]
    assert original.size() == 2

    # Modified should have new element
    assert modified.list() == [1, 2, 3]
    assert modified.size() == 3


def test_chained_insertions():
    """Test chaining multiple insertions"""
    result = PSet.empty(int).insert(5).insert(2).insert(8).insert(1).insert(9)

    assert result.size() == 5
    assert result.list() == [1, 2, 5, 8, 9]


def test_insert_same_value_returns_same_instance():
    """Test that inserting an existing value returns the same set"""
    set_obj = PSet.empty(int).insert(42)
    same_set = set_obj.insert(42)

    # Should be the same object since value already exists
    assert set_obj is same_set
    assert set_obj.size() == 1
    assert same_set.size() == 1


def test_balanced_tree_property():
    """Test that tree remains reasonably balanced after many insertions"""
    set_obj = PSet.empty(int)

    # Insert sequential numbers (worst case for unbalanced trees)
    for i in range(50):
        set_obj = set_obj.insert(i)

    # If tree is balanced, operations should still be efficient
    # This is more of a performance test, but we can at least verify correctness
    assert set_obj.size() == 50
    assert set_obj.list() == list(range(50))

    # Test that we can still insert efficiently
    set_obj = set_obj.insert(100)
    assert 100 in set_obj.list()
    assert set_obj.size() == 51


def test_merge_empty_sets():
    """Test merging empty sets"""
    empty1 = PSet.empty(int)
    empty2 = PSet.empty(int)
    result = empty1.merge(empty2)

    assert result.null()
    assert result.size() == 0
    assert result.list() == []


def test_merge_empty_with_non_empty():
    """Test merging empty set with non-empty set"""
    empty_set = PSet.empty(int)
    non_empty = PSet.mk([1, 2, 3])

    result1 = empty_set.merge(non_empty)
    result2 = non_empty.merge(empty_set)

    assert result1.list() == [1, 2, 3]
    assert result2.list() == [1, 2, 3]
    assert result1.size() == 3
    assert result2.size() == 3


def test_merge_disjoint_sets():
    """Test merging sets with no common elements"""
    set1 = PSet.mk([1, 3, 5])
    set2 = PSet.mk([2, 4, 6])

    result = set1.merge(set2)

    assert result.size() == 6
    assert result.list() == [1, 2, 3, 4, 5, 6]


def test_merge_overlapping_sets():
    """Test merging sets with some common elements"""
    set1 = PSet.mk([1, 2, 3, 4])
    set2 = PSet.mk([3, 4, 5, 6])

    result = set1.merge(set2)

    assert result.size() == 6
    assert result.list() == [1, 2, 3, 4, 5, 6]


def test_merge_identical_sets():
    """Test merging identical sets"""
    set1 = PSet.mk([1, 2, 3])
    set2 = PSet.mk([1, 2, 3])

    result = set1.merge(set2)

    assert result.size() == 3
    assert result.list() == [1, 2, 3]


def test_merge_subset_sets():
    """Test merging when one set is a subset of another"""
    set1 = PSet.mk([1, 2, 3, 4, 5])
    set2 = PSet.mk([2, 4])

    result1 = set1.merge(set2)
    result2 = set2.merge(set1)

    assert result1.list() == [1, 2, 3, 4, 5]
    assert result2.list() == [1, 2, 3, 4, 5]
    assert result1.size() == 5
    assert result2.size() == 5


def test_merge_single_element_sets():
    """Test merging single element sets"""
    set1 = PSet.singleton(1)
    set2 = PSet.singleton(2)
    set3 = PSet.singleton(1)  # Same element

    result1 = set1.merge(set2)
    result2 = set1.merge(set3)

    assert result1.list() == [1, 2]
    assert result1.size() == 2

    assert result2.list() == [1]
    assert result2.size() == 1


def test_merge_large_sets():
    """Test merging large sets"""
    set1 = PSet.mk(range(0, 100, 2))  # Even numbers 0-98
    set2 = PSet.mk(range(1, 100, 2))  # Odd numbers 1-99

    result = set1.merge(set2)

    assert result.size() == 100
    assert result.list() == list(range(100))


def test_merge_string_sets():
    """Test merging sets of strings"""
    set1 = PSet.mk(["apple", "banana", "cherry"])
    set2 = PSet.mk(["banana", "date", "elderberry"])

    result = set1.merge(set2)

    expected = ["apple", "banana", "cherry", "date", "elderberry"]
    assert result.list() == expected
    assert result.size() == 5


def test_merge_operator_plus():
    """Test merge using + operator"""
    set1 = PSet.mk([1, 2, 3])
    set2 = PSet.mk([3, 4, 5])

    result = set1 + set2

    assert result.list() == [1, 2, 3, 4, 5]
    assert result.size() == 5


def test_merge_chaining():
    """Test chaining multiple merge operations"""
    set1 = PSet.mk([1, 2])
    set2 = PSet.mk([3, 4])
    set3 = PSet.mk([5, 6])

    result = set1.merge(set2).merge(set3)

    assert result.list() == [1, 2, 3, 4, 5, 6]
    assert result.size() == 6


def test_merge_persistence():
    """Test that merge operations don't modify original sets"""
    set1 = PSet.mk([1, 2, 3])
    set2 = PSet.mk([4, 5, 6])

    original1_list = set1.list()
    original2_list = set2.list()

    result = set1.merge(set2)

    # Original sets should be unchanged
    assert set1.list() == original1_list
    assert set2.list() == original2_list
    assert set1.size() == 3
    assert set2.size() == 3

    # Result should contain all elements
    assert result.list() == [1, 2, 3, 4, 5, 6]
    assert result.size() == 6


def test_merge_negative_numbers():
    """Test merging sets with negative numbers"""
    set1 = PSet.mk([-3, -1, 1, 3])
    set2 = PSet.mk([-2, 0, 2])

    result = set1.merge(set2)

    assert result.list() == [-3, -2, -1, 0, 1, 2, 3]
    assert result.size() == 7


def test_find_min_empty_set():
    """Test find_min on empty set returns None"""
    empty_set = PSet.empty(int)
    result = empty_set.find_min()
    assert result is None


def test_find_min_singleton():
    """Test find_min on singleton set"""
    single_set = PSet.singleton(42)
    result = single_set.find_min()

    assert result is not None
    min_val, remaining = result
    assert min_val == 42
    assert remaining.null()
    assert remaining.size() == 0


def test_find_min_multiple_elements():
    """Test find_min on set with multiple elements"""
    set_obj = PSet.mk([5, 2, 8, 1, 9])
    result = set_obj.find_min()

    assert result is not None
    min_val, remaining = result
    assert min_val == 1
    assert remaining.size() == 4
    assert remaining.list() == [2, 5, 8, 9]


def test_find_min_negative_numbers():
    """Test find_min with negative numbers"""
    set_obj = PSet.mk([-5, -1, 0, 3, -10, 7])
    result = set_obj.find_min()

    assert result is not None
    min_val, remaining = result
    assert min_val == -10
    assert remaining.size() == 5
    assert remaining.list() == [-5, -1, 0, 3, 7]


def test_find_min_strings():
    """Test find_min with string elements"""
    set_obj = PSet.mk(["zebra", "apple", "banana", "cherry"])
    result = set_obj.find_min()

    assert result is not None
    min_val, remaining = result
    assert min_val == "apple"
    assert remaining.size() == 3
    assert remaining.list() == ["banana", "cherry", "zebra"]


def test_find_min_persistence():
    """Test that find_min doesn't modify original set"""
    original = PSet.mk([3, 1, 4, 2])
    original_list = original.list()
    original_size = original.size()

    result = original.find_min()

    # Original should be unchanged
    assert original.list() == original_list
    assert original.size() == original_size

    # Result should be correct
    assert result is not None
    min_val, remaining = result
    assert min_val == 1
    assert remaining.list() == [2, 3, 4]


def test_find_min_repeated_calls():
    """Test repeated calls to find_min to extract all elements"""
    original = PSet.mk([5, 2, 8, 1, 9, 3])
    extracted = []
    current = original

    while True:
        result = current.find_min()
        if result is None:
            break
        min_val, remaining = result
        extracted.append(min_val)
        current = remaining

    assert extracted == [1, 2, 3, 5, 8, 9]
    assert current.null()


def test_find_min_large_set():
    """Test find_min on large set to verify performance"""
    values = list(range(100, 0, -1))  # [100, 99, 98, ..., 1]
    set_obj = PSet.mk(values)

    result = set_obj.find_min()

    assert result is not None
    min_val, remaining = result
    assert min_val == 1
    assert remaining.size() == 99
    # Verify the remaining set doesn't contain the minimum
    assert 1 not in remaining.list()


def test_delete_min_empty_set():
    """Test delete_min on empty set returns None"""
    empty_set = PSet.empty(int)
    result = empty_set.delete_min()
    assert result is None


def test_delete_min_singleton():
    """Test delete_min on singleton set"""
    single_set = PSet.singleton(42)
    result = single_set.delete_min()

    assert result is not None
    assert result.null()
    assert result.size() == 0


def test_delete_min_multiple_elements():
    """Test delete_min on set with multiple elements"""
    set_obj = PSet.mk([5, 2, 8, 1, 9])
    result = set_obj.delete_min()

    assert result is not None
    assert result.size() == 4
    assert result.list() == [2, 5, 8, 9]


def test_delete_min_consistency_with_find_min():
    """Test that delete_min is consistent with find_min"""
    set_obj = PSet.mk([7, 3, 11, 1, 9, 5])

    find_result = set_obj.find_min()
    delete_result = set_obj.delete_min()

    assert find_result is not None
    assert delete_result is not None

    _, remaining_from_find = find_result

    # delete_min result should be same as remaining set from find_min
    assert delete_result.list() == remaining_from_find.list()
    assert delete_result.size() == remaining_from_find.size()


def test_find_max_empty_set():
    """Test find_max on empty set returns None"""
    empty_set = PSet.empty(int)
    result = empty_set.find_max()
    assert result is None


def test_find_max_singleton():
    """Test find_max on singleton set"""
    single_set = PSet.singleton(42)
    result = single_set.find_max()

    assert result is not None
    remaining, max_val = result
    assert max_val == 42
    assert remaining.null()
    assert remaining.size() == 0


def test_find_max_multiple_elements():
    """Test find_max on set with multiple elements"""
    set_obj = PSet.mk([5, 2, 8, 1, 9])
    result = set_obj.find_max()

    assert result is not None
    remaining, max_val = result
    assert max_val == 9
    assert remaining.size() == 4
    assert remaining.list() == [1, 2, 5, 8]


def test_find_max_negative_numbers():
    """Test find_max with negative numbers"""
    set_obj = PSet.mk([-5, -1, 0, 3, -10, 7])
    result = set_obj.find_max()

    assert result is not None
    remaining, max_val = result
    assert max_val == 7
    assert remaining.size() == 5
    assert remaining.list() == [-10, -5, -1, 0, 3]


def test_find_max_strings():
    """Test find_max with string elements"""
    set_obj = PSet.mk(["zebra", "apple", "banana", "cherry"])
    result = set_obj.find_max()

    assert result is not None
    remaining, max_val = result
    assert max_val == "zebra"
    assert remaining.size() == 3
    assert remaining.list() == ["apple", "banana", "cherry"]


def test_find_max_persistence():
    """Test that find_max doesn't modify original set"""
    original = PSet.mk([3, 1, 4, 2])
    original_list = original.list()
    original_size = original.size()

    result = original.find_max()

    # Original should be unchanged
    assert original.list() == original_list
    assert original.size() == original_size

    # Result should be correct
    assert result is not None
    remaining, max_val = result
    assert max_val == 4
    assert remaining.list() == [1, 2, 3]


def test_find_max_repeated_calls():
    """Test repeated calls to find_max to extract all elements"""
    original = PSet.mk([5, 2, 8, 1, 9, 3])
    extracted = []
    current = original

    while True:
        result = current.find_max()
        if result is None:
            break
        remaining, max_val = result
        extracted.append(max_val)
        current = remaining

    assert extracted == [9, 8, 5, 3, 2, 1]
    assert current.null()


def test_find_max_large_set():
    """Test find_max on large set to verify performance"""
    values = list(range(1, 101))  # [1, 2, 3, ..., 100]
    set_obj = PSet.mk(values)

    result = set_obj.find_max()

    assert result is not None
    remaining, max_val = result
    assert max_val == 100
    assert remaining.size() == 99
    # Verify the remaining set doesn't contain the maximum
    assert 100 not in remaining.list()


def test_delete_max_empty_set():
    """Test delete_max on empty set returns None"""
    empty_set = PSet.empty(int)
    result = empty_set.delete_max()
    assert result is None


def test_delete_max_singleton():
    """Test delete_max on singleton set"""
    single_set = PSet.singleton(42)
    result = single_set.delete_max()

    assert result is not None
    assert result.null()
    assert result.size() == 0


def test_delete_max_multiple_elements():
    """Test delete_max on set with multiple elements"""
    set_obj = PSet.mk([5, 2, 8, 1, 9])
    result = set_obj.delete_max()

    assert result is not None
    assert result.size() == 4
    assert result.list() == [1, 2, 5, 8]


def test_delete_max_consistency_with_find_max():
    """Test that delete_max is consistent with find_max"""
    set_obj = PSet.mk([7, 3, 11, 1, 9, 5])

    find_result = set_obj.find_max()
    delete_result = set_obj.delete_max()

    assert find_result is not None
    assert delete_result is not None

    remaining_from_find, _ = find_result

    # delete_max result should be same as remaining set from find_max
    assert delete_result.list() == remaining_from_find.list()
    assert delete_result.size() == remaining_from_find.size()


def test_find_min_max_symmetry():
    """Test that find_min and find_max work correctly together"""
    set_obj = PSet.mk([5, 2, 8, 1, 9, 3])

    min_result = set_obj.find_min()
    max_result = set_obj.find_max()

    assert min_result is not None
    assert max_result is not None

    min_val, _ = min_result
    _, max_val = max_result

    assert min_val == 1
    assert max_val == 9

    # Test that removing min and max from original gives same result as
    # removing max then min (or min then max)
    after_min_removal = set_obj.delete_min()
    after_max_removal = set_obj.delete_max()

    assert after_min_removal is not None
    assert after_max_removal is not None

    min_then_max = after_min_removal.delete_max()
    max_then_min = after_max_removal.delete_min()

    assert min_then_max is not None
    assert max_then_min is not None
    assert min_then_max.list() == max_then_min.list()


def test_find_max_single_element_after_operations():
    """Test find_max behavior after various operations"""
    set_obj = PSet.mk([1, 2, 3])

    # Remove minimum twice
    after_first_min = set_obj.delete_min()
    assert after_first_min is not None
    after_second_min = after_first_min.delete_min()
    assert after_second_min is not None

    # Should have only element 3 left
    result = after_second_min.find_max()
    assert result is not None
    remaining, max_val = result
    assert max_val == 3
    assert remaining.null()
