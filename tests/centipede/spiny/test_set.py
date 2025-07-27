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
