from centipede.spiny.heap import PHeap


def test_empty_heap():
    """Test creating an empty PHeap and asserting it is empty"""
    heap = PHeap.empty(int)
    assert heap.null()
    assert heap.size() == 0

    # Test find_min returns None for empty heap
    find_result = heap.find_min()
    assert find_result is None

    # Test delete_min returns None for empty heap
    del_result = heap.delete_min()
    assert del_result is None


def test_singleton():
    """Test creating a singleton heap"""
    heap = PHeap.singleton(5)
    assert not heap.null()
    assert heap.size() == 1

    # Test find_min returns the single element
    result = heap.find_min()
    assert result is not None
    value, rest = result
    assert value == 5
    assert rest.null()


def test_insert_single():
    """Test inserting a single element into an empty heap"""
    heap = PHeap.empty(int)
    heap_with_one = heap.insert(10)

    assert not heap_with_one.null()
    assert heap_with_one.size() == 1

    result = heap_with_one.find_min()
    assert result is not None
    value, rest = result
    assert value == 10
    assert rest.null()


def test_insert_multiple():
    """Test inserting multiple elements maintains min-heap property"""
    heap = PHeap.empty(int)
    heap = heap.insert(5)
    heap = heap.insert(2)
    heap = heap.insert(8)
    heap = heap.insert(1)
    heap = heap.insert(7)

    assert heap.size() == 5

    # Should find minimum element (1)
    result = heap.find_min()
    assert result is not None
    value, _ = result
    assert value == 1


def test_delete_min():
    """Test delete_min removes minimum element"""
    heap = PHeap.empty(int)
    heap = heap.insert(5)
    heap = heap.insert(2)
    heap = heap.insert(8)
    heap = heap.insert(1)

    # Delete minimum (1)
    result = heap.delete_min()
    assert result is not None
    assert result.size() == 3

    # Next minimum should be 2
    min_result = result.find_min()
    assert min_result is not None
    value, _ = min_result
    assert value == 2


def test_merge():
    """Test merging two heaps"""
    heap1 = PHeap.empty(int)
    heap1 = heap1.insert(1).insert(3).insert(5)

    heap2 = PHeap.empty(int)
    heap2 = heap2.insert(2).insert(4).insert(6)

    merged = heap1.merge(heap2)
    assert merged.size() == 6

    # Should find minimum from both heaps
    result = merged.find_min()
    assert result is not None
    value, _ = result
    assert value == 1


def test_merge_operator():
    """Test merging with + operator"""
    heap1 = PHeap.singleton(3)
    heap2 = PHeap.singleton(1)

    merged = heap1 + heap2
    assert merged.size() == 2

    result = merged.find_min()
    assert result is not None
    value, _ = result
    assert value == 1


def test_mk_from_iterable():
    """Test creating heap from iterable"""
    values = [5, 2, 8, 1, 9, 3]
    heap = PHeap.mk(values)

    assert heap.size() == 6

    # Should find minimum
    result = heap.find_min()
    assert result is not None
    value, _ = result
    assert value == 1


def test_iteration():
    """Test iterating through heap in sorted order"""
    values = [5, 2, 8, 1, 9, 3]
    heap = PHeap.mk(values)

    sorted_values = list(heap.iter())
    assert sorted_values == sorted(values)


def test_persistence():
    """Test that operations don't modify original heap"""
    original = PHeap.mk([3, 1, 4])
    inserted = original.insert(2)

    # Original should be unchanged
    assert original.size() == 3
    assert inserted.size() == 4

    # Delete from inserted shouldn't affect original
    deleted = inserted.delete_min()
    assert deleted is not None
    assert original.size() == 3
    assert inserted.size() == 4
    assert deleted.size() == 3


def test_ordering_with_duplicates():
    """Test heap with duplicate elements"""
    heap = PHeap.mk([3, 1, 3, 2, 1])
    assert heap.size() == 5

    # Should handle duplicates correctly
    sorted_values = list(heap.iter())
    assert sorted_values == [1, 1, 2, 3, 3]


def test_large_heap():
    """Test with larger number of elements"""
    values = list(range(100, 0, -1))  # 100 down to 1
    heap = PHeap.mk(values)

    assert heap.size() == 100

    # Minimum should be 1
    result = heap.find_min()
    assert result is not None
    value, _ = result
    assert value == 1

    # All elements should come out in sorted order
    sorted_values = list(heap.iter())
    assert sorted_values == list(range(1, 101))


def test_string_values():
    """Test heap with string values"""
    words = ["zebra", "apple", "banana", "cherry"]
    heap = PHeap.mk(words)

    result = heap.find_min()
    assert result is not None
    value, _ = result
    assert value == "apple"

    sorted_words = list(heap.iter())
    assert sorted_words == sorted(words)
