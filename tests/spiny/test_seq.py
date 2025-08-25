from typing import Dict, List, Optional, Tuple

from spiny.seq import PSeq


def test_empty_seq():
    """Test creating an empty PSeq and asserting it is empty"""
    seq = PSeq.empty(int)
    assert seq.null()

    # Test derived properties
    assert seq.size() == 0
    assert seq.list() == []


def test_cons_uncons():
    """Test that uncons returns what was consed"""
    seq = PSeq.empty(int)

    # Test with single element
    seq_with_one = seq.cons(42)
    result = seq_with_one.uncons()
    assert result is not None
    head, tail = result
    assert head == 42
    assert tail.null()

    # Test derived properties
    assert seq_with_one.size() == 1
    assert seq_with_one.list() == [42]

    # Test with multiple elements
    seq_with_many = seq.cons(1).cons(2).cons(3)
    result = seq_with_many.uncons()
    assert result is not None
    head, tail = result
    assert head == 3

    # Verify we can uncons the tail
    result2 = tail.uncons()
    assert result2 is not None
    head2, tail2 = result2
    assert head2 == 2

    # And the tail of that
    result3 = tail2.uncons()
    assert result3 is not None
    head3, tail3 = result3
    assert head3 == 1
    assert tail3.null()

    # Test derived properties
    assert seq_with_many.size() == 3
    assert seq_with_many.list() == [3, 2, 1]

    # Test empty seq returns None
    empty_result: Optional[Tuple[int, PSeq[int]]] = PSeq.empty(int).uncons()
    assert empty_result is None


def test_snoc_unsnoc():
    """Test that unsnoc returns what was snoced"""
    seq: PSeq[int] = PSeq.empty(int)

    # Test with single element
    seq_with_one = seq.snoc(42)
    result = seq_with_one.unsnoc()
    assert result is not None
    init, last = result
    assert last == 42
    assert init.null()

    # Test derived properties
    assert seq_with_one.size() == 1
    assert seq_with_one.list() == [42]

    # Test with multiple elements
    seq_with_many = seq.snoc(1).snoc(2).snoc(3)
    result = seq_with_many.unsnoc()
    assert result is not None
    init, last = result
    assert last == 3

    # Verify we can unsnoc the init
    result2 = init.unsnoc()
    assert result2 is not None
    init2, last2 = result2
    assert last2 == 2

    # And the init of that
    result3 = init2.unsnoc()
    assert result3 is not None
    init3, last3 = result3
    assert last3 == 1
    assert init3.null()

    # Test derived properties
    assert seq_with_many.size() == 3
    assert seq_with_many.list() == [1, 2, 3]

    # Test empty seq returns None
    empty_result: Optional[Tuple[PSeq[int], int]] = PSeq.empty(int).unsnoc()
    assert empty_result is None


def test_concat():
    """Test concatenating sequences"""
    empty: PSeq[int] = PSeq.empty(int)
    single1 = PSeq.singleton(1)
    single2 = PSeq.singleton(2)

    # Test empty + empty = empty
    result1 = empty.concat(empty)
    assert result1.null()
    assert result1.list() == []

    # Test empty + single = single
    result2 = empty.concat(single1)
    assert result2.list() == [1]

    # Test single + empty = single
    result3 = single1.concat(empty)
    assert result3.list() == [1]

    # Test single + single = deep with 2 elements
    result4 = single1.concat(single2)
    assert result4.list() == [1, 2]

    # Test single + deep sequence
    deep_seq = empty.snoc(3).snoc(4).snoc(5)
    result5 = single1.concat(deep_seq)
    assert result5.list() == [1, 3, 4, 5]

    # Test deep + single
    result6 = deep_seq.concat(single2)
    assert result6.list() == [3, 4, 5, 2]

    # Test deep + deep
    deep_seq2 = empty.snoc(6).snoc(7)
    result7 = deep_seq.concat(deep_seq2)
    assert result7.list() == [3, 4, 5, 6, 7]


def test_lookup_empty():
    """Test lookup on empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)

    # Any index should return None
    assert empty.lookup(0) is None
    assert empty.lookup(1) is None
    assert empty.lookup(-1) is None


def test_lookup_single():
    """Test lookup on single element sequence"""
    single = PSeq.singleton(42)

    # Index 0 should return the element
    assert single.lookup(0) == 42

    # Any other index should return None
    assert single.lookup(1) is None
    assert single.lookup(-1) is None
    assert single.lookup(100) is None


def test_lookup_multiple():
    """Test lookup on sequences with multiple elements"""
    # Test with cons operations
    seq_cons = PSeq.empty(int).cons(1).cons(2).cons(3)  # [3, 2, 1]
    assert seq_cons.lookup(0) == 3
    assert seq_cons.lookup(1) == 2
    assert seq_cons.lookup(2) == 1
    assert seq_cons.lookup(3) is None
    assert seq_cons.lookup(-1) is None

    # Test with snoc operations
    seq_snoc: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)  # [1, 2, 3]
    assert seq_snoc.lookup(0) == 1
    assert seq_snoc.lookup(1) == 2
    assert seq_snoc.lookup(2) == 3
    assert seq_snoc.lookup(3) is None
    assert seq_snoc.lookup(-1) is None


def test_lookup_deep_sequence():
    """Test lookup on deep sequences that trigger complex finger tree structure"""
    # Create a larger sequence to test deep structure
    seq: PSeq[int] = PSeq.empty(int)
    values = list(range(20))  # [0, 1, 2, ..., 19]

    # Build sequence with snoc
    for val in values:
        seq = seq.snoc(val)

    # Get the actual sequence order (finger tree may rebalance elements)
    actual_list = seq.list()

    # Test all valid indices against the actual list representation
    for i, expected in enumerate(actual_list):
        assert seq.lookup(i) == expected

    # Test out of bounds
    assert seq.lookup(len(actual_list)) is None
    assert seq.lookup(100) is None
    assert seq.lookup(-1) is None


def test_lookup_mixed_operations():
    """Test lookup on sequences built with mixed cons/snoc operations"""
    seq: PSeq[int] = PSeq.empty(int).snoc(5).snoc(6).cons(4).cons(3)  # [3, 4, 5, 6]

    assert seq.lookup(0) == 3
    assert seq.lookup(1) == 4
    assert seq.lookup(2) == 5
    assert seq.lookup(3) == 6
    assert seq.lookup(4) is None

    # Verify against list for consistency
    list_repr = seq.list()
    for i in range(len(list_repr)):
        assert seq.lookup(i) == list_repr[i]


def test_lookup_after_concat():
    """Test lookup on sequences after concatenation"""
    seq1: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2)  # [1, 2]
    seq2: PSeq[int] = PSeq.empty(int).snoc(3).snoc(4)  # [3, 4]
    concat_seq = seq1.concat(seq2)  # [1, 2, 3, 4]

    assert concat_seq.lookup(0) == 1
    assert concat_seq.lookup(1) == 2
    assert concat_seq.lookup(2) == 3
    assert concat_seq.lookup(3) == 4
    assert concat_seq.lookup(4) is None

    # Verify against list for consistency
    list_repr = concat_seq.list()
    for i in range(len(list_repr)):
        assert concat_seq.lookup(i) == list_repr[i]


def test_mk():
    """Test creating a sequence from an Iterable"""
    assert PSeq.mk(range(10)).list() == list(range(10))


def test_update_empty():
    """Test updating an empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)

    # Updating any index should return the same empty sequence
    assert empty.update(0, 42) == empty
    assert empty.update(1, 42) == empty
    assert empty.update(-1, 42) == empty


def test_update_single():
    """Test updating a single element sequence"""
    single = PSeq.singleton(42)

    # Valid update at index 0
    updated = single.update(0, 100)
    assert updated.lookup(0) == 100
    assert updated.size() == 1
    assert updated.list() == [100]

    # Invalid indices should return unchanged sequence
    assert single.update(1, 999) == single
    assert single.update(-1, 999) == single
    assert single.update(10, 999) == single


def test_update_multiple():
    """Test updating sequences with multiple elements"""
    # Test with snoc operations: [1, 2, 3]
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)

    # Update each valid index
    updated0 = seq.update(0, 10)
    assert updated0.list() == [10, 2, 3]
    assert updated0.size() == 3

    updated1 = seq.update(1, 20)
    assert updated1.list() == [1, 20, 3]
    assert updated1.size() == 3

    updated2 = seq.update(2, 30)
    assert updated2.list() == [1, 2, 30]
    assert updated2.size() == 3

    # Invalid indices should return unchanged sequence
    assert seq.update(3, 999) == seq
    assert seq.update(-1, 999) == seq
    assert seq.update(100, 999) == seq

    # Original sequence should be unchanged (immutable)
    assert seq.list() == [1, 2, 3]


def test_update_cons_sequence():
    """Test updating a sequence built with cons operations"""
    # [3, 2, 1] from cons operations
    seq = PSeq.empty(int).cons(1).cons(2).cons(3)

    # Update each valid index
    updated0 = seq.update(0, 30)
    assert updated0.list() == [30, 2, 1]

    updated1 = seq.update(1, 20)
    assert updated1.list() == [3, 20, 1]

    updated2 = seq.update(2, 10)
    assert updated2.list() == [3, 2, 10]

    # Original sequence unchanged
    assert seq.list() == [3, 2, 1]


def test_update_deep_sequence():
    """Test updating deep sequences that trigger complex finger tree structure"""
    # Create a larger sequence to test deep structure
    seq: PSeq[int] = PSeq.empty(int)
    values = list(range(20))  # [0, 1, 2, ..., 19]

    # Build sequence with snoc
    for val in values:
        seq = seq.snoc(val)

    original_list = seq.list()

    # Test updating various indices
    for i in range(len(original_list)):
        new_value = original_list[i] + 100
        updated = seq.update(i, new_value)

        # Check the updated value
        assert updated.lookup(i) == new_value
        assert updated.size() == seq.size()

        # Check that other values are unchanged
        updated_list = updated.list()
        for j in range(len(updated_list)):
            if j == i:
                assert updated_list[j] == new_value
            else:
                assert updated_list[j] == original_list[j]

    # Test out of bounds updates
    assert seq.update(len(original_list), 999) == seq
    assert seq.update(-1, 999) == seq

    # Original sequence should be unchanged
    assert seq.list() == original_list


def test_update_mixed_operations():
    """Test updating sequences built with mixed cons/snoc operations"""
    # [3, 4, 5, 6] from mixed operations
    seq: PSeq[int] = PSeq.empty(int).snoc(5).snoc(6).cons(4).cons(3)

    # Update each position
    updated0 = seq.update(0, 30)
    assert updated0.list() == [30, 4, 5, 6]

    updated1 = seq.update(1, 40)
    assert updated1.list() == [3, 40, 5, 6]

    updated2 = seq.update(2, 50)
    assert updated2.list() == [3, 4, 50, 6]

    updated3 = seq.update(3, 60)
    assert updated3.list() == [3, 4, 5, 60]

    # Original unchanged
    assert seq.list() == [3, 4, 5, 6]


def test_update_after_concat():
    """Test updating sequences after concatenation"""
    seq1: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2)  # [1, 2]
    seq2: PSeq[int] = PSeq.empty(int).snoc(3).snoc(4)  # [3, 4]
    concat_seq = seq1.concat(seq2)  # [1, 2, 3, 4]

    # Update each position in concatenated sequence
    updated0 = concat_seq.update(0, 10)
    assert updated0.list() == [10, 2, 3, 4]

    updated1 = concat_seq.update(1, 20)
    assert updated1.list() == [1, 20, 3, 4]

    updated2 = concat_seq.update(2, 30)
    assert updated2.list() == [1, 2, 30, 4]

    updated3 = concat_seq.update(3, 40)
    assert updated3.list() == [1, 2, 3, 40]

    # Out of bounds
    assert concat_seq.update(4, 999) == concat_seq

    # Original sequences and concatenated sequence unchanged
    assert seq1.list() == [1, 2]
    assert seq2.list() == [3, 4]
    assert concat_seq.list() == [1, 2, 3, 4]


def test_update_persistence():
    """Test that updates create new sequences without modifying originals"""
    original: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)

    # Multiple updates creating different sequences
    updated1 = original.update(0, 100)
    updated2 = original.update(1, 200)
    updated3 = original.update(2, 300)

    # All sequences should be different
    assert original.list() == [1, 2, 3]
    assert updated1.list() == [100, 2, 3]
    assert updated2.list() == [1, 200, 3]
    assert updated3.list() == [1, 2, 300]

    # Chain updates
    chained = original.update(0, 10).update(1, 20).update(2, 30)
    assert chained.list() == [10, 20, 30]
    assert original.list() == [1, 2, 3]  # Still unchanged


def test_map_empty():
    """Test mapping over an empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)
    mapped = empty.map(lambda x: x * 2)
    assert mapped.null()
    assert mapped.list() == []


def test_map_single():
    """Test mapping over a single element sequence"""
    single = PSeq.singleton(5)
    mapped = single.map(lambda x: x * 2)
    assert mapped.list() == [10]
    assert mapped.size() == 1


def test_map_multiple():
    """Test mapping over sequences with multiple elements"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)
    mapped = seq.map(lambda x: x * 2)
    assert mapped.list() == [2, 4, 6]
    assert mapped.size() == 3

    # Original sequence unchanged
    assert seq.list() == [1, 2, 3]


def test_map_type_change():
    """Test mapping that changes the element type"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)
    mapped = seq.map(lambda x: str(x))
    assert mapped.list() == ["1", "2", "3"]
    assert mapped.size() == 3


def test_map_deep_sequence():
    """Test mapping over deep sequences"""
    seq: PSeq[int] = PSeq.empty(int)
    for i in range(20):
        seq = seq.snoc(i)

    mapped = seq.map(lambda x: x * 3)
    expected = [i * 3 for i in range(20)]
    assert mapped.list() == expected
    assert mapped.size() == 20


def test_map_cons_sequence():
    """Test mapping over sequences built with cons"""
    seq = PSeq.empty(int).cons(1).cons(2).cons(3)  # [3, 2, 1]
    mapped = seq.map(lambda x: x + 10)
    assert mapped.list() == [13, 12, 11]


def test_filter_empty():
    """Test filtering an empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)
    filtered = empty.filter(lambda x: x > 0)
    assert filtered.null()
    assert filtered.list() == []


def test_filter_single_match():
    """Test filtering a single element that matches"""
    single = PSeq.singleton(5)
    filtered = single.filter(lambda x: x > 0)
    assert filtered.list() == [5]
    assert filtered.size() == 1


def test_filter_single_no_match():
    """Test filtering a single element that doesn't match"""
    single = PSeq.singleton(-5)
    filtered = single.filter(lambda x: x > 0)
    assert filtered.null()
    assert filtered.list() == []


def test_filter_multiple():
    """Test filtering sequences with multiple elements"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3).snoc(4).snoc(5)
    filtered = seq.filter(lambda x: x % 2 == 0)  # Even numbers
    assert filtered.list() == [2, 4]
    assert filtered.size() == 2

    # Original sequence unchanged
    assert seq.list() == [1, 2, 3, 4, 5]


def test_filter_all_match():
    """Test filtering where all elements match"""
    seq: PSeq[int] = PSeq.empty(int).snoc(2).snoc(4).snoc(6)
    filtered = seq.filter(lambda x: x % 2 == 0)
    assert filtered.list() == [2, 4, 6]
    assert filtered.size() == 3


def test_filter_none_match():
    """Test filtering where no elements match"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(3).snoc(5)
    filtered = seq.filter(lambda x: x % 2 == 0)
    assert filtered.null()
    assert filtered.list() == []


def test_filter_deep_sequence():
    """Test filtering deep sequences"""
    seq: PSeq[int] = PSeq.empty(int)
    for i in range(20):
        seq = seq.snoc(i)

    filtered = seq.filter(lambda x: x % 3 == 0)  # Multiples of 3
    expected = [i for i in range(20) if i % 3 == 0]
    assert filtered.list() == expected


def test_flat_map_empty():
    """Test flat_map over an empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)
    flat_mapped = empty.flat_map(lambda x: PSeq.singleton(x).snoc(x + 1))
    assert flat_mapped.null()
    assert flat_mapped.list() == []


def test_flat_map_single():
    """Test flat_map over a single element"""
    single = PSeq.singleton(5)
    flat_mapped = single.flat_map(lambda x: PSeq.singleton(x).snoc(x * 2))
    assert flat_mapped.list() == [5, 10]
    assert flat_mapped.size() == 2


def test_flat_map_multiple():
    """Test flat_map over multiple elements"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)
    flat_mapped = seq.flat_map(lambda x: PSeq.singleton(x).snoc(x * 2))
    assert flat_mapped.list() == [1, 2, 2, 4, 3, 6]
    assert flat_mapped.size() == 6

    # Original sequence unchanged
    assert seq.list() == [1, 2, 3]


def test_flat_map_empty_results():
    """Test flat_map where some results are empty"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3).snoc(4)
    flat_mapped = seq.flat_map(
        lambda x: PSeq.singleton(x) if x % 2 == 0 else PSeq.empty()
    )
    assert flat_mapped.list() == [2, 4]
    assert flat_mapped.size() == 2


def test_flat_map_varying_lengths():
    """Test flat_map with varying result lengths"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)
    flat_mapped = seq.flat_map(lambda x: PSeq.mk(range(x)))
    # x=1 -> [0], x=2 -> [0,1], x=3 -> [0,1,2]
    assert flat_mapped.list() == [0, 0, 1, 0, 1, 2]
    assert flat_mapped.size() == 6


def test_flat_map_type_change():
    """Test flat_map that changes the element type"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2)
    flat_mapped = seq.flat_map(lambda x: PSeq.singleton(str(x)).snoc(str(x * 2)))
    assert flat_mapped.list() == ["1", "2", "2", "4"]
    assert flat_mapped.size() == 4


def test_method_chaining():
    """Test chaining map, filter, and flat_map operations"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3).snoc(4).snoc(5)

    # Chain: filter evens, map to double, flat_map to duplicate
    result = (
        seq.filter(lambda x: x % 2 == 0)  # [2, 4]
        .map(lambda x: x * 2)  # [4, 8]
        .flat_map(lambda x: PSeq.singleton(x).snoc(x))
    )  # [4, 4, 8, 8]

    assert result.list() == [4, 4, 8, 8]
    assert result.size() == 4

    # Original sequence unchanged
    assert seq.list() == [1, 2, 3, 4, 5]


def test_fold_empty():
    """Test folding an empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)
    result = empty.fold(lambda acc, x: acc + x, 0)
    assert result == 0

    # Test with different accumulator
    result2 = empty.fold(lambda acc, x: acc * x, 1)
    assert result2 == 1


def test_fold_single():
    """Test folding a single element sequence"""
    single = PSeq.singleton(5)

    # Sum operation
    result = single.fold(lambda acc, x: acc + x, 0)
    assert result == 5

    # Product operation
    result2 = single.fold(lambda acc, x: acc * x, 1)
    assert result2 == 5

    # String concatenation
    single_str = PSeq.singleton("hello")
    result3 = single_str.fold(lambda acc, x: acc + x, "")
    assert result3 == "hello"


def test_fold_multiple():
    """Test folding sequences with multiple elements"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3).snoc(4).snoc(5)

    # Sum operation
    result = seq.fold(lambda acc, x: acc + x, 0)
    assert result == 15

    # Product operation
    result2 = seq.fold(lambda acc, x: acc * x, 1)
    assert result2 == 120

    # Build list (should preserve order)
    result3: List[int] = seq.fold(lambda acc, x: acc + [x], [])
    assert result3 == [1, 2, 3, 4, 5]

    # Original sequence unchanged
    assert seq.list() == [1, 2, 3, 4, 5]


def test_fold_cons_sequence():
    """Test folding sequences built with cons"""
    seq = PSeq.empty(int).cons(1).cons(2).cons(3)  # [3, 2, 1]

    # Sum operation
    result = seq.fold(lambda acc, x: acc + x, 0)
    assert result == 6

    # Build list (should preserve cons order)
    result2: List[int] = seq.fold(lambda acc, x: acc + [x], [])
    assert result2 == [3, 2, 1]


def test_fold_type_change():
    """Test folding that changes the accumulator type"""
    seq: PSeq[int] = PSeq.empty(int).snoc(1).snoc(2).snoc(3)

    # Convert numbers to string representation
    result = seq.fold(lambda acc, x: acc + str(x), "")
    assert result == "123"

    # Count elements
    result2 = seq.fold(lambda acc, x: acc + 1, 0)
    assert result2 == 3


def test_fold_deep_sequence():
    """Test folding deep sequences"""
    seq: PSeq[int] = PSeq.empty(int)
    for i in range(20):
        seq = seq.snoc(i)

    # Sum all elements
    result = seq.fold(lambda acc, x: acc + x, 0)
    expected_sum = sum(range(20))
    assert result == expected_sum

    # Count elements
    count = seq.fold(lambda acc, x: acc + 1, 0)
    assert count == 20


def test_fold_with_index_empty():
    """Test fold_with_index on an empty sequence"""
    empty: PSeq[int] = PSeq.empty(int)
    result = empty.fold_with_index(lambda acc, i, x: acc + x + i, 0)
    assert result == 0

    # Test with different accumulator
    result2: List[str] = empty.fold_with_index(lambda acc, i, x: acc + [f"{i}:{x}"], [])
    assert result2 == []


def test_fold_with_index_single():
    """Test fold_with_index on a single element sequence"""
    single = PSeq.singleton(5)

    # Add value and index
    result = single.fold_with_index(lambda acc, i, x: acc + x + i, 0)
    assert result == 5  # 0 + 5 + 0

    # Format with index
    result2: List[str] = single.fold_with_index(
        lambda acc, i, x: acc + [f"{i}:{x}"], []
    )
    assert result2 == ["0:5"]


def test_fold_with_index_multiple():
    """Test fold_with_index on sequences with multiple elements"""
    seq: PSeq[int] = PSeq.empty(int).snoc(10).snoc(20).snoc(30)

    # Sum values and indices
    result = seq.fold_with_index(lambda acc, i, x: acc + x + i, 0)
    assert result == 63  # 0 + (10+0) + (20+1) + (30+2) = 63

    # Build indexed list
    result2: List[str] = seq.fold_with_index(lambda acc, i, x: acc + [f"{i}:{x}"], [])
    assert result2 == ["0:10", "1:20", "2:30"]

    # Verify index order matches element order
    indexed_pairs: List[Tuple[int, int]] = seq.fold_with_index(
        lambda acc, i, x: acc + [(i, x)], []
    )
    assert indexed_pairs == [(0, 10), (1, 20), (2, 30)]

    # Original sequence unchanged
    assert seq.list() == [10, 20, 30]


def test_fold_with_index_cons_sequence():
    """Test fold_with_index on sequences built with cons"""
    seq = PSeq.empty(int).cons(10).cons(20).cons(30)  # [30, 20, 10]

    # Build indexed list
    result: List[str] = seq.fold_with_index(lambda acc, i, x: acc + [f"{i}:{x}"], [])
    assert result == ["0:30", "1:20", "2:10"]

    # Verify indices correspond to iteration order, not construction order
    indexed_pairs: List[Tuple[int, int]] = seq.fold_with_index(
        lambda acc, i, x: acc + [(i, x)], []
    )
    assert indexed_pairs == [(0, 30), (1, 20), (2, 10)]


def test_fold_with_index_type_change():
    """Test fold_with_index that changes accumulator type"""
    seq: PSeq[str] = PSeq.empty(str).snoc("a").snoc("b").snoc("c")

    # Create dictionary mapping indices to values
    result: Dict[int, str] = seq.fold_with_index(lambda acc, i, x: {**acc, i: x}, {})
    assert result == {0: "a", 1: "b", 2: "c"}

    # Count characters including index contribution
    result2 = seq.fold_with_index(lambda acc, i, x: acc + len(x) + i, 0)
    assert result2 == 6  # (1+0) + (1+1) + (1+2) = 6


def test_fold_with_index_deep_sequence():
    """Test fold_with_index on deep sequences"""
    seq: PSeq[int] = PSeq.empty(int)
    for i in range(10):
        seq = seq.snoc(i * 10)  # [0, 10, 20, ..., 90]

    # Sum values plus their indices
    result = seq.fold_with_index(lambda acc, i, x: acc + x + i, 0)
    # Sum of values: 0+10+20+...+90 = 450
    # Sum of indices: 0+1+2+...+9 = 45
    # Total: 495
    assert result == 495

    # Verify all indices are correct
    indices: List[int] = seq.fold_with_index(lambda acc, i, x: acc + [i], [])
    assert indices == list(range(10))


def test_fold_with_index_mixed_operations():
    """Test fold_with_index on sequences built with mixed operations"""
    # [30, 40, 50, 60] from mixed cons/snoc
    seq: PSeq[int] = PSeq.empty(int).snoc(50).snoc(60).cons(40).cons(30)

    # Build detailed info with indices
    result: List[Dict[str, int]] = seq.fold_with_index(
        lambda acc, i, x: acc + [{"index": i, "value": x, "product": i * x}], []
    )
    expected = [
        {"index": 0, "value": 30, "product": 0},
        {"index": 1, "value": 40, "product": 40},
        {"index": 2, "value": 50, "product": 100},
        {"index": 3, "value": 60, "product": 180},
    ]
    assert result == expected
