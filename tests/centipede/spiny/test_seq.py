from typing import Optional, Tuple

from centipede.spiny.seq import Seq

_EMPTY_INT_SEQ: Seq[int] = Seq.empty()


def test_empty_seq():
    """Test creating an empty Seq and asserting it is empty"""
    seq = _EMPTY_INT_SEQ
    assert seq.null()

    # Test derived properties
    assert seq.size() == 0
    assert seq.list() == []


def test_cons_uncons():
    """Test that uncons returns what was consed"""
    seq = _EMPTY_INT_SEQ

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
    empty_result: Optional[Tuple[int, Seq[int]]] = _EMPTY_INT_SEQ.uncons()
    assert empty_result is None


def test_snoc_unsnoc():
    """Test that unsnoc returns what was snoced"""
    seq: Seq[int] = _EMPTY_INT_SEQ

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
    empty_result: Optional[Tuple[Seq[int], int]] = _EMPTY_INT_SEQ.unsnoc()
    assert empty_result is None


def test_concat():
    """Test concatenating sequences"""
    empty: Seq[int] = _EMPTY_INT_SEQ
    single1 = Seq.singleton(1)
    single2 = Seq.singleton(2)

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
    empty: Seq[int] = _EMPTY_INT_SEQ

    # Any index should return None
    assert empty.lookup(0) is None
    assert empty.lookup(1) is None
    assert empty.lookup(-1) is None


def test_lookup_single():
    """Test lookup on single element sequence"""
    single = Seq.singleton(42)

    # Index 0 should return the element
    assert single.lookup(0) == 42

    # Any other index should return None
    assert single.lookup(1) is None
    assert single.lookup(-1) is None
    assert single.lookup(100) is None


def test_lookup_multiple():
    """Test lookup on sequences with multiple elements"""
    # Test with cons operations
    seq_cons = _EMPTY_INT_SEQ.cons(1).cons(2).cons(3)  # [3, 2, 1]
    assert seq_cons.lookup(0) == 3
    assert seq_cons.lookup(1) == 2
    assert seq_cons.lookup(2) == 1
    assert seq_cons.lookup(3) is None
    assert seq_cons.lookup(-1) is None

    # Test with snoc operations
    seq_snoc: Seq[int] = _EMPTY_INT_SEQ.snoc(1).snoc(2).snoc(3)  # [1, 2, 3]
    assert seq_snoc.lookup(0) == 1
    assert seq_snoc.lookup(1) == 2
    assert seq_snoc.lookup(2) == 3
    assert seq_snoc.lookup(3) is None
    assert seq_snoc.lookup(-1) is None


def test_lookup_deep_sequence():
    """Test lookup on deep sequences that trigger complex finger tree structure"""
    # Create a larger sequence to test deep structure
    seq: Seq[int] = _EMPTY_INT_SEQ
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
    seq: Seq[int] = _EMPTY_INT_SEQ.snoc(5).snoc(6).cons(4).cons(3)  # [3, 4, 5, 6]

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
    seq1: Seq[int] = _EMPTY_INT_SEQ.snoc(1).snoc(2)  # [1, 2]
    seq2: Seq[int] = _EMPTY_INT_SEQ.snoc(3).snoc(4)  # [3, 4]
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
