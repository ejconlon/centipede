from typing import Optional, Tuple

from centipede.spiny.seq import Seq


def test_empty_seq():
    """Test creating an empty Seq and asserting it is empty"""
    seq: Seq[int] = Seq.empty()
    assert seq.null()

    # Test derived properties
    assert seq.size() == 0
    assert seq.to_list() == []


def test_cons_uncons():
    """Test that uncons returns what was consed"""
    seq: Seq[int] = Seq.empty()

    # Test with single element
    seq_with_one = seq.cons(42)
    result = seq_with_one.uncons()
    assert result is not None
    head, tail = result
    assert head == 42
    assert tail.null()

    # Test derived properties
    assert seq_with_one.size() == 1
    assert seq_with_one.to_list() == [42]

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
    assert seq_with_many.to_list() == [3, 2, 1]

    # Test empty seq returns None
    empty_result: Optional[Tuple[int, Seq[int]]] = Seq.empty().uncons()
    assert empty_result is None


def test_snoc_unsnoc():
    """Test that unsnoc returns what was snoced"""
    seq: Seq[int] = Seq.empty()

    # Test with single element
    seq_with_one = seq.snoc(42)
    result = seq_with_one.unsnoc()
    assert result is not None
    init, last = result
    assert last == 42
    assert init.null()

    # Test derived properties
    assert seq_with_one.size() == 1
    assert seq_with_one.to_list() == [42]

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
    assert seq_with_many.to_list() == [1, 2, 3]

    # Test empty seq returns None
    empty_result: Optional[Tuple[Seq[int], int]] = Seq.empty().unsnoc()
    assert empty_result is None


def test_concat():
    """Test concatenating sequences"""
    empty: Seq[int] = Seq.empty()
    single1 = Seq.singleton(1)
    single2 = Seq.singleton(2)

    # Test empty + empty = empty
    result1 = empty.concat(empty)
    assert result1.null()
    assert result1.to_list() == []

    # Test empty + single = single
    result2 = empty.concat(single1)
    assert result2.to_list() == [1]

    # Test single + empty = single
    result3 = single1.concat(empty)
    assert result3.to_list() == [1]

    # Test single + single = deep with 2 elements
    result4 = single1.concat(single2)
    assert result4.to_list() == [1, 2]

    # Test single + deep sequence
    deep_seq = empty.snoc(3).snoc(4).snoc(5)
    result5 = single1.concat(deep_seq)
    assert result5.to_list() == [1, 3, 4, 5]

    # Test deep + single
    result6 = deep_seq.concat(single2)
    assert result6.to_list() == [3, 4, 5, 2]

    # Test deep + deep
    deep_seq2 = empty.snoc(6).snoc(7)
    result7 = deep_seq.concat(deep_seq2)
    assert result7.to_list() == [3, 4, 5, 6, 7]
