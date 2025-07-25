from typing import Optional, Tuple

from centipede.heap import Seq


def test_empty_seq():
    """Test creating an empty Seq and asserting it is empty"""
    seq: Seq[int] = Seq.empty()
    assert seq.null()


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

    # Test empty seq returns None
    empty_result: Optional[Tuple[Seq[int], int]] = Seq.empty().unsnoc()
    assert empty_result is None


def test_cons_snoc_uncons_unsnoc_combination():
    """Test combining cons, snoc, uncons, and unsnoc operations"""
    seq: Seq[int] = Seq.empty()

    # Build a sequence using both cons and snoc: [10, 5, 1, 2, 3, 20]
    # Start with [1, 2, 3] using snoc
    seq = seq.snoc(1).snoc(2).snoc(3)

    # Add to front using cons: [5, 1, 2, 3]
    seq = seq.cons(5)

    # Add to front again: [10, 5, 1, 2, 3]
    seq = seq.cons(10)

    # Add to end using snoc: [10, 5, 1, 2, 3, 20]
    seq = seq.snoc(20)

    # Now test uncons - should get 10 and [5, 1, 2, 3, 20]
    uncons_result = seq.uncons()
    assert uncons_result is not None
    head, tail = uncons_result
    assert head == 10

    # Test unsnoc on the tail - should get [5, 1, 2, 3] and 20
    unsnoc_result = tail.unsnoc()
    assert unsnoc_result is not None
    init, last = unsnoc_result
    assert last == 20

    # Test uncons on the remaining sequence - should get 5 and [1, 2, 3]
    uncons_result2 = init.uncons()
    assert uncons_result2 is not None
    head2, tail2 = uncons_result2
    assert head2 == 5

    # Test unsnoc on what remains - should get [1, 2] and 3
    unsnoc_result2 = tail2.unsnoc()
    assert unsnoc_result2 is not None
    init2, last2 = unsnoc_result2
    assert last2 == 3

    # Verify the middle sequence [1, 2] by uncons
    uncons_result3 = init2.uncons()
    assert uncons_result3 is not None
    head3, tail3 = uncons_result3
    assert head3 == 1

    # Final uncons should give us 2 and empty
    uncons_result4 = tail3.uncons()
    assert uncons_result4 is not None
    head4, tail4 = uncons_result4
    assert head4 == 2
    assert tail4.null()
