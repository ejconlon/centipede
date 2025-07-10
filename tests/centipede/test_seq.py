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
    empty_result = Seq.empty().uncons()
    assert empty_result is None
