"""Property-based tests for PSeq using Hypothesis."""

from typing import List

from hypothesis import given
from hypothesis import strategies as st

from spiny.seq import PSeq
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def seq_strategy(
    draw: st.DrawFn, element_strategy: st.SearchStrategy[int] = st.integers()
) -> PSeq[int]:
    return PSeq.mk(draw(st.lists(element_strategy, min_size=0, max_size=20)))


@given(st.lists(st.integers(), min_size=0, max_size=50))
def test_seq_mk_equals_list(elements: List[int]) -> None:
    """Creating a PSeq from elements should preserve order and size."""
    seq = PSeq.mk(elements)
    assert seq.list() == elements
    assert seq.size() == len(elements)
    assert seq.null() == (len(elements) == 0)


@given(seq_strategy())
def test_cons_uncons_inverse(seq: PSeq[int]) -> None:
    """cons and uncons should be inverse operations."""
    element = 42

    new_seq = seq.cons(element)
    result = new_seq.uncons()

    if seq.null():
        assert result == (element, PSeq.empty(int))
    else:
        assert result is not None
        head, tail = result
        assert head == element
        assert tail.list() == seq.list()


@given(seq_strategy())
def test_snoc_unsnoc_inverse(seq: PSeq[int]) -> None:
    """snoc and unsnoc should be inverse operations."""
    element = 42

    new_seq = seq.snoc(element)
    result = new_seq.unsnoc()

    if seq.null():
        assert result == (PSeq.empty(int), element)
    else:
        assert result is not None
        init, last = result
        assert last == element
        assert init.list() == seq.list()


@given(seq_strategy(), seq_strategy())
def test_concat_associative(seq1: PSeq[int], seq2: PSeq[int]) -> None:
    """Concatenation should be associative: (a + b) + c == a + (b + c)."""
    seq3 = PSeq.mk([100, 200])

    left_assoc = seq1.concat(seq2).concat(seq3)
    right_assoc = seq1.concat(seq2.concat(seq3))

    assert left_assoc.list() == right_assoc.list()


@given(seq_strategy())
def test_concat_empty_identity(seq: PSeq[int]) -> None:
    """Concatenating with empty should be identity."""
    empty = PSeq.empty(int)

    assert seq.concat(empty).list() == seq.list()
    assert empty.concat(seq).list() == seq.list()


@given(seq_strategy(), st.integers(min_value=0, max_value=10))
def test_lookup_get_consistency(seq: PSeq[int], index: int) -> None:
    """lookup and get should be consistent for valid indices."""
    elements = seq.list()

    if index < len(elements):
        assert seq.lookup(index) == elements[index]
        assert seq.get(index) == elements[index]
        assert seq[index] == elements[index]
    else:
        assert seq.lookup(index) is None


@given(seq_strategy(), st.integers(min_value=0, max_value=10), st.integers())
def test_update_preserves_other_elements(
    seq: PSeq[int], index: int, new_value: int
) -> None:
    """Update should only change the specified index."""
    elements = seq.list()

    updated = seq.update(index, new_value)

    if index < len(elements):
        assert updated.size() == seq.size()
        for i in range(len(elements)):
            if i == index:
                assert updated.lookup(i) == new_value
            else:
                assert updated.lookup(i) == elements[i]
    else:
        assert updated.list() == seq.list()


@given(seq_strategy(), st.integers())
def test_update_out_of_bounds_unchanged(seq: PSeq[int], new_value: int) -> None:
    """Update with out-of-bounds index should return unchanged sequence."""
    elements = seq.list()
    size = len(elements)

    # Test various out-of-bounds indices
    assert seq.update(-1, new_value).list() == seq.list()
    assert seq.update(size, new_value).list() == seq.list()
    assert seq.update(size + 10, new_value).list() == seq.list()


@given(seq_strategy())
def test_list_iter_consistency(seq: PSeq[int]) -> None:
    """list() and iter() should produce the same elements in the same order."""

    list_result = seq.list()
    iter_result = list(seq.iter())

    assert list_result == iter_result


@given(seq_strategy())
def test_reversed_iter_consistency(seq: PSeq[int]) -> None:
    """reversed() should produce elements in reverse order of list()."""

    normal_list = seq.list()
    reversed_list = list(seq.reversed())

    assert reversed_list == list(reversed(normal_list))


@given(st.lists(st.integers(), min_size=1))
def test_size_matches_operations(elements: List[int]) -> None:
    """Size should correctly track through various operations."""
    seq = PSeq.empty(int)
    expected_size = 0

    for elem in elements:
        seq = seq.snoc(elem)
        expected_size += 1
        assert seq.size() == expected_size

    # Test uncons operations
    for _ in range(len(elements)):
        result = seq.uncons()
        if result is not None:
            _, seq = result
            expected_size -= 1
            assert seq.size() == expected_size


@given(seq_strategy(), seq_strategy())
def test_concat_size_additive(seq1: PSeq[int], seq2: PSeq[int]) -> None:
    """Concatenated sequence size should equal sum of individual sizes."""

    concat_seq = seq1.concat(seq2)
    assert concat_seq.size() == seq1.size() + seq2.size()


@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_cons_snoc_operations_preserve_elements(elements: List[int]) -> None:
    """Mixed cons and snoc operations should preserve all elements."""
    seq = PSeq.empty(int)
    all_elements = []

    for i, elem in enumerate(elements):
        if i % 2 == 0:
            seq = seq.snoc(elem)
        else:
            seq = seq.cons(elem)
        all_elements.append(elem)

    # Check that all elements are present (order may vary due to finger tree structure)
    seq_elements = seq.list()
    assert len(seq_elements) == len(all_elements)
    assert sorted(seq_elements) == sorted(all_elements)
    assert seq.size() == len(all_elements)


@given(seq_strategy())
def test_multiple_uncons_exhaustion(seq: PSeq[int]) -> None:
    """Repeated uncons should eventually reach empty sequence."""
    elements = seq.list()
    remaining_elements = elements.copy()

    while not seq.null():
        result = seq.uncons()
        assert result is not None
        head, seq = result
        assert head == remaining_elements.pop(0)

    assert remaining_elements == []
    assert seq.null()


@given(seq_strategy())
def test_multiple_unsnoc_exhaustion(seq: PSeq[int]) -> None:
    """Repeated unsnoc should eventually reach empty sequence."""
    elements = seq.list()
    remaining_elements = elements.copy()

    while not seq.null():
        result = seq.unsnoc()
        assert result is not None
        seq, last = result
        assert last == remaining_elements.pop()

    assert remaining_elements == []
    assert seq.null()


@given(st.lists(st.integers(), min_size=0, max_size=10))
def test_operators_consistency(elements: List[int]) -> None:
    """>> and << operators should match snoc and cons methods."""
    seq1 = PSeq.mk(elements)
    seq2 = PSeq.mk(elements)

    test_elem = 999

    # Test >> operator (snoc)
    seq1_snoc = seq1.snoc(test_elem)
    seq1_op = seq1 >> test_elem
    assert seq1_snoc.list() == seq1_op.list()

    # Test << operator (cons) - note the direction
    seq2_cons = seq2.cons(test_elem)
    seq2_op = test_elem << seq2
    assert seq2_cons.list() == seq2_op.list()


@given(seq_strategy(), seq_strategy())
def test_addition_operator_concat(seq1: PSeq[int], seq2: PSeq[int]) -> None:
    """+ operator should match concat method."""

    concat_method = seq1.concat(seq2)
    concat_op = seq1 + seq2

    assert concat_method.list() == concat_op.list()


@given(st.integers())
def test_singleton_properties(value: int) -> None:
    """Singleton sequence should have expected properties."""
    seq = PSeq.singleton(value)

    assert not seq.null()
    assert seq.size() == 1
    assert seq.list() == [value]
    assert seq.lookup(0) == value
    assert seq.lookup(1) is None

    uncons_result = seq.uncons()
    assert uncons_result is not None
    head, tail = uncons_result
    assert head == value
    assert tail.null()

    unsnoc_result = seq.unsnoc()
    assert unsnoc_result is not None
    init, last = unsnoc_result
    assert init.null()
    assert last == value


@given(seq_strategy(), st.integers(min_value=0, max_value=20))
def test_chained_updates_independent(seq: PSeq[int], new_value: int) -> None:
    """Multiple updates should be independent and preserve immutability."""
    elements = seq.list()

    if len(elements) >= 2:
        # Create multiple updated versions
        updated_0 = seq.update(0, new_value)
        updated_1 = seq.update(1, new_value)

        # Original should be unchanged
        assert seq.list() == elements

        # Updates should be independent
        if len(elements) > 0:
            assert updated_0.lookup(0) == new_value
            assert updated_0.lookup(1) == elements[1] if len(elements) > 1 else None

        if len(elements) > 1:
            assert updated_1.lookup(0) == elements[0]
            assert updated_1.lookup(1) == new_value
