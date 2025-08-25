"""Property-based tests for PArray using Hypothesis."""

from hypothesis import assume, given
from hypothesis import strategies as st

from spiny.array import PArray
from tests.spiny.hypo import configure_hypo

configure_hypo()


@st.composite
def array_strategy(draw, element_strategy=st.integers(), max_size=20):
    """Generate an PArray with random size and fill element."""
    size = draw(st.integers(min_value=0, max_value=max_size))
    fill = draw(element_strategy)
    arr = PArray.new(size, fill)

    # Randomly set some elements
    num_sets = draw(st.integers(min_value=0, max_value=min(size, 10)))
    for _ in range(num_sets):
        if size > 0:
            index = draw(st.integers(min_value=0, max_value=size - 1))
            value = draw(element_strategy)
            arr = arr.set(index, value)

    return arr


@given(st.integers(min_value=0, max_value=100), st.integers())
def test_array_creation_properties(size, fill):
    """Test basic properties of PArray creation."""
    arr = PArray.new(size, fill)

    assert arr.size() == size
    assert len(arr) == size
    assert arr.null() == (size == 0)

    # All unset elements should return fill
    for i in range(size):
        assert arr.get(i) == fill
        assert arr.lookup(i) is None


@given(array_strategy())
def test_array_size_invariants(arr):
    """Test that PArray size is consistent across operations."""
    size = arr.size()

    assert len(arr) == size
    assert arr.null() == (size == 0)
    assert bool(arr) == (size > 0)

    # Iteration should yield exactly size elements
    elements = list(arr)
    assert len(elements) == size


@given(array_strategy(), st.integers())
def test_get_bounds_checking(arr, index):
    """Test that get method properly checks bounds."""
    size = arr.size()

    if 0 <= index < size:
        # Valid index should not raise
        result = arr.get(index)
        assert result is not None or result is None  # Always succeeds
    else:
        # Invalid index should raise KeyError
        try:
            arr.get(index)
            assert False, f"Should raise KeyError for index {index} with size {size}"
        except KeyError:
            pass


@given(array_strategy(), st.integers())
def test_lookup_bounds_checking(arr, index):
    """Test that lookup method properly checks bounds."""
    size = arr.size()

    if 0 <= index < size:
        # Valid index should not raise
        result = arr.lookup(index)
        assert result is None or result is not None  # Always succeeds
    else:
        # Invalid index should raise KeyError
        try:
            arr.lookup(index)
            assert False, f"Should raise KeyError for index {index} with size {size}"
        except KeyError:
            pass


@given(array_strategy(), st.integers())
def test_set_immutability(arr, value):
    """Test that set operations don't modify the original array."""
    size = arr.size()

    # Only test if array is not empty
    assume(size > 0)

    # Pick a valid index
    index = size // 2  # Use middle index

    original_elements = list(arr)
    new_arr = arr.set(index, value)

    # Original array should be unchanged
    assert list(arr) == original_elements

    # New array should have the set value
    assert new_arr.get(index) == value
    assert new_arr.lookup(index) == value

    # Other elements should be the same
    for i in range(size):
        if i != index:
            assert new_arr.get(i) == arr.get(i)


@given(array_strategy(), st.integers(min_value=0, max_value=100))
def test_resize_properties(arr, new_size):
    """Test properties of resize operation."""
    original_size = arr.size()
    resized = arr.resize(new_size)

    assert resized.size() == new_size

    # Elements within both sizes should be preserved
    min_size = min(original_size, new_size)
    for i in range(min_size):
        assert resized.get(i) == arr.get(i)
        assert resized.lookup(i) == arr.lookup(i)


@given(array_strategy())
def test_iteration_consistency(arr):
    """Test that different iteration methods are consistent."""
    size = arr.size()

    # All iteration methods should yield the same elements
    iter_elements = list(arr.iter())
    list_elements = list(arr)
    getitem_elements = [arr[i] for i in range(size)]

    assert iter_elements == list_elements == getitem_elements
    assert len(iter_elements) == size


@given(array_strategy(), array_strategy())
def test_comparison_properties(arr1, arr2):
    """Test lexicographic comparison properties."""
    # Reflexivity
    assert arr1 == arr1
    assert arr2 == arr2

    # Antisymmetry of <
    if arr1 < arr2:
        assert not (arr2 < arr1)
        assert arr1 != arr2

    # Transitivity is harder to test without more arrays
    # but we can test some basic properties
    if arr1 == arr2:
        assert not (arr1 < arr2)
        assert not (arr2 < arr1)
        assert arr1 <= arr2
        assert arr2 <= arr1


@given(st.integers(min_value=0, max_value=20), st.integers())
def test_empty_vs_filled_array(size, fill):
    """Test comparing empty vs filled arrays."""
    empty_arr = PArray.new(size, fill)

    if size > 0:
        filled_arr = (
            empty_arr.set(0, fill + 1)
            if fill != fill + 1
            else empty_arr.set(0, fill - 1)
        )

        # PArrays with different elements should not be equal
        if list(empty_arr) != list(filled_arr):
            assert empty_arr != filled_arr


@given(array_strategy())
def test_roundtrip_properties(arr):
    """Test roundtrip properties like resize(size()) is identity-like."""
    size = arr.size()

    # Resizing to same size should preserve all elements
    resized = arr.resize(size)

    for i in range(size):
        assert resized.get(i) == arr.get(i)
        assert resized.lookup(i) == arr.lookup(i)


@given(st.integers(min_value=1, max_value=20), st.integers())
def test_set_get_consistency(size, fill):
    """Test that set and get operations are consistent."""
    arr = PArray.new(size, fill)

    for i in range(size):
        new_value = fill + i + 1
        arr_with_set = arr.set(i, new_value)

        # Get should return the set value
        assert arr_with_set.get(i) == new_value
        assert arr_with_set.lookup(i) == new_value

        # Other indices should still return fill or their set values
        for j in range(size):
            if j != i:
                assert arr_with_set.get(j) == arr.get(j)


@given(st.integers(min_value=0, max_value=20), st.integers())
def test_magic_method_consistency(size, fill):
    """Test that magic methods are consistent with regular methods."""
    arr = PArray.new(size, fill)

    # __len__ should match size()
    assert len(arr) == arr.size()

    # __bool__ should match not null()
    assert bool(arr) == (not arr.null())

    # __getitem__ should match get() for valid indices
    for i in range(size):
        assert arr[i] == arr.get(i)

    # __iter__ should match iter()
    assert list(arr) == list(arr.iter())
