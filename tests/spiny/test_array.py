"""Tests for PArray implementation"""

from typing import Dict, List, Optional, Tuple

from spiny.array import PArray


def test_empty_array() -> None:
    """Test creating an empty PArray and asserting it is empty"""
    arr = PArray.new(0, "fill")
    assert arr.null()
    assert arr.size() == 0
    assert list(arr.iter()) == []
    assert len(arr) == 0


def test_non_empty_array() -> None:
    """Test creating a non-empty PArray with fill element"""
    arr = PArray.new(5, "default")
    assert not arr.null()
    assert arr.size() == 5
    assert len(arr) == 5

    # All elements should return the fill element initially
    for i in range(5):
        assert arr.get(i) == "default"
        assert arr[i] == "default"  # Test __getitem__

    assert list(arr.iter()) == ["default"] * 5
    assert list(arr) == ["default"] * 5  # Test __iter__


def test_array_construction_validation() -> None:
    """Test PArray construction with invalid parameters"""
    # Test negative size
    try:
        PArray.new(-1, "fill")
        assert False, "Should raise ValueError for negative size"
    except ValueError as e:
        assert "non-negative" in str(e)

    # Test zero size is valid
    arr = PArray.new(0, "fill")
    assert arr.size() == 0


def test_get_method() -> None:
    """Test PArray get method with various scenarios"""
    arr = PArray.new(5, "fill")

    # Test getting from empty array (all indices should return fill)
    for i in range(5):
        assert arr.get(i) == "fill"

    # Test bounds checking
    try:
        arr.get(-1)
        assert False, "Should raise KeyError for negative index"
    except KeyError:
        pass

    try:
        arr.get(5)
        assert False, "Should raise KeyError for index >= size"
    except KeyError:
        pass

    try:
        arr.get(100)
        assert False, "Should raise KeyError for large index"
    except KeyError:
        pass


def test_lookup_method() -> None:
    """Test PArray lookup method"""
    arr = PArray.new(5, "fill")

    # Test lookup on empty array (all indices should return None)
    for i in range(5):
        assert arr.lookup(i) is None

    # Test bounds checking
    try:
        arr.lookup(-1)
        assert False, "Should raise KeyError for negative index"
    except KeyError:
        pass

    try:
        arr.lookup(5)
        assert False, "Should raise KeyError for index >= size"
    except KeyError:
        pass


def test_set_method() -> None:
    """Test PArray set method and immutability"""
    arr = PArray.new(3, "fill")

    # Test setting values
    arr1 = arr.set(0, "first")
    arr2 = arr1.set(2, "third")

    # Original array should be unchanged
    assert arr.get(0) == "fill"
    assert arr.get(1) == "fill"
    assert arr.get(2) == "fill"

    # First modification
    assert arr1.get(0) == "first"
    assert arr1.get(1) == "fill"
    assert arr1.get(2) == "fill"

    # Second modification
    assert arr2.get(0) == "first"
    assert arr2.get(1) == "fill"
    assert arr2.get(2) == "third"

    # Test bounds checking for set
    try:
        arr.set(-1, "invalid")
        assert False, "Should raise KeyError for negative index"
    except KeyError:
        pass

    try:
        arr.set(3, "invalid")
        assert False, "Should raise KeyError for index >= size"
    except KeyError:
        pass


def test_set_and_lookup_interaction() -> None:
    """Test interaction between set, get, and lookup methods"""
    arr = PArray.new(3, "fill")
    arr = arr.set(1, "middle")

    # Test get
    assert arr.get(0) == "fill"  # Not set, returns fill
    assert arr.get(1) == "middle"  # Set value
    assert arr.get(2) == "fill"  # Not set, returns fill

    # Test lookup
    assert arr.lookup(0) is None  # Not set, returns None
    assert arr.lookup(1) == "middle"  # Set value
    assert arr.lookup(2) is None  # Not set, returns None


def test_resize_method() -> None:
    """Test PArray resize functionality"""
    arr = PArray.new(3, "fill")
    arr = arr.set(0, "first").set(2, "third")

    # Test resizing larger
    arr_larger = arr.resize(5)
    assert arr_larger.size() == 5
    assert arr_larger.get(0) == "first"  # Preserved
    assert arr_larger.get(1) == "fill"  # Fill element
    assert arr_larger.get(2) == "third"  # Preserved
    assert arr_larger.get(3) == "fill"  # New fill element
    assert arr_larger.get(4) == "fill"  # New fill element

    # Test resizing smaller
    arr_smaller = arr.resize(2)
    assert arr_smaller.size() == 2
    assert arr_smaller.get(0) == "first"  # Preserved
    assert arr_smaller.get(1) == "fill"  # Fill element
    # Index 2 should be out of bounds now
    try:
        arr_smaller.get(2)
        assert False, "Should raise KeyError for out of bounds"
    except KeyError:
        pass

    # Test resizing to zero
    arr_empty = arr.resize(0)
    assert arr_empty.size() == 0
    assert arr_empty.null()

    # Test invalid resize
    try:
        arr.resize(-1)
        assert False, "Should raise ValueError for negative size"
    except ValueError:
        pass


def test_iteration() -> None:
    """Test PArray iteration methods"""
    arr = PArray.new(4, "default")
    arr = arr.set(1, "one").set(3, "three")

    # Test iter()
    elements = list(arr.iter())
    expected = ["default", "one", "default", "three"]
    assert elements == expected

    # Test __iter__
    elements = list(arr)
    assert elements == expected

    # Test empty array iteration
    empty_arr = PArray.new(0, "fill")
    assert list(empty_arr.iter()) == []
    assert list(empty_arr) == []


def test_magic_methods() -> None:
    """Test Python magic methods"""
    arr = PArray.new(3, "fill")
    arr = arr.set(1, "middle")

    # Test __len__
    assert len(arr) == 3

    # Test __getitem__
    assert arr[0] == "fill"
    assert arr[1] == "middle"
    assert arr[2] == "fill"

    # Test bounds checking in __getitem__
    try:
        _ = arr[-1]
        assert False, "Should raise KeyError for negative index"
    except KeyError:
        pass

    try:
        _ = arr[3]
        assert False, "Should raise KeyError for index >= size"
    except KeyError:
        pass


def test_lexicographic_comparison() -> None:
    """Test lexicographic comparison of PArrays"""
    arr1 = PArray.new(3, "a")
    arr2 = PArray.new(3, "a")
    arr3 = PArray.new(3, "b")
    arr4 = PArray.new(2, "a")
    arr5 = PArray.new(4, "a")

    # Test equality
    assert arr1 == arr2
    assert not (arr1 != arr2)

    # Test inequality by fill element
    assert arr1 != arr3
    assert arr1 < arr3
    assert arr3 > arr1

    # Test inequality by size (shorter is less)
    assert arr4 < arr1
    assert arr1 > arr4
    assert arr1 < arr5
    assert arr5 > arr1

    # Test with set values
    arr1_modified = arr1.set(1, "x")
    arr2_modified = arr2.set(1, "y")

    assert arr1_modified < arr2_modified
    assert arr2_modified > arr1_modified


def test_complex_operations() -> None:
    """Test complex combinations of PArray operations"""
    # Create and populate an array
    arr = PArray.new(5, 0)
    for i in range(5):
        arr = arr.set(i, i * i)  # Set to squares: [0, 1, 4, 9, 16]

    # Verify all values
    expected = [0, 1, 4, 9, 16]
    for i, expected_val in enumerate(expected):
        assert arr.get(i) == expected_val
        assert arr[i] == expected_val
        assert arr.lookup(i) == expected_val

    # Test resize preserving values
    arr_resized = arr.resize(7)
    for i, expected_val in enumerate(expected):
        assert arr_resized.get(i) == expected_val
    assert arr_resized.get(5) == 0  # Fill element
    assert arr_resized.get(6) == 0  # Fill element

    # Test resize shrinking
    arr_shrunk = arr.resize(3)
    for i in range(3):
        assert arr_shrunk.get(i) == expected[i]

    # Original array should be unchanged
    for i, expected_val in enumerate(expected):
        assert arr.get(i) == expected_val


def test_edge_cases() -> None:
    """Test edge cases and boundary conditions"""
    # Test with None as fill element
    arr_none: PArray[Optional[str]] = PArray.new(2, None)
    assert arr_none.get(0) is None
    assert arr_none.get(1) is None
    assert arr_none.lookup(0) is None  # Both None, but different meanings

    arr_none = arr_none.set(0, "not_none")
    assert arr_none.get(0) == "not_none"
    assert arr_none.get(1) is None
    assert arr_none.lookup(0) == "not_none"
    assert arr_none.lookup(1) is None

    # Test with different data types
    arr_int = PArray.new(2, 42)
    arr_str = PArray.new(2, "default")
    arr_list: PArray[List[int]] = PArray.new(2, [])

    assert arr_int.get(0) == 42
    assert arr_str.get(0) == "default"
    assert arr_list.get(0) == []

    # Test large array
    large_arr = PArray.new(1000, "fill")
    assert large_arr.size() == 1000
    assert large_arr.get(999) == "fill"

    large_arr = large_arr.set(500, "middle")
    assert large_arr.get(500) == "middle"
    assert large_arr.lookup(499) is None
    assert large_arr.lookup(500) == "middle"
    assert large_arr.lookup(501) is None


def test_array_with_various_types() -> None:
    """Test PArray with different element types"""
    # String array
    str_arr = PArray.new(3, "")
    str_arr = str_arr.set(1, "hello")
    assert str_arr.get(0) == ""
    assert str_arr.get(1) == "hello"

    # Integer array
    int_arr = PArray.new(3, -1)
    int_arr = int_arr.set(2, 100)
    assert int_arr.get(0) == -1
    assert int_arr.get(2) == 100

    # Boolean array
    bool_arr = PArray.new(3, False)
    bool_arr = bool_arr.set(0, True)
    assert bool_arr.get(0) is True
    assert bool_arr.get(1) is False

    # List array (mutable elements)
    list_arr: PArray[List[int]] = PArray.new(2, [])
    list_arr = list_arr.set(0, [1, 2, 3])
    assert list_arr.get(0) == [1, 2, 3]
    assert list_arr.get(1) == []


def test_fold_empty() -> None:
    """Test folding an empty array"""
    empty = PArray.new(0, "fill")
    result = empty.fold(lambda acc, x: acc + len(x), 0)
    assert result == 0


def test_fold_single() -> None:
    """Test folding a single element array"""
    arr = PArray.new(1, "hello")

    # Sum of string lengths
    result = arr.fold(lambda acc, x: acc + len(x), 0)
    assert result == 5

    # String concatenation
    result2 = arr.fold(lambda acc, x: acc + x, "prefix:")
    assert result2 == "prefix:hello"


def test_fold_multiple() -> None:
    """Test folding arrays with multiple elements"""
    arr = PArray.new(5, 0)
    arr = arr.set(0, 1).set(1, 2).set(2, 3).set(3, 4).set(4, 5)

    # Sum operation
    result = arr.fold(lambda acc, x: acc + x, 0)
    assert result == 15

    # Product operation
    result2 = arr.fold(lambda acc, x: acc * x, 1)
    assert result2 == 120

    # Build list (should preserve order)
    result3: List[int] = arr.fold(lambda acc, x: acc + [x], [])
    assert result3 == [1, 2, 3, 4, 5]


def test_fold_with_fill_elements() -> None:
    """Test folding arrays that include fill elements"""
    arr = PArray.new(4, 10)  # Fill element is 10
    arr = arr.set(1, 20).set(3, 30)  # [10, 20, 10, 30]

    # Sum all elements including fill elements
    result = arr.fold(lambda acc, x: acc + x, 0)
    assert result == 70  # 10 + 20 + 10 + 30

    # Count non-fill elements
    result2 = arr.fold(lambda acc, x: acc + (1 if x != 10 else 0), 0)
    assert result2 == 2  # Only positions 1 and 3 have non-fill values


def test_fold_with_index_empty() -> None:
    """Test fold_with_index on an empty array"""
    empty = PArray.new(0, "fill")
    result = empty.fold_with_index(lambda acc, i, x: acc + i + len(x), 0)
    assert result == 0


def test_fold_with_index_single() -> None:
    """Test fold_with_index on a single element array"""
    arr = PArray.new(1, "test")

    # Add value length and index
    result = arr.fold_with_index(lambda acc, i, x: acc + len(x) + i, 0)
    assert result == 4  # 0 + 4 + 0

    # Format with index
    result2: List[str] = arr.fold_with_index(lambda acc, i, x: acc + [f"{i}:{x}"], [])
    assert result2 == ["0:test"]


def test_fold_with_index_multiple() -> None:
    """Test fold_with_index on arrays with multiple elements"""
    arr = PArray.new(3, "a")
    arr = arr.set(0, "x").set(1, "y").set(2, "z")

    # Sum string lengths plus indices
    result = arr.fold_with_index(lambda acc, i, x: acc + len(x) + i, 0)
    assert result == 6  # (1+0) + (1+1) + (1+2) = 6

    # Build indexed list
    result2: List[str] = arr.fold_with_index(lambda acc, i, x: acc + [f"{i}:{x}"], [])
    assert result2 == ["0:x", "1:y", "2:z"]

    # Create dictionary mapping indices to values
    result3: Dict[int, str] = arr.fold_with_index(lambda acc, i, x: {**acc, i: x}, {})
    assert result3 == {0: "x", 1: "y", 2: "z"}


def test_fold_with_index_mixed_values() -> None:
    """Test fold_with_index with mixed set and fill values"""
    arr = PArray.new(4, 0)
    arr = arr.set(1, 10).set(3, 30)  # [0, 10, 0, 30]

    # Build list of (index, value) pairs for non-zero values
    result: List[Tuple[int, int]] = arr.fold_with_index(
        lambda acc, i, x: acc + [(i, x)] if x != 0 else acc, []
    )
    assert result == [(1, 10), (3, 30)]

    # Sum values times their indices
    result2 = arr.fold_with_index(lambda acc, i, x: acc + (x * i), 0)
    assert result2 == 100  # (0*0) + (10*1) + (0*2) + (30*3) = 10 + 90 = 100


def test_fold_type_change() -> None:
    """Test fold operations that change accumulator types"""
    arr = PArray.new(3, 1)
    arr = arr.set(0, 5).set(1, 10).set(2, 15)

    # Convert numbers to string representation
    result = arr.fold(lambda acc, x: acc + str(x), "")
    assert result == "51015"

    # Count elements
    result2 = arr.fold(lambda acc, x: acc + 1, 0)
    assert result2 == 3


def test_fold_persistence() -> None:
    """Test that fold operations don't modify the original array"""
    original = PArray.new(3, 1)
    original = original.set(0, 2).set(1, 4).set(2, 6)

    # Perform fold operations
    sum_result = original.fold(lambda acc, x: acc + x, 0)
    index_result: List[int] = original.fold_with_index(lambda acc, i, x: acc + [x], [])

    assert sum_result == 12
    assert index_result == [2, 4, 6]

    # Original array should be unchanged
    assert original.get(0) == 2
    assert original.get(1) == 4
    assert original.get(2) == 6
    assert list(original.iter()) == [2, 4, 6]
