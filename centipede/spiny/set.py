"""Persistent set implementation based on weight-balanced trees"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Tuple, Type, override

from centipede.spiny.common import (
    Box,
    Impossible,
    LexComparable,
    Ordering,
    Sized,
    compare,
)

__all__ = ["PSet"]


# sealed
class PSet[T](Sized, LexComparable[T, "PSet[T]"]):
    @staticmethod
    def empty(_ty: Optional[Type[T]] = None) -> PSet[T]:
        """Create an empty set.

        Args:
            _ty: Optional type hint (unused).

        Returns:
            An empty set instance.
        """
        return _PSET_EMPTY

    @staticmethod
    def singleton(value: T) -> PSet[T]:
        """Create a set containing a single element.

        Args:
            value: The single element for the set.

        Returns:
            A set containing only the given element.
        """
        return PSetBranch(1, _PSET_EMPTY, value, _PSET_EMPTY)

    @staticmethod
    def mk(values: Iterable[T]) -> PSet[T]:
        """Create a set from an iterable of values.

        Args:
            values: Iterable of values to include in the set.

        Returns:
            A sequence containing all the given values in order.
        """
        box: Box[PSet[T]] = Box(PSet.empty())
        for value in values:
            box.value = box.value.insert(value)
        return box.value

    @override
    def null(self) -> bool:
        match self:
            case PSetEmpty():
                return True
            case PSetBranch():
                return False
            case _:
                raise Impossible

    @override
    def size(self) -> int:
        match self:
            case PSetEmpty():
                return 0
            case PSetBranch(_size, _, _, _):
                return _size
            case _:
                raise Impossible

    @override
    def iter(self) -> Generator[T]:
        match self:
            case PSetEmpty():
                return
            case PSetBranch(_, left, value, right):
                yield from left.iter()
                yield value
                yield from right.iter()

    def insert(self, value: T) -> PSet[T]:
        """Insert a value into the set.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Args:
            value: The value to insert.

        Returns:
            A new set containing the inserted value.
        """
        return _pset_insert(self, value)

    def find_min(self) -> Optional[Tuple[T, PSet[T]]]:
        """Find the minimum element in the set.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            None if the set is empty, otherwise a tuple containing:
            - The minimum element
            - A new set with the minimum element removed
        """
        return _pset_find_min(self)

    def delete_min(self) -> Optional[PSet[T]]:
        """Remove the minimum element from the set.

        Returns:
            None if the set is empty, otherwise a new set with the
            minimum element removed.
        """
        result = self.find_min()
        return None if result is None else result[1]

    def find_max(self) -> Optional[Tuple[PSet[T], T]]:
        """Find the maximum element in the set.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            None if the set is empty, otherwise a tuple containing:
            - A new set with the maximum element removed
            - The maximum element
        """
        return _pset_find_max(self)

    def delete_max(self) -> Optional[PSet[T]]:
        """Remove the maximum element from the set.

        Returns:
            None if the set is empty, otherwise a new set with the
            maximum element removed.
        """
        result = self.find_max()
        return None if result is None else result[0]

    def split(self, pivot: T) -> Tuple[PSet[T], bool, PSet[T]]:
        """Split a set into elements smaller and larger than the pivot.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            A tuple (smaller, pivot_found, larger) where:
            - smaller contains elements < pivot
            - pivot_found indicates whether pivot was in the original set
            - larger contains elements > pivot
            The pivot itself is excluded from both smaller and larger.
        """
        return _pset_split(self, pivot)

    def contains(self, value: T) -> bool:
        """Check if a value is present in the set.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for recursion stack

        Args:
            value: The value to search for.

        Returns:
            True if the value is in the set, False otherwise.
        """
        return _pset_contains(self, value)

    def __contains__(self, value: T) -> bool:
        """Support 'in' operator for membership testing.

        Args:
            value: The value to search for.

        Returns:
            True if the value is in the set, False otherwise.
        """
        return self.contains(value)

    def union(self, other: PSet[T]) -> PSet[T]:
        """Return the union of two sets.

        Time Complexity: O(m + n) where m, n are sizes of the sets
        Space Complexity: O(log(m + n)) for recursion

        Args:
            other: The set to union with this one.

        Returns:
            A new set containing all elements from both sets.
        """
        return _pset_merge(self, other)

    def intersection(self, other: PSet[T]) -> PSet[T]:
        """Return the intersection of two sets.

        Time Complexity: O(m + n) where m, n are sizes of the sets
        Space Complexity: O(log(min(m, n))) for recursion

        Args:
            other: The set to intersect with this one.

        Returns:
            A new set containing only elements present in both sets.
        """
        return _pset_intersection(self, other)

    def difference(self, other: PSet[T]) -> PSet[T]:
        """Return the difference of two sets (elements in self but not in other).

        Time Complexity: O(m + n) where m, n are sizes of the sets
        Space Complexity: O(log m) for recursion

        Args:
            other: The set to subtract from this one.

        Returns:
            A new set containing elements in self but not in other.
        """
        return _pset_difference(self, other)

    def symdiff(self, other: PSet[T]) -> PSet[T]:
        """Symmetric difference.

        Args:
            other: The set to compute symmetric difference with.

        Returns:
            A new set containing elements in either set but not in both.
        """
        return self.union(other).difference(self.intersection(other))

    def __rshift__(self, value: T) -> PSet[T]:
        """Insert element using >> operator (element on right).

        Args:
            value: The element to insert.

        Returns:
            A new set with the element inserted.
        """
        return self.insert(value)

    def __rlshift__(self, value: T) -> PSet[T]:
        """Insert element using << operator (element on left).

        Args:
            value: The element to insert.

        Returns:
            A new set with the element inserted.
        """
        return self.insert(value)

    def __or__(self, other: PSet[T]) -> PSet[T]:
        """Union using | operator (Python set-like behavior).

        Args:
            other: The set to union with this one.

        Returns:
            A new set containing elements from both sets.
        """
        return self.union(other)

    def __and__(self, other: PSet[T]) -> PSet[T]:
        """Intersection using & operator (Python set-like behavior).

        Args:
            other: The set to intersect with this one.

        Returns:
            A new set containing only elements present in both sets.
        """
        return self.intersection(other)

    def __sub__(self, other: PSet[T]) -> PSet[T]:
        """Difference using - operator (Python set-like behavior).

        Args:
            other: The set to subtract from this one.

        Returns:
            A new set containing elements in self but not in other.
        """
        return self.difference(other)

    def __xor__(self, other: PSet[T]) -> PSet[T]:
        """Symmetric difference using ^ operator (Python set-like behavior).

        Args:
            other: The set to compute symmetric difference with.

        Returns:
            A new set containing elements in either set but not in both.
        """
        return self.symdiff(other)


@dataclass(frozen=True, eq=False)
class PSetEmpty[T](PSet[T]):
    pass


_PSET_EMPTY: PSet[Any] = PSetEmpty()


@dataclass(frozen=True, eq=False)
class PSetBranch[T](PSet[T]):
    _size: int
    _left: PSet[T]
    _value: T
    _right: PSet[T]


def _pset_insert[T](pset: PSet[T], value: T) -> PSet[T]:
    match pset:
        case PSetEmpty():
            return PSetBranch(1, _PSET_EMPTY, value, _PSET_EMPTY)
        case PSetBranch(_, left, branch_value, right):
            cmp = compare(value, branch_value)
            if cmp == Ordering.Lt:
                new_left = _pset_insert(left, value)
                return _pset_balance(new_left, branch_value, right)
            elif cmp == Ordering.Gt:
                new_right = _pset_insert(right, value)
                return _pset_balance(left, branch_value, new_right)
            else:
                # Value already exists, return unchanged
                return pset
        case _:
            raise Impossible


def _pset_merge[T](left_set: PSet[T], right_set: PSet[T]) -> PSet[T]:
    match (left_set, right_set):
        case (PSetEmpty(), _):
            return right_set
        case (_, PSetEmpty()):
            return left_set
        case (PSetBranch(_, left_left, left_value, left_right), _):
            # Split right_set around left_value and merge recursively
            smaller, _, larger = _pset_split(right_set, left_value)
            merged_left = _pset_merge(left_left, smaller)
            merged_right = _pset_merge(left_right, larger)
            return _pset_balance(merged_left, left_value, merged_right)
        case _:
            raise Impossible


def _pset_split[T](pset: PSet[T], pivot: T) -> tuple[PSet[T], bool, PSet[T]]:
    match pset:
        case PSetEmpty():
            return (_PSET_EMPTY, False, _PSET_EMPTY)
        case PSetBranch(_, left, value, right):
            cmp = compare(pivot, value)
            if cmp == Ordering.Lt:
                # pivot < value, so value goes to larger side
                left_smaller, pivot_found, left_larger = _pset_split(left, pivot)
                larger = _pset_balance(left_larger, value, right)
                return (left_smaller, pivot_found, larger)
            elif cmp == Ordering.Gt:
                # pivot > value, so value goes to smaller side
                right_smaller, pivot_found, right_larger = _pset_split(right, pivot)
                smaller = _pset_balance(left, value, right_smaller)
                return (smaller, pivot_found, right_larger)
            else:
                # pivot == value, found it and exclude the value
                return (left, True, right)
        case _:
            raise Impossible


def _pset_find_min[T](pset: PSet[T]) -> Optional[Tuple[T, PSet[T]]]:
    match pset:
        case PSetEmpty():
            return None
        case PSetBranch(_, left, value, right):
            if left.null():
                # This node contains the minimum value
                return (value, right)
            else:
                # Minimum is in the left subtree
                min_result = _pset_find_min(left)
                if min_result is None:
                    raise Impossible
                min_value, new_left = min_result
                new_tree = _pset_balance(new_left, value, right)
                return (min_value, new_tree)
        case _:
            raise Impossible


def _pset_find_max[T](pset: PSet[T]) -> Optional[Tuple[PSet[T], T]]:
    match pset:
        case PSetEmpty():
            return None
        case PSetBranch(_, left, value, right):
            if right.null():
                # This node contains the maximum value
                return (left, value)
            else:
                # Maximum is in the right subtree
                max_result = _pset_find_max(right)
                if max_result is None:
                    raise Impossible
                new_right, max_value = max_result
                new_tree = _pset_balance(left, value, new_right)
                return (new_tree, max_value)
        case _:
            raise Impossible


def _pset_balance[T](left: PSet[T], value: T, right: PSet[T]) -> PSet[T]:
    left_size = left.size()
    right_size = right.size()
    total_size = left_size + 1 + right_size

    # Weight-balanced tree invariant: neither subtree should be more than
    # 3 times larger than the other
    if left_size > 3 * right_size:
        # Left is too heavy, need to rotate right
        match left:
            case PSetBranch(_, left_left, left_value, left_right):
                left_left_size = left_left.size()
                left_right_size = left_right.size()
                if left_left_size >= left_right_size:
                    # Single rotation right
                    return PSetBranch(
                        total_size,
                        left_left,
                        left_value,
                        PSetBranch(
                            1 + left_right_size + right_size, left_right, value, right
                        ),
                    )
                else:
                    # Double rotation left-right
                    match left_right:
                        case PSetBranch(
                            _, left_right_left, left_right_value, left_right_right
                        ):
                            return PSetBranch(
                                total_size,
                                PSetBranch(
                                    1 + left_left_size + left_right_left.size(),
                                    left_left,
                                    left_value,
                                    left_right_left,
                                ),
                                left_right_value,
                                PSetBranch(
                                    1 + left_right_right.size() + right_size,
                                    left_right_right,
                                    value,
                                    right,
                                ),
                            )
    elif right_size > 3 * left_size:
        # Right is too heavy, need to rotate left
        match right:
            case PSetBranch(_, right_left, right_value, right_right):
                right_left_size = right_left.size()
                right_right_size = right_right.size()
                if right_right_size >= right_left_size:
                    # Single rotation left
                    return PSetBranch(
                        total_size,
                        PSetBranch(
                            1 + left_size + right_left_size, left, value, right_left
                        ),
                        right_value,
                        right_right,
                    )
                else:
                    # Double rotation right-left
                    match right_left:
                        case PSetBranch(
                            _, right_left_left, right_left_value, right_left_right
                        ):
                            return PSetBranch(
                                total_size,
                                PSetBranch(
                                    1 + left_size + right_left_left.size(),
                                    left,
                                    value,
                                    right_left_left,
                                ),
                                right_left_value,
                                PSetBranch(
                                    1 + right_left_right.size() + right_right_size,
                                    right_left_right,
                                    right_value,
                                    right_right,
                                ),
                            )

    # Tree is balanced or no rotation needed
    return PSetBranch(total_size, left, value, right)


def _pset_contains[T](pset: PSet[T], value: T) -> bool:
    """Check if a value is present in the set using binary search."""
    match pset:
        case PSetEmpty():
            return False
        case PSetBranch(_, left, branch_value, right):
            cmp = compare(value, branch_value)
            if cmp == Ordering.Lt:
                return _pset_contains(left, value)
            elif cmp == Ordering.Gt:
                return _pset_contains(right, value)
            else:
                # Found exact match
                return True
        case _:
            raise Impossible


def _pset_intersection[T](set1: PSet[T], set2: PSet[T]) -> PSet[T]:
    """Compute intersection of two sets efficiently using tree structure."""
    match (set1, set2):
        case (PSetEmpty(), _):
            return _PSET_EMPTY
        case (_, PSetEmpty()):
            return _PSET_EMPTY
        case (PSetBranch(_, left1, value1, right1), _):
            # Split set2 around value1
            smaller2, value1_in_set2, larger2 = _pset_split(set2, value1)
            left_intersection = _pset_intersection(left1, smaller2)
            right_intersection = _pset_intersection(right1, larger2)

            # Include value1 only if it's in set2
            if value1_in_set2:
                return _pset_balance(left_intersection, value1, right_intersection)
            else:
                return _pset_merge(left_intersection, right_intersection)
        case _:
            raise Impossible


def _pset_difference[T](set1: PSet[T], set2: PSet[T]) -> PSet[T]:
    """Compute difference of two sets (set1 - set2) using tree structure."""
    match (set1, set2):
        case (PSetEmpty(), _):
            return _PSET_EMPTY
        case (_, PSetEmpty()):
            return set1
        case (PSetBranch(_, left1, value1, right1), _):
            # Split set2 around value1
            smaller2, value1_in_set2, larger2 = _pset_split(set2, value1)
            left_difference = _pset_difference(left1, smaller2)
            right_difference = _pset_difference(right1, larger2)

            # Include value1 only if it's NOT in set2
            if not value1_in_set2:
                return _pset_balance(left_difference, value1, right_difference)
            else:
                return _pset_merge(left_difference, right_difference)
        case _:
            raise Impossible
