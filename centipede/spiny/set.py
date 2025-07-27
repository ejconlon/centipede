"""Persistent set implementation based on weight-balanced trees"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Type, override

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

        Args:
            value: The value to insert.

        Returns:
            A new set containing the inserted value.
        """
        return _pset_insert(self, value)

    def merge(self, _other: PSet[T]) -> PSet[T]:
        raise Exception("TODO")

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

    def __add__(self, other: PSet[T]) -> PSet[T]:
        """Merge sequences using + operator.

        Args:
            other: The set to merge with this one.

        Returns:
            A new set containing elements from both sets.
        """
        return self.merge(other)


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
    """Internal insert method using match blocks."""
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


def _pset_balance[T](left: PSet[T], value: T, right: PSet[T]) -> PSet[T]:
    """Create a balanced tree from left subtree, value, and right subtree."""
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
