"""Persistent set implementation based on weight-balanced trees"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Type, override

from centipede.spiny.common import Box, LexComparable, Sized

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
        raise Exception("TODO")

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
        raise Exception("TODO")

    @override
    def size(self) -> int:
        raise Exception("TODO")

    @override
    def iter(self) -> Generator[T]:
        raise Exception("TODO")

    def insert(self, value: T) -> PSet[T]:
        raise Exception("TODO")

    def merge(self, other: PSet[T]) -> PSet[T]:
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
