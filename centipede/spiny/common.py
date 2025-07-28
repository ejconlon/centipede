"""Common utility types and functions for the spiny data structure library.

This module provides fundamental types and comparison utilities used throughout
the persistent data structures implementation.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generator, List, Tuple, cast, override

__all__ = [
    "Box",
    "Comparable",
    "Entry",
    "Flip",
    "LexComparable",
    "Impossible",
    "Ordering",
    "Iterating",
    "Sized",
    "Unit",
    "compare",
    "compare_lex",
    "group_runs",
]


class Impossible(Exception):
    """Exception raised when encountering theoretically impossible states.

    Used to indicate internal consistency violations in data structure operations.
    """

    pass


@dataclass
class Box[T]:
    """Mutable container for a single value.

    Provides a reference type wrapper for values that need to be updated
    within immutable contexts.
    """

    value: T

    def __iadd__(self, other):
        """Defer += operator to the underlying value."""
        self.value = self.value + other
        return self

    def __isub__(self, other):
        """Defer -= operator to the underlying value."""
        self.value = self.value - other
        return self

    def __imul__(self, other):
        """Defer *= operator to the underlying value."""
        self.value = self.value * other
        return self

    def __itruediv__(self, other):
        """Defer /= operator to the underlying value."""
        self.value = self.value / other
        return self

    def __ifloordiv__(self, other):
        """Defer //= operator to the underlying value."""
        self.value = self.value // other
        return self

    def __imod__(self, other):
        """Defer %= operator to the underlying value."""
        self.value = self.value % other
        return self

    def __ipow__(self, other):
        """Defer **= operator to the underlying value."""
        self.value = self.value**other
        return self

    def __ilshift__(self, other):
        """Defer <<= operator to the underlying value."""
        self.value = other << self.value
        return self

    def __irshift__(self, other):
        """Defer >>= operator to the underlying value."""
        self.value = self.value >> other
        return self

    def __iand__(self, other):
        """Defer &= operator to the underlying value."""
        self.value = self.value & other
        return self

    def __ior__(self, other):
        """Defer |= operator to the underlying value."""
        self.value = self.value | other
        return self

    def __ixor__(self, other):
        """Defer ^= operator to the underlying value."""
        self.value = self.value ^ other
        return self


@dataclass(frozen=True)
class Unit:
    """Singleton unit type representing no meaningful value.

    Similar to void in other languages, used where a type is required
    but no actual data needs to be stored.
    """

    @staticmethod
    def instance() -> Unit:
        """Get the singleton Unit instance.

        Returns:
            The global Unit instance.
        """
        return _UNIT


_UNIT = Unit()


class Sized(metaclass=ABCMeta):
    @abstractmethod
    def size(self) -> int: ...

    def null(self) -> bool:
        return self.size() == 0

    def __bool__(self) -> bool:
        return not self.null()

    def __len__(self) -> int:
        return self.size()


class Iterating[U](metaclass=ABCMeta):
    @abstractmethod
    def iter(self) -> Generator[U]: ...

    def list(self) -> List[U]:
        return list(self.iter())

    def __iter__(self) -> Generator[U]:
        return self.iter()

    def __list__(self) -> List[U]:
        return self.list()


class Ordering(Enum):
    """Enumeration representing the result of a comparison operation."""

    Lt = -1
    Eq = 0
    Gt = 1


class Comparable[T](metaclass=ABCMeta):
    @abstractmethod
    def compare(self, other: T) -> Ordering: ...

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self.compare(cast(T, other)) == Ordering.Eq
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: T) -> bool:
        return self.compare(other) == Ordering.Lt

    def __le__(self, other: T) -> bool:
        return not self.__gt__(other)

    def __gt__(self, other: T) -> bool:
        return self.compare(other) == Ordering.Gt

    def __ge__(self, other: T) -> bool:
        return not self.__lt__(other)


class LexComparable[U, T](Iterating[U], Comparable[T]):
    @override
    def compare(self, other: T) -> Ordering:
        return compare_lex(self.iter(), getattr(other, "iter")())


@dataclass(frozen=True, eq=False)
class Entry[K, V](Comparable["Entry[K, V]"]):
    """A key-value entry that compares only on the key.

    This allows us to store entries in a set or heap and lookup by key only,
    while still maintaining the associated values.
    """

    key: K
    value: V

    @override
    def compare(self, other: Entry[K, V]) -> Ordering:
        """Compare entries based on their keys only."""
        return compare(self.key, other.key)


@dataclass(frozen=True, eq=False)
class Flip[T](Comparable["Flip[T]"]):
    """A wrapper that flips the comparison result of the wrapped value.

    This is useful for converting min-heaps to max-heaps by reversing
    the comparison order of elements.

    Example:
        >>> from centipede.spiny.common import Flip, compare, Ordering
        >>> compare(1, 2)  # Normal comparison
        <Ordering.Lt: -1>
        >>> compare(Flip(1), Flip(2))  # Flipped comparison
        <Ordering.Gt: 1>
    """

    value: T

    @override
    def compare(self, other: Flip[T]) -> Ordering:
        """Compare by flipping the result of comparing the wrapped values."""
        result = compare(self.value, other.value)
        if result == Ordering.Lt:
            return Ordering.Gt
        elif result == Ordering.Gt:
            return Ordering.Lt
        else:
            return Ordering.Eq


def compare[T](a: T, b: T) -> Ordering:
    """Compare two values and return their ordering relationship.

    Uses the objects' __eq__ and __lt__ methods to determine the comparison result.

    Args:
        a: First value to compare.
        b: Second value to compare.

    Returns:
        Ordering indicating the relationship between a and b.
    """
    # Unsafe eq/lt because generic protocols are half-baked
    if getattr(a, "__eq__")(b):
        return Ordering.Eq
    elif getattr(a, "__lt__")(b):
        return Ordering.Lt
    else:
        return Ordering.Gt


def compare_lex[T](agen: Generator[T], bgen: Generator[T]) -> Ordering:
    """Perform lexicographic comparison of two sequences via generators.

    Compares elements from both generators in order, returning the first
    non-equal comparison result. If one generator is exhausted first,
    the shorter sequence is considered less than the longer one.

    Args:
        agen: Generator producing elements from the first sequence.
        bgen: Generator producing elements from the second sequence.

    Returns:
        Ordering indicating the lexicographic relationship between the sequences.
    """
    while True:
        try:
            a = next(agen)
        except StopIteration:
            try:
                _ = next(bgen)
                return Ordering.Lt
            except StopIteration:
                return Ordering.Eq
        try:
            b = next(bgen)
            r = compare(a, b)
            if r != Ordering.Eq:
                return r
        except StopIteration:
            return Ordering.Gt


def group_runs[K, V](gen: Generator[Tuple[K, V]]) -> Generator[Tuple[K, List[V]]]:
    """Group consecutive elements with equal keys into runs.

    Takes a generator of (key, value) pairs and yields (key, sequence) pairs where
    each sequence contains all consecutive values that had the same key.

    Args:
        gen: Generator producing (key, value) tuples.

    Yields:
        Tuples of (key, sequence)

    Example:
        >>> list(group_runs(iter([('a', 1), ('a', 2), ('b', 3), ('b', 4), ('a', 5)])))
        [('a', [1, 2]), ('b', [3, 4]), ('a', [5])]
    """
    try:
        current_key, current_value = next(gen)
    except StopIteration:
        return

    current_run = [current_value]

    for key, value in gen:
        if key == current_key:
            current_run.append(value)
        else:
            yield (current_key, current_run)
            current_key = key
            current_run = [value]

    yield (current_key, current_run)
