"""Common utility types and functions for the spiny data structure library.

This module provides fundamental types and comparison utilities used throughout
the persistent data structures implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generator

__all__ = ["Box", "Impossible", "Ordering", "Unit", "compare", "compare_lex"]


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


class Ordering(Enum):
    """Enumeration representing the result of a comparison operation.

    Values correspond to standard comparison semantics:
    - Lt: left operand is less than right operand
    - Eq: operands are equal
    - Gt: left operand is greater than right operand
    """

    Lt = -1
    Eq = 0
    Gt = 1


def compare[T](a: T, b: T) -> Ordering:
    """Compare two values and return their ordering relationship.

    Uses the objects' __eq__ and __lt__ methods to determine the comparison result.
    Note: Uses getattr for method access due to limitations in generic protocols.

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
