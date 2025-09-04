"""Common types and constants for minipat pattern system."""

from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from fractions import Fraction
from typing import Any, Callable, NewType, Self, override


class PartialMatchException(Exception):
    def __init__(self, val: Any):
        super().__init__(f"Unmatched type: {type(val)}")


def ignore_arg[A, B](fn: Callable[[A], B]) -> Callable[[None, A], B]:
    def wrapper(_: None, arg: A) -> B:
        return fn(arg)

    return wrapper


CycleTime = NewType("CycleTime", Fraction)
"""Time measured in elapsed cycles as fractions."""

PosixTime = NewType("PosixTime", float)
"""Time measured as POSIX timestamp (seconds since epoch)."""

CycleDelta = NewType("CycleDelta", Fraction)
"""Type for cycle time deltas represented as fractions."""

PosixDelta = NewType("PosixDelta", float)
"""Type for POSIX time deltas represented as floats."""

type Factor = Fraction
"""Type alias for scaling factors represented as fractions."""

ZERO = Fraction(0)
"""The constant 0 as a fraction."""

ONE = Fraction(1)
"""The constant 1 as a fraction."""

ONE_HALF = Fraction(1, 2)
"""The constant 1/2 as a fraction."""


def format_fraction(frac: Fraction) -> str:
    """Format a fraction according to the printing rules.

    Rules:
    - Always print fractional representations in parentheses
    - Handle integers appropriately (no parentheses needed)

    Args:
        frac: The fraction to format

    Returns:
        String representation of the fraction
    """
    if frac.denominator == 1:
        # It's a whole number
        return str(frac.numerator)
    else:
        # Always use parenthesized fraction format
        return f"({frac.numerator}/{frac.denominator})"


class TimeOps[T, D](metaclass=ABCMeta):
    """Operations on times (T) and their deltas (D)."""

    def __new__(cls) -> Self:
        raise Exception("Cannot instantiate namespace")

    @classmethod
    @abstractmethod
    def diff(cls, end: T, start: T) -> D:
        """Returns end-start as a delta."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def add(cls, base: T, delta: D) -> T:
        """Returns base+delta as a time."""
        raise NotImplementedError


class CycleTimeOps(TimeOps[CycleTime, CycleDelta]):
    """Time operations for CycleTime."""

    @override
    @classmethod
    def diff(cls, end: CycleTime, start: CycleTime) -> CycleDelta:
        return CycleDelta(end - start)

    @override
    @classmethod
    def add(cls, base: CycleTime, delta: CycleDelta) -> CycleTime:
        return CycleTime(base + delta)


class PosixTimeOps(TimeOps[PosixTime, PosixDelta]):
    """Time operations for PosixTime."""

    @override
    @classmethod
    def diff(cls, end: PosixTime, start: PosixTime) -> PosixDelta:
        return PosixDelta(end - start)

    @override
    @classmethod
    def add(cls, base: PosixTime, delta: PosixDelta) -> PosixTime:
        return PosixTime(base + delta)


def current_posix_time() -> PosixTime:
    """Get the current POSIX time.

    Returns:
        The current time as a POSIX timestamp
    """
    return PosixTime(time.time())
