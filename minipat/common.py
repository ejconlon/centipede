"""Common types and constants for minipat pattern system."""

from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from fractions import Fraction
from typing import Any, Callable, NewType, Self, Union, override


class PartialMatchException(Exception):
    def __init__(self, val: Any):
        super().__init__(f"Unmatched type: {type(val)}")


def ignore_arg[A, B](fn: Callable[[A], B]) -> Callable[[None, A], B]:
    def wrapper(_: None, arg: A) -> B:
        return fn(arg)

    return wrapper


# =============================================================================
# Numeric Utilities
# =============================================================================

Numeric = Union[int, float, Fraction]
"""Type alias for numeric values that can be converted to Fraction."""


def is_numeric(value: Any) -> bool:
    """Check if a value is a numeric type.

    Args:
        value: The value to check

    Returns:
        True if the value is int, float, or Fraction
    """
    return isinstance(value, (int, float, Fraction))


def numeric_frac(numeric: Numeric) -> Fraction:
    """Convert a numeric value to a Fraction.

    Args:
        numeric: The numeric value to convert

    Returns:
        The value as a Fraction
    """
    if isinstance(numeric, Fraction):
        return numeric
    return Fraction(numeric)


def frac_ceil(f: Fraction) -> int:
    """Ceiling function for Fraction that maintains exact arithmetic.

    Avoids precision issues from converting to float.

    Args:
        f: The fraction to take the ceiling of

    Returns:
        The smallest integer greater than or equal to f
    """
    if f.numerator % f.denominator == 0:
        # Already an integer
        return f.numerator // f.denominator
    elif f >= 0:
        # Positive: round up
        return f.numerator // f.denominator + 1
    else:
        # Negative: ceiling means towards positive infinity
        # For negative numbers, we use the identity: ceil(x) = -floor(-x)
        return -((-f.numerator) // f.denominator)


# =============================================================================
# Time Types
# =============================================================================

CycleTime = NewType("CycleTime", Fraction)
"""Time measured in elapsed cycles as fractions."""

PosixTime = NewType("PosixTime", float)
"""Time measured as POSIX timestamp (seconds since epoch)."""

CycleDelta = NewType("CycleDelta", Fraction)
"""Type for cycle time deltas represented as fractions."""

PosixDelta = NewType("PosixDelta", float)
"""Type for POSIX time deltas represented as floats."""

StepTime = NewType("StepTime", int)
"""Time measured as step number (increments since start)."""

StepDelta = NewType("StepDelta", int)
"""Type for step time deltas represented as integers."""

Cps = NewType("Cps", Fraction)
"""Cycles per second represented as a fraction."""

Bpc = NewType("Bpc", int)
"""Beats per cycle represented as an integer."""

Tempo = NewType("Tempo", Fraction)
"""Tempo in beats per minute represented as a fraction."""

type Factor = Fraction
"""Type alias for scaling factors represented as fractions."""

# Type aliases for constructor functions
type CpsLike = Union[Cps, Numeric]
"""Type alias for values that can be converted to Cps."""

type BpcLike = Union[Bpc, int]
"""Type alias for values that can be converted to Bpc."""

type TempoLike = Union[Tempo, Numeric]
"""Type alias for values that can be converted to Tempo."""

type CycleTimeLike = Union[CycleTime, Numeric]
"""Type alias for values that can be converted to CycleTime."""

type CycleDeltaLike = Union[CycleDelta, Numeric]
"""Type alias for values that can be converted to CycleDelta."""

type PosixTimeLike = Union[PosixTime, float]
"""Type alias for values that can be converted to PosixTime."""

type PosixDeltaLike = Union[PosixDelta, float]
"""Type alias for values that can be converted to PosixDelta."""

type StepTimeLike = Union[StepTime, int]
"""Type alias for values that can be converted to StepTime."""

type StepDeltaLike = Union[StepDelta, int]
"""Type alias for values that can be converted to StepDelta."""

ZERO = Fraction(0)
"""The constant 0 as a fraction."""

ONE = Fraction(1)
"""The constant 1 as a fraction."""

ONE_HALF = Fraction(1, 2)
"""The constant 1/2 as a fraction."""


def format_fraction(frac: Fraction) -> str:
    """Format a fraction according to the printing rules.

    Rules:
    - Always print fractional representations with % syntax
    - Handle integers appropriately (no % needed)

    Args:
        frac: The fraction to format

    Returns:
        String representation of the fraction
    """
    if frac.denominator == 1:
        # It's a whole number
        return str(frac.numerator)
    else:
        # Always use % fraction format
        return f"{frac.numerator}%{frac.denominator}"


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

    @classmethod
    @abstractmethod
    def negate(cls, delta: D) -> D:
        """Negates delta"""
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

    @override
    @classmethod
    def negate(cls, delta: CycleDelta) -> CycleDelta:
        return CycleDelta(-delta)


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

    @override
    @classmethod
    def negate(cls, delta: PosixDelta) -> PosixDelta:
        return PosixDelta(-delta)


class StepTimeOps(TimeOps[StepTime, StepDelta]):
    """Time operations for StepTime."""

    @override
    @classmethod
    def diff(cls, end: StepTime, start: StepTime) -> StepDelta:
        return StepDelta(end - start)

    @override
    @classmethod
    def add(cls, base: StepTime, delta: StepDelta) -> StepTime:
        return StepTime(base + delta)

    @override
    @classmethod
    def negate(cls, delta: StepDelta) -> StepDelta:
        return StepDelta(-delta)


def current_posix_time() -> PosixTime:
    """Get the current POSIX time.

    Returns:
        The current time as a POSIX timestamp
    """
    return PosixTime(time.time())


# =============================================================================
# Constructor Functions
# =============================================================================


def mk_bpc(value: BpcLike) -> Bpc:
    """Create a Bpc from an int or existing Bpc.

    Args:
        value: An integer or existing Bpc value

    Returns:
        A Bpc value

    Raises:
        ValueError: If value is not an int or Bpc
    """
    if isinstance(value, int):
        return Bpc(value)
    else:
        raise ValueError(f"Cannot create Bpc from {type(value)}")


def mk_cps(value: CpsLike) -> Cps:
    """Create a Cps from a numeric value or existing Cps.

    Args:
        value: A numeric value (int, float, Fraction) or existing Cps

    Returns:
        A Cps value

    Raises:
        ValueError: If value is not numeric or Cps
    """
    if is_numeric(value):
        return Cps(numeric_frac(value))
    else:
        raise ValueError(f"Cannot create Cps from {type(value)}")


def mk_tempo(value: TempoLike) -> Tempo:
    """Create a Tempo from a numeric value or existing Tempo.

    Args:
        value: A numeric value (int, float, Fraction) or existing Tempo

    Returns:
        A Tempo value

    Raises:
        ValueError: If value is not numeric or Tempo
    """
    if is_numeric(value):
        return Tempo(numeric_frac(value))
    else:
        raise ValueError(f"Cannot create Tempo from {type(value)}")


def mk_cycle_time(value: CycleTimeLike) -> CycleTime:
    """Create a CycleTime from a numeric value or existing CycleTime.

    Args:
        value: A numeric value (int, float, Fraction) or existing CycleTime

    Returns:
        A CycleTime value

    Raises:
        ValueError: If value is not numeric or CycleTime
    """
    if is_numeric(value):
        return CycleTime(numeric_frac(value))
    else:
        raise ValueError(f"Cannot create CycleTime from {type(value)}")


def mk_cycle_delta(value: CycleDeltaLike) -> CycleDelta:
    """Create a CycleDelta from a numeric value or existing CycleDelta.

    Args:
        value: A numeric value (int, float, Fraction) or existing CycleDelta

    Returns:
        A CycleDelta value

    Raises:
        ValueError: If value is not numeric or CycleDelta
    """
    if is_numeric(value):
        return CycleDelta(numeric_frac(value))
    else:
        raise ValueError(f"Cannot create CycleDelta from {type(value)}")


def mk_posix_time(value: PosixTimeLike) -> PosixTime:
    """Create a PosixTime from a float or existing PosixTime.

    Args:
        value: A float or existing PosixTime

    Returns:
        A PosixTime value

    Raises:
        ValueError: If value is not a float or PosixTime
    """
    if isinstance(value, float):
        return PosixTime(value)
    else:
        raise ValueError(f"Cannot create PosixTime from {type(value)}")


def mk_posix_delta(value: PosixDeltaLike) -> PosixDelta:
    """Create a PosixDelta from a float or existing PosixDelta.

    Args:
        value: A float or existing PosixDelta

    Returns:
        A PosixDelta value

    Raises:
        ValueError: If value is not a float or PosixDelta
    """
    if isinstance(value, float):
        return PosixDelta(value)
    else:
        raise ValueError(f"Cannot create PosixDelta from {type(value)}")


def mk_step_time(value: StepTimeLike) -> StepTime:
    """Create a StepTime from an int or existing StepTime.

    Args:
        value: An int or existing StepTime

    Returns:
        A StepTime value

    Raises:
        ValueError: If value is not an int or StepTime
    """
    if isinstance(value, int):
        return StepTime(value)
    else:
        raise ValueError(f"Cannot create StepTime from {type(value)}")


def mk_step_delta(value: StepDeltaLike) -> StepDelta:
    """Create a StepDelta from an int or existing StepDelta.

    Args:
        value: An int or existing StepDelta

    Returns:
        A StepDelta value

    Raises:
        ValueError: If value is not an int or StepDelta
    """
    if isinstance(value, int):
        return StepDelta(value)
    else:
        raise ValueError(f"Cannot create StepDelta from {type(value)}")
