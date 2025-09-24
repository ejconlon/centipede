"""Comprehensive time, numeric, and arc functionality for minipat patterns.

This module consolidates all temporal abstractions, numeric utilities, and arc types
to eliminate circular dependencies and provide a unified interface for time handling.
"""

from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import (
    Any,
    Iterator,
    NewType,
    Optional,
    Self,
    Union,
)

# =============================================================================
# Core Numeric Types and Utilities
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

    Raises:
        ValueError: If the value cannot be converted to a Fraction
    """
    if isinstance(numeric, Fraction):
        return numeric
    elif isinstance(numeric, int):
        return Fraction(numeric)
    elif isinstance(numeric, float):
        return Fraction(numeric).limit_denominator()
    else:
        raise ValueError(f"Cannot convert {type(numeric)} to Fraction")


# =============================================================================
# Core Time Types
# =============================================================================

CycleTime = NewType("CycleTime", Fraction)
"""Time in musical cycles."""

CycleDelta = NewType("CycleDelta", Fraction)
"""Duration in musical cycles."""

PosixTime = NewType("PosixTime", float)
"""Wall clock time (seconds since epoch)."""

PosixDelta = NewType("PosixDelta", float)
"""Wall clock duration in seconds."""

StepTime = NewType("StepTime", int)
"""Discrete step time (for step sequencers)."""

StepDelta = NewType("StepDelta", int)
"""Duration in discrete steps."""

Tempo = NewType("Tempo", Fraction)
"""Tempo in beats per minute."""

Bpc = NewType("Bpc", int)
"""Beats per cycle."""

Cps = NewType("Cps", Fraction)
"""Cycles per second."""


# =============================================================================
# Time Operation Interfaces
# =============================================================================


class TimeOps[T, D](metaclass=ABCMeta):
    """Abstract interface for time operations."""

    @abstractmethod
    def plus(self, time: T, delta: D) -> T:
        """Add a duration to a time."""
        raise NotImplementedError()

    @abstractmethod
    def minus(self, time: T, delta: D) -> T:
        """Subtract a duration from a time."""
        raise NotImplementedError()

    @abstractmethod
    def diff(self, end: T, start: T) -> D:
        """Calculate the duration between two times."""
        raise NotImplementedError()

    @abstractmethod
    def scale(self, delta: D, factor: Fraction) -> D:
        """Scale a duration by a factor."""
        raise NotImplementedError()


class CycleTimeOps(TimeOps[CycleTime, CycleDelta]):
    """Time operations for cycle time."""

    def plus(self, time: CycleTime, delta: CycleDelta) -> CycleTime:
        return CycleTime(time + delta)

    def minus(self, time: CycleTime, delta: CycleDelta) -> CycleTime:
        return CycleTime(time - delta)

    def diff(self, end: CycleTime, start: CycleTime) -> CycleDelta:
        return CycleDelta(end - start)

    def scale(self, delta: CycleDelta, factor: Fraction) -> CycleDelta:
        return CycleDelta(delta * factor)


class PosixTimeOps(TimeOps[PosixTime, PosixDelta]):
    """Time operations for POSIX time."""

    def plus(self, time: PosixTime, delta: PosixDelta) -> PosixTime:
        return PosixTime(time + delta)

    def minus(self, time: PosixTime, delta: PosixDelta) -> PosixTime:
        return PosixTime(time - delta)

    def diff(self, end: PosixTime, start: PosixTime) -> PosixDelta:
        return PosixDelta(end - start)

    def scale(self, delta: PosixDelta, factor: Fraction) -> PosixDelta:
        return PosixDelta(delta * float(factor))


class StepTimeOps(TimeOps[StepTime, StepDelta]):
    """Time operations for step time."""

    def plus(self, time: StepTime, delta: StepDelta) -> StepTime:
        return StepTime(time + delta)

    def minus(self, time: StepTime, delta: StepDelta) -> StepTime:
        return StepTime(time - delta)

    def diff(self, end: StepTime, start: StepTime) -> StepDelta:
        return StepDelta(end - start)

    def scale(self, delta: StepDelta, factor: Fraction) -> StepDelta:
        return StepDelta(int(delta * factor))


# =============================================================================
# Type Aliases for Conversion Functions
# =============================================================================

type BpcLike = Union[Bpc, int]
"""Type alias for values that can be converted to Bpc."""

type CpsLike = Union[Cps, Numeric]
"""Type alias for values that can be converted to Cps."""

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


# =============================================================================
# Arc Types
# =============================================================================


class Arc[T, D](metaclass=ABCMeta):
    """Abstract base class for time intervals."""

    @property
    @abstractmethod
    def start(self) -> T:
        """Get the start time of the arc."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def end(self) -> T:
        """Get the end time of the arc."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def time_ops(self) -> TimeOps[T, D]:
        """Get the time operations for this arc type."""
        raise NotImplementedError()

    @abstractmethod
    def length(self) -> D:
        """Calculate the duration of the arc."""
        raise NotImplementedError()

    @abstractmethod
    def shift(self, delta: D) -> Self:
        """Shift the arc by a duration."""
        raise NotImplementedError()

    @abstractmethod
    def null(self) -> bool:
        """Check if the arc is null/empty (end <= start)."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def mk(cls, start: T, end: T) -> Self:
        """Create an arc from start and end times."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def normalize(cls, arc: Arc[T, D]) -> Self:
        """Normalize an arc to this type."""
        raise NotImplementedError()

    @abstractmethod
    def union(self, other: Arc[T, D]) -> Self:
        """Get the union of two arcs."""
        raise NotImplementedError()

    @abstractmethod
    def intersect(self, other: Arc[T, D]) -> Self:
        """Get the intersection of two arcs."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def union_all(cls, arcs: Iterable[Arc[T, D]]) -> Self:
        """Get the union of multiple arcs."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def intersect_all(cls, arcs: Iterable[Arc[T, D]]) -> Self:
        """Get the intersection of multiple arcs."""
        raise NotImplementedError()


@dataclass(frozen=True, order=True)
class CycleArc(Arc[CycleTime, CycleDelta]):
    """Arc representing a time interval in cycle time."""

    _start: CycleTime
    _end: CycleTime

    @property
    def start(self) -> CycleTime:
        return self._start

    @property
    def end(self) -> CycleTime:
        return self._end

    @property
    def time_ops(self) -> CycleTimeOps:
        return CycleTimeOps()

    def length(self) -> CycleDelta:
        return CycleDelta(self._end - self._start)

    def shift(self, delta: CycleDelta) -> CycleArc:
        return CycleArc(CycleTime(self._start + delta), CycleTime(self._end + delta))

    def null(self) -> bool:
        return self._start >= self._end

    @classmethod
    def mk(cls, start: CycleTime, end: CycleTime) -> CycleArc:
        return cls(start, end)

    @classmethod
    def normalize(cls, arc: Arc[CycleTime, CycleDelta]) -> CycleArc:
        if isinstance(arc, cls):
            return arc
        return cls(arc.start, arc.end)

    def union(self, other: Arc[CycleTime, CycleDelta]) -> CycleArc:
        if self.null():
            return CycleArc.normalize(other)
        elif other.null():
            return self
        else:
            start = CycleTime(min(self._start, other.start))
            end = CycleTime(max(self._end, other.end))
            return CycleArc(start, end)

    def intersect(self, other: Arc[CycleTime, CycleDelta]) -> CycleArc:
        if self.null() or other.null():
            return CycleArc.empty()
        else:
            start = CycleTime(max(self._start, other.start))
            end = CycleTime(min(self._end, other.end))
            return CycleArc(start, end)

    @classmethod
    def union_all(cls, arcs: Iterable[Arc[CycleTime, CycleDelta]]) -> CycleArc:
        result = cls.empty()
        for arc in arcs:
            result = result.union(arc)
        return result

    @classmethod
    def intersect_all(cls, arcs: Iterable[Arc[CycleTime, CycleDelta]]) -> CycleArc:
        arc_list = list(arcs)
        if not arc_list:
            return cls.empty()

        result = cls.normalize(arc_list[0])
        for arc in arc_list[1:]:
            result = result.intersect(arc)
        return result

    @classmethod
    def empty(cls) -> CycleArc:
        """Create an empty arc."""
        return cls(CycleTime(Fraction(0)), CycleTime(Fraction(0)))

    @classmethod
    def cycle(cls, cyc: int) -> CycleArc:
        """Create an arc representing a full cycle.

        Args:
            cyc: The cycle number

        Returns:
            An arc from cyc to cyc+1
        """
        return cls(CycleTime(Fraction(cyc)), CycleTime(Fraction(cyc + 1)))

    def scale(self, factor: Fraction) -> CycleArc:
        """Scale the arc by a factor.

        Args:
            factor: The scaling factor

        Returns:
            A new arc scaled by the factor
        """
        return CycleArc(CycleTime(self._start * factor), CycleTime(self._end * factor))

    def clip(self, factor: Fraction) -> CycleArc:
        """Clip the arc to a fraction of its length.

        Args:
            factor: The clipping factor (0 to 1)

        Returns:
            A new arc clipped to the specified fraction
        """
        length = self._end - self._start
        new_end = CycleTime(self._start + length * factor)
        return CycleArc(self._start, new_end)

    def split_cycles(
        self, bounds: Optional[Union[tuple[int, int], CycleArc]] = None
    ) -> Iterator[tuple[int, CycleArc]]:
        """Split the arc into individual cycles.

        Args:
            bounds: Optional (min_cycle, max_cycle) bounds or CycleArc bounds

        Yields:
            Tuples of (cycle_index, cycle_arc)
        """
        if self.null():
            return

        start_cycle = frac_floor(self._start)
        end_cycle = frac_ceil(self._end)

        if bounds:
            if isinstance(bounds, CycleArc):
                min_cycle = frac_floor(bounds._start)
                max_cycle = frac_ceil(bounds._end)
            else:
                min_cycle, max_cycle = bounds
            start_cycle = max(start_cycle, min_cycle)
            end_cycle = min(end_cycle, max_cycle)

        for cycle_index in range(start_cycle, end_cycle):
            cycle_start = CycleTime(max(self._start, Fraction(cycle_index)))
            cycle_end = CycleTime(min(self._end, Fraction(cycle_index + 1)))

            if cycle_start < cycle_end:
                yield cycle_index, CycleArc(cycle_start, cycle_end)


@dataclass(frozen=True, order=True)
class StepArc(Arc[StepTime, StepDelta]):
    """Arc representing a time interval in step time."""

    _start: StepTime
    _end: StepTime

    @property
    def start(self) -> StepTime:
        return self._start

    @property
    def end(self) -> StepTime:
        return self._end

    @property
    def time_ops(self) -> StepTimeOps:
        return StepTimeOps()

    def length(self) -> StepDelta:
        return StepDelta(self._end - self._start)

    def shift(self, delta: StepDelta) -> StepArc:
        return StepArc(StepTime(self._start + delta), StepTime(self._end + delta))

    def null(self) -> bool:
        return self._start >= self._end

    @classmethod
    def mk(cls, start: StepTime, end: StepTime) -> StepArc:
        return cls(start, end)

    @classmethod
    def normalize(cls, arc: Arc[StepTime, StepDelta]) -> StepArc:
        if isinstance(arc, cls):
            return arc
        return cls(arc.start, arc.end)

    def union(self, other: Arc[StepTime, StepDelta]) -> StepArc:
        if self.null():
            return StepArc.normalize(other)
        elif other.null():
            return self
        else:
            start = StepTime(min(self._start, other.start))
            end = StepTime(max(self._end, other.end))
            return StepArc(start, end)

    def intersect(self, other: Arc[StepTime, StepDelta]) -> StepArc:
        if self.null() or other.null():
            return StepArc.empty()
        else:
            start = StepTime(max(self._start, other.start))
            end = StepTime(min(self._end, other.end))
            return StepArc(start, end)

    @classmethod
    def union_all(cls, arcs: Iterable[Arc[StepTime, StepDelta]]) -> StepArc:
        result = cls.empty()
        for arc in arcs:
            result = result.union(arc)
        return result

    @classmethod
    def intersect_all(cls, arcs: Iterable[Arc[StepTime, StepDelta]]) -> StepArc:
        arc_list = list(arcs)
        if not arc_list:
            return cls.empty()

        result = cls.normalize(arc_list[0])
        for arc in arc_list[1:]:
            result = result.intersect(arc)
        return result

    @classmethod
    def empty(cls) -> StepArc:
        """Create an empty arc."""
        return cls(StepTime(0), StepTime(0))


# =============================================================================
# Span Types (Arcs with optional "whole" context)
# =============================================================================


@dataclass(frozen=True, order=True)
class CycleSpan:
    """Span in cycle time with whole context."""

    _whole: CycleArc
    _active: CycleArc

    @property
    def active(self) -> CycleArc:
        return self._active

    @property
    def whole(self) -> CycleArc:
        return self._whole

    def shift(self, delta: CycleDelta) -> CycleSpan:
        new_active = self.active.shift(delta)
        new_whole = self.whole.shift(delta)
        return CycleSpan.mk(whole=new_whole, active=new_active)

    def scale(self, factor: Fraction) -> CycleSpan:
        """Scale the span by a factor."""
        new_active = self.active.scale(factor)
        new_whole = self.whole.scale(factor)
        return CycleSpan.mk(whole=new_whole, active=new_active)

    def clip(self, factor: Fraction) -> CycleSpan:
        """Clip the span to a fraction of its length."""
        new_active = self.active.clip(factor)
        new_whole = self.whole.clip(factor)
        return CycleSpan.mk(whole=new_whole, active=new_active)

    def contains_start(self) -> bool:
        """Check if the whole start time is within the active arc."""
        return (
            self.whole.start >= self.active.start
            and self.whole.start <= self.active.end
        )

    def contains_end(self) -> bool:
        """Check if the whole end time is within the active arc."""
        return self.whole.end >= self.active.start and self.whole.end <= self.active.end

    def valid(self) -> bool:
        """Check if active is contained within whole."""
        return self.active.null() or (
            self.active.start >= self.whole.start and self.active.end <= self.whole.end
        )

    @classmethod
    def mk(cls, whole: CycleArc, active: CycleArc) -> CycleSpan:
        span = cls(_whole=whole, _active=active)
        assert span.valid(), (
            f"Active arc {active} must be contained within whole arc {whole}"
        )
        return span

    @classmethod
    def empty(cls) -> CycleSpan:
        empty_arc = CycleArc.empty()
        return cls.mk(whole=empty_arc, active=empty_arc)


# Type aliases for arc conversion
type CycleArcLike = Union[CycleArc, tuple[Numeric, Numeric]]
"""Type alias for values that can be converted to CycleArc."""


# =============================================================================
# Utility Functions for Time Fractions
# =============================================================================


def frac_floor(frac: Fraction) -> int:
    """Floor division for Fraction values.

    Args:
        frac: The fraction to floor

    Returns:
        The floor of the fraction as an integer
    """
    return floor(float(frac))


def frac_ceil(frac: Fraction) -> int:
    """Ceiling division for Fraction values.

    Args:
        frac: The fraction to ceiling

    Returns:
        The ceiling of the fraction as an integer
    """
    return ceil(float(frac))


# =============================================================================
# Creation Functions
# =============================================================================


def mk_bpc(value: BpcLike) -> Bpc:
    """Create a Bpc from an int or existing Bpc.

    Args:
        value: An int or existing Bpc

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


def mk_cycle_arc(value: CycleArcLike) -> CycleArc:
    """Create a CycleArc from various arc-like values.

    Args:
        value: A CycleArc or (start, end) tuple

    Returns:
        A CycleArc value

    Raises:
        ValueError: If value cannot be converted to CycleArc
    """
    if isinstance(value, CycleArc):
        return value
    elif isinstance(value, tuple) and len(value) == 2:
        # (start, end) tuple
        start, end = value
        return CycleArc(mk_cycle_time(start), mk_cycle_time(end))
    else:
        raise ValueError(f"Cannot create CycleArc from {type(value)}")


# =============================================================================
# Convenience Functions
# =============================================================================


def arc(start: Numeric, end: Numeric) -> CycleArc:
    """Create a CycleArc from numeric start and end values.

    Args:
        start: The start time as a numeric value
        end: The end time as a numeric value

    Returns:
        A CycleArc representing the time interval

    Examples:
        >>> arc(0, 1)
        CycleArc(_start=Fraction(0, 1), _end=Fraction(1, 1))
        >>> arc(0.5, 2.5)
        CycleArc(_start=Fraction(1, 2), _end=Fraction(5, 2))
    """
    return CycleArc(mk_cycle_time(start), mk_cycle_time(end))


def now() -> PosixTime:
    """Get the current POSIX time.

    Returns:
        Current wall clock time as PosixTime
    """
    return PosixTime(time.time())
