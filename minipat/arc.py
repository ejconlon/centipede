"""Arc type for representing time intervals in minipat patterns."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Iterator, Optional, Self, Tuple, Type, override

from minipat.common import ZERO, CycleDelta, CycleTime, CycleTimeOps, Factor, TimeOps


class Arc[T, D](metaclass=ABCMeta):
    @property
    @abstractmethod
    def start(self) -> T:
        raise NotImplementedError()

    @property
    @abstractmethod
    def end(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def length(self) -> D:
        """Get the length of the arc.

        Returns:
            The duration of the arc (end - start)
        """
        raise NotImplementedError()

    @abstractmethod
    def null(self) -> bool:
        """Check if the arc is null (empty).

        Returns:
            True if start >= end, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def union(self, other: Arc[T, D]) -> Self:
        """Create the union of this arc with another.

        Args:
            other: The other arc to unite with

        Returns:
            The union of both arcs
        """
        raise NotImplementedError()

    @abstractmethod
    def intersect(self, other: Arc[T, D]) -> Self:
        """Create the intersection of this arc with another.

        Args:
            other: The other arc to intersect with

        Returns:
            The intersection of both arcs
        """
        raise NotImplementedError()

    @abstractmethod
    def shift(self, delta: D) -> Self:
        """Shift the arc by a given delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new arc shifted by delta
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def time_ops(cls) -> Type[TimeOps[T, D]]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def normalize(cls, arc: Arc[T, D]) -> Self:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def mk(cls, start: T, end: T) -> Self:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        """Create an empty arc (start >= end).

        Returns:
            An empty arc
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def union_all(cls, arcs: Iterable[Arc[T, D]]) -> Self:
        """Create the union of all given arcs.

        Args:
            arcs: The arcs to unite

        Returns:
            The union of all arcs
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def intersect_all(cls, arcs: Iterable[Arc[T, D]]) -> Self:
        """Create the intersection of all given arcs.

        Args:
            arcs: The arcs to intersect

        Returns:
            The intersection of all arcs
        """
        raise NotImplementedError()


@dataclass(frozen=True, order=True)
class CycleArc(Arc[CycleTime, CycleDelta]):
    _start: CycleTime
    _end: CycleTime

    @property
    @override
    def start(self) -> CycleTime:
        return self._start

    @property
    @override
    def end(self) -> CycleTime:
        return self._end

    @override
    def length(self) -> CycleDelta:
        return CycleTimeOps.diff(self.end, self.start)

    @override
    def null(self) -> bool:
        return self.start >= self.end

    @override
    def union(self, other: Arc[CycleTime, CycleDelta]) -> CycleArc:
        if self.null():
            return CycleArc.normalize(other)
        elif other.null():
            return self
        else:
            start = CycleTime(min(self.start, other.start))
            end = CycleTime(max(self.end, other.end))
            if start < end:
                return CycleArc(start, end)
            else:
                return CycleArc.empty()

    @override
    def intersect(self, other: Arc[CycleTime, CycleDelta]) -> CycleArc:
        if self.null():
            return CycleArc.normalize(self)
        elif other.null():
            return CycleArc.normalize(other)
        else:
            start = CycleTime(max(self.start, other.start))
            end = CycleTime(min(self.end, other.end))
            if start < end:
                return CycleArc(start, end)
            else:
                return CycleArc.empty()

    def shift(self, delta: CycleDelta) -> CycleArc:
        """Shift the arc by a given delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new arc shifted by delta
        """
        if self.null() or delta == 0:
            return CycleArc.normalize(self)
        else:
            return CycleArc(CycleTime(self.start + delta), CycleTime(self.end + delta))

    def split_cycles(
        self, bounds: Optional[CycleArc] = None
    ) -> Iterator[Tuple[int, CycleArc]]:
        """Split the arc into cycles, yielding (cycle_index, arc) pairs.

        Args:
            bounds: Optional bounds to constrain the splitting

        Yields:
            Tuples of (cycle_index, arc) for each cycle
        """
        start = self.start if bounds is None else max(self.start, bounds.start)
        end = self.end if bounds is None else min(self.end, bounds.end)
        if start < end:
            left_ix = floor(start)
            right_ix = ceil(end)
            for cyc in range(left_ix, right_ix):
                s = CycleTime(max(Fraction(cyc), self.start))
                e = CycleTime(min(Fraction(cyc + 1), self.end))
                yield (cyc, CycleArc(s, e))

    def scale(self, factor: Factor) -> CycleArc:
        """Scale the arc by a given factor.

        Args:
            factor: The scaling factor

        Returns:
            A new arc scaled by the factor
        """
        if self.null() or factor == 1:
            return CycleArc.normalize(self)
        elif factor <= 0:
            return CycleArc.empty()
        else:
            return CycleArc(
                CycleTime(self.start * factor), CycleTime(self.end * factor)
            )

    def clip(self, factor: Factor) -> CycleArc:
        """Clip the arc to a fraction of its length.

        Args:
            factor: The fraction to clip to (0 to 1)

        Returns:
            A new arc clipped to the given fraction
        """
        if self.null() or factor == 1:
            return CycleArc.normalize(self)
        elif factor <= 0:
            return CycleArc.empty()
        else:
            end = CycleTime(self.start + (self.end - self.start) * factor)
            return CycleArc(self.start, end)

    @override
    @classmethod
    def time_ops(cls) -> Type[CycleTimeOps]:
        return CycleTimeOps

    @override
    @classmethod
    def normalize(cls, arc: Arc[CycleTime, CycleDelta]) -> CycleArc:
        if arc.start < arc.end or (arc.start == 0 and arc.end == 0):
            if isinstance(arc, CycleArc):
                return arc
            else:
                return CycleArc(arc.start, arc.end)
        else:
            return CycleArc.empty()

    @override
    @classmethod
    def mk(cls, start: CycleTime, end: CycleTime) -> CycleArc:
        return cls(start, end)

    @override
    @classmethod
    def empty(cls) -> CycleArc:
        return _EMPTY_CYCLE_ARC

    @classmethod
    def cycle(cls, cyc: int) -> CycleArc:
        """Create an arc representing a full cycle.

        Args:
            cyc: The cycle number

        Returns:
            An arc from cyc to cyc+1
        """
        return CycleArc(CycleTime(Fraction(cyc)), CycleTime(Fraction(cyc + 1)))

    @override
    @classmethod
    def union_all(cls, arcs: Iterable[Arc[CycleTime, CycleDelta]]) -> CycleArc:
        out = CycleArc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = CycleArc.normalize(arc)
            else:
                out = out.union(arc)
        return out

    @override
    @classmethod
    def intersect_all(cls, arcs: Iterable[Arc[CycleTime, CycleDelta]]) -> CycleArc:
        out = CycleArc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = CycleArc.normalize(arc)
            else:
                if out.null():
                    break
                else:
                    out = out.intersect(arc)
        return out


_EMPTY_CYCLE_ARC = CycleArc(CycleTime(ZERO), CycleTime(ZERO))


class Span[T, D, A](metaclass=ABCMeta):
    @property
    @abstractmethod
    def active(self) -> A:
        raise NotImplementedError()

    @property
    @abstractmethod
    def whole(self) -> Optional[A]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def time_ops(cls) -> Type[TimeOps[T, D]]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def mk(cls, active: A, whole: Optional[A]) -> Span[T, D, A]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def empty(cls) -> Span[T, D, A]:
        """Create an empty span

        Returns:
            An empty span
        """
        raise NotImplementedError()

    @abstractmethod
    def shift(self, delta: D) -> Span[T, D, A]:
        """Shift the span by a given delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new span shifted by delta
        """
        raise NotImplementedError()


@dataclass(frozen=True, order=True)
class CycleSpan(Span[CycleTime, CycleDelta, CycleArc]):
    """Annotates an Arc optionally contained within a wider Arc.
    This is useful to communicate that certain intervals belong
    to larger intervals.

    Args:
        active: The interval in question
        whole: If present, a wider interval containing active
    """

    _active: CycleArc
    _whole: Optional[CycleArc] = None

    @property
    @override
    def active(self) -> CycleArc:
        return self._active

    @property
    @override
    def whole(self) -> Optional[CycleArc]:
        return self._whole

    @override
    def shift(self, delta: CycleDelta) -> CycleSpan:
        """Shift the span by a given delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new span shifted by delta
        """
        new_active = self._active.shift(delta)
        new_whole = self._whole.shift(delta) if self._whole is not None else None
        return CycleSpan(new_active, new_whole)

    def scale(self, factor: Factor) -> CycleSpan:
        """Scale the span by a given factor.

        Args:
            factor: The scaling factor

        Returns:
            A new span scaled by the factor
        """
        new_active = self._active.scale(factor)
        new_whole = self._whole.scale(factor) if self._whole is not None else None
        return CycleSpan(new_active, new_whole)

    def clip(self, factor: Factor) -> CycleSpan:
        """Clip the span to a fraction of its length.

        Args:
            factor: The fraction to clip to (0 to 1)

        Returns:
            A new span clipped to the given fraction
        """
        new_active = self._active.clip(factor)
        new_whole = self._whole.clip(factor) if self._whole is not None else None
        return CycleSpan(new_active, new_whole)

    @override
    @classmethod
    def time_ops(cls) -> Type[CycleTimeOps]:
        return CycleTimeOps

    @override
    @classmethod
    def mk(
        cls,
        active: CycleArc,
        whole: Optional[CycleArc],
    ) -> CycleSpan:
        return cls(active, whole)

    @override
    @classmethod
    def empty(cls) -> CycleSpan:
        return _EMPTY_CYCLE_SPAN


_EMPTY_CYCLE_SPAN = CycleSpan(_EMPTY_CYCLE_ARC, None)
