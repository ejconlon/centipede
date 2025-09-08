"""Arc type for representing time intervals in minipat patterns."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Iterator, Optional, Self, Tuple, Type, override

from minipat.common import (
    ZERO,
    CycleDelta,
    CycleTime,
    CycleTimeOps,
    Factor,
    PosixDelta,
    PosixTime,
    PosixTimeOps,
    StepDelta,
    StepTime,
    StepTimeOps,
    TimeOps,
)


class Arc[T, D](metaclass=ABCMeta):
    @property
    @abstractmethod
    def start(self) -> T:
        """Get the start time of the arc.

        Returns:
            The start time of the arc
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def end(self) -> T:
        """Get the end time of the arc.

        Returns:
            The end time of the arc
        """
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
        """Get the time operations class for this arc type.

        Returns:
            The time operations class
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def normalize(cls, arc: Arc[T, D]) -> Self:
        """Normalize an arc to the concrete type.

        Args:
            arc: The arc to normalize

        Returns:
            The normalized arc
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def mk(cls, start: T, end: T) -> Self:
        """Create an arc with the given start and end times.

        Args:
            start: The start time
            end: The end time

        Returns:
            A new arc
        """
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
        if self.null() or factor == 1:
            return CycleArc.normalize(self)
        elif factor <= 0:
            return CycleArc.empty()
        else:
            return CycleArc(
                CycleTime(self.start * factor), CycleTime(self.end * factor)
            )

    def clip(self, factor: Factor) -> CycleArc:
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


@dataclass(frozen=True, order=True)
class PosixArc(Arc[PosixTime, PosixDelta]):
    _start: PosixTime
    _end: PosixTime

    @property
    @override
    def start(self) -> PosixTime:
        return self._start

    @property
    @override
    def end(self) -> PosixTime:
        return self._end

    @override
    def length(self) -> PosixDelta:
        return PosixTimeOps.diff(self.end, self.start)

    @override
    def null(self) -> bool:
        return self.start >= self.end

    @override
    def union(self, other: Arc[PosixTime, PosixDelta]) -> PosixArc:
        if self.null():
            return PosixArc.normalize(other)
        elif other.null():
            return self
        else:
            start = PosixTime(min(self.start, other.start))
            end = PosixTime(max(self.end, other.end))
            if start < end:
                return PosixArc(start, end)
            else:
                return PosixArc.empty()

    @override
    def intersect(self, other: Arc[PosixTime, PosixDelta]) -> PosixArc:
        if self.null():
            return PosixArc.normalize(self)
        elif other.null():
            return PosixArc.normalize(other)
        else:
            start = PosixTime(max(self.start, other.start))
            end = PosixTime(min(self.end, other.end))
            if start < end:
                return PosixArc(start, end)
            else:
                return PosixArc.empty()

    def shift(self, delta: PosixDelta) -> PosixArc:
        if self.null() or delta == 0.0:
            return PosixArc.normalize(self)
        else:
            return PosixArc(PosixTime(self.start + delta), PosixTime(self.end + delta))

    def scale(self, factor: Factor) -> PosixArc:
        if self.null() or factor == 1:
            return PosixArc.normalize(self)
        elif factor <= 0:
            return PosixArc.empty()
        else:
            return PosixArc(
                PosixTime(self.start * float(factor)),
                PosixTime(self.end * float(factor)),
            )

    def clip(self, factor: Factor) -> PosixArc:
        if self.null() or factor == 1:
            return PosixArc.normalize(self)
        elif factor <= 0:
            return PosixArc.empty()
        else:
            end = PosixTime(self.start + (self.end - self.start) * float(factor))
            return PosixArc(self.start, end)

    @override
    @classmethod
    def time_ops(cls) -> Type[PosixTimeOps]:
        return PosixTimeOps

    @override
    @classmethod
    def normalize(cls, arc: Arc[PosixTime, PosixDelta]) -> PosixArc:
        if arc.start < arc.end or (arc.start == 0.0 and arc.end == 0.0):
            if isinstance(arc, PosixArc):
                return arc
            else:
                return PosixArc(arc.start, arc.end)
        else:
            return PosixArc.empty()

    @override
    @classmethod
    def mk(cls, start: PosixTime, end: PosixTime) -> PosixArc:
        return cls(start, end)

    @override
    @classmethod
    def empty(cls) -> PosixArc:
        return _EMPTY_POSIX_ARC

    @override
    @classmethod
    def union_all(cls, arcs: Iterable[Arc[PosixTime, PosixDelta]]) -> PosixArc:
        out = PosixArc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = PosixArc.normalize(arc)
            else:
                out = out.union(arc)
        return out

    @override
    @classmethod
    def intersect_all(cls, arcs: Iterable[Arc[PosixTime, PosixDelta]]) -> PosixArc:
        out = PosixArc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = PosixArc.normalize(arc)
            else:
                if out.null():
                    break
                else:
                    out = out.intersect(arc)
        return out


_EMPTY_POSIX_ARC = PosixArc(PosixTime(0.0), PosixTime(0.0))


@dataclass(frozen=True, order=True)
class StepArc(Arc[StepTime, StepDelta]):
    _start: StepTime
    _end: StepTime

    @property
    @override
    def start(self) -> StepTime:
        return self._start

    @property
    @override
    def end(self) -> StepTime:
        return self._end

    @override
    def length(self) -> StepDelta:
        return StepTimeOps.diff(self.end, self.start)

    @override
    def null(self) -> bool:
        return self.start >= self.end

    @override
    def union(self, other: Arc[StepTime, StepDelta]) -> StepArc:
        if self.null():
            return StepArc.normalize(other)
        elif other.null():
            return self
        else:
            start = StepTime(min(self.start, other.start))
            end = StepTime(max(self.end, other.end))
            if start < end:
                return StepArc(start, end)
            else:
                return StepArc.empty()

    @override
    def intersect(self, other: Arc[StepTime, StepDelta]) -> StepArc:
        if self.null():
            return StepArc.normalize(self)
        elif other.null():
            return StepArc.normalize(other)
        else:
            start = StepTime(max(self.start, other.start))
            end = StepTime(min(self.end, other.end))
            if start < end:
                return StepArc(start, end)
            else:
                return StepArc.empty()

    def shift(self, delta: StepDelta) -> StepArc:
        if self.null() or delta == 0:
            return StepArc.normalize(self)
        else:
            return StepArc(StepTime(self.start + delta), StepTime(self.end + delta))

    def scale(self, factor: Factor) -> StepArc:
        if self.null() or factor == 1:
            return StepArc.normalize(self)
        elif factor <= 0:
            return StepArc.empty()
        else:
            return StepArc(
                StepTime(int(self.start * factor)), StepTime(int(self.end * factor))
            )

    def clip(self, factor: Factor) -> StepArc:
        if self.null() or factor == 1:
            return StepArc.normalize(self)
        elif factor <= 0:
            return StepArc.empty()
        else:
            end = StepTime(self.start + int((self.end - self.start) * factor))
            return StepArc(self.start, end)

    @override
    @classmethod
    def time_ops(cls) -> Type[StepTimeOps]:
        return StepTimeOps

    @override
    @classmethod
    def normalize(cls, arc: Arc[StepTime, StepDelta]) -> StepArc:
        if arc.start < arc.end or (arc.start == 0 and arc.end == 0):
            if isinstance(arc, StepArc):
                return arc
            else:
                return StepArc(arc.start, arc.end)
        else:
            return StepArc.empty()

    @override
    @classmethod
    def mk(cls, start: StepTime, end: StepTime) -> StepArc:
        return cls(start, end)

    @override
    @classmethod
    def empty(cls) -> StepArc:
        return _EMPTY_STEP_ARC

    @override
    @classmethod
    def union_all(cls, arcs: Iterable[Arc[StepTime, StepDelta]]) -> StepArc:
        out = StepArc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = StepArc.normalize(arc)
            else:
                out = out.union(arc)
        return out

    @override
    @classmethod
    def intersect_all(cls, arcs: Iterable[Arc[StepTime, StepDelta]]) -> StepArc:
        out = StepArc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = StepArc.normalize(arc)
            else:
                if out.null():
                    break
                else:
                    out = out.intersect(arc)
        return out


_EMPTY_STEP_ARC = StepArc(StepTime(0), StepTime(0))


class Span[T, D, A](metaclass=ABCMeta):
    @property
    @abstractmethod
    def active(self) -> A:
        """Get the active arc of the span.

        Returns:
            The active arc
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def whole(self) -> Optional[A]:
        """Get the whole arc of the span, if present.

        Returns:
            The whole arc, or None if not present
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def time_ops(cls) -> Type[TimeOps[T, D]]:
        """Get the time operations class for this span type.

        Returns:
            The time operations class
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def mk(cls, active: A, whole: Optional[A]) -> Span[T, D, A]:
        """Create a span with the given active and whole arcs.

        Args:
            active: The active arc
            whole: The whole arc, or None

        Returns:
            A new span
        """
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
        new_active = self._active.shift(delta)
        new_whole = self._whole.shift(delta) if self._whole is not None else None
        return CycleSpan(new_active, new_whole)

    def scale(self, factor: Factor) -> CycleSpan:
        new_active = self._active.scale(factor)
        new_whole = self._whole.scale(factor) if self._whole is not None else None
        return CycleSpan(new_active, new_whole)

    def clip(self, factor: Factor) -> CycleSpan:
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


@dataclass(frozen=True, order=True)
class PosixSpan(Span[PosixTime, PosixDelta, PosixArc]):
    """Annotates an Arc optionally contained within a wider Arc.
    This is useful to communicate that certain intervals belong
    to larger intervals.

    Args:
        active: The interval in question
        whole: If present, a wider interval containing active
    """

    _active: PosixArc
    _whole: Optional[PosixArc] = None

    @property
    @override
    def active(self) -> PosixArc:
        return self._active

    @property
    @override
    def whole(self) -> Optional[PosixArc]:
        return self._whole

    @override
    def shift(self, delta: PosixDelta) -> PosixSpan:
        new_active = self._active.shift(delta)
        new_whole = self._whole.shift(delta) if self._whole is not None else None
        return PosixSpan(new_active, new_whole)

    def scale(self, factor: Factor) -> PosixSpan:
        new_active = self._active.scale(factor)
        new_whole = self._whole.scale(factor) if self._whole is not None else None
        return PosixSpan(new_active, new_whole)

    def clip(self, factor: Factor) -> PosixSpan:
        new_active = self._active.clip(factor)
        new_whole = self._whole.clip(factor) if self._whole is not None else None
        return PosixSpan(new_active, new_whole)

    @override
    @classmethod
    def time_ops(cls) -> Type[PosixTimeOps]:
        return PosixTimeOps

    @override
    @classmethod
    def mk(
        cls,
        active: PosixArc,
        whole: Optional[PosixArc],
    ) -> PosixSpan:
        return cls(active, whole)

    @override
    @classmethod
    def empty(cls) -> PosixSpan:
        return _EMPTY_POSIX_SPAN


_EMPTY_POSIX_SPAN = PosixSpan(_EMPTY_POSIX_ARC, None)


@dataclass(frozen=True, order=True)
class StepSpan(Span[StepTime, StepDelta, StepArc]):
    """Annotates an Arc optionally contained within a wider Arc.
    This is useful to communicate that certain intervals belong
    to larger intervals.

    Args:
        active: The interval in question
        whole: If present, a wider interval containing active
    """

    _active: StepArc
    _whole: Optional[StepArc] = None

    @property
    @override
    def active(self) -> StepArc:
        return self._active

    @property
    @override
    def whole(self) -> Optional[StepArc]:
        return self._whole

    @override
    def shift(self, delta: StepDelta) -> StepSpan:
        new_active = self._active.shift(delta)
        new_whole = self._whole.shift(delta) if self._whole is not None else None
        return StepSpan(new_active, new_whole)

    def scale(self, factor: Factor) -> StepSpan:
        new_active = self._active.scale(factor)
        new_whole = self._whole.scale(factor) if self._whole is not None else None
        return StepSpan(new_active, new_whole)

    def clip(self, factor: Factor) -> StepSpan:
        new_active = self._active.clip(factor)
        new_whole = self._whole.clip(factor) if self._whole is not None else None
        return StepSpan(new_active, new_whole)

    @override
    @classmethod
    def time_ops(cls) -> Type[StepTimeOps]:
        return StepTimeOps

    @override
    @classmethod
    def mk(
        cls,
        active: StepArc,
        whole: Optional[StepArc],
    ) -> StepSpan:
        return cls(active, whole)

    @override
    @classmethod
    def empty(cls) -> StepSpan:
        return _EMPTY_STEP_SPAN


_EMPTY_STEP_SPAN = StepSpan(_EMPTY_STEP_ARC, None)
