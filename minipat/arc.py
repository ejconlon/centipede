"""Arc type for representing time intervals in minipat patterns."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Iterator, Optional, Tuple

from minipat.common import ZERO, CycleDelta, CycleTime, CycleTimeOps, Factor


@dataclass(frozen=True, order=True)
class Arc:
    """Represents a time interval with start and end points.

    Args:
        start: The start time of the arc
        end: The end time of the arc
    """

    start: CycleTime
    end: CycleTime

    @staticmethod
    def empty() -> Arc:
        """Create an empty arc (start >= end).

        Returns:
            An empty arc
        """
        return _EMPTY_ARC

    @staticmethod
    def cycle(cyc: int) -> Arc:
        """Create an arc representing a full cycle.

        Args:
            cyc: The cycle number

        Returns:
            An arc from cyc to cyc+1
        """
        return Arc(CycleTime(Fraction(cyc)), CycleTime(Fraction(cyc + 1)))

    @staticmethod
    def union_all(arcs: Iterable[Arc]) -> Arc:
        """Create the union of all given arcs.

        Args:
            arcs: The arcs to unite

        Returns:
            The union of all arcs
        """
        out = Arc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = arc._normalize()
            else:
                out = out.union(arc)
        return out

    @staticmethod
    def intersect_all(arcs: Iterable[Arc]) -> Arc:
        """Create the intersection of all given arcs.

        Args:
            arcs: The arcs to intersect

        Returns:
            The intersection of all arcs
        """
        out = Arc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = arc._normalize()
            else:
                if out.null():
                    break
                else:
                    out = out.intersect(arc)
        return out

    def length(self) -> CycleDelta:
        """Get the length of the arc.

        Returns:
            The duration of the arc (end - start)
        """
        return CycleTimeOps.diff(self.end, self.start)

    def null(self) -> bool:
        """Check if the arc is null (empty).

        Returns:
            True if start >= end, False otherwise
        """
        return self.start >= self.end

    def _normalize(self) -> Arc:
        if self.start < self.end or (self.start == 0 and self.end == 0):
            return self
        else:
            return Arc.empty()

    def split_cycles(self, bounds: Optional[Arc] = None) -> Iterator[Tuple[int, Arc]]:
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
                yield (cyc, Arc(s, e))

    def union(self, other: Arc) -> Arc:
        """Create the union of this arc with another.

        Args:
            other: The other arc to unite with

        Returns:
            The union of both arcs
        """
        if self.null():
            return other._normalize()
        elif other.null():
            return self
        else:
            start = CycleTime(min(self.start, other.start))
            end = CycleTime(max(self.end, other.end))
            if start < end:
                return Arc(start, end)
            else:
                return Arc.empty()

    def intersect(self, other: Arc) -> Arc:
        """Create the intersection of this arc with another.

        Args:
            other: The other arc to intersect with

        Returns:
            The intersection of both arcs
        """
        if self.null():
            return self._normalize()
        elif other.null():
            return other._normalize()
        else:
            start = CycleTime(max(self.start, other.start))
            end = CycleTime(min(self.end, other.end))
            if start < end:
                return Arc(start, end)
            else:
                return Arc.empty()

    def shift(self, delta: CycleDelta) -> Arc:
        """Shift the arc by a given delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new arc shifted by delta
        """
        if self.null() or delta == 0:
            return self._normalize()
        else:
            return Arc(CycleTime(self.start + delta), CycleTime(self.end + delta))

    def scale(self, factor: Factor) -> Arc:
        """Scale the arc by a given factor.

        Args:
            factor: The scaling factor

        Returns:
            A new arc scaled by the factor
        """
        if self.null() or factor == 1:
            return self._normalize()
        elif factor <= 0:
            return Arc.empty()
        else:
            return Arc(CycleTime(self.start * factor), CycleTime(self.end * factor))

    def clip(self, factor: Factor) -> Arc:
        """Clip the arc to a fraction of its length.

        Args:
            factor: The fraction to clip to (0 to 1)

        Returns:
            A new arc clipped to the given fraction
        """
        if self.null() or factor == 1:
            return self._normalize()
        elif factor <= 0:
            return Arc.empty()
        else:
            end = CycleTime(self.start + (self.end - self.start) * factor)
            return Arc(self.start, end)


_EMPTY_ARC = Arc(start=CycleTime(ZERO), end=CycleTime(ZERO))


@dataclass(frozen=True, order=True)
class Span:
    """Annotates an Arc optionally contained within a wider Arc.
    This is useful to communicate that certain intervals belong
    to larger intervals.

    Args:
        active: The interval in question
        whole: If present, a wider interval containing active
    """

    active: Arc
    whole: Optional[Arc] = None

    @staticmethod
    def empty() -> Span:
        """Create an empty span

        Returns:
            An empty span
        """
        return _EMPTY_SPAN

    def shift(self, delta: CycleDelta) -> Span:
        """Shift the span by a given delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new span shifted by delta
        """
        new_active = self.active.shift(delta)
        new_whole = self.whole.shift(delta) if self.whole is not None else None
        return Span(active=new_active, whole=new_whole)

    def scale(self, factor: Factor) -> Span:
        """Scale the span by a given factor.

        Args:
            factor: The scaling factor

        Returns:
            A new span scaled by the factor
        """
        new_active = self.active.scale(factor)
        new_whole = self.whole.scale(factor) if self.whole is not None else None
        return Span(active=new_active, whole=new_whole)

    def clip(self, factor: Factor) -> Span:
        """Clip the span to a fraction of its length.

        Args:
            factor: The fraction to clip to (0 to 1)

        Returns:
            A new span clipped to the given fraction
        """
        new_active = self.active.clip(factor)
        new_whole = self.whole.clip(factor) if self.whole is not None else None
        return Span(active=new_active, whole=new_whole)


_EMPTY_SPAN = Span(active=_EMPTY_ARC, whole=None)
