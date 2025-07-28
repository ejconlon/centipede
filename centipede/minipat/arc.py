from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Iterator, Optional, Tuple

from centipede.minipat.common import ZERO, Delta, Factor, Time


@dataclass(frozen=True, order=True)
class Arc:
    start: Time
    end: Time

    @staticmethod
    def empty() -> Arc:
        return _EMPTY_ARC

    @staticmethod
    def cycle(cyc: int) -> Arc:
        return Arc(Fraction(cyc), Fraction(cyc + 1))

    @staticmethod
    def union_all(arcs: Iterable[Arc]) -> Arc:
        out = Arc.empty()
        for ix, arc in enumerate(arcs):
            if ix == 0:
                out = arc._normalize()
            else:
                out = out.union(arc)
        return out

    @staticmethod
    def intersect_all(arcs: Iterable[Arc]) -> Arc:
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

    def length(self) -> Delta:
        return self.end - self.start

    def null(self) -> bool:
        return self.start >= self.end

    def _normalize(self) -> Arc:
        if self.start < self.end or (self.start == 0 and self.end == 0):
            return self
        else:
            return Arc.empty()

    def split_cycles(self, bounds: Optional[Arc] = None) -> Iterator[Tuple[int, Arc]]:
        start = self.start if bounds is None else max(self.start, bounds.start)
        end = self.end if bounds is None else min(self.end, bounds.end)
        if start < end:
            left_ix = floor(start)
            right_ix = ceil(end)
            for cyc in range(left_ix, right_ix):
                s = max(Fraction(cyc), self.start)
                e = min(Fraction(cyc + 1), self.end)
                yield (cyc, Arc(s, e))

    def union(self, other: Arc) -> Arc:
        if self.null():
            return other._normalize()
        elif other.null():
            return self
        else:
            start = min(self.start, other.start)
            end = max(self.end, other.end)
            if start < end:
                return Arc(start, end)
            else:
                return Arc.empty()

    def intersect(self, other: Arc) -> Arc:
        if self.null():
            return self._normalize()
        elif other.null():
            return other._normalize()
        else:
            start = max(self.start, other.start)
            end = min(self.end, other.end)
            if start < end:
                return Arc(start, end)
            else:
                return Arc.empty()

    def shift(self, delta: Delta) -> Arc:
        if self.null() or delta == 0:
            return self._normalize()
        else:
            return Arc(self.start + delta, self.end + delta)

    def scale(self, factor: Factor) -> Arc:
        if self.null() or factor == 1:
            return self._normalize()
        elif factor <= 0:
            return Arc.empty()
        else:
            return Arc(self.start * factor, self.end * factor)

    def clip(self, factor: Factor) -> Arc:
        if self.null() or factor == 1:
            return self._normalize()
        elif factor <= 0:
            return Arc.empty()
        else:
            end = self.start + (self.end - self.start) * factor
            return Arc(self.start, end)


_EMPTY_ARC = Arc(ZERO, ZERO)
