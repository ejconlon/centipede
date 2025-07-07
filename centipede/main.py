from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Generator, Generic, Optional, Tuple, TypeVar, override

import numpy as np
import numpy.typing as npt
import plotext as plt
from pyrsistent import PVector

type Time = Fraction
type Delta = Fraction
type Factor = Fraction
type Freq = Fraction
type Phase = Fraction
type Rate = int
type Array = npt.NDArray[np.float64]


_ZERO = Fraction(0)
_TAU = np.float64(2 * np.pi)


def mk_lspace(start: Time, end: Time, rate: Rate) -> Array:
    assert start <= end
    assert rate > 0
    start_ix = round(rate * start)
    end_ix = max(start_ix, round(rate * end) - 1)
    num = end_ix - start_ix + 1
    start_rnd = start_ix / rate
    end_rnd = end_ix / rate
    return np.linspace(start=start_rnd, stop=end_rnd, num=num, dtype=np.float64)


def mk_pspace(lspace: Array, freq: Freq, phase: Phase = _ZERO) -> Array:
    assert freq > 0
    arr = lspace.copy()
    np.multiply(arr, np.float64(freq) * _TAU, out=arr)
    if phase.numerator != 0:
        np.add(arr, np.float64(phase), out=arr)
    np.mod(arr, _TAU, out=arr)
    return arr


def mk_sin(lspace: Array, freq: Freq, phase: Phase = _ZERO) -> Array:
    arr = mk_pspace(lspace=lspace, freq=freq, phase=phase)
    np.sin(arr, out=arr)
    return arr


def plot(spc: Array, arr: Array) -> None:
    plt.plot(spc, arr)
    plt.show()


def test_plot() -> None:
    lspace = mk_lspace(Fraction(0), Fraction(1), 1024)
    arr = mk_sin(lspace=lspace, freq=Fraction(4))
    plot(lspace, arr)


@dataclass(frozen=True)
class Arc:
    start: Time
    end: Time

    @classmethod
    def empty(cls) -> Arc:
        return _EMPTY_ARC

    @classmethod
    def cycle(cls, cyc: int) -> Arc:
        return Arc(Fraction(cyc), Fraction(cyc + 1))

    @classmethod
    def union_all(cls, arcs: Sequence[Arc]) -> Arc:
        if len(arcs) == 0:
            return Arc.empty()
        else:
            out = arcs[0]._normalize()
            for i in range(1, len(arcs)):
                out = out.union(arcs[i])
            return out

    @classmethod
    def intersect_all(cls, arcs: Sequence[Arc]) -> Arc:
        if len(arcs) == 0:
            return Arc.empty()
        else:
            out = arcs[0]._normalize()
            for i in range(1, len(arcs)):
                if out.null():
                    break
                out = out.intersect(arcs[i])
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

    def split_cycles(self, bounds: Optional[Arc] = None) -> Generator[Tuple[int, Arc]]:
        start = self.start if bounds is None else max(self.start, bounds.start)
        end = self.end if bounds is None else min(self.end, bounds.end)
        if start < end:
            left_ix = math.floor(start)
            right_ix = math.ceil(end)
            for cyc in range(left_ix, right_ix):
                s = max(Fraction(cyc), self.start)
                e = min(Fraction(cyc + 1), self.end)
                yield (cyc, Arc(s, e))

    def render_lspace(self, rate: Rate) -> Array:
        return mk_lspace(start=self.start, end=self.end, rate=rate)

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


_EMPTY_ARC = Arc(_ZERO, _ZERO)


_T = TypeVar("_T")


@dataclass(frozen=True)
class Ev(Generic[_T]):
    arc: Arc
    val: _T

    def shift(self, delta: Delta) -> Ev[_T]:
        return Ev(self.arc.shift(delta), self.val)

    def scale(self, factor: Factor) -> Ev[_T]:
        return Ev(self.arc.scale(factor), self.val)

    def clip(self, factor: Factor) -> Ev[_T]:
        return Ev(self.arc.clip(factor), self.val)


class Pat(Generic[_T], metaclass=ABCMeta):
    @classmethod
    def empty(cls) -> Pat[_T]:
        return PatEmpty()

    @classmethod
    def pure(cls, val: _T) -> Pat[_T]:
        return PatPure(val)

    @abstractmethod
    def active(self, bounds: Arc) -> Arc:
        raise NotImplementedError()

    @abstractmethod
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        raise NotImplementedError()

    def shift(self, delta: Delta) -> Pat[_T]:
        if delta == 0:
            return self
        else:
            match self:
                case PatShift(d, p):
                    return p.shift(d + delta)
                case _:
                    return PatShift(delta, self)

    def scale(self, factor: Factor) -> Pat[_T]:
        if factor <= 0:
            return PatEmpty()
        elif factor == 1:
            return self
        else:
            match self:
                case PatScale(f, p):
                    return p.scale(f * factor)
                case _:
                    return PatScale(factor, self)


@dataclass(frozen=True)
class PatEmpty(Pat[_T]):
    @override
    def active(self, bounds: Arc) -> Arc:
        return Arc.empty()

    @override
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        yield from ()


@dataclass(frozen=True)
class PatPure(Pat[_T]):
    val: _T

    @override
    def active(self, bounds: Arc) -> Arc:
        return bounds

    @override
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        for _, sub_arc in bounds.split_cycles():
            yield Ev(sub_arc, self.val)


@dataclass(frozen=True)
class PatMask(Pat[_T]):
    mask: Arc
    sub_pat: Pat[_T]

    @override
    def active(self, bounds: Arc) -> Arc:
        sub_bounds = bounds.intersect(self.mask)
        if sub_bounds.null():
            return sub_bounds
        else:
            return self.sub_pat.active(sub_bounds)

    @override
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        sub_bounds = bounds.intersect(self.mask)
        if sub_bounds.null():
            return
        else:
            yield from self.sub_pat.rep(sub_bounds)


@dataclass(frozen=True)
class PatShift(Pat[_T]):
    delta: Delta
    sub_pat: Pat[_T]

    @override
    def active(self, bounds: Arc) -> Arc:
        return self.sub_pat.active(bounds.shift(self.delta)).shift(-self.delta)

    @override
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        yield from map(
            lambda ev: ev.shift(-self.delta), self.sub_pat.rep(bounds.shift(self.delta))
        )


@dataclass(frozen=True)
class PatScale(Pat[_T]):
    factor: Factor
    sub_pat: Pat[_T]

    @override
    def active(self, bounds: Arc) -> Arc:
        raise NotImplementedError()

    @override
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        raise NotImplementedError()


@dataclass(frozen=True)
class PatSeq(Pat[_T]):
    sub_pats: PVector[Pat[_T]]

    @override
    def active(self, bounds: Arc) -> Arc:
        raise NotImplementedError()

    @override
    def rep(self, bounds: Arc) -> Generator[Ev[_T]]:
        raise NotImplementedError()


def main():
    test_plot()


if __name__ == "__main__":
    main()
