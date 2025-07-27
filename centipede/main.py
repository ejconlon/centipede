from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from math import ceil, floor
from typing import Any, Callable, Generator, Optional, Tuple

import numpy as np
import numpy.typing as npt
import plotext as plt

from centipede.spiny import Box, PSeq


def ignore_arg[A, B](fn: Callable[[A], B]) -> Callable[[None, A], B]:
    def wrapper(_: None, arg: A) -> B:
        return fn(arg)

    return wrapper


type Time = Fraction
type Delta = Fraction
type Factor = Fraction
type Freq = Fraction
type Phase = Fraction
type Rate = int
type Array = npt.NDArray[np.float64]


_acc = Fraction(0)
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


def mk_pspace(lspace: Array, freq: Freq, phase: Phase = _acc) -> Array:
    assert freq > 0
    arr = lspace.copy()
    np.multiply(arr, np.float64(freq) * _TAU, out=arr)
    if phase.numerator != 0:
        np.add(arr, np.float64(phase), out=arr)
    np.mod(arr, _TAU, out=arr)
    return arr


def mk_sin(lspace: Array, freq: Freq, phase: Phase = _acc) -> Array:
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
    def union_all(arcs: Sequence[Arc]) -> Arc:
        if len(arcs) == 0:
            return Arc.empty()
        else:
            out = arcs[0]._normalize()
            for i in range(1, len(arcs)):
                out = out.union(arcs[i])
            return out

    @staticmethod
    def intersect_all(arcs: Sequence[Arc]) -> Arc:
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
            left_ix = floor(start)
            right_ix = ceil(end)
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


_EMPTY_ARC = Arc(_acc, _acc)


@dataclass(frozen=True)
class Ev[T]:
    arc: Arc
    val: T

    def shift(self, delta: Delta) -> Ev[T]:
        return Ev(self.arc.shift(delta), self.val)

    def scale(self, factor: Factor) -> Ev[T]:
        return Ev(self.arc.scale(factor), self.val)

    def clip(self, factor: Factor) -> Ev[T]:
        return Ev(self.arc.clip(factor), self.val)


# type EvHeap[T] = PHeap[Arc, Ev[T]]
#
#
# def ev_heap_empty[T]() -> EvHeap[T]:
#     return PHeap.empty()
#
#
# def ev_heap_push[T](ev: Ev[T], heap: EvHeap[T]) -> EvHeap[T]:
#     raise NotImplementedError()


# sealed
class PatF[T, R]:
    pass


@dataclass(frozen=True)
class Pat[T]:
    unwrap: PatF[T, Pat[T]]

    @staticmethod
    def empty() -> Pat[T]:
        return _PAT_EMPTY

    @staticmethod
    def pure(val: T) -> Pat[T]:
        return Pat(PatPure(val))

    @staticmethod
    def seq(pats: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatSeq(PSeq.mk(pats)))

    @staticmethod
    def par(pats: Iterable[Pat[T]]) -> Pat[T]:
        return Pat(PatPar(PSeq.mk(pats)))

    def mask(self, arc: Arc) -> Pat[T]:
        if arc.null():
            return Pat.empty()
        else:
            match self.unwrap:
                case PatMask(a, p):
                    return p.mask(a.intersect(arc))
                case _:
                    return Pat(PatMask(arc, self))

    def shift(self, delta: Delta) -> Pat[T]:
        if delta == 0:
            return self
        else:
            match self.unwrap:
                case PatShift(d, p):
                    return p.shift(d + delta)
                case _:
                    return Pat(PatShift(delta, self))

    def scale(self, factor: Factor) -> Pat[T]:
        if factor <= 0:
            return Pat.empty()
        elif factor == 1:
            return self
        else:
            match self.unwrap:
                case PatScale(f, p):
                    return p.scale(f * factor)
                case _:
                    return Pat(PatScale(factor, self))

    def clip(self, factor: Factor) -> Pat[T]:
        if factor <= 0:
            return Pat.empty()
        elif factor == 1:
            return self
        else:
            match self.unwrap:
                case PatClip(f, p):
                    return p.clip(f * factor)
                case _:
                    return Pat(PatClip(factor, self))

    def map[U](self, fn: Callable[[T], U]) -> Pat[U]:
        return pat_map(fn)(self)

    def fold[Z](self, start: Z, fn: Callable[[Box[Z], T], None]) -> Z:
        return pat_fold(fn)(start, self)

    def cata[Z](self, fn: Callable[[PatF[T, Z]], Z]) -> Z:
        return pat_cata(fn)(self)

    def cata_state[S, Z](
        self, start: S, fn: Callable[[Box[S], PatF[T, Z]], Z]
    ) -> Tuple[S, Z]:
        return pat_cata_state(fn)(start, self)


@dataclass(frozen=True)
class PatEmpty(PatF[Any, Any]):
    pass


_PAT_EMPTY = Pat(PatEmpty())


@dataclass(frozen=True)
class PatPure[T](PatF[T, Any]):
    val: T


@dataclass(frozen=True)
class PatMask[T, R](PatF[T, R]):
    arc: Arc
    child: R

    # @override
    # def active(self, bounds: Arc) -> Arc:
    #     sub_bounds = bounds.intersect(self.mask)
    #     if sub_bounds.null():
    #         return sub_bounds
    #     else:
    #         return self.sub_pat.active(sub_bounds)
    #
    # @override
    # def rep(self, bounds: Arc) -> Generator[Ev[T]]:
    #     sub_bounds = bounds.intersect(self.mask)
    #     if sub_bounds.null():
    #         return
    #     else:
    #         yield from self.sub_pat.rep(sub_bounds)


@dataclass(frozen=True)
class PatShift[T, R](PatF[T, R]):
    delta: Delta
    child: R

    # @override
    # def active(self, bounds: Arc) -> Arc:
    #     return self.sub_pat.active(bounds.shift(self.delta)).shift(-self.delta)
    #
    # @override
    # def rep(self, bounds: Arc) -> Generator[Ev[T]]:
    #     yield from map(
    #         lambda ev: ev.shift(-self.delta), self.sub_pat.rep(bounds.shift(self.delta))
    #     )


@dataclass(frozen=True)
class PatScale[T, R](PatF[T, R]):
    factor: Factor
    child: R


@dataclass(frozen=True)
class PatClip[T, R](PatF[T, R]):
    factor: Factor
    child: R


@dataclass(frozen=True)
class PatSeq[T, R](PatF[T, R]):
    children: PSeq[R]


@dataclass(frozen=True)
class PatPar[T, R](PatF[T, R]):
    children: PSeq[R]


class PartialMatchException(Exception):
    def __init__(self, val: Any):
        super().__init__(f"Unmatched type: {type(val)}")


def pat_cata_env[V, T, Z](fn: Callable[[V, PatF[T, Z]], Z]) -> Callable[[V, Pat[T]], Z]:
    def wrapper(env: V, pat: Pat[T]) -> Z:
        pf = pat.unwrap
        match pf:
            case PatEmpty():
                return fn(env, pf)
            case PatPure(_):
                return fn(env, pf)
            case PatClip(f, c):
                cz = wrapper(env, c)
                return fn(env, PatClip(f, cz))
            case PatMask(a, c):
                cz = wrapper(env, c)
                return fn(env, PatMask(a, cz))
            case PatScale(f, c):
                cz = wrapper(env, c)
                return fn(env, PatScale(f, cz))
            case PatSeq(cs):
                czs = PSeq.mk(wrapper(env, c) for c in cs)
                return fn(env, PatSeq(czs))
            case PatPar(cs):
                czs = PSeq.mk(wrapper(env, c) for c in cs)
                return fn(env, PatPar(czs))
            case _:
                raise PartialMatchException(pf)

    return wrapper


def pat_cata_state[S, T, Z](
    fn: Callable[[Box[S], PatF[T, Z]], Z],
) -> Callable[[S, Pat[T]], Tuple[S, Z]]:
    k: Callable[[Box[S], Pat[T]], Z] = pat_cata_env(fn)

    def unwrap(start: S, pat: Pat[T]) -> Tuple[S, Z]:
        box = Box(start)
        out = k(box, pat)
        return (box.value, out)

    return unwrap


def pat_cata[T, Z](fn: Callable[[PatF[T, Z]], Z]) -> Callable[[Pat[T]], Z]:
    k: Callable[[None, Pat[T]], Z] = pat_cata_env(ignore_arg(fn))
    return partial(k, None)


def pat_map[T, U](fn: Callable[[T], U]) -> Callable[[Pat[T]], Pat[U]]:
    def elim(pf: PatF[T, Pat[U]]) -> Pat[U]:
        match pf:
            case PatEmpty():
                return Pat.empty()
            case PatPure(val):
                return Pat(PatPure(fn(val)))
            case PatClip(f, c):
                return Pat(PatClip(f, c))
            case PatMask(a, c):
                return Pat(PatMask(a, c))
            case PatScale(f, c):
                return Pat(PatScale(f, c))
            case PatSeq(cs):
                return Pat(PatSeq(cs))
            case PatPar(cs):
                return Pat(PatPar(cs))
            case _:
                raise PartialMatchException(pf)

    return pat_cata(elim)


def pat_fold[T, Z](fn: Callable[[Box[Z], T], None]) -> Callable[[Z, Pat[T]], Z]:
    def visit(box: Box[Z], pf: PatF[T, None]) -> None:
        match pf:
            case PatPure(val):
                fn(box, val)
            case _:
                pass

    k: Callable[[Box[Z], Pat[T]], None] = pat_cata_env(visit)

    def wrapper(start: Z, pat: Pat[T]) -> Z:
        box = Box(start)
        k(box, pat)
        return box.value

    return wrapper


# # @dataclass(frozen=True)
# # class CellKey[K]:
# #     row: K
# #     col: K
# #
# #
# # @dataclass(frozen=True)
# # class CellIx:
# #     row: int
# #     col: int
# #
# #
# # def trim_entries[V](rows: int, cols: int, entries: PMap[CellIx, V]) -> PMap[CellIx, V]:
# #     ev = entries.evolver()
# #     for ix in entries:
# #         if ix.row >= rows or ix.col >= cols:
# #             del ev[ix]
# #     return ev.persistent()
# #
# #
# # @dataclass(frozen=True)
# # class Table[K, V]:
# #     rows: int
# #     col_lookup: PMap[K, int]
# #     entries: PMap[CellIx, V]
# #
# #     @property
# #     def cols(self) -> int:
# #         return len(self.col_lookup)
# #
# #     # def append_row(self) -> Table[K, V]:
# #     #     return self.set_size(rows=self.rows + 1, cols=self.cols)
# #     #
# #     # def add_col(self) -> Table[K, V]:
# #     #     return self.set_size(rows=self.rows, cols=self.cols + 1)
# #     #
# #     # def del_row(self) -> Table[K, V]:
# #     #     return self.set_size(rows=self.rows - 1, cols=self.cols)
# #     #
# #     # def del_col(self) -> Table[K, V]:
# #     #     return self.set_size(rows=self.rows, cols=self.cols - 1)
# #     #
# #     # def set_size(self, rows: int, cols: int) -> Table[K, V]:
# #     #     if rows >= self.rows and cols >= self.cols:
# #     #         return replace(self, rows=rows, cols=cols)
# #     #     else:
# #     #         entries = trim_entries(rows=rows, cols=cols, entries=self.entries)
# #     #         return Table(rows=rows, cols=cols, entries=entries)


def main():
    test_plot()


if __name__ == "__main__":
    main()
