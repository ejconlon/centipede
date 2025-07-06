from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generator, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import plotext as plt

type Time = float
type Delta = float
type Freq = float
type Phase = float
type Rate = int
type Array = npt.NDArray[np.float64]


TAU = np.float64(2 * np.pi)


def mk_lspace(start: Time, end: Time, rate: Rate) -> Array:
    assert start <= end
    assert rate > 0
    start_ix = round(rate * start)
    end_ix = max(start_ix, round(rate * end) - 1)
    num = end_ix - start_ix + 1
    start_rnd = start_ix / rate
    end_rnd = end_ix / rate
    return np.linspace(start=start_rnd, stop=end_rnd, num=num, dtype=np.float64)


def mk_pspace(lspace: Array, freq: Freq, phase: Phase = 0) -> Array:
    assert freq > 0
    arr = lspace.copy()
    np.multiply(arr, freq * TAU, out=arr)
    if phase != 0:
        np.add(arr, phase, out=arr)
    np.mod(arr, TAU, out=arr)
    return arr


def mk_sin(lspace: Array, freq: Freq, phase: Phase = 0) -> Array:
    arr = mk_pspace(lspace=lspace, freq=freq, phase=phase)
    np.sin(arr, out=arr)
    return arr


def plot(spc: Array, arr: Array) -> None:
    plt.plot(spc, arr)
    plt.show()


def test_plot() -> None:
    lspace = mk_lspace(0, 1, 1024)
    arr = mk_sin(lspace=lspace, freq=4)
    plot(lspace, arr)


@dataclass(frozen=True)
class Arc:
    start: Time
    end: Time

    @classmethod
    def empty(cls) -> Arc:
        return Arc(0, 0)

    @classmethod
    def cycle(cls, cyc: int) -> Arc:
        return Arc(cyc, cyc + 1)

    def null(self) -> bool:
        return self.start >= self.end

    def split_cycles(self, bounds: Optional[Arc] = None) -> Generator[Tuple[int, Arc]]:
        start = self.start if bounds is None else max(self.start, bounds.start)
        end = self.end if bounds is None else min(self.end, bounds.end)
        left_ix = math.floor(start)
        right_ix = math.ceil(end)
        for cyc in range(left_ix, right_ix):
            s = max(cyc, self.start)
            e = min(cyc + 1, self.end)
            yield (cyc, Arc(s, e))

    def render_lspace(self, rate: Rate) -> Array:
        return mk_lspace(start=self.start, end=self.end, rate=rate)


T = TypeVar("T")


@dataclass(frozen=True)
class Ev(Generic[T]):
    arc: Arc
    val: T


class Pat(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def active(self, bounds: Arc) -> Arc:
        raise NotImplementedError()

    @abstractmethod
    def rep(self, bounds: Arc) -> Generator[Ev[T]]:
        raise NotImplementedError()


@dataclass(frozen=True)
class PatPure(Pat[T]):
    val: T

    def active(self, bounds: Arc) -> Arc:
        return bounds

    def rep(self, bounds: Arc) -> Generator[Ev[T]]:
        for _, sub_arc in bounds.split_cycles():
            yield Ev(sub_arc, self.val)


@dataclass(frozen=True)
class PatSeq(Pat[T]):
    sub_pats: List[Pat[T]]

    def active(self, bounds: Arc) -> Arc:
        raise NotImplementedError()

    def rep(self, bounds: Arc) -> Generator[Ev[T]]:
        raise NotImplementedError()


# @dataclass(frozen=True)
# class Periodic:
#     arc: Arc
#     freq: Freq
#     phase: Phase = 0
#
#     def render(self, rate: Rate) -> Array:
#         raise Exception('TODO')


def main():
    test_plot()


if __name__ == "__main__":
    main()
