from __future__ import annotations

from fractions import Fraction

import numpy as np
import numpy.typing as npt
import plotext as plt

from minipat.arc import Arc
from minipat.common import ONE, ZERO, Time

type Freq = Fraction
type Phase = Fraction
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


def mk_pspace(lspace: Array, freq: Freq, phase: Phase = ZERO) -> Array:
    assert freq > 0
    arr = lspace.copy()
    np.multiply(arr, np.float64(freq) * TAU, out=arr)
    if phase.numerator != 0:
        np.add(arr, np.float64(phase), out=arr)
    np.mod(arr, TAU, out=arr)
    return arr


def mk_sin(lspace: Array, freq: Freq, phase: Phase = ZERO) -> Array:
    arr = mk_pspace(lspace=lspace, freq=freq, phase=phase)
    np.sin(arr, out=arr)
    return arr


def render_arc(arc: Arc, rate: Rate) -> Array:
    return mk_lspace(start=arc.start, end=arc.end, rate=rate)


def plot(spc: Array, arr: Array) -> None:
    plt.plot(spc, arr)
    plt.show()


def test_plot() -> None:
    lspace = mk_lspace(ZERO, ONE, 1024)
    arr = mk_sin(lspace=lspace, freq=Fraction(4))
    plot(lspace, arr)
