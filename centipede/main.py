import math

import numpy as np
import numpy.typing as npt
import plotext as plt

type Time = float
type Freq = float
type Phase = float
type Rate = int
type Space = npt.NDArray[npt.int32]
type Array = npt.NDArray[npt.float32]


TAU = 2 * math.pi


def main():
    print("Hello, world!")


def mk_space(start: Time, end: Time, rate: Rate) -> Space:
    assert start <= end
    assert rate > 0
    start_point = round(start * rate)
    end_point = round(end * rate)
    num = end_point - start_point
    return np.linspace(start_point, end_point, num=num)


def mk_sin(spc: Space, freq: Freq, phase: Phase) -> Array:
    assert freq > 0
    return np.sin(spc * TAU + phase)


def plot(spc: Space, arr: Array) -> None:
    plt.plot(spc, arr)
    plt.show()


if __name__ == "__main__":
    main()
