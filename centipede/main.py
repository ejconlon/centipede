import numpy as np
import numpy.typing as npt
import plotext as plt

type Time = float
type Freq = float
type Phase = float
type Rate = int
type Space = npt.NDArray[np.int64]
type Array = npt.NDArray[np.float64]


TAU = np.float64(2 * np.pi)


def main():
    print("Hello, world!")


def mk_space(start: Time, end: Time, rate: Rate) -> Space:
    assert start <= end
    assert rate > 0
    start_point = round(start * rate)
    end_point = max(start_point, round(end * rate) - 1)
    num = end_point - start_point + 1
    return np.linspace(start=start_point, stop=end_point, num=num, dtype=np.int64)


def mk_sin(spc: Space, rate: Rate, freq: Freq, phase: Phase) -> Array:
    assert freq > 0
    return np.sin(np.mod(spc.astype(np.float64) * (freq * TAU / rate) + phase, TAU))


def plot(spc: Space, arr: Array) -> None:
    plt.plot(spc, arr)
    plt.show()


if __name__ == "__main__":
    main()
