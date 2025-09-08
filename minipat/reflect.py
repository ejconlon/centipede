"""Reflecting sequences of events back to patterns."""

from dataclasses import dataclass
from fractions import Fraction

from minipat.common import CycleDelta
from minipat.pat import Pat
from spiny.seq import PSeq


@dataclass(frozen=True)
class DeltaVal[T]:
    """A value annotated with a fractional length"""

    delta: CycleDelta
    val: T


@dataclass(frozen=True)
class StepVal[T]:
    """A value annotated with an integral length"""

    steps: int
    val: T


type DeltaSeq[T] = DeltaVal[PSeq[DeltaVal[T]]]
"""A sequence of events annotated with total fractional length.
Invariant: root delta is the sum of all child deltas.
"""


type StepSeq[T] = StepVal[PSeq[StepVal[T]]]
"""A sequence of events annotated with total fractional length
Invariant: root steps is the sum of all child steps.
"""


def quantize[T](ds: DeltaSeq[T]) -> StepSeq[T]:
    """Quantizes a sequence of events with fractional lengths into
    an equivalent sequence with integral lengths.
    """
    raise Exception("TODO")


def step_delta[T](ds: DeltaSeq[T], ss: StepSeq[T]) -> CycleDelta:
    """Returns the fractional length of a single step."""
    return CycleDelta(ds.delta * Fraction(1, ss.steps))


def reflect[T](ss: StepSeq[T]) -> Pat[T]:
    """Assembles a compact representation of the quantized sequence
    as a pattern."""
    raise Exception("TODO")
