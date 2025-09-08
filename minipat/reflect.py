"""Reflecting sequences of events back to patterns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd

from minipat.common import CycleDelta
from minipat.pat import Pat, PatStretch
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


def _lcm(a: int, b: int) -> int:
    """Compute least common multiple of two integers."""
    return abs(a * b) // gcd(a, b) if a and b else 0


def _collect_denominators[T](seq: PSeq[DeltaVal[T]]) -> list[int]:
    """Collect all denominators from a sequence of DeltaVals."""
    denoms = []
    for item in seq.iter():
        denoms.append(item.delta.denominator)
    return denoms


def quantize[T](ds: DeltaSeq[T]) -> StepSeq[T]:
    """Quantizes a sequence of events with fractional lengths into
    an equivalent sequence with integral lengths.
    """
    if ds.val.null():
        return StepVal(0, PSeq.empty())

    denoms = _collect_denominators(ds.val)
    if not denoms:
        return StepVal(0, PSeq.empty())

    common_denom = reduce(_lcm, denoms)

    quantized_items = []
    for item in ds.val.iter():
        steps = int(item.delta * common_denom)
        quantized_items.append(StepVal(steps, item.val))

    total_steps = sum(item.steps for item in quantized_items)
    return StepVal(total_steps, PSeq.mk(quantized_items))


def step_delta[T](ds: DeltaSeq[T], ss: StepSeq[T]) -> CycleDelta:
    """Returns the fractional length of a single step."""
    return CycleDelta(ds.delta * Fraction(1, ss.steps))


def unquantize[T](ss: StepSeq[T], total_delta: CycleDelta) -> DeltaSeq[T]:
    """Converts a sequence with integral step lengths back to fractional lengths.

    Args:
        ss: The step sequence to convert
        total_delta: The total fractional length to distribute across the steps

    Returns:
        A DeltaSeq with fractional lengths proportional to the step counts
    """
    if ss.val.null() or ss.steps == 0:
        return DeltaVal(CycleDelta(Fraction(0)), PSeq.empty())

    # Calculate the fractional length per step
    delta_per_step = Fraction(total_delta) / ss.steps

    # Convert each StepVal to a DeltaVal
    delta_items = []
    for item in ss.val.iter():
        item_delta = CycleDelta(delta_per_step * item.steps)
        delta_items.append(DeltaVal(item_delta, item.val))

    return DeltaVal(total_delta, PSeq.mk(delta_items))


def reflect[T](ss: StepSeq[T]) -> Pat[T]:
    """Assembles a compact representation of the quantized sequence
    as a pattern."""
    if ss.val.null():
        return Pat.silent()

    pats = []
    for item in ss.val.iter():
        if item.steps == 0:
            continue
        elif item.steps == 1:
            pats.append(Pat.pure(item.val))
        else:
            # For items taking multiple steps, we need to stretch them
            # We'll use Pat.stretch to make a pattern take up more space
            base_pat = Pat.pure(item.val)
            # Stretch by the number of steps
            stretched = Pat(PatStretch(base_pat, Fraction(item.steps)))
            pats.append(stretched)

    if len(pats) == 0:
        return Pat.silent()
    elif len(pats) == 1:
        return pats[0]
    else:
        return Pat.seq(pats)
