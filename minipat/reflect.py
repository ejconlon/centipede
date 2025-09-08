"""Reflecting sequences of events back to patterns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Callable, List, Optional, Sequence

from minipat.common import CycleDelta
from minipat.pat import Pat, PatSeq, PatSpeed, PatStretch, SpeedOp
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


def _collect_denominators[T](seq: PSeq[DeltaVal[T]]) -> List[int]:
    """Collect all denominators from a sequence of DeltaVals."""
    denoms = []
    for item in seq.iter():
        denoms.append(item.delta.denominator)
    return denoms


# =============================================================================
# Pattern Minimization Functions
# =============================================================================

type PatMinimizer[T] = Callable[[Pat[T]], Optional[Pat[T]]]
"""Type alias for functions that minimize patterns.

Returns the minimized pattern if a change was made, None if no change."""


def minimize_seq_repetition[T](pat: Pat[T]) -> Optional[Pat[T]]:
    """Minimize sequences with repeated patterns using PatSpeed.

    Converts [p p p] -> p*3
    Returns None if no repetition found.
    """
    match pat.unwrap:
        case PatSeq(pats):
            items = list(pats.iter())
            if len(items) < 2:
                return None

            # Check if all patterns are identical
            first = items[0]
            if all(p == first for p in items):
                # All identical - use speed operator
                return Pat(PatSpeed(first, SpeedOp.Fast, Fraction(len(items))))

            # Check for longer repetitions
            n = len(items)
            for period in range(2, n // 2 + 1):
                if n % period == 0:
                    repetitions = n // period
                    base_pattern = items[:period]

                    # Check if pattern repeats
                    is_repeating = True
                    for i in range(repetitions):
                        for j in range(period):
                            if items[i * period + j] != base_pattern[j]:
                                is_repeating = False
                                break
                        if not is_repeating:
                            break

                    if is_repeating:
                        if period == 1:
                            base = base_pattern[0]
                        else:
                            base = Pat.seq(base_pattern)
                        return Pat(PatSpeed(base, SpeedOp.Fast, Fraction(repetitions)))
        case _:
            pass

    return None


def minimize_single_seq[T](pat: Pat[T]) -> Optional[Pat[T]]:
    """Remove unnecessary single-element sequences.

    Converts [p] -> p
    Returns None if sequence has multiple elements.
    """
    match pat.unwrap:
        case PatSeq(pats):
            items = list(pats.iter())
            if len(items) == 1:
                return items[0]
        case _:
            pass

    return None


def run_minimizers[T](
    pat: Pat[T], minimizers: Sequence[PatMinimizer[T]], max_iterations: int = 10
) -> Pat[T]:
    """Run minimizers to saturation or until max_iterations reached."""
    current = pat

    for _ in range(max_iterations):
        changed = False

        # Apply all minimizers
        for minimizer in minimizers:
            result = minimizer(current)
            if result is not None:
                current = result
                changed = True

        # If no change, we've reached saturation
        if not changed:
            break

    return current


def minimize_pattern[T](pat: Pat[T]) -> Pat[T]:
    """Apply all available minimizers to a pattern until saturation."""
    minimizers = [
        minimize_single_seq,
        minimize_seq_repetition,
    ]

    return run_minimizers(pat, minimizers)


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


def reflect_minimal[T](ss: StepSeq[T]) -> Pat[T]:
    """Reflect a StepSeq to a minimized Pat.

    First reflects normally, then applies all available minimizers until saturation.
    """
    base_pattern = reflect(ss)
    return minimize_pattern(base_pattern)
