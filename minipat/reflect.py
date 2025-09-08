"""Reflecting sequences of events back to patterns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Callable, List, Optional, Sequence

from minipat.common import CycleDelta
from minipat.pat import (
    Pat,
    PatSeq,
    PatSpeed,
    PatStretch,
    SpeedOp,
)
from minipat.pat_dag import Find, PatDag, PatNode
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

type DagMinimizer[T] = Callable[[PatDag[T], Find], Optional[Find]]
"""Type alias for functions that minimize DAG patterns.

Returns the minimized Find handle if a change was made, None if no change."""


def minimize_seq_repetition_dag[T](dag: PatDag[T], find: Find) -> Optional[Find]:
    """Minimize sequences with repeated patterns using PatSpeed in a DAG.

    Converts [p p p] -> p*3
    Returns None if no repetition found.
    """

    pat_f = dag.get_node(find)
    match pat_f:
        case PatSeq(pats):
            items = list(pats.iter())
            if len(items) < 2:
                return None

            # Check if all patterns are identical (using root node ID after canonicalization)
            first_root = items[0].root_node_id()
            if all(p.root_node_id() == first_root for p in items):
                # All identical - use speed operator
                speed_node = dag.add_node(
                    PatSpeed(items[0], SpeedOp.Fast, Fraction(len(items)))
                )
                return speed_node

            # Check for longer repetitions
            n = len(items)
            for period in range(2, n // 2 + 1):
                if n % period == 0:
                    repetitions = n // period
                    base_pattern = items[:period]

                    # Check if pattern repeats (using root node IDs)
                    is_repeating = True
                    for i in range(repetitions):
                        for j in range(period):
                            current_root = items[i * period + j].root_node_id()
                            base_root = base_pattern[j].root_node_id()
                            if current_root != base_root:
                                is_repeating = False
                                break
                        if not is_repeating:
                            break

                    if is_repeating:
                        if period == 1:
                            base = base_pattern[0]
                        else:
                            base = dag.add_node(PatSeq(PSeq.mk(base_pattern)))
                        speed_node = dag.add_node(
                            PatSpeed(base, SpeedOp.Fast, Fraction(repetitions))
                        )
                        return speed_node
        case _:
            pass

    return None


def minimize_single_seq_dag[T](dag: PatDag[T], find: Find) -> Optional[Find]:
    """Remove unnecessary single-element sequences in DAG.

    Converts [p] -> p
    Returns None if sequence has multiple elements.
    """

    pat_f = dag.get_node(find)
    match pat_f:
        case PatSeq(pats):
            items = list(pats.iter())
            if len(items) == 1:
                return items[0]
        case _:
            pass

    return None


def run_dag_minimizers[T](
    dag: PatDag[T], minimizers: Sequence[DagMinimizer[T]], max_iterations: int = 10
) -> None:
    """Run DAG minimizers to saturation or until max_iterations reached.

    Processes nodes in bottom-up order to ensure children are minimized before parents.
    Modifies the DAG in-place by replacing nodes with minimized versions.
    """
    for _ in range(max_iterations):
        changed = False

        # Process nodes in bottom-up postorder
        node_ids = dag.postorder()

        for node_id in node_ids:
            if node_id not in dag.nodes:  # Node might have been removed
                continue

            find = dag.nodes[node_id].find

            # Apply all minimizers to this node
            for minimizer in minimizers:
                result = minimizer(dag, find)
                if result is not None:
                    # Replace the current node with the minimized result
                    minimized_content = dag.get_node(result)
                    # Update the PatNode with new content but keep the same Find handle
                    dag.nodes[node_id] = PatNode(find, minimized_content)
                    changed = True
                    break  # Only apply one minimizer per iteration per node

        # If no change, we've reached saturation
        if not changed:
            break


def minimize_pattern_dag[T](pat: Pat[T]) -> Pat[T]:
    """Apply all available minimizers using DAG representation.

    This version operates entirely on DAG nodes for maximum efficiency,
    only converting to Pat at the very end.
    """
    # Convert to DAG
    dag = PatDag.from_pat(pat)

    # Canonicalize to find and merge equivalent subpatterns
    # This makes Find equality work properly for minimizers
    dag.canonicalize()

    # Apply DAG-based minimization rules
    # These now use root node IDs for equality checks
    dag_minimizers = [
        minimize_single_seq_dag,
        minimize_seq_repetition_dag,
    ]

    run_dag_minimizers(dag, dag_minimizers)

    # Run canonicalization again after minimization to catch new equivalences
    dag.canonicalize()

    # Garbage collect unused nodes
    dag.collect()

    # Convert back to Pat only at the very end
    return dag.to_pat()


def minimize_pattern[T](pat: Pat[T]) -> Pat[T]:
    """Apply all available minimizers to a pattern until saturation.

    Uses DAG representation for efficient pattern minimization.
    """
    return minimize_pattern_dag(pat)


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
    Uses DAG representation for efficient equality checking during minimization.
    """
    base_pattern = reflect(ss)
    return minimize_pattern_dag(base_pattern)
