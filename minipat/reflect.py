"""Reflecting sequences of events back to patterns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Callable, List, Optional, Sequence

from minipat.arc import Arc
from minipat.common import CycleDelta, CycleTime
from minipat.pat import (
    Pat,
    PatRepeat,
    PatSeq,
    PatStretch,
)
from minipat.pat_dag import PatDag, PatFind
from minipat.stream import Stream
from spiny.seq import PSeq


@dataclass(frozen=True)
class DeltaVal[T]:
    """A value annotated with fractional offset and duration"""

    offset: CycleDelta
    duration: CycleDelta
    val: T


@dataclass(frozen=True)
class StepVal[T]:
    """A value annotated with integral offset and duration"""

    offset: int
    duration: int
    val: T


type DeltaSeq[T] = PSeq[DeltaVal[T]]
"""A sequence of events with fractional offsets and durations.

Invariants:
- Root offset is 0 (first event starts at beginning)
- Root duration is >= max(offset + duration) across all events
- Offsets must be non-decreasing (sorted by start time)
"""


type StepSeq[T] = PSeq[StepVal[T]]
"""A sequence of events with integral offsets and durations.

Invariants:
- Root offset is 0 (first event starts at beginning)
- Root duration is >= max(offset + duration) across all events  
- Offsets must be non-decreasing (sorted by start time)
"""


def _lcm(a: int, b: int) -> int:
    """Compute least common multiple of two integers."""
    return abs(a * b) // gcd(a, b) if a and b else 0


def _collect_denominators[T](seq: DeltaSeq[T]) -> List[int]:
    """Collect all denominators from a sequence of DeltaVals."""
    denoms = []
    for item in seq.iter():
        denoms.append(item.offset.denominator)
        denoms.append(item.duration.denominator)
    return denoms


# =============================================================================
# Pattern Minimization Functions
# =============================================================================

type DagMinimizer[T] = Callable[[PatDag[T], PatFind], Optional[PatFind]]
"""Type alias for functions that minimize DAG patterns.

Returns the minimized Find handle if a change was made, None if no change."""


def minimize_seq_repetition_dag[T](dag: PatDag[T], find: PatFind) -> Optional[PatFind]:
    """Minimize sequences with repeated patterns using PatSpeed in a DAG.

    Converts [p p p] -> p*3
    Returns None if no repetition found.
    """

    patf = dag.get_node(find)
    match patf:
        case PatSeq(pats):
            items = list(pats.iter())
            if len(items) < 2:
                return None

            # Check if all patterns are identical (using root node ID after canonicalization)
            first_root = items[0].root_node_id()
            if all(p.root_node_id() == first_root for p in items):
                # All identical - use repeat operator
                repeat_node = dag.add_node(PatRepeat(items[0], Fraction(len(items))))
                return repeat_node

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
                        repeat_node = dag.add_node(
                            PatRepeat(base, Fraction(repetitions))
                        )
                        return repeat_node
        case _:
            pass

    return None


def minimize_single_seq_dag[T](dag: PatDag[T], find: PatFind) -> Optional[PatFind]:
    """Remove unnecessary single-element sequences in DAG.

    Converts [p] -> p
    Returns None if sequence has multiple elements.
    """

    patf = dag.get_node(find)
    match patf:
        case PatSeq(pats):
            items: List[PatFind] = list(pats.iter())
            if len(items) == 1:
                return items[0]
        case _:
            pass

    return None


def minimize_seq_gcd_dag[T](dag: PatDag[T], find: PatFind) -> Optional[PatFind]:
    """Minimize sequences by factoring out GCD of child durations.

    Converts [a@4 b@4] -> [a b]@4
    Converts [a@2 b@4] -> [a b@2]@2
    Returns None if no GCD can be factored out.
    """
    from functools import reduce
    from math import gcd

    patf = dag.get_node(find)
    match patf:
        case PatSeq(pats):
            items: List[PatFind] = list(pats.iter())
            if len(items) < 2:
                return None

            # Collect stretch factors from child patterns
            stretch_factors = []
            for child in items:
                child_patf = dag.get_node(child)
                match child_patf:
                    case PatStretch(_, count):
                        # Convert Fraction to int if it's a whole number
                        if count.denominator == 1:
                            stretch_factors.append(count.numerator)
                        else:
                            # Can't handle fractional stretches for GCD
                            return None
                    case _:
                        # Non-stretched patterns have implicit stretch of 1
                        stretch_factors.append(1)

            if not stretch_factors or len(stretch_factors) != len(items):
                return None

            # Find GCD of all stretch factors
            common_gcd = reduce(gcd, stretch_factors)

            # Only proceed if GCD > 1 (there's something to factor out)
            if common_gcd <= 1:
                return None

            # Create new children with reduced stretch factors
            new_children = []
            for child, factor in zip(items, stretch_factors):
                new_factor = factor // common_gcd
                child_patf = dag.get_node(child)

                match child_patf:
                    case PatStretch(inner, _):
                        if new_factor == 1:
                            # Stretch factor becomes 1, just use the inner pattern
                            new_children.append(inner)
                        else:
                            # Create new stretch with reduced factor
                            new_stretch = dag.add_node(
                                PatStretch(inner, Fraction(new_factor))
                            )
                            new_children.append(new_stretch)
                    case _:
                        # Was implicitly stretched by 1, now need explicit stretch
                        if new_factor == 1:
                            # Still stretch of 1, no change needed
                            new_children.append(child)
                        else:
                            # This shouldn't happen given our logic above, but handle it
                            new_stretch = dag.add_node(
                                PatStretch(child, Fraction(new_factor))
                            )
                            new_children.append(new_stretch)

            # Create the new sequence
            new_seq = dag.add_node(PatSeq(PSeq.mk(new_children)))

            # Apply the common GCD as a stretch to the whole sequence
            result = dag.add_node(PatStretch(new_seq, Fraction(common_gcd)))
            return result

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
        finds = dag.postorder()

        for find in finds:
            # Apply all minimizers to this node
            for minimizer in minimizers:
                result = minimizer(dag, find)
                if result is not None:
                    # Replace the current node with the minimized result
                    minimized_content = dag.get_node(result)
                    # Update the node's content through its Find handle
                    dag.update_node(find, minimized_content)
                    changed = True
                    break  # Only apply one minimizer per iteration per node

        # If no change, we've reached saturation
        if not changed:
            break


def minimize_pattern[T](pat: Pat[T]) -> Pat[T]:
    """Apply all available minimizers to a pattern until saturation.

    Uses DAG representation for efficient pattern minimization.
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
        minimize_seq_gcd_dag,
    ]

    run_dag_minimizers(dag, dag_minimizers)

    # Run canonicalization again after minimization to catch new equivalences
    dag.canonicalize()

    # Convert back to Pat only at the very end
    return dag.to_pat()


def quantize[T](ds: DeltaSeq[T]) -> StepSeq[T]:
    """Quantizes a sequence of events with fractional offsets/durations into
    an equivalent sequence with integral offsets/durations.
    """
    if ds.null():
        return PSeq.empty()

    denoms = _collect_denominators(ds)
    if not denoms:
        return PSeq.empty()

    common_denom = reduce(_lcm, denoms)

    quantized_items = []
    for item in ds.iter():
        offset_steps = int(item.offset * common_denom)
        duration_steps = int(item.duration * common_denom)
        quantized_items.append(StepVal(offset_steps, duration_steps, item.val))

    return PSeq.mk(quantized_items)


def get_min_total_duration[T](ds: DeltaSeq[T]) -> CycleDelta:
    """Returns the minimum total duration of a DeltaSeq (max offset + duration)."""
    if ds.null():
        return CycleDelta(Fraction(0))

    max_end = CycleDelta(Fraction(0))
    for item in ds.iter():
        end_time = CycleDelta(item.offset + item.duration)
        if end_time > max_end:
            max_end = end_time

    return max_end


def unquantize[T](ss: StepSeq[T], step_duration: CycleDelta) -> DeltaSeq[T]:
    """Converts a sequence with integral offsets/durations back to fractional lengths.

    Args:
        ss: The step sequence to convert
        step_duration: The fractional duration of a single step

    Returns:
        A DeltaSeq with fractional offsets/durations scaled by step_duration
    """
    if ss.null():
        return PSeq.empty()

    # Convert each StepVal to a DeltaVal by scaling with step_duration
    delta_items = []
    for item in ss.iter():
        fractional_offset = CycleDelta(step_duration * item.offset)
        fractional_duration = CycleDelta(step_duration * item.duration)
        delta_items.append(DeltaVal(fractional_offset, fractional_duration, item.val))

    return PSeq.mk(delta_items)


def reflect[T](ss: StepSeq[T]) -> Pat[T]:
    """Assembles a compact representation of the quantized sequence
    as a pattern."""
    if ss.null():
        return Pat.silent()

    # Sort events by offset to ensure proper temporal order
    sorted_items = sorted(ss.iter(), key=lambda item: item.offset)

    pats = []
    current_offset = 0

    for item in sorted_items:
        # Add silence for any gap before this event
        if item.offset > current_offset:
            gap_duration = item.offset - current_offset
            if gap_duration > 0:
                # For now, we'll skip gaps - this needs to be handled properly
                # when we support silences
                pass

        # Create pattern for this event
        if item.duration == 0:
            continue
        elif item.duration == 1:
            pats.append(Pat.pure(item.val))
        else:
            # For items taking multiple steps, we need to stretch them
            base_pat = Pat.pure(item.val)
            # Stretch by the number of steps
            stretched = Pat(PatStretch(base_pat, Fraction(item.duration)))
            pats.append(stretched)

        current_offset = item.offset + item.duration

    if len(pats) == 0:
        return Pat.silent()
    elif len(pats) == 1:
        return pats[0]
    else:
        return Pat.seq(pats)


def pat_to_deltaseq[T](
    pat: Pat[T], total_delta: CycleDelta, start_time: Optional[CycleTime] = None
) -> DeltaSeq[T]:
    """Convert a Pat back to a DeltaSeq by evaluating it over the given time arc.

    This streams the pattern from start_time to start_time + total_delta and collects the events.
    Used for semantic equivalence testing of minimized patterns.

    The pattern must produce a monophonic stream (no overlapping events).
    If overlaps are detected, a ValueError will be raised.

    Args:
        pat: The pattern to evaluate (must be monophonic)
        total_delta: The total time duration to evaluate over
        start_time: The start time of the evaluation (defaults to None, which means cycle 0)

    Returns:
        A DeltaSeq representing the events produced by the pattern

    Raises:
        ValueError: If the pattern produces overlapping events
    """
    if total_delta == 0:
        return PSeq.empty()

    # Convert pattern to stream
    stream = Stream.pat(pat)

    # Use cycle 0 if start_time is None
    if start_time is None:
        start_time = CycleTime(Fraction(0))

    # Create arc from start_time to start_time + total_delta
    end_time = CycleTime(start_time + total_delta)
    arc = Arc(start_time, end_time)

    # Get events by streaming over the arc
    events = stream.unstream(arc)

    # Convert events back to DeltaSeq format, sorting by start time
    event_list = []
    for _, ev in events:
        event_list.append(ev)

    # Sort events by start time to maintain temporal order
    sorted_events = sorted(event_list, key=lambda ev: ev.span.active.start)

    if not sorted_events:
        return PSeq.empty()

    # Process events to ensure monophonic stream (no overlaps)
    delta_items = []
    current_time = start_time

    for i, ev in enumerate(sorted_events):
        event_start = ev.span.active.start
        event_end = ev.span.active.end

        # Check for overlaps
        if event_start < current_time:
            overlap_amount = CycleDelta(current_time - event_start)
            raise ValueError(
                f"Overlap detected in monophonic stream at time {event_start}: overlap of {overlap_amount}"
            )

        # Calculate offset from start_time and duration
        offset = CycleDelta(event_start - start_time)
        duration = CycleDelta(event_end - event_start)
        delta_items.append(DeltaVal(offset, duration, ev.val))
        current_time = event_end

    return PSeq.mk(delta_items)


def reflect_minimal[T](ss: StepSeq[T]) -> Pat[T]:
    """Reflect a StepSeq to a minimized Pat.

    First reflects normally, then applies all available minimizers until saturation.
    Uses DAG representation for efficient equality checking during minimization.
    """
    base_pattern = reflect(ss)
    return minimize_pattern(base_pattern)
