"""Stream implementation for converting patterns to timed events."""

from __future__ import annotations

import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
from typing import Callable, List, Optional, override

from minipat.arc import Arc, Span
from minipat.common import CycleDelta, CycleTime
from minipat.ev import Ev, ev_heap_empty, ev_heap_push, ev_heap_singleton
from minipat.pat import (
    Pat,
    PatAlt,
    PatEuc,
    PatPar,
    PatPoly,
    PatProb,
    PatPure,
    PatRand,
    PatRepeat,
    PatSeq,
    PatSilent,
    PatSpeed,
    PatStretch,
    SpeedOp,
)
from spiny import PSeq
from spiny.heapmap import PHeapMap


class MergeStrat(Enum):
    """Merge strategy for combining stream events."""

    Inner = auto()
    Outer = auto()
    Mixed = auto()


def _create_span(original_arc: Arc, query_arc: Arc) -> Optional[Span]:
    """Create a span for an event within a query arc.

    Args:
        original_arc: The original arc of the event
        query_arc: The query arc to intersect with

    Returns:
        A span with active arc as intersection, whole arc as original if different,
        or None if intersection is null
    """
    active = original_arc.intersect(query_arc)
    if active.null():
        return None

    # Only set whole if original extends beyond the active area
    if original_arc == active:
        whole = None
    else:
        whole = original_arc

    return Span(active=active, whole=whole)


# sealed
class Stream[T](metaclass=ABCMeta):
    """A stream of events in time."""

    @abstractmethod
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        """Emit all events that start or end in the given arc.

        Args:
            arc: The time arc to query for events

        Returns:
            A heap map of events within the arc
        """
        raise NotImplementedError

    @staticmethod
    def silent() -> Stream[T]:
        """Create a silent stream.

        Returns:
            A stream representing silence
        """
        return SilenceStream()

    @staticmethod
    def pure(val: T) -> Stream[T]:
        """Create a stream with a single value.

        Args:
            val: The value to wrap

        Returns:
            A stream containing the single value
        """
        return PureStream(val)

    @staticmethod
    def seq(streams: PSeq[Stream[T]]) -> Stream[T]:
        """Create a sequential stream.

        Args:
            streams: The streams to sequence

        Returns:
            A stream that plays the given streams in sequence
        """
        return SeqStream(streams)

    @staticmethod
    def par(streams: PSeq[Stream[T]]) -> Stream[T]:
        """Create a parallel stream.

        Args:
            streams: The streams to play in parallel

        Returns:
            A stream that plays the given streams simultaneously
        """
        return ParStream(streams)

    @staticmethod
    def rand(choices: PSeq[Stream[T]]) -> Stream[T]:
        """Create a choice stream.

        Args:
            choices: The streams to choose from

        Returns:
            A stream that cycles through the given choices
        """
        return ChoiceStream(choices)

    @staticmethod
    def euc(stream: Stream[T], hits: int, steps: int, rotation: int = 0) -> Stream[T]:
        """Create a Euclidean rhythm stream.

        Args:
            stream: The stream to distribute
            hits: Number of hits to distribute
            steps: Total number of steps
            rotation: Optional rotation offset

        Returns:
            A stream with Euclidean rhythm distribution
        """
        return EuclideanStream.create(stream, hits, steps, rotation)

    @staticmethod
    def poly(patterns: PSeq[Stream[T]], subdiv: Optional[int] = None) -> Stream[T]:
        """Create a polymetric stream.

        Args:
            patterns: The streams to play polymetrically
            subdiv: Optional subdivision factor

        Returns:
            A polymetric stream with or without subdivision
        """
        return PolymetricStream(patterns, subdiv)

    @staticmethod
    def speed(stream: Stream[T], op: SpeedOp, factor: Fraction) -> Stream[T]:
        """Create a speed stream.

        Args:
            stream: The stream to speed up/down
            op: The speed operator (Fast or Slow)
            factor: The speed factor

        Returns:
            A stream with the specified speed
        """
        return RepetitionStream(stream, op, factor)

    @staticmethod
    def stretch(stream: Stream[T], count: int) -> Stream[T]:
        """Create a stretched stream.

        Args:
            stream: The stream to stretch
            count: The stretch count

        Returns:
            A stream stretched by the given count
        """
        return ElongationStream(stream, count)

    @staticmethod
    def prob(stream: Stream[T], chance: Fraction) -> Stream[T]:
        """Create a probabilistic stream.

        Args:
            stream: The stream to apply probability to
            chance: The probability (0 to 1 as a Fraction)

        Returns:
            A stream that plays with the given probability
        """
        return ProbabilityStream(stream, chance)

    @staticmethod
    def alt(patterns: PSeq[Stream[T]]) -> Stream[T]:
        """Create an alternating stream.

        Args:
            patterns: The streams to alternate between

        Returns:
            A stream that alternates between the given streams
        """
        return AlternatingStream(patterns)

    @staticmethod
    def repeat(stream: Stream[T], count: int) -> Stream[T]:
        """Create a repeat stream.

        Args:
            stream: The stream to repeat
            count: The number of times to repeat

        Returns:
            A repeated stream
        """
        return ReplicateStream(stream, count)

    @staticmethod
    def pat(pattern: Pat[T]) -> Stream[T]:
        """Create a stream from a pattern.

        Args:
            pattern: The pattern to convert to a stream

        Returns:
            A specialized stream for the pattern
        """
        return pat_stream(pattern)

    def map[U](self, func: Callable[[T], U]) -> Stream[U]:
        """Map a function over the stream values.

        Args:
            func: The function to apply to each value

        Returns:
            A new stream with transformed values
        """
        return MapStream(self, func)

    def filter(self, predicate: Callable[[T], bool]) -> Stream[T]:
        """Filter events in a stream based on a predicate.

        Args:
            predicate: Function to test each event value

        Returns:
            A new stream with only events passing the predicate
        """
        return FilterStream(self, predicate)

    def bind[B](
        self, merge_strat: MergeStrat, func: Callable[[T], Stream[B]]
    ) -> Stream[B]:
        """Bind a stream with a merge strategy.

        Args:
            merge_strat: Strategy for merging overlapping events
            func: Function to transform each event value into a new stream

        Returns:
            A new stream with transformed and merged events
        """
        return BindStream(self, merge_strat, func)

    def apply[B, C](
        self, merge_strat: MergeStrat, func: Callable[[T, B], C], other: Stream[B]
    ) -> Stream[C]:
        """Apply a function across two streams.

        Args:
            merge_strat: Strategy for merging overlapping events
            func: Function to combine values from both streams
            other: The other stream to combine with

        Returns:
            A new stream with combined events
        """
        return ApplyStream(self, merge_strat, func, other)

    def fast_by(self, factor: Fraction) -> Stream[T]:
        """Speed up stream events by a given factor.

        Args:
            factor: Factor to speed up by (> 1 makes it faster)

        Returns:
            A new stream with events sped up
        """
        return FastStream(self, factor)

    def slow_by(self, factor: Fraction) -> Stream[T]:
        """Slow down stream events by a given factor.

        Args:
            factor: Factor to slow down by (> 1 makes it slower)

        Returns:
            A new stream with events slowed down
        """
        return SlowStream(self, factor)

    def early_by(self, delta: CycleDelta) -> Stream[T]:
        """Shift stream events earlier in time.

        Args:
            delta: Amount to shift earlier

        Returns:
            A new stream with events shifted earlier
        """
        return EarlyStream(self, delta)

    def late_by(self, delta: CycleDelta) -> Stream[T]:
        """Shift stream events later in time.

        Args:
            delta: Amount to shift later

        Returns:
            A new stream with events shifted later
        """
        return LateStream(self, delta)


@dataclass(frozen=True)
class SilenceStream[T](Stream[T]):
    """Specialized stream for silence patterns."""

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        return ev_heap_empty()


@dataclass(frozen=True)
class PureStream[T](Stream[T]):
    """Specialized stream for pure value patterns."""

    value: T

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null():
            return ev_heap_empty()
        span = Span(active=arc, whole=None)
        return ev_heap_singleton(Ev(span, self.value))


@dataclass(frozen=True)
class SeqStream[T](Stream[T]):
    """Specialized stream for sequential patterns."""

    children: PSeq[Stream[T]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.children) == 0:
            return ev_heap_empty()

        seq_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()
        child_duration = arc.length() / len(self.children)

        for i, child in enumerate(self.children):
            child_start = arc.start + i * child_duration
            child_arc = Arc(
                CycleTime(child_start), CycleTime(child_start + child_duration)
            )

            intersection = child_arc.intersect(arc)
            if not intersection.null():
                child_events = child.unstream(child_arc)
                for _, ev in child_events:
                    # Create proper span for this event within the query arc
                    span = _create_span(ev.span.active, arc)
                    if span is not None:
                        # Preserve the child's whole information if it exists, otherwise use the child's active as whole
                        if ev.span.whole is not None:
                            span = Span(active=span.active, whole=ev.span.whole)
                        elif ev.span.active != span.active:
                            span = Span(active=span.active, whole=ev.span.active)
                        new_ev = Ev(span, ev.val)
                        seq_result = ev_heap_push(new_ev, seq_result)

        return seq_result


@dataclass(frozen=True)
class ParStream[T](Stream[T]):
    """Specialized stream for parallel patterns."""

    children: PSeq[Stream[T]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null():
            return ev_heap_empty()

        par_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()
        for child in self.children:
            child_events = child.unstream(arc)
            for _, ev in child_events:
                par_result = ev_heap_push(ev, par_result)
        return par_result


@dataclass(frozen=True)
class ChoiceStream[T](Stream[T]):
    """Specialized stream for choice patterns."""

    choices: PSeq[Stream[T]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.choices) == 0:
            return ev_heap_empty()

        cycle_index = int(arc.start) % len(self.choices)
        chosen_stream = self.choices[cycle_index]
        return chosen_stream.unstream(arc)


@dataclass(frozen=True)
class EuclideanStream[T](Stream[T]):
    """Specialized stream for Euclidean patterns."""

    atom: Stream[T]
    hits: int
    steps: int
    rotation: int
    pattern: List[bool]  # Pre-computed Euclidean pattern

    @classmethod
    def create(
        cls, atom: Stream[T], hits: int, steps: int, rotation: int
    ) -> EuclideanStream[T]:
        pattern = _generate_euclidean(hits, steps, rotation)
        return cls(atom, hits, steps, rotation, pattern)

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.steps <= 0 or self.hits <= 0:
            return ev_heap_empty()

        euc_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()
        step_duration = arc.length() / self.steps

        for i, is_hit in enumerate(self.pattern):
            if is_hit:
                step_start = arc.start + i * step_duration
                step_arc = Arc(
                    CycleTime(step_start), CycleTime(step_start + step_duration)
                )

                if not arc.intersect(step_arc).null():
                    atom_events = self.atom.unstream(step_arc)
                    for _, ev in atom_events:
                        # Create span for this step within the query arc
                        span = _create_span(step_arc, arc)
                        if span is not None:
                            step_ev = Ev(span, ev.val)
                            euc_result = ev_heap_push(step_ev, euc_result)

        return euc_result


@dataclass(frozen=True)
class PolymetricStream[T](Stream[T]):
    """Specialized stream for polymetric patterns."""

    patterns: PSeq[Stream[T]]
    subdiv: Optional[int]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.patterns) == 0:
            return ev_heap_empty()

        polymetric_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        if self.subdiv is None:
            # All patterns play simultaneously
            for pattern in self.patterns:
                pattern_events = pattern.unstream(arc)
                for _, ev in pattern_events:
                    polymetric_result = ev_heap_push(ev, polymetric_result)
        else:
            # With subdivision
            if self.subdiv <= 0:
                return ev_heap_empty()
            sub_arc = arc.scale(Fraction(1, self.subdiv))
            for pattern in self.patterns:
                pattern_events = pattern.unstream(sub_arc)
                for _, ev in pattern_events:
                    scaled_ev = ev.scale(Fraction(self.subdiv))
                    span = _create_span(scaled_ev.span.active, arc)
                    if span is not None:
                        # Preserve the whole information from scaling
                        if scaled_ev.span.whole is not None:
                            span = Span(active=span.active, whole=scaled_ev.span.whole)
                        elif scaled_ev.span.active != span.active:
                            span = Span(active=span.active, whole=scaled_ev.span.active)
                        new_ev = Ev(span, scaled_ev.val)
                        polymetric_result = ev_heap_push(new_ev, polymetric_result)

        return polymetric_result


@dataclass(frozen=True)
class RepetitionStream[T](Stream[T]):
    """Specialized stream for repetition patterns."""

    pattern: Stream[T]
    operator: SpeedOp
    count: Fraction

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.count <= 0:
            return ev_heap_empty()

        rep_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        match self.operator:
            case SpeedOp.Fast:
                # Handle fractional repetitions: x*2.5 = 2 full + 0.5 partial repetition
                if self.count > 0:
                    # Get integer and fractional parts
                    int_part = int(self.count)  # Number of full repetitions
                    frac_part = (
                        self.count - int_part
                    )  # Fractional part for partial repetition

                    rep_duration = (
                        arc.length() / self.count
                    )  # Duration of each repetition

                    # Unstream once for a single repetition
                    full_rep_arc = Arc(arc.start, CycleTime(arc.start + rep_duration))
                    full_pattern_events = self.pattern.unstream(full_rep_arc)

                    # Add full repetitions
                    for i in range(int_part):
                        rep_start = arc.start + i * rep_duration
                        for span, ev in full_pattern_events:
                            # Shift the event to the correct repetition position
                            offset = rep_start - arc.start
                            shifted_span = Span(
                                active=Arc(
                                    CycleTime(span.active.start + offset),
                                    CycleTime(span.active.end + offset),
                                ),
                                whole=span.whole,
                            )

                            clipped_span = _create_span(shifted_span.active, arc)
                            if clipped_span is not None:
                                if shifted_span.whole is not None:
                                    clipped_span = Span(
                                        active=clipped_span.active,
                                        whole=shifted_span.whole,
                                    )
                                elif shifted_span.active != clipped_span.active:
                                    clipped_span = Span(
                                        active=clipped_span.active,
                                        whole=shifted_span.active,
                                    )
                                new_ev = Ev(clipped_span, ev.val)
                                rep_result = ev_heap_push(new_ev, rep_result)

                    # Add partial repetition if needed
                    if frac_part > 0:
                        partial_start = arc.start + int_part * rep_duration
                        partial_end = partial_start + frac_part * rep_duration
                        partial_arc = Arc(
                            CycleTime(partial_start), CycleTime(partial_end)
                        )

                        # Unstream the partial repetition
                        partial_pattern_events = self.pattern.unstream(partial_arc)
                        for _, ev in partial_pattern_events:
                            clipped_span = _create_span(ev.span.active, arc)
                            if clipped_span is not None:
                                if ev.span.whole is not None:
                                    span = Span(
                                        active=clipped_span.active, whole=ev.span.whole
                                    )
                                elif ev.span.active != clipped_span.active:
                                    span = Span(
                                        active=clipped_span.active, whole=ev.span.active
                                    )
                                else:
                                    span = clipped_span
                                new_ev = Ev(span, ev.val)
                                rep_result = ev_heap_push(new_ev, rep_result)

            case SpeedOp.Slow:
                # Slow by factor N is just fast by 1/N
                fast_stream = RepetitionStream(
                    self.pattern, SpeedOp.Fast, Fraction(1) / self.count
                )
                return fast_stream.unstream(arc)

        return rep_result


@dataclass(frozen=True)
class ElongationStream[T](Stream[T]):
    """Specialized stream for elongation patterns."""

    pattern: Stream[T]
    count: int

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.count <= 0:
            return ev_heap_empty()

        stretched_arc = arc.scale(Fraction(self.count))
        pattern_events = self.pattern.unstream(stretched_arc)
        elong_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        for _, ev in pattern_events:
            elongated_ev = ev.scale(Fraction(1, self.count))
            span = _create_span(elongated_ev.span.active, arc)
            if span is not None:
                # Preserve the whole information from scaling
                if elongated_ev.span.whole is not None:
                    span = Span(active=span.active, whole=elongated_ev.span.whole)
                elif elongated_ev.span.active != span.active:
                    span = Span(active=span.active, whole=elongated_ev.span.active)
                new_ev = Ev(span, elongated_ev.val)
                elong_result = ev_heap_push(new_ev, elong_result)

        return elong_result


@dataclass(frozen=True)
class ProbabilityStream[T](Stream[T]):
    """Specialized stream for probability patterns."""

    pattern: Stream[T]
    chance: Fraction

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null():
            return ev_heap_empty()

        random.seed(hash(arc.start))
        if random.random() < self.chance:
            return self.pattern.unstream(arc)
        return ev_heap_empty()


@dataclass(frozen=True)
class AlternatingStream[T](Stream[T]):
    """Specialized stream for alternating patterns."""

    patterns: PSeq[Stream[T]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.patterns) == 0:
            return ev_heap_empty()

        cycle_index = int(arc.start) % len(self.patterns)
        chosen_stream = self.patterns[cycle_index]
        return chosen_stream.unstream(arc)


@dataclass(frozen=True)
class ReplicateStream[T](Stream[T]):
    """Specialized stream for replicate patterns."""

    pattern: Stream[T]
    count: int

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.count <= 0:
            return ev_heap_empty()

        replicate_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()
        rep_duration = arc.length() / self.count

        for i in range(self.count):
            rep_start = arc.start + i * rep_duration
            rep_arc = Arc(CycleTime(rep_start), CycleTime(rep_start + rep_duration))
            if not arc.intersect(rep_arc).null():
                child_events = self.pattern.unstream(rep_arc)
                for _, ev in child_events:
                    span = _create_span(ev.span.active, arc)
                    if span is not None:
                        # Preserve the child's whole information if it exists
                        if ev.span.whole is not None:
                            span = Span(active=span.active, whole=ev.span.whole)
                        elif ev.span.active != span.active:
                            span = Span(active=span.active, whole=ev.span.active)
                        new_ev = Ev(span, ev.val)
                        replicate_result = ev_heap_push(new_ev, replicate_result)
        return replicate_result


@dataclass(frozen=True)
class FilterStream[T](Stream[T]):
    """Stream that filters events based on a predicate."""

    source: Stream[T]
    predicate: Callable[[T], bool]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        source_events = self.source.unstream(arc)
        result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        for _, ev in source_events:
            if self.predicate(ev.val):
                result = ev_heap_push(ev, result)

        return result


@dataclass(frozen=True)
class BindStream[A, B](Stream[B]):
    """Stream that binds another stream with a transformation function."""

    source: Stream[A]
    merge_strat: MergeStrat
    func: Callable[[A], Stream[B]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[B]]:
        source_events = self.source.unstream(arc)
        result: PHeapMap[Span, Ev[B]] = ev_heap_empty()

        for _, ev in source_events:
            inner_stream = self.func(ev.val)
            # Use the event's span to determine the query arc for the inner stream
            query_arc = ev.span.whole if ev.span.whole else ev.span.active
            inner_events = inner_stream.unstream(query_arc)

            for _, inner_ev in inner_events:
                # Apply merge strategy here if needed
                result = ev_heap_push(inner_ev, result)

        return result


@dataclass(frozen=True)
class ApplyStream[A, B, C](Stream[C]):
    """Stream that applies a function across two streams."""

    left: Stream[A]
    merge_strat: MergeStrat
    func: Callable[[A, B], C]
    right: Stream[B]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[C]]:
        left_events = self.left.unstream(arc)
        right_events = self.right.unstream(arc)
        result: PHeapMap[Span, Ev[C]] = ev_heap_empty()

        # Simple inner join strategy - only combine events that overlap
        for _, left_ev in left_events:
            for _, right_ev in right_events:
                left_arc = (
                    left_ev.span.whole if left_ev.span.whole else left_ev.span.active
                )
                right_arc = (
                    right_ev.span.whole if right_ev.span.whole else right_ev.span.active
                )

                intersection = left_arc.intersect(right_arc)
                if not intersection.null():
                    combined_val = self.func(left_ev.val, right_ev.val)
                    combined_span = Span(active=intersection, whole=None)
                    combined_ev = Ev(combined_span, combined_val)
                    result = ev_heap_push(combined_ev, result)

        return result


@dataclass(frozen=True)
class FastStream[T](Stream[T]):
    """Stream that speeds up events by a factor."""

    source: Stream[T]
    factor: Fraction

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.factor == 0:
            return ev_heap_empty()

        # Scale the query arc down by the factor
        scaled_arc = arc.scale(Fraction(1) / self.factor)
        source_events = self.source.unstream(scaled_arc)
        result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        for _, ev in source_events:
            # Scale the event back up by the factor
            fast_ev = ev.scale(self.factor)
            span = _create_span(fast_ev.span.active, arc)
            if span is not None:
                if fast_ev.span.whole is not None:
                    span = Span(active=span.active, whole=fast_ev.span.whole)
                elif fast_ev.span.active != span.active:
                    span = Span(active=span.active, whole=fast_ev.span.active)
                new_ev = Ev(span, fast_ev.val)
                result = ev_heap_push(new_ev, result)

        return result


@dataclass(frozen=True)
class SlowStream[T](Stream[T]):
    """Stream that slows down events by a factor."""

    source: Stream[T]
    factor: Fraction

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.factor == 0:
            return ev_heap_empty()

        # Scale the query arc up by the factor
        scaled_arc = arc.scale(self.factor)
        source_events = self.source.unstream(scaled_arc)
        result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        for _, ev in source_events:
            # Scale the event down by the factor
            slow_ev = ev.scale(Fraction(1) / self.factor)
            span = _create_span(slow_ev.span.active, arc)
            if span is not None:
                if slow_ev.span.whole is not None:
                    span = Span(active=span.active, whole=slow_ev.span.whole)
                elif slow_ev.span.active != span.active:
                    span = Span(active=span.active, whole=slow_ev.span.active)
                new_ev = Ev(span, slow_ev.val)
                result = ev_heap_push(new_ev, result)

        return result


@dataclass(frozen=True)
class EarlyStream[T](Stream[T]):
    """Stream that shifts events earlier in time."""

    source: Stream[T]
    delta: CycleDelta

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null():
            return ev_heap_empty()

        # Shift the query arc later to get events that will be shifted earlier
        shifted_arc = Arc(
            CycleTime(arc.start + self.delta), CycleTime(arc.end + self.delta)
        )
        source_events = self.source.unstream(shifted_arc)
        result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        for _, ev in source_events:
            # Shift the event earlier
            early_active = Arc(
                CycleTime(ev.span.active.start - self.delta),
                CycleTime(ev.span.active.end - self.delta),
            )
            early_whole = None
            if ev.span.whole is not None:
                early_whole = Arc(
                    CycleTime(ev.span.whole.start - self.delta),
                    CycleTime(ev.span.whole.end - self.delta),
                )

            span = _create_span(early_active, arc)
            if span is not None:
                if early_whole is not None:
                    span = Span(active=span.active, whole=early_whole)
                elif early_active != span.active:
                    span = Span(active=span.active, whole=early_active)
                new_ev = Ev(span, ev.val)
                result = ev_heap_push(new_ev, result)

        return result


@dataclass(frozen=True)
class MapStream[T, U](Stream[U]):
    """Stream that maps a function over values."""

    source: Stream[T]
    func: Callable[[T], U]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[U]]:
        source_events = self.source.unstream(arc)
        result: PHeapMap[Span, Ev[U]] = ev_heap_empty()

        for _, ev in source_events:
            mapped_val = self.func(ev.val)
            mapped_ev = Ev(ev.span, mapped_val)
            result = ev_heap_push(mapped_ev, result)

        return result


@dataclass(frozen=True)
class LateStream[T](Stream[T]):
    """Stream that shifts events later in time."""

    source: Stream[T]
    delta: CycleDelta

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null():
            return ev_heap_empty()

        # Shift the query arc earlier to get events that will be shifted later
        shifted_arc = Arc(
            CycleTime(arc.start - self.delta), CycleTime(arc.end - self.delta)
        )
        source_events = self.source.unstream(shifted_arc)
        result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        for _, ev in source_events:
            # Shift the event later
            late_active = Arc(
                CycleTime(ev.span.active.start + self.delta),
                CycleTime(ev.span.active.end + self.delta),
            )
            late_whole = None
            if ev.span.whole is not None:
                late_whole = Arc(
                    CycleTime(ev.span.whole.start + self.delta),
                    CycleTime(ev.span.whole.end + self.delta),
                )

            span = _create_span(late_active, arc)
            if span is not None:
                if late_whole is not None:
                    span = Span(active=span.active, whole=late_whole)
                elif late_active != span.active:
                    span = Span(active=span.active, whole=late_active)
                new_ev = Ev(span, ev.val)
                result = ev_heap_push(new_ev, result)

        return result


def stream_rand[T](streams: PSeq[Stream[T]]) -> Stream[T]:
    """Randomly select a stream from a sequence.

    Args:
        streams: Sequence of streams to choose from

    Returns:
        A stream that randomly selects from the input streams
    """
    return RandStream(streams)


def stream_alt[T](streams: PSeq[Stream[T]]) -> Stream[T]:
    """Alternately select streams from a sequence.

    Args:
        streams: Sequence of streams to cycle through

    Returns:
        A stream that alternates through the input streams
    """
    return AltStream(streams)


def stream_par[T](streams: PSeq[Stream[T]]) -> Stream[T]:
    """Combine multiple streams in parallel.

    Args:
        streams: Sequence of streams to combine

    Returns:
        A stream that plays all input streams simultaneously
    """
    return ParStream(streams)


@dataclass(frozen=True)
class RandStream[T](Stream[T]):
    """Stream that randomly selects from a sequence of streams."""

    streams: PSeq[Stream[T]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.streams) == 0:
            return ev_heap_empty()

        # Use arc start as seed for deterministic randomness
        random.seed(hash(arc.start))
        selected_stream = self.streams[random.randint(0, len(self.streams) - 1)]
        return selected_stream.unstream(arc)


@dataclass(frozen=True)
class AltStream[T](Stream[T]):
    """Stream that alternates through a sequence of streams."""

    streams: PSeq[Stream[T]]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.streams) == 0:
            return ev_heap_empty()

        # Cycle through streams based on cycle position
        cycle_index = int(arc.start) % len(self.streams)
        selected_stream = self.streams[cycle_index]
        return selected_stream.unstream(arc)


def pat_stream[T](pat: Pat[T]) -> Stream[T]:
    """Create a specialized stream for the given pattern.

    Args:
        pat: The pattern to create a stream for

    Returns:
        A specialized stream optimized for the pattern's constructor
    """
    match pat.unwrap:
        case PatSilent():
            return SilenceStream()
        case PatPure(val):
            return PureStream(val)
        case PatSeq(children):
            child_streams = PSeq.mk(pat_stream(child) for child in children)
            return SeqStream(child_streams)
        case PatPar(pats):
            child_streams = PSeq.mk(pat_stream(child) for child in pats)
            return ParStream(child_streams)
        case PatRand(pats):
            choice_streams = PSeq.mk(pat_stream(choice) for choice in pats)
            return ChoiceStream(choice_streams)
        case PatEuc(pat, hits, steps, rotation):
            atom_stream = pat_stream(pat)
            return EuclideanStream.create(atom_stream, hits, steps, rotation)
        case PatPoly(pats, subdiv):
            pattern_streams = PSeq.mk(pat_stream(pattern) for pattern in pats)
            return PolymetricStream(pattern_streams, subdiv)
        case PatSpeed(pat, op, factor):
            pattern_stream = pat_stream(pat)
            return RepetitionStream(pattern_stream, op, factor)
        case PatStretch(pat, count):
            pattern_stream = pat_stream(pat)
            return ElongationStream(pattern_stream, count)
        case PatProb(pat, chance):
            pattern_stream = pat_stream(pat)
            return ProbabilityStream(pattern_stream, chance)
        case PatAlt(pats):
            pattern_streams = PSeq.mk(pat_stream(pattern) for pattern in pats)
            return AlternatingStream(pattern_streams)
        case PatRepeat(pat, count):
            pattern_stream = pat_stream(pat)
            return ReplicateStream(pattern_stream, count)
        case _:
            # This should never happen if all pattern types are handled above
            raise Exception(f"Unhandled pattern type: {type(pat.unwrap).__name__}")


def _generate_euclidean(hits: int, steps: int, rotation: int) -> List[bool]:
    """Generata a Euclidean rhythm pattern using Bresenham's line algorithm.

    Args:
        hits: Number of hits to distribute
        steps: Total number of steps
        rotation: Optional rotation offset

    Returns:
        A list of booleans representing the Euclidean rhythm
    """
    if steps <= 0 or hits <= 0:
        return []

    if hits >= steps:
        return [True] * steps

    # Use Bresenham's line algorithm to distribute hits evenly
    pattern = [False] * steps
    slope = hits / steps
    previous = 0.0

    for i in range(steps):
        current = (i + 1) * slope
        if int(current) != int(previous):
            pattern[i] = True
        previous = current

    # Apply rotation
    if rotation != 0:
        rotation = rotation % steps
        pattern = pattern[rotation:] + pattern[:rotation]

    return pattern
