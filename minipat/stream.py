"""Stream implementation for converting patterns to timed events."""

import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, override

from minipat.arc import Arc, Span
from minipat.common import CycleTime
from minipat.ev import Ev, ev_heap_empty, ev_heap_push, ev_heap_singleton
from minipat.pat import (
    Pat,
    PatAlternating,
    PatChoice,
    PatElongation,
    PatEuclidean,
    PatPar,
    PatPolymetric,
    PatProbability,
    PatPure,
    PatRepetition,
    PatReplicate,
    PatSelect,
    PatSeq,
    PatSilence,
    RepetitionOp,
)
from spiny import PSeq
from spiny.heapmap import PHeapMap


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
    ) -> "EuclideanStream[T]":
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
    subdivision: Optional[int]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or len(self.patterns) == 0:
            return ev_heap_empty()

        polymetric_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        if self.subdivision is None:
            # All patterns play simultaneously
            for pattern in self.patterns:
                pattern_events = pattern.unstream(arc)
                for _, ev in pattern_events:
                    polymetric_result = ev_heap_push(ev, polymetric_result)
        else:
            # With subdivision
            if self.subdivision <= 0:
                return ev_heap_empty()
            sub_arc = arc.scale(Fraction(1, self.subdivision))
            for pattern in self.patterns:
                pattern_events = pattern.unstream(sub_arc)
                for _, ev in pattern_events:
                    scaled_ev = ev.scale(Fraction(self.subdivision))
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
    operator: RepetitionOp
    count: int

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null() or self.count <= 0:
            return ev_heap_empty()

        rep_result: PHeapMap[Span, Ev[T]] = ev_heap_empty()

        match self.operator:
            case RepetitionOp.Fast:
                if hasattr(self.count, "denominator") and self.count.denominator != 1:
                    scaled_arc = arc.scale(Fraction(1) / Fraction(self.count))
                    pattern_events = self.pattern.unstream(scaled_arc)
                    for _, ev in pattern_events:
                        fast_ev = ev.scale(Fraction(self.count))
                        span = _create_span(fast_ev.span.active, arc)
                        if span is not None:
                            # Preserve the whole information from scaling
                            if fast_ev.span.whole is not None:
                                span = Span(
                                    active=span.active, whole=fast_ev.span.whole
                                )
                            elif fast_ev.span.active != span.active:
                                span = Span(
                                    active=span.active, whole=fast_ev.span.active
                                )
                            new_ev = Ev(span, fast_ev.val)
                            rep_result = ev_heap_push(new_ev, rep_result)
                else:
                    int_count = (
                        int(self.count)
                        if hasattr(self.count, "numerator")
                        else self.count
                    )
                    if int_count > 0:
                        rep_duration = arc.length() / int_count
                        for i in range(int_count):
                            rep_start = arc.start + i * rep_duration
                            rep_arc = Arc(
                                CycleTime(rep_start),
                                CycleTime(rep_start + rep_duration),
                            )

                            if not arc.intersect(rep_arc).null():
                                pattern_events = self.pattern.unstream(rep_arc)
                                for _, ev in pattern_events:
                                    span = _create_span(ev.span.active, arc)
                                    if span is not None:
                                        # Preserve the child's whole information if it exists
                                        if ev.span.whole is not None:
                                            span = Span(
                                                active=span.active, whole=ev.span.whole
                                            )
                                        elif ev.span.active != span.active:
                                            span = Span(
                                                active=span.active, whole=ev.span.active
                                            )
                                        new_ev = Ev(span, ev.val)
                                        rep_result = ev_heap_push(new_ev, rep_result)

            case RepetitionOp.Slow:
                stretched_arc = arc.scale(Fraction(self.count))
                pattern_events = self.pattern.unstream(stretched_arc)
                for _, ev in pattern_events:
                    slow_ev = ev.scale(Fraction(1, self.count))
                    span = _create_span(slow_ev.span.active, arc)
                    if span is not None:
                        # Preserve the whole information from scaling
                        if slow_ev.span.whole is not None:
                            span = Span(active=span.active, whole=slow_ev.span.whole)
                        elif slow_ev.span.active != span.active:
                            span = Span(active=span.active, whole=slow_ev.span.active)
                        new_ev = Ev(span, slow_ev.val)
                        rep_result = ev_heap_push(new_ev, rep_result)

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
    prob: Fraction

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        if arc.null():
            return ev_heap_empty()

        random.seed(hash(arc.start))
        if random.random() < self.prob:
            return self.pattern.unstream(arc)
        return ev_heap_empty()


@dataclass(frozen=True)
class SelectStream[T](Stream[T]):
    """Specialized stream for select patterns."""

    pattern: Stream[T]
    selector: str

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Span, Ev[T]]:
        return self.pattern.unstream(arc)


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


def pat_stream[T](pat: Pat[T]) -> Stream[T]:
    """Create a specialized stream for the given pattern.

    Args:
        pat: The pattern to create a stream for

    Returns:
        A specialized stream optimized for the pattern's constructor
    """
    match pat.unwrap:
        case PatSilence():
            return SilenceStream()
        case PatPure(val):
            return PureStream(val)
        case PatSeq(children):
            child_streams = PSeq.mk(pat_stream(child) for child in children)
            return SeqStream(child_streams)
        case PatPar(children):
            child_streams = PSeq.mk(pat_stream(child) for child in children)
            return ParStream(child_streams)
        case PatChoice(choices):
            choice_streams = PSeq.mk(pat_stream(choice) for choice in choices)
            return ChoiceStream(choice_streams)
        case PatEuclidean(atom, hits, steps, rotation):
            atom_stream = pat_stream(atom)
            return EuclideanStream.create(atom_stream, hits, steps, rotation)
        case PatPolymetric(patterns, subdivision):
            pattern_streams = PSeq.mk(pat_stream(pattern) for pattern in patterns)
            return PolymetricStream(pattern_streams, subdivision)
        case PatRepetition(pattern, operator, count):
            pattern_stream = pat_stream(pattern)
            return RepetitionStream(pattern_stream, operator, count)
        case PatElongation(pattern, count):
            pattern_stream = pat_stream(pattern)
            return ElongationStream(pattern_stream, count)
        case PatProbability(pattern, prob):
            pattern_stream = pat_stream(pattern)
            return ProbabilityStream(pattern_stream, prob)
        case PatSelect(pattern, selector):
            pattern_stream = pat_stream(pattern)
            return SelectStream(pattern_stream, selector)
        case PatAlternating(patterns):
            pattern_streams = PSeq.mk(pat_stream(pattern) for pattern in patterns)
            return AlternatingStream(pattern_streams)
        case PatReplicate(pattern, count):
            pattern_stream = pat_stream(pattern)
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
