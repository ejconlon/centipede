from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, override

from centipede.minipat.arc import Arc
from centipede.minipat.ev import Ev, ev_heap_empty, ev_heap_push, ev_heap_singleton
from centipede.minipat.pat import (
    Pat,
    PatAlternating,
    PatChoice,
    PatElongation,
    PatEuclidean,
    PatGroup,
    PatPar,
    PatPolymetric,
    PatProbability,
    PatPure,
    PatRepetition,
    PatScale,
    PatSelect,
    PatSeq,
    PatSilence,
    RepetitionOp,
)
from centipede.spiny.heapmap import PHeapMap


# sealed
class Stream[T](metaclass=ABCMeta):
    """A stream of events in time."""

    @abstractmethod
    def unstream(self, arc: Arc) -> PHeapMap[Arc, Ev[T]]:
        """Emit all events that start or end in the given arc."""
        ...


@dataclass(frozen=True)
class PatStream[T](Stream[T]):
    pat: Pat[T]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Arc, Ev[T]]:
        def process_pattern(pf) -> PHeapMap[Arc, Ev[T]]:
            match pf:
                case PatSilence():
                    return ev_heap_empty()

                case PatPure(val):
                    if not arc.null():
                        event = Ev(arc, val)
                        return ev_heap_singleton(event)
                    return ev_heap_empty()

                case PatScale(factor, child_result):
                    scaled_events: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    for _, ev in child_result:
                        scaled_ev = ev.scale(factor)
                        if not arc.intersect(scaled_ev.arc).null():
                            scaled_events = ev_heap_push(scaled_ev, scaled_events)
                    return scaled_events

                case PatSeq(children):
                    seq_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    if len(children) == 0:
                        return seq_result

                    child_duration = arc.length() / len(children)
                    for i, child_result in enumerate(children):
                        child_start = arc.start + i * child_duration
                        child_arc = Arc(child_start, child_start + child_duration)
                        if not child_arc.intersect(arc).null():
                            for _, ev in child_result:
                                if not arc.intersect(ev.arc).null():
                                    seq_result = ev_heap_push(ev, seq_result)
                    return seq_result

                case PatGroup(children):
                    # Groups behave like sequences
                    group_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    if len(children) == 0:
                        return group_result

                    child_duration = arc.length() / len(children)
                    for i, child_result in enumerate(children):
                        child_start = arc.start + i * child_duration
                        child_arc = Arc(child_start, child_start + child_duration)
                        if not child_arc.intersect(arc).null():
                            for _, ev in child_result:
                                if not arc.intersect(ev.arc).null():
                                    group_result = ev_heap_push(ev, group_result)
                    return group_result

                case PatPar(children):
                    par_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    for child_result in children:
                        for _, ev in child_result:
                            if not arc.intersect(ev.arc).null():
                                par_result = ev_heap_push(ev, par_result)
                    return par_result

                case PatChoice(choices):
                    # For simplicity, take the first choice
                    if len(choices) > 0:
                        return choices[0]
                    return ev_heap_empty()

                case PatEuclidean(atom_result, hits, steps, rotation):
                    euclidean_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    if steps <= 0 or hits <= 0:
                        return euclidean_result

                    # Generate euclidean rhythm
                    step_duration = arc.length() / steps
                    euclidean_hits = _generate_euclidean(hits, steps, rotation)

                    for i, is_hit in enumerate(euclidean_hits):
                        if is_hit:
                            step_start = arc.start + i * step_duration
                            step_arc = Arc(step_start, step_start + step_duration)
                            for _, ev in atom_result:
                                shifted_ev = Ev(step_arc, ev.val)
                                if not arc.intersect(shifted_ev.arc).null():
                                    euclidean_result = ev_heap_push(
                                        shifted_ev, euclidean_result
                                    )
                    return euclidean_result

                case PatPolymetric(patterns):
                    polymetric_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    for pattern_result in patterns:
                        for _, ev in pattern_result:
                            if not arc.intersect(ev.arc).null():
                                polymetric_result = ev_heap_push(ev, polymetric_result)
                    return polymetric_result

                case PatRepetition(pattern_result, operator, count):
                    repetition_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    if count <= 0:
                        return repetition_result

                    match operator:
                        case RepetitionOp.FAST:
                            # Faster repetition - compress time
                            rep_duration = arc.length() / count
                            for i in range(count):
                                for _, ev in pattern_result:
                                    scaled_ev = ev.scale(1 / count).shift(
                                        i * rep_duration
                                    )
                                    if not arc.intersect(scaled_ev.arc).null():
                                        repetition_result = ev_heap_push(
                                            scaled_ev, repetition_result
                                        )
                        case RepetitionOp.SLOW:
                            # Slower repetition - stretch time
                            for _, ev in pattern_result:
                                stretched_ev = ev.scale(count)
                                if not arc.intersect(stretched_ev.arc).null():
                                    repetition_result = ev_heap_push(
                                        stretched_ev, repetition_result
                                    )
                    return repetition_result

                case PatElongation(pattern_result, count):
                    elongation_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    for _, ev in pattern_result:
                        elongated_ev = ev.scale(count)
                        if not arc.intersect(elongated_ev.arc).null():
                            elongation_result = ev_heap_push(
                                elongated_ev, elongation_result
                            )
                    return elongation_result

                case PatProbability(pattern_result, prob):
                    # For simplicity, include events based on probability threshold
                    import random

                    if random.random() < prob:
                        return pattern_result
                    return ev_heap_empty()

                case PatSelect(pattern_result, _):
                    # For simplicity, return the pattern as-is
                    return pattern_result

                case PatAlternating(patterns):
                    # For simplicity, cycle through patterns
                    alternating_result: PHeapMap[Arc, Ev[T]] = ev_heap_empty()
                    for i, pattern_result in enumerate(patterns):
                        for _, ev in pattern_result:
                            if not arc.intersect(ev.arc).null():
                                alternating_result = ev_heap_push(
                                    ev, alternating_result
                                )
                    return alternating_result

                case _:
                    return ev_heap_empty()

        return self.pat.cata(process_pattern)


def _generate_euclidean(hits: int, steps: int, rotation: int = 0) -> List[bool]:
    """Generate a Euclidean rhythm pattern."""
    if steps <= 0 or hits <= 0:
        return []

    if hits >= steps:
        return [True] * steps

    # Euclidean algorithm for rhythm generation
    pattern = [False] * steps
    bucket = 0

    for i in range(steps):
        bucket += hits
        if bucket >= steps:
            bucket -= steps
            pattern[i] = True

    # Apply rotation
    if rotation != 0:
        rotation = rotation % steps
        pattern = pattern[rotation:] + pattern[:rotation]

    return pattern
