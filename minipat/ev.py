"""Event type for representing timed values in minipat patterns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from minipat.time import CycleDelta, CycleSpan
from spiny import PHeapMap


@dataclass(frozen=True)
class Ev[T]:
    """An event with a time arc and a value.

    Args:
        arc: The time interval of the event
        val: The value of the event
    """

    span: CycleSpan
    val: T

    def shift(self, delta: CycleDelta) -> Ev[T]:
        """Shift the event by a time delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new event shifted by the delta
        """
        return Ev(self.span.shift(delta), self.val)

    def scale(self, factor: Fraction) -> Ev[T]:
        """Scale the event's span by a factor.

        Args:
            factor: The scaling factor

        Returns:
            A new event with scaled arc
        """
        return Ev(self.span.scale(factor), self.val)

    def clip(self, factor: Fraction) -> Ev[T]:
        """Clip the event's span to a fraction of its length.

        Args:
            factor: The fraction to clip to (0 to 1)

        Returns:
            A new event with clipped span
        """
        return Ev(self.span.clip(factor), self.val)


type EvHeap[T] = PHeapMap[CycleSpan, Ev[T]]
"""Type alias for a heap map of events indexed by their arcs."""


def ev_heap_empty[T]() -> EvHeap[T]:
    """Create an empty event heap.

    Returns:
        An empty event heap
    """
    return PHeapMap.empty()


def ev_heap_singleton[T](ev: Ev[T]) -> EvHeap[T]:
    """Create an event heap with a single event.

    Args:
        ev: The event to include

    Returns:
        An event heap containing only the given event
    """
    return PHeapMap.singleton(ev.span, ev)


def ev_heap_push[T](ev: Ev[T], heap: EvHeap[T]) -> EvHeap[T]:
    """Push an event into an event heap.

    Args:
        ev: The event to add
        heap: The heap to add to

    Returns:
        A new event heap with the event added
    """
    return heap.insert(ev.span, ev)
