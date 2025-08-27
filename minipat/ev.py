"""Event type for representing timed values in minipat patterns."""

from __future__ import annotations

from dataclasses import dataclass

from minipat.arc import Arc
from minipat.common import CycleDelta, Factor
from spiny import PHeapMap


@dataclass(frozen=True)
class Ev[T]:
    """An event with a time arc and a value.

    Args:
        arc: The time interval of the event
        val: The value of the event
    """

    arc: Arc
    val: T

    def shift(self, delta: CycleDelta) -> Ev[T]:
        """Shift the event by a time delta.

        Args:
            delta: The amount to shift by

        Returns:
            A new event shifted by the delta
        """
        return Ev(self.arc.shift(delta), self.val)

    def scale(self, factor: Factor) -> Ev[T]:
        """Scale the event's arc by a factor.

        Args:
            factor: The scaling factor

        Returns:
            A new event with scaled arc
        """
        return Ev(self.arc.scale(factor), self.val)

    def clip(self, factor: Factor) -> Ev[T]:
        """Clip the event's arc to a fraction of its length.

        Args:
            factor: The fraction to clip to (0 to 1)

        Returns:
            A new event with clipped arc
        """
        return Ev(self.arc.clip(factor), self.val)


type EvHeap[T] = PHeapMap[Arc, Ev[T]]
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
    return PHeapMap.singleton(ev.arc, ev)


def ev_heap_push[T](ev: Ev[T], heap: EvHeap[T]) -> EvHeap[T]:
    """Push an event into an event heap.

    Args:
        ev: The event to add
        heap: The heap to add to

    Returns:
        A new event heap with the event added
    """
    return heap.insert(ev.arc, ev)
