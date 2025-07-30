from __future__ import annotations

from dataclasses import dataclass

from centipede.minipat.arc import Arc
from centipede.minipat.common import Delta, Factor
from centipede.spiny import PHeapMap


@dataclass(frozen=True)
class Ev[T]:
    arc: Arc
    val: T

    def shift(self, delta: Delta) -> Ev[T]:
        return Ev(self.arc.shift(delta), self.val)

    def scale(self, factor: Factor) -> Ev[T]:
        return Ev(self.arc.scale(factor), self.val)

    def clip(self, factor: Factor) -> Ev[T]:
        return Ev(self.arc.clip(factor), self.val)


type EvHeap[T] = PHeapMap[Arc, Ev[T]]


def ev_heap_empty[T]() -> EvHeap[T]:
    return PHeapMap.empty()


def ev_heap_singleton[T](ev: Ev[T]) -> EvHeap[T]:
    return PHeapMap.singleton(ev.arc, ev)


def ev_heap_push[T](ev: Ev[T], heap: EvHeap[T]) -> EvHeap[T]:
    return heap.insert(ev.arc, ev)
