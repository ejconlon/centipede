""" A Brodal-Okasaki persistent min-heap """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple


class Comparable(Protocol):
    def __le__[S](self: S, other: S) -> bool:
        raise NotImplementedError()


class Seq[T]:
    @staticmethod
    def empty() -> Seq[T]:
        raise NotImplementedError()

    @staticmethod
    def singleton(val: T) -> Seq[T]:
        raise NotImplementedError()

    def null(self) -> bool:
        raise NotImplementedError()

    def uncons(self) -> Optional[Tuple[T, Seq[T]]]:
        raise NotImplementedError()

    def cons(self, val: T) -> Seq[T]:
        raise NotImplementedError()

    def concat(self, other: Seq[T]):
        raise NotImplementedError()


@dataclass(frozen=True, eq=False)
class HeapNode[K: Comparable, V]:
    key: K
    value: V
    rank: int
    rest: Heap[K, V]


@dataclass(frozen=True, eq=False)
class Heap[K: Comparable, V]:
    _unwrap: Seq[HeapNode[K, V]]

    @staticmethod
    def empty() -> Heap[K, V]:
        return _HEAP_EMPTY

    @staticmethod
    def singleton(key: K, value: V) -> Heap[K, V]:
        return Heap(Seq.singleton(HeapNode(key, value, 0, Heap.empty())))

    def null(self) -> bool:
        return bool(self._unwrap)

    def findMin(self) -> Optional[HeapNode[K, V]]:
        raise NotImplementedError()

    def insert(self, key: K, value: V) -> Heap[K, V]:
        cand = HeapNode(key, value, 0, Heap.empty())
        return _heap_insert(cand, self)

    def meld(self, other: Heap[K, V]) -> Heap[K, V]:
        return _heap_meld(self, other)

    def deleteMin(self) -> Optional[Heap[K, V]]:
        raise NotImplementedError()


_HEAP_EMPTY: Heap[Any, Any] = Heap(Seq.empty())


def _heap_insert[K: Comparable, V](cand: HeapNode[K, V], heap: Heap[K, V]) -> Heap[K, V]:
    match heap._unwrap.uncons():
        case None:
            return Heap(Seq.singleton(cand))
        case (head, tail):
            if cand.rank < head.rank:
                return Heap(heap._unwrap.cons(cand))
            else:
                new_node = _heap_link(cand, head)
                return _heap_insert(new_node, Heap(tail))
        case _:
            raise Exception('impossible')


def _heap_link[K: Comparable, V](first: HeapNode[K, V], second: HeapNode[K, V]) -> HeapNode[K, V]:
    if first.key <= second.key:
        return HeapNode(
            first.key,
            first.value,
            first.rank + 1,
            Heap(first.rest._unwrap.cons(second))
        )
    else:
        return HeapNode(
            second.key,
            second.value,
            second.rank + 1,
            Heap(second.rest._unwrap.cons(first))
        )


def _heap_meld[K: Comparable, V](first: Heap[K, V], second: Heap[K, V]) -> Heap[K, V]:
    match first._unwrap.uncons():
        case None:
            return second
        case (first_head, first_tail):
            match second._unwrap.uncons():
                case None:
                    return first
                case (second_head, second_tail):
                    raise Exception('TODO')
                case _:
                    raise Exception('impossible')
        case _:
            raise Exception('impossible')
