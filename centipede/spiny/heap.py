"""Persistent min-heap implementation using Brodal-Okasaki binomial heaps.

This module provides a functional min-heap data structure that supports
efficient insertion, deletion, and melding operations while maintaining
persistence (immutability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Type

from centipede.spiny.common import Impossible, Ordering, compare
from centipede.spiny.seq import Seq

__all__ = ["Heap"]


@dataclass(frozen=True, eq=False)
class HeapNode[K, V]:
    """Internal node representation for the binomial heap.

    Each node contains a key-value pair, a rank indicating the size of the
    subtree, and a reference to child heaps.

    Attributes:
        key: The key used for heap ordering.
        value: The value associated with the key.
        rank: The rank (size) of this heap node.
        rest: Child heap containing remaining elements.
    """

    key: K
    value: V
    rank: int
    rest: Heap[K, V]


@dataclass(frozen=True, eq=False)
class Heap[K, V]:
    """A Brodal-Okasaki persistent min-heap"""

    _unwrap: Seq[HeapNode[K, V]]

    @staticmethod
    def empty(
        _kty: Optional[Type[K]] = None, _vty: Optional[Type[V]] = None
    ) -> Heap[K, V]:
        """Create an empty heap.

        Args:
            _kty: Optional type hint for keys (unused).
            _vty: Optional type hint for values (unused).

        Returns:
            An empty heap instance.
        """
        return _HEAP_EMPTY

    @staticmethod
    def singleton(key: K, value: V) -> Heap[K, V]:
        """Create a heap containing a single key-value pair.

        Args:
            key: The key for the single element.
            value: The value for the single element.

        Returns:
            A heap containing only the given key-value pair.
        """
        return Heap(Seq.singleton(HeapNode(key, value, 0, Heap.empty())))

    @staticmethod
    def mk(entries: Iterable[Tuple[K, V]]) -> Heap[K, V]:
        """Create a heap from an iterable of key-value pairs.

        Args:
            entries: Iterable of (key, value) tuples to insert into the heap.

        Returns:
            A heap containing all the given entries.
        """
        heap: Heap[K, V] = Heap.empty()
        for key, value in entries:
            heap = heap.insert(key, value)
        return heap

    def null(self) -> bool:
        """Check if the heap is empty.

        Returns:
            True if the heap contains no elements, False otherwise.
        """
        return self._unwrap.null()

    def find_min(self) -> Optional[Tuple[K, V, Heap[K, V]]]:
        """Find the minimum element in the heap.

        Returns:
            None if the heap is empty, otherwise a tuple containing:
            - The minimum key
            - The corresponding value
            - A new heap with the minimum element removed
        """
        return _heap_find_min(self)

    def insert(self, key: K, value: V) -> Heap[K, V]:
        """Insert a new key-value pair into the heap.

        Args:
            key: The key to insert.
            value: The value associated with the key.

        Returns:
            A new heap containing the inserted element.
        """
        cand = HeapNode(key, value, 0, Heap.empty())
        return _heap_insert(cand, self)

    def meld(self, other: Heap[K, V]) -> Heap[K, V]:
        """Merge this heap with another heap.

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        return _heap_meld(self, other)

    def delete_min(self) -> Optional[Heap[K, V]]:
        """Remove the minimum element from the heap.

        Returns:
            None if the heap is empty, otherwise a new heap with the
            minimum element removed.
        """
        return _heap_delete_min(self)

    def __add__(self, other: Heap[K, V]) -> Heap[K, V]:
        """Merge heaps using the + operator.

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        return self.meld(other)

    def __bool__(self) -> bool:
        """Check if the heap is non-empty for boolean contexts.

        Returns:
            True if the heap contains elements, False if empty.
        """
        return not self.null()

    # def __list__(self) -> List[T]:
    #     return self.list()
    #
    # def __iter__(self) -> Generator[T]:
    #     return self.iter()
    #
    # def __reversed__(self) -> Generator[T]:
    #     return self.reversed()


_HEAP_EMPTY: Heap[Any, Any] = Heap(Seq.empty())


def _heap_insert[K, V](cand: HeapNode[K, V], heap: Heap[K, V]) -> Heap[K, V]:
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
            raise Impossible


def _heap_link[K, V](first: HeapNode[K, V], second: HeapNode[K, V]) -> HeapNode[K, V]:
    if compare(second.key, first.key) == Ordering.Gt:
        return HeapNode(
            first.key,
            first.value,
            first.rank + 1,
            Heap(first.rest._unwrap.cons(second)),
        )
    else:
        return HeapNode(
            second.key,
            second.value,
            second.rank + 1,
            Heap(second.rest._unwrap.cons(first)),
        )


def _heap_meld[K, V](first: Heap[K, V], second: Heap[K, V]) -> Heap[K, V]:
    match first._unwrap.uncons():
        case None:
            return second
        case (first_head, first_tail):
            match second._unwrap.uncons():
                case None:
                    return first
                case (second_head, second_tail):
                    if first_head.rank < second_head.rank:
                        tail = _heap_meld(Heap(first_tail), second)
                        return Heap(tail._unwrap.cons(first_head))
                    elif second_head.rank < first_head.rank:
                        tail = _heap_meld(first, Heap(second_tail))
                        return Heap(tail._unwrap.cons(second_head))
                    else:
                        head = _heap_link(first_head, second_head)
                        tail = _heap_meld(Heap(first_tail), Heap(second_tail))
                        return _heap_insert(head, tail)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _heap_find_min[K, V](
    heap: Heap[K, V],
) -> Optional[Tuple[K, V, Heap[K, V]]]:
    match heap._unwrap.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return (head.key, head.value, head.rest)
            else:
                cand = _heap_find_min(Heap(tail))
                if cand is None or compare(head.key, cand[0]) == Ordering.Lt:
                    rest = _heap_meld(head.rest, Heap(tail))
                    return (head.key, head.value, rest)
                else:
                    rest = _heap_meld(Heap(tail), cand[2])
                    return (cand[0], cand[1], rest)
        case _:
            raise Impossible


def _heap_delete_min[K, V](heap: Heap[K, V]) -> Optional[Heap[K, V]]:
    match heap._unwrap.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return head.rest
            else:
                cand = _heap_find_min(Heap(tail))
                if cand is None or compare(head.key, cand[0]) == Ordering.Lt:
                    return _heap_meld(head.rest, Heap(tail))
                else:
                    return _heap_meld(Heap(tail), cand[2])
        case _:
            raise Impossible
