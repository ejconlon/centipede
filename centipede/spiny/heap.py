"""Persistent min-heap implementation using Brodal-Okasaki binomial heaps.

This module provides a functional min-heap data structure that supports
efficient insertion, deletion, and melding operations while maintaining
persistence (immutability).

Each key is associated with a single value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Tuple, Type, override

from centipede.spiny.common import Impossible, Iterating, Ordering, Sized, compare
from centipede.spiny.seq import Seq

__all__ = ["Heap"]


@dataclass(frozen=True, eq=False)
class HeapNode[K, V]:
    """Internal node representation for the binomial heap.

    Each node contains a key-values pair, a rank indicating the size of the
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
class Heap[K, V](Sized, Iterating[Tuple[K, V]]):
    """A Brodal-Okasaki persistent min-heap"""

    _size: int
    _children: Seq[HeapNode[K, V]]

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
        return Heap(1, Seq.singleton(HeapNode(key, value, 0, Heap.empty())))

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

    @override
    def null(self) -> bool:
        """Check if the heap is empty.

        Returns:
            True if the heap contains no elements, False otherwise.
        """
        return self._children.null()

    def find_min(self) -> Optional[Tuple[K, V, Heap[K, V]]]:
        """Find the minimum element in the heap.

        Returns:
            None if the heap is empty, otherwise a tuple containing:
            - The minimum key
            - The corresponding value
            - A new heap with the minimum element removed
        """
        result = _heap_find_min(self)
        if result is None:
            return None
        key, value, remaining = result
        return (key, value, Heap(self._size - 1, remaining._children))

    def insert(self, key: K, value: V) -> Heap[K, V]:
        """Insert a new key-value pair into the heap.

        Args:
            key: The key to insert.
            value: The value associated with the key.

        Returns:
            A new heap containing the inserted element.
        """
        cand = HeapNode(key, value, 0, Heap.empty())
        new_heap = _heap_insert(cand, self)
        return Heap(self._size + 1, new_heap._children)

    def meld(self, other: Heap[K, V]) -> Heap[K, V]:
        """Merge this heap with another heap.

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        new_heap = _heap_meld(self, other)
        return Heap(self._size + other._size, new_heap._children)

    def delete_min(self) -> Optional[Heap[K, V]]:
        """Remove the minimum element from the heap.

        Returns:
            None if the heap is empty, otherwise a new heap with the
            minimum element removed.
        """
        result = _heap_delete_min(self)
        if result is None:
            return None
        return Heap(self._size - 1, result._children)

    @override
    def size(self) -> int:
        """Return the number of elements in the heap.

        Returns:
            The total number of key-value pairs in the heap.
        """
        return self._size

    @override
    def iter(self) -> Generator[Tuple[K, V]]:
        """Iterate through the heap in ascending order.

        Yields:
            Tuples of (key, value) pairs in ascending order by key.
        """
        return _heap_iter(self)

    def __add__(self, other: Heap[K, V]) -> Heap[K, V]:
        """Merge heaps using the + operator.

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        return self.meld(other)


_HEAP_EMPTY: Heap[Any, Any] = Heap(0, Seq.empty())


def _heap_insert[K, V](cand: HeapNode[K, V], heap: Heap[K, V]) -> Heap[K, V]:
    match heap._children.uncons():
        case None:
            return Heap(0, Seq.singleton(cand))
        case (head, tail):
            if cand.rank < head.rank:
                return Heap(0, heap._children.cons(cand))
            else:
                new_node = _heap_link(cand, head)
                return _heap_insert(new_node, Heap(0, tail))
        case _:
            raise Impossible


def _heap_link[K, V](first: HeapNode[K, V], second: HeapNode[K, V]) -> HeapNode[K, V]:
    match compare(first.key, second.key):
        case Ordering.Gt:
            return HeapNode(
                second.key,
                second.value,
                second.rank + 1,
                Heap(0, second.rest._children.cons(first)),
            )
        case _:
            return HeapNode(
                first.key,
                first.value,
                first.rank + 1,
                Heap(0, first.rest._children.cons(second)),
            )


def _heap_meld[K, V](first: Heap[K, V], second: Heap[K, V]) -> Heap[K, V]:
    match first._children.uncons():
        case None:
            return second
        case (first_head, first_tail):
            match second._children.uncons():
                case None:
                    return first
                case (second_head, second_tail):
                    if first_head.rank < second_head.rank:
                        tail = _heap_meld(Heap(0, first_tail), second)
                        return Heap(0, tail._children.cons(first_head))
                    elif second_head.rank < first_head.rank:
                        tail = _heap_meld(first, Heap(0, second_tail))
                        return Heap(0, tail._children.cons(second_head))
                    else:
                        head = _heap_link(first_head, second_head)
                        tail = _heap_meld(Heap(0, first_tail), Heap(0, second_tail))
                        return _heap_insert(head, tail)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _heap_find_min[K, V](
    heap: Heap[K, V],
) -> Optional[Tuple[K, V, Heap[K, V]]]:
    match heap._children.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return (head.key, head.value, head.rest)
            else:
                cand = _heap_find_min(Heap(0, tail))
                if cand is None or compare(head.key, cand[0]) == Ordering.Lt:
                    rest = _heap_meld(head.rest, Heap(0, tail))
                    return (head.key, head.value, rest)
                else:
                    rest = _heap_meld(head.rest, cand[2])
                    return (cand[0], cand[1], rest)
        case _:
            raise Impossible


def _heap_delete_min[K, V](heap: Heap[K, V]) -> Optional[Heap[K, V]]:
    match heap._children.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return head.rest
            else:
                cand = _heap_find_min(Heap(0, tail))
                if cand is None or compare(head.key, cand[0]) == Ordering.Lt:
                    return _heap_meld(head.rest, Heap(0, tail))
                else:
                    return _heap_meld(head.rest, cand[2])
        case _:
            raise Impossible


def _heap_iter[K, V](heap: Heap[K, V]) -> Generator[Tuple[K, V]]:
    while not heap.null():
        min_result = heap.find_min()
        if min_result is None:
            break
        key, value, remaining = min_result
        yield (key, value)
        heap = remaining
