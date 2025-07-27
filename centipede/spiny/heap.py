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
from centipede.spiny.seq import PSeq

__all__ = ["PHeap"]


@dataclass(frozen=True, eq=False)
class PHeapNode[K, V]:
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
    rest: PHeap[K, V]


@dataclass(frozen=True, eq=False)
class PHeap[K, V](Sized, Iterating[Tuple[K, V]]):
    """A Brodal-Okasaki persistent min-heap"""

    _size: int
    _children: PSeq[PHeapNode[K, V]]

    @staticmethod
    def empty(
        _kty: Optional[Type[K]] = None, _vty: Optional[Type[V]] = None
    ) -> PHeap[K, V]:
        """Create an empty heap.

        Args:
            _kty: Optional type hint for keys (unused).
            _vty: Optional type hint for values (unused).

        Returns:
            An empty heap instance.
        """
        return _HEAP_EMPTY

    @staticmethod
    def singleton(key: K, value: V) -> PHeap[K, V]:
        """Create a heap containing a single key-value pair.

        Args:
            key: The key for the single element.
            value: The value for the single element.

        Returns:
            A heap containing only the given key-value pair.
        """
        return PHeap(1, PSeq.singleton(PHeapNode(key, value, 0, PHeap.empty())))

    @staticmethod
    def mk(entries: Iterable[Tuple[K, V]]) -> PHeap[K, V]:
        """Create a heap from an iterable of key-value pairs.

        Args:
            entries: Iterable of (key, value) tuples to insert into the heap.

        Returns:
            A heap containing all the given entries.
        """
        heap: PHeap[K, V] = PHeap.empty()
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

    def find_min(self) -> Optional[Tuple[K, V, PHeap[K, V]]]:
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
        return (key, value, remaining)

    def insert(self, key: K, value: V) -> PHeap[K, V]:
        """Insert a new key-value pair into the heap.

        Args:
            key: The key to insert.
            value: The value associated with the key.

        Returns:
            A new heap containing the inserted element.
        """
        cand = PHeapNode(key, value, 0, PHeap.empty())
        new_heap = _heap_insert(cand, self)
        return PHeap(self._size + 1, new_heap._children)

    def meld(self, other: PHeap[K, V]) -> PHeap[K, V]:
        """Merge this heap with another heap.

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        new_heap = _heap_meld(self, other)
        return PHeap(self._size + other._size, new_heap._children)

    def delete_min(self) -> Optional[PHeap[K, V]]:
        """Remove the minimum element from the heap.

        Returns:
            None if the heap is empty, otherwise a new heap with the
            minimum element removed.
        """
        result = self.find_min()
        return None if result is None else result[2]

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

    def __add__(self, other: PHeap[K, V]) -> PHeap[K, V]:
        """Merge heaps using the + operator.

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        return self.meld(other)


_HEAP_EMPTY: PHeap[Any, Any] = PHeap(0, PSeq.empty())


def _calculate_node_size[K, V](node: PHeapNode[K, V]) -> int:
    """Calculate the size of a heap node (1 + size of its rest heap)."""
    return 1 + node.rest._size


def _heap_insert[K, V](cand: PHeapNode[K, V], heap: PHeap[K, V]) -> PHeap[K, V]:
    cand_size = _calculate_node_size(cand)
    match heap._children.uncons():
        case None:
            return PHeap(cand_size, PSeq.singleton(cand))
        case (head, tail):
            if cand.rank < head.rank:
                return PHeap(heap._size + cand_size, heap._children.cons(cand))
            else:
                new_node = _heap_link(cand, head)
                tail_size = heap._size - _calculate_node_size(head)
                return _heap_insert(new_node, PHeap(tail_size, tail))
        case _:
            raise Impossible


def _heap_link[K, V](
    first: PHeapNode[K, V], second: PHeapNode[K, V]
) -> PHeapNode[K, V]:
    match compare(first.key, second.key):
        case Ordering.Gt:
            new_rest_size = second.rest._size + _calculate_node_size(first)
            return PHeapNode(
                second.key,
                second.value,
                second.rank + 1,
                PHeap(new_rest_size, second.rest._children.cons(first)),
            )
        case _:
            new_rest_size = first.rest._size + _calculate_node_size(second)
            return PHeapNode(
                first.key,
                first.value,
                first.rank + 1,
                PHeap(new_rest_size, first.rest._children.cons(second)),
            )


def _heap_meld[K, V](first: PHeap[K, V], second: PHeap[K, V]) -> PHeap[K, V]:
    match first._children.uncons():
        case None:
            return second
        case (first_head, first_tail):
            match second._children.uncons():
                case None:
                    return first
                case (second_head, second_tail):
                    if first_head.rank < second_head.rank:
                        first_tail_size = first._size - _calculate_node_size(first_head)
                        tail = _heap_meld(PHeap(first_tail_size, first_tail), second)
                        return PHeap(
                            tail._size + _calculate_node_size(first_head),
                            tail._children.cons(first_head),
                        )
                    elif second_head.rank < first_head.rank:
                        second_tail_size = second._size - _calculate_node_size(
                            second_head
                        )
                        tail = _heap_meld(first, PHeap(second_tail_size, second_tail))
                        return PHeap(
                            tail._size + _calculate_node_size(second_head),
                            tail._children.cons(second_head),
                        )
                    else:
                        head = _heap_link(first_head, second_head)
                        first_tail_size = first._size - _calculate_node_size(first_head)
                        second_tail_size = second._size - _calculate_node_size(
                            second_head
                        )
                        tail = _heap_meld(
                            PHeap(first_tail_size, first_tail),
                            PHeap(second_tail_size, second_tail),
                        )
                        return _heap_insert(head, tail)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _heap_find_min[K, V](
    heap: PHeap[K, V],
) -> Optional[Tuple[K, V, PHeap[K, V]]]:
    match heap._children.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return (head.key, head.value, head.rest)
            else:
                tail_size = heap._size - _calculate_node_size(head)
                cand = _heap_find_min(PHeap(tail_size, tail))
                if cand is None or compare(head.key, cand[0]) != Ordering.Gt:
                    # Choose head when it's smaller or equal (prefer head for ties)
                    rest = _heap_meld(head.rest, PHeap(tail_size, tail))
                    return (head.key, head.value, rest)
                else:
                    # Only choose candidate when head is strictly greater
                    head_as_heap = PHeap(
                        _calculate_node_size(head), PSeq.singleton(head)
                    )
                    rest = _heap_meld(head_as_heap, cand[2])
                    return (cand[0], cand[1], rest)
        case _:
            raise Impossible


def _heap_iter[K, V](heap: PHeap[K, V]) -> Generator[Tuple[K, V]]:
    while not heap.null():
        min_result = heap.find_min()
        if min_result is None:
            break
        key, value, _ = min_result
        yield (key, value)

        delete_result = heap.delete_min()
        if delete_result is None:
            break
        heap = delete_result
