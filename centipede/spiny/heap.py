"""Persistent min-heap implementation using Brodal-Okasaki binomial heaps.

This module provides a functional min-heap data structure that supports
efficient insertion, deletion, and merging operations while maintaining
persistence (immutability).

The heap can be used with Entry for map-like functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Type, override

from centipede.spiny.common import Impossible, Iterating, Ordering, Sized, compare
from centipede.spiny.seq import PSeq

__all__ = ["PHeap"]


@dataclass(frozen=True, eq=False)
class PHeapNode[T]:
    """Internal node representation for the binomial heap.

    Each node contains a value, a rank indicating the depth of the
    subtree, and a reference to child heaps.

    Attributes:
        value: The value used for heap ordering.
        rank: The rank (depth) of this heap node.
        rest: Child heap containing remaining elements.
    """

    value: T
    rank: int
    rest: PHeap[T]


@dataclass(frozen=True, eq=False)
class PHeap[T](Sized, Iterating[T]):
    """A Brodal-Okasaki persistent min-heap"""

    _size: int
    _children: PSeq[PHeapNode[T]]

    @staticmethod
    def empty(_ty: Optional[Type[T]] = None) -> PHeap[T]:
        """Create an empty heap.

        Args:
            _ty: Optional type hint for elements (unused).

        Returns:
            An empty heap instance.
        """
        return _HEAP_EMPTY

    @staticmethod
    def singleton(value: T) -> PHeap[T]:
        """Create a heap containing a single element.

        Args:
            value: The singleton element.

        Returns:
            A heap containing only the given element.
        """
        return PHeap(1, PSeq.singleton(PHeapNode(value, 0, PHeap.empty())))

    @staticmethod
    def mk(values: Iterable[T]) -> PHeap[T]:
        """Create a heap from an iterable of elements.

        Args:
            values: Iterable of elements to insert into the heap.

        Returns:
            A heap containing all the given elements.
        """
        heap: PHeap[T] = PHeap.empty()
        for value in values:
            heap = heap.insert(value)
        return heap

    @override
    def null(self) -> bool:
        """Check if the heap is empty.

        Returns:
            True if the heap contains no elements, False otherwise.
        """
        return self._children.null()

    @override
    def size(self) -> int:
        """Return the number of elements in the heap.

        Returns:
            The total number of elements in the heap.
        """
        return self._size

    def insert(self, value: T) -> PHeap[T]:
        """Insert a new element into the heap.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Args:
            value: The element to insert.

        Returns:
            A new heap containing the inserted element.
        """
        cand = PHeapNode(value, 0, PHeap.empty())
        new_heap = _heap_insert(cand, self)
        return PHeap(self._size + 1, new_heap._children)

    def merge(self, other: PHeap[T]) -> PHeap[T]:
        """Merge this heap with another heap.

        Time Complexity: O(log(m + n)) where m, n are sizes of the heaps
        Space Complexity: O(log(m + n)) for path copying

        Args:
            other: The heap to merge with this one.

        Returns:
            A new heap containing all elements from both heaps.
        """
        new_heap = _heap_merge(self, other)
        return PHeap(self._size + other._size, new_heap._children)

    def find_min(self) -> Optional[Tuple[T, PHeap[T]]]:
        """Find the minimum element in the heap.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            None if the heap is empty, otherwise a tuple containing:
            - The minimum element
            - A new heap with the minimum element removed
        """
        result = _heap_find_min(self)
        if result is None:
            return None
        value, remaining = result
        return (value, remaining)

    def delete_min(self) -> Optional[PHeap[T]]:
        """Remove the minimum element from the heap.

        Returns:
            None if the heap is empty, otherwise a new heap with the
            minimum element removed.
        """
        result = self.find_min()
        return None if result is None else result[1]

    @override
    def iter(self) -> Iterator[T]:
        """Iterate through the heap in ascending order.

        Yields:
            Elements in ascending order.
        """
        return _heap_iter(self)

    def fold[Z](self, fn: Callable[[Z, T], Z], acc: Z) -> Z:
        """Fold the heap from left to right with an accumulator.

        Args:
            fn: A function that takes an accumulator and element, returns new accumulator.
            acc: The initial accumulator value.

        Returns:
            The final accumulator value after processing all elements.
        """
        result = acc
        for item in self.iter():
            result = fn(result, item)
        return result

    def __add__(self, other: PHeap[T]) -> PHeap[T]:
        """Alias for merge()."""
        return self.merge(other)

    def __rshift__(self, value: T) -> PHeap[T]:
        """Alias for insert()."""
        return self.insert(value)

    def __rlshift__(self, value: T) -> PHeap[T]:
        """Alias for insert()."""
        return self.insert(value)


_HEAP_EMPTY: PHeap[Any] = PHeap(0, PSeq.empty())


def _calculate_node_size[T](node: PHeapNode[T]) -> int:
    """Calculate the size of a heap node (1 + size of its rest heap)."""
    return 1 + node.rest._size


def _heap_insert[T](cand: PHeapNode[T], heap: PHeap[T]) -> PHeap[T]:
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


def _heap_link[T](first: PHeapNode[T], second: PHeapNode[T]) -> PHeapNode[T]:
    match compare(first.value, second.value):
        case Ordering.Gt:
            new_rest_size = second.rest._size + _calculate_node_size(first)
            return PHeapNode(
                second.value,
                second.rank + 1,
                PHeap(new_rest_size, second.rest._children.cons(first)),
            )
        case _:
            new_rest_size = first.rest._size + _calculate_node_size(second)
            return PHeapNode(
                first.value,
                first.rank + 1,
                PHeap(new_rest_size, first.rest._children.cons(second)),
            )


def _heap_merge[T](first: PHeap[T], second: PHeap[T]) -> PHeap[T]:
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
                        tail = _heap_merge(PHeap(first_tail_size, first_tail), second)
                        return PHeap(
                            tail._size + _calculate_node_size(first_head),
                            tail._children.cons(first_head),
                        )
                    elif second_head.rank < first_head.rank:
                        second_tail_size = second._size - _calculate_node_size(
                            second_head
                        )
                        tail = _heap_merge(first, PHeap(second_tail_size, second_tail))
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
                        tail = _heap_merge(
                            PHeap(first_tail_size, first_tail),
                            PHeap(second_tail_size, second_tail),
                        )
                        return _heap_insert(head, tail)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _heap_find_min[T](
    heap: PHeap[T],
) -> Optional[Tuple[T, PHeap[T]]]:
    match heap._children.uncons():
        case None:
            return None
        case (head, tail):
            if tail.null():
                return (head.value, head.rest)
            else:
                tail_size = heap._size - _calculate_node_size(head)
                cand = _heap_find_min(PHeap(tail_size, tail))
                if cand is None or compare(head.value, cand[0]) != Ordering.Gt:
                    # Choose head when it's smaller or equal (prefer head for ties)
                    rest = _heap_merge(head.rest, PHeap(tail_size, tail))
                    return (head.value, rest)
                else:
                    # Only choose candidate when head is strictly greater
                    head_as_heap = PHeap(
                        _calculate_node_size(head), PSeq.singleton(head)
                    )
                    rest = _heap_merge(head_as_heap, cand[1])
                    return (cand[0], rest)
        case _:
            raise Impossible


def _heap_iter[T](heap: PHeap[T]) -> Iterator[T]:
    while not heap.null():
        min_result = heap.find_min()
        if min_result is None:
            break
        value, _ = min_result
        yield value

        delete_result = heap.delete_min()
        if delete_result is None:
            break
        heap = delete_result
