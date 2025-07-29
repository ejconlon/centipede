"""Persistent heap-based priority map implementation wrapping PHeap[Entry[K, V]]"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Tuple, Type, override

from centipede.spiny.common import (
    Entry,
    LexComparable,
    Sized,
)
from centipede.spiny.heap import PHeap

__all__ = ["PHeapMap"]


@dataclass(frozen=True, eq=False)
class PHeapMap[K, V](Sized, LexComparable[Tuple[K, V], "PHeapMap[K, V]"]):
    """Persistent heap-based priority map implementation.

    This wraps a PHeap[Entry[K, V]] to provide a map interface where entries
    are ordered by key using heap ordering (minimum key at the top).
    """

    _heap: PHeap[Entry[K, V]]

    @staticmethod
    def empty(
        _kty: Optional[Type[K]] = None, _vty: Optional[Type[V]] = None
    ) -> PHeapMap[K, V]:
        """Create an empty heap map.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            _kty: Optional key type hint (unused).
            _vty: Optional value type hint (unused).

        Returns:
            An empty heap map instance.
        """
        return PHeapMap(PHeap.empty())

    @staticmethod
    def singleton(key: K, value: V) -> PHeapMap[K, V]:
        """Create a heap map containing a single key-value pair.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            key: The key for the entry.
            value: The value for the entry.

        Returns:
            A heap map containing only the given key-value pair.
        """
        entry = Entry(key, value)
        return PHeapMap(PHeap.singleton(entry))

    @staticmethod
    def mk(pairs: Iterable[Tuple[K, V]]) -> PHeapMap[K, V]:
        """Create a heap map from an iterable of key-value pairs.

        Time Complexity: O(n log n) where n is the number of pairs
        Space Complexity: O(log n) for the heap structure

        Args:
            pairs: Iterable of (key, value) tuples.

        Returns:
            A heap map containing all the given key-value pairs.
        """
        entries = [Entry(key, value) for key, value in pairs]
        return PHeapMap(PHeap.mk(entries))

    @override
    def null(self) -> bool:
        """Check if the heap map is empty."""
        return self._heap.null()

    @override
    def size(self) -> int:
        """Get the number of key-value pairs in the heap map."""
        return self._heap.size()

    @override
    def iter(self) -> Iterator[Tuple[K, V]]:
        """Iterate over all key-value pairs in the heap map in heap order (sorted by key).

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        for entry in self._heap.iter():
            yield (entry.key, entry.value)

    def keys(self) -> Iterator[K]:
        """Iterate over all keys in the heap map in sorted order.

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        for key, _ in self.iter():
            yield key

    def values(self) -> Iterator[V]:
        """Iterate over all values in the heap map in key order.

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        for _, value in self.iter():
            yield value

    def insert(self, key: K, value: V) -> PHeapMap[K, V]:
        """Insert a key-value pair into the heap map.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Note: Unlike regular maps, heap maps can contain duplicate keys.
        This operation always adds a new entry.

        Args:
            key: The key to insert.
            value: The value to associate with the key.

        Returns:
            A new heap map with the key-value pair inserted.
        """
        entry = Entry(key, value)
        return PHeapMap(self._heap.insert(entry))

    def find_min(self) -> Optional[Tuple[K, V, PHeapMap[K, V]]]:
        """Find the minimum key-value pair in the heap map.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            None if the heap map is empty, otherwise a tuple containing:
            - The minimum key
            - The value associated with the minimum key
            - A new heap map with the minimum entry removed
        """
        result = self._heap.find_min()
        if result is None:
            return None
        min_entry, new_heap = result
        return (min_entry.key, min_entry.value, PHeapMap(new_heap))

    def delete_min(self) -> Optional[PHeapMap[K, V]]:
        """Remove the minimum key-value pair from the heap map.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            None if the heap map is empty, otherwise a new heap map with the
            minimum entry removed.
        """
        result = self._heap.find_min()
        if result is None:
            return None
        _, new_heap = result
        return PHeapMap(new_heap)

    def merge(self, other: PHeapMap[K, V]) -> PHeapMap[K, V]:
        """Merge this heap map with another heap map.

        Time Complexity: O(log(m + n)) where m, n are sizes of the heaps
        Space Complexity: O(log(m + n)) for path copying

        Args:
            other: The heap map to merge with this one.

        Returns:
            A new heap map containing entries from both heap maps.
        """
        return PHeapMap(self._heap.merge(other._heap))

    def fold_with_key[Z](self, fn: Callable[[Z, K, V], Z], acc: Z) -> Z:
        """Fold the heap map from left to right with an accumulator, key, and value.

        Time Complexity: O(n) for iteration plus cost of fn
        Space Complexity: O(log n) for recursion stack during iteration

        Args:
            fn: A function that takes an accumulator, key, and value, returns new accumulator.
            acc: The initial accumulator value.

        Returns:
            The final accumulator value after processing all key-value pairs.
        """
        result = acc
        for key, value in self.iter():
            result = fn(result, key, value)
        return result

    def filter_keys(self, predicate: Callable[[K], bool]) -> PHeapMap[K, V]:
        """Filter entries based on their keys.

        Time Complexity: O(n + m log m) where n is total entries, m is filtered entries
        Space Complexity: O(log m) for the resulting heap structure

        Args:
            predicate: A function that returns True for keys to keep.

        Returns:
            A new heap map containing only entries whose keys satisfy the predicate.
        """
        # Collect filtered entries
        filtered_entries = []
        for entry in self._heap.iter():
            if predicate(entry.key):
                filtered_entries.append(entry)

        # Create new heap from filtered entries
        return PHeapMap(PHeap.mk(filtered_entries))

    def map_values[W](self, fn: Callable[[V], W]) -> PHeapMap[K, W]:
        """Transform each value using the given function while preserving heap structure.

        Time Complexity: O(n log n) where n is the number of entries
        Space Complexity: O(log n) for the resulting heap structure

        Args:
            fn: A function that transforms values from type V to type W.

        Returns:
            A new heap map with each value transformed by fn, preserving the heap order.
        """
        # Collect transformed entries
        transformed_entries = []
        for entry in self._heap.iter():
            new_entry = Entry(entry.key, fn(entry.value))
            transformed_entries.append(new_entry)

        # Create new heap from transformed entries
        return PHeapMap(PHeap.mk(transformed_entries))

    def __rshift__(self, pair: Tuple[K, V]) -> PHeapMap[K, V]:
        """Alias for insert()."""
        key, value = pair
        return self.insert(key, value)

    def __rlshift__(self, pair: Tuple[K, V]) -> PHeapMap[K, V]:
        """Alias for insert()."""
        key, value = pair
        return self.insert(key, value)

    def __add__(self, other: PHeapMap[K, V]) -> PHeapMap[K, V]:
        """Alias for merge()."""
        return self.merge(other)
