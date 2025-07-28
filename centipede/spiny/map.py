"""Persistent map implementation based on persistent sets"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple, Type, override

from centipede.spiny.common import (
    Box,
    Entry,
    LexComparable,
    Ordering,
    Sized,
    compare,
)
from centipede.spiny.set import PSet, PSetBranch, PSetEmpty

__all__ = ["PMap", "Entry"]


@dataclass(frozen=True, eq=False)
class PMap[K, V](Sized, LexComparable[Entry[K, V], "PMap[K, V]"]):
    """Persistent map implementation using a PSet of Entry objects internally."""

    _entries: PSet[Entry[K, V]]

    @staticmethod
    def empty(
        _kty: Optional[Type[K]] = None, _vty: Optional[Type[V]] = None
    ) -> PMap[K, V]:
        """Create an empty map.

        Args:
            _kty: Optional key type hint (unused).
            _vty: Optional value type hint (unused).

        Returns:
            An empty map instance.
        """
        return PMap(PSet.empty())

    @staticmethod
    def singleton(key: K, value: V) -> PMap[K, V]:
        """Create a map containing a single key-value pair.

        Args:
            key: The key for the entry.
            value: The value for the entry.

        Returns:
            A map containing only the given key-value pair.
        """
        entry = Entry(key, value)
        return PMap(PSet.singleton(entry))

    @staticmethod
    def mk(pairs: Iterable[Tuple[K, V]]) -> PMap[K, V]:
        """Create a map from an iterable of key-value pairs.

        Args:
            pairs: Iterable of (key, value) tuples.

        Returns:
            A map containing all the given key-value pairs.
        """
        box: Box[PMap[K, V]] = Box(PMap.empty())
        for key, value in pairs:
            box.value = box.value.put(key, value)
        return box.value

    @override
    def null(self) -> bool:
        """Check if the map is empty."""
        return self._entries.null()

    @override
    def size(self) -> int:
        """Get the number of key-value pairs in the map."""
        return self._entries.size()

    @override
    def iter(self) -> Generator[Entry[K, V]]:
        """Iterate over all entries in the map in key order."""
        yield from self._entries.iter()

    def keys(self) -> Generator[K]:
        """Iterate over all keys in the map in sorted order."""
        for entry in self._entries.iter():
            yield entry.key

    def values(self) -> Generator[V]:
        """Iterate over all values in the map in key order."""
        for entry in self._entries.iter():
            yield entry.value

    def items(self) -> Generator[Tuple[K, V]]:
        """Iterate over all key-value pairs in the map in key order."""
        for entry in self._entries.iter():
            yield (entry.key, entry.value)

    def get(self, key: K) -> Optional[V]:
        """Get the value associated with a key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if the key is not found.
        """
        return _map_get(self, key)

    def contains(self, key: K) -> bool:
        """Check if the map contains the given key.

        Args:
            key: The key to check for.

        Returns:
            True if the key exists in the map, False otherwise.
        """
        return _map_contains(self, key)

    def put(self, key: K, value: V) -> PMap[K, V]:
        """Insert or update a key-value pair in the map.

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.

        Returns:
            A new map with the key-value pair inserted or updated.
        """
        entry = Entry(key, value)
        # Remove any existing entry with this key, then insert the new one
        new_entries = _map_remove(self, key)._entries.insert(entry)
        return PMap(new_entries)

    def remove(self, key: K) -> PMap[K, V]:
        """Remove a key-value pair from the map.

        Args:
            key: The key to remove.

        Returns:
            A new map with the key-value pair removed.
        """
        return _map_remove(self, key)

    def merge(self, other: PMap[K, V]) -> PMap[K, V]:
        """Merge this map with another map.

        If both maps contain the same key, the value from this map takes precedence.

        Args:
            other: The map to merge with this one.

        Returns:
            A new map containing entries from both maps.
        """
        result = other
        for key, value in self.items():
            result = result.put(key, value)
        return result

    def find_min(self) -> Optional[Tuple[K, V, PMap[K, V]]]:
        """Find the minimum key-value pair in the map.

        Returns:
            None if the map is empty, otherwise a tuple containing:
            - The minimum key
            - The value associated with the minimum key
            - A new map with the minimum entry removed
        """
        result = self._entries.find_min()
        if result is None:
            return None
        entry, remaining_entries = result
        return (entry.key, entry.value, PMap(remaining_entries))

    def find_max(self) -> Optional[Tuple[PMap[K, V], K, V]]:
        """Find the maximum key-value pair in the map.

        Returns:
            None if the map is empty, otherwise a tuple containing:
            - A new map with the maximum entry removed
            - The maximum key
            - The value associated with the maximum key
        """
        result = self._entries.find_max()
        if result is None:
            return None
        remaining_entries, entry = result
        return (PMap(remaining_entries), entry.key, entry.value)

    def delete_min(self) -> Optional[PMap[K, V]]:
        """Remove the minimum key-value pair from the map.

        Returns:
            None if the map is empty, otherwise a new map with the
            minimum entry removed.
        """
        result = self.find_min()
        return None if result is None else result[2]

    def delete_max(self) -> Optional[PMap[K, V]]:
        """Remove the maximum key-value pair from the map.

        Returns:
            None if the map is empty, otherwise a new map with the
            maximum entry removed.
        """
        result = self.find_max()
        return None if result is None else result[0]

    def __rshift__(self, pair: Tuple[K, V]) -> PMap[K, V]:
        """Insert key-value pair using >> operator.

        Args:
            pair: A (key, value) tuple to insert.

        Returns:
            A new map with the key-value pair inserted.
        """
        key, value = pair
        return self.put(key, value)

    def __rlshift__(self, pair: Tuple[K, V]) -> PMap[K, V]:
        """Insert key-value pair using << operator.

        Args:
            pair: A (key, value) tuple to insert.

        Returns:
            A new map with the key-value pair inserted.
        """
        key, value = pair
        return self.put(key, value)

    def __add__(self, other: PMap[K, V]) -> PMap[K, V]:
        """Merge maps using + operator.

        Args:
            other: The map to merge with this one.

        Returns:
            A new map containing entries from both maps.
        """
        return self.merge(other)


def _map_get[K, V](pmap: PMap[K, V], key: K) -> Optional[V]:
    current = pmap._entries
    while True:
        match current:
            case PSetEmpty():
                return None
            case PSetBranch(_, left, entry, right):
                cmp = compare(key, entry.key)
                if cmp == Ordering.Eq:
                    return entry.value
                elif cmp == Ordering.Lt:
                    current = left
                else:
                    current = right
            case _:
                return None


def _map_contains[K, V](pmap: PMap[K, V], key: K) -> bool:
    current = pmap._entries
    while True:
        match current:
            case PSetEmpty():
                return False
            case PSetBranch(_, left, entry, right):
                cmp = compare(key, entry.key)
                if cmp == Ordering.Eq:
                    return True
                elif cmp == Ordering.Lt:
                    current = left
                else:
                    current = right
            case _:
                return False


def _map_remove[K, V](pmap: PMap[K, V], key: K) -> PMap[K, V]:
    # HACK Since Entry comparison won't even look at value,
    # we can fill in whatever to use as a pivot
    dummy_entry = Entry(key, None)
    smaller, larger = pmap._entries.split(dummy_entry)  # type: ignore
    return PMap(smaller.merge(larger))
