"""Persistent isomorphic map implementation.

A PIsoMap maintains two PMap instances to provide efficient bidirectional
lookup between keys and values. All operations maintain the invariant that
the forward and backward maps are consistent with each other.
"""

from __future__ import annotations

from typing import Iterator, Optional, TypeVar

from spiny.map import PMap

K = TypeVar("K")
V = TypeVar("V")


class PIsoMap[K, V]:
    """A persistent bidirectional map backed by two PMap instances.

    Maintains forward (K -> V) and backward (V -> K) mappings that are
    kept in sync for efficient bidirectional lookup.

    All operations return new instances, preserving immutability.
    """

    def __init__(self, fwd: PMap[K, V], bwd: PMap[V, K]) -> None:
        """Initialize with existing forward and backward maps.

        Args:
            fwd: Forward mapping from keys to values
            bwd: Backward mapping from values to keys

        Note:
            The maps must be consistent with each other.
            Use mk() or empty() for safe construction.
        """
        self._fwd = fwd
        self._bwd = bwd

    @classmethod
    def empty(cls) -> PIsoMap[K, V]:
        """Create an empty bidirectional map."""
        return cls(PMap.empty(), PMap.empty())

    @classmethod
    def mk(cls, pairs: list[tuple[K, V]]) -> PIsoMap[K, V]:
        """Create a bidirectional map from key-value pairs.

        Args:
            pairs: List of (key, value) tuples

        Returns:
            New PIsoMap instance

        Note:
            If there are duplicate keys or values, later pairs take precedence.
        """
        result = cls.empty()
        for k, v in pairs:
            result = result.insert(k, v)
        return result

    def get_fwd(self, key: K) -> Optional[V]:
        """Look up value by key (forward direction).

        Args:
            key: Key to look up

        Returns:
            Associated value if found, None otherwise
        """
        return self._fwd.lookup(key)

    def get_bwd(self, value: V) -> Optional[K]:
        """Look up key by value (backward direction).

        Args:
            value: Value to look up

        Returns:
            Associated key if found, None otherwise
        """
        return self._bwd.lookup(value)

    def insert(self, key: K, value: V) -> PIsoMap[K, V]:
        """Insert a key-value pair, returning a new instance.

        Args:
            key: Key to insert
            value: Value to insert

        Returns:
            New PIsoMap with the pair inserted

        Note:
            If the key or value already exists, the old mappings are removed
            to maintain bidirectional consistency.
        """
        # Remove any existing mappings for this key or value
        result = self.remove_key(key).remove_value(value)

        # Insert the new mapping in both directions
        new_fwd = result._fwd.put(key, value)
        new_bwd = result._bwd.put(value, key)

        return PIsoMap(new_fwd, new_bwd)

    def remove_key(self, key: K) -> PIsoMap[K, V]:
        """Remove a mapping by key, returning a new instance.

        Args:
            key: Key to remove

        Returns:
            New PIsoMap with the mapping removed
        """
        value = self._fwd.lookup(key)
        if value is None:
            return self

        new_fwd = self._fwd.remove(key)
        new_bwd = self._bwd.remove(value)

        return PIsoMap(new_fwd, new_bwd)

    def remove_value(self, value: V) -> PIsoMap[K, V]:
        """Remove a mapping by value, returning a new instance.

        Args:
            value: Value to remove

        Returns:
            New PIsoMap with the mapping removed
        """
        key = self._bwd.lookup(value)
        if key is None:
            return self

        new_fwd = self._fwd.remove(key)
        new_bwd = self._bwd.remove(value)

        return PIsoMap(new_fwd, new_bwd)

    def contains_key(self, key: K) -> bool:
        """Check if a key exists in the map.

        Args:
            key: Key to check

        Returns:
            True if the key exists, False otherwise
        """
        return self._fwd.contains(key)

    def contains_value(self, value: V) -> bool:
        """Check if a value exists in the map.

        Args:
            value: Value to check

        Returns:
            True if the value exists, False otherwise
        """
        return self._bwd.contains(value)

    def keys(self) -> Iterator[K]:
        """Iterate over all keys."""
        return self._fwd.keys()

    def values(self) -> Iterator[V]:
        """Iterate over all values."""
        return self._bwd.keys()

    def items(self) -> Iterator[tuple[K, V]]:
        """Iterate over all key-value pairs."""
        return self._fwd.items()

    def size(self) -> int:
        """Get the number of mappings in the map."""
        return self._fwd.size()

    def is_empty(self) -> bool:
        """Check if the map is empty."""
        return self._fwd.null()

    def __len__(self) -> int:
        """Get the number of mappings (Python len() support)."""
        return self.size()

    def __bool__(self) -> bool:
        """Check if the map is non-empty (Python bool() support)."""
        return not self.is_empty()

    def __contains__(self, key: K) -> bool:
        """Check if a key exists (Python 'in' operator support)."""
        return self.contains_key(key)

    def __getitem__(self, key: K) -> V:
        """Get value by key (Python bracket operator support).

        Args:
            key: Key to look up

        Returns:
            Associated value

        Raises:
            KeyError: If the key is not found
        """
        value = self.get_fwd(key)
        if value is None:
            raise KeyError(key)
        return value

    def __repr__(self) -> str:
        """String representation of the map."""
        items = list(self.items())
        return f"PIsoMap({items})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another PIsoMap."""
        if not isinstance(other, PIsoMap):
            return NotImplemented
        return self._fwd == other._fwd and self._bwd == other._bwd
