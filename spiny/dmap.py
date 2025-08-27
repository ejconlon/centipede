from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, Optional, Self, Type, Union

from spiny.map import PMap


@dataclass(frozen=True)
class Missing:
    pass


_MISSING = Missing()


class DKey[K, V](metaclass=ABCMeta):
    @classmethod
    def instance(cls) -> Self:
        return cls()

    @classmethod
    def key(cls) -> str:
        return cls.__name__


@dataclass(frozen=True)
class DValue[V]:
    value: V


class DMap[K]:
    def __init__(self):
        self._map: PMap[str, Any] = PMap.empty()

    @staticmethod
    def empty(_kty: Optional[Type[K]] = None) -> DMap[K]:
        """Create an empty DMap.

        Returns:
            A new empty DMap
        """
        return DMap[K]()

    @staticmethod
    def singleton[V](dkey: DKey[K, V], value: V) -> DMap[K]:
        """Create a DMap containing a single key-value pair.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            dkey: The dependent key for the entry.
            value: The value for the entry.

        Returns:
            A DMap containing only the given key-value pair.
        """
        new_dmap = DMap[K]()
        new_dmap._map = PMap.singleton(dkey.key(), DValue(value))
        return new_dmap

    def lookup[V](self, dkey: DKey[K, V]) -> Optional[V]:
        """Lookup a value by its dependent key.

        Args:
            dkey: The dependent key to lookup

        Returns:
            The value associated with the key, or None if not found
        """
        dvalue = self._map.lookup(dkey.key())
        return dvalue.value if dvalue is not None else None

    def get[V](self, dkey: DKey[K, V], default: Union[V, Missing] = _MISSING) -> V:
        """Get the value associated with a dependent key.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for recursion stack

        Args:
            dkey: The dependent key to look up.
            default: Value to return if key is not found. If not provided and key
                    is not found, raises KeyError.

        Returns:
            The value associated with the key, or default if key is not found
            and default is provided.

        Raises:
            KeyError: If key is not found and no default is provided.
        """
        dvalue = self._map.lookup(dkey.key())
        if dvalue is None:
            if isinstance(default, Missing):
                raise KeyError(dkey.key())
            else:
                return default
        else:
            return dvalue.value

    def put[V](self, dkey: DKey[K, V], value: V) -> DMap[K]:
        """Store a value with its dependent key.

        Args:
            dkey: The dependent key
            value: The value to store

        Returns:
            A new DMap with the key-value pair added
        """
        new_dmap = DMap[K]()
        new_dmap._map = self._map.put(dkey.key(), DValue(value))
        return new_dmap

    def contains[V](self, dkey: DKey[K, V]) -> bool:
        """Check if the map contains the given dependent key.

        Args:
            dkey: The dependent key to check for

        Returns:
            True if the key exists in the map, False otherwise
        """
        return self._map.contains(dkey.key())

    def remove[V](self, dkey: DKey[K, V]) -> DMap[K]:
        """Remove a key-value pair from the map.

        Args:
            dkey: The dependent key to remove

        Returns:
            A new DMap with the key-value pair removed
        """
        new_dmap = DMap[K]()
        new_dmap._map = self._map.remove(dkey.key())
        return new_dmap

    def null(self) -> bool:
        """Check if the map is empty.

        Returns:
            True if the map contains no entries, False otherwise
        """
        return self._map.null()

    def size(self) -> int:
        """Get the number of key-value pairs in the map.

        Returns:
            The number of entries in the map
        """
        return self._map.size()
