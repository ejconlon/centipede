"""Fixed-size array implementation backed by PMap"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, override

from centipede.spiny.common import LexComparable, Sized
from centipede.spiny.map import PMap

__all__ = ["Array"]


@dataclass(frozen=True, eq=False)
class Array[T](Sized, LexComparable[T, "Array[T]"]):
    """Fixed-size array backed by a PMap.

    Acts like a fixed-size array but is implemented using a PMap for efficiency.
    Supports get/set operations with bounds checking and resizing.
    """

    _size: int
    _fill: T
    _pmap: PMap[int, T]

    @staticmethod
    def new(size: int, fill: T) -> Array[T]:
        """Create a new Array with the given size and fill element.

        Args:
            size: The size of the array (must be >= 0)
            fill: The element to return for unset indices

        Raises:
            ValueError: If size is negative
        """
        if size < 0:
            raise ValueError("Array size must be non-negative")
        return Array(_size=size, _fill=fill, _pmap=PMap.empty())

    @override
    def size(self) -> int:
        """Get the size of the array.

        Returns:
            The current size of the array
        """
        return self._size

    @override
    def null(self) -> bool:
        """Check if the array has size 0.

        Returns:
            True if the array has size 0, False otherwise
        """
        return self._size == 0

    @override
    def iter(self) -> Iterator[T]:
        """Iterate over all elements in the array.

        Yields elements in index order from 0 to size-1.
        """
        for i in range(self._size):
            value = self.lookup(i)
            yield value if value is not None else self._fill

    def get(self, index: int) -> T:
        """Get the element at the given index.

        Args:
            index: The index to get (must be in range [0, size-1])
            default: Value to return if index is not found in the PMap.
                    If not provided, returns fill for valid indices.

        Returns:
            The element at the index, the default if provided and index not in PMap,
            or the fill element if no default provided and index not in PMap.

        Raises:
            KeyError: If index is outside the valid range
        """
        if not (0 <= index < self._size):
            raise KeyError(
                f"Index {index} out of bounds for array of size {self._size}"
            )

        return self._pmap.get(index, self._fill)

    def lookup(self, index: int) -> Optional[T]:
        """Get the element at the given index, returning None if not found.

        Args:
            index: The index to get (must be in range [0, size-1])

        Returns:
            The element at the index if found in the PMap, None if index is
            valid but not in the PMap, or raises KeyError if index is invalid.

        Raises:
            KeyError: If index is outside the valid range
        """
        if not (0 <= index < self._size):
            raise KeyError(
                f"Index {index} out of bounds for array of size {self._size}"
            )

        return self._pmap.lookup(index)

    def set(self, index: int, value: T) -> Array[T]:
        """Set the element at the given index.

        Args:
            index: The index to set (must be in range [0, size-1])
            value: The value to set at the index

        Returns:
            A new Array with the value set at the index

        Raises:
            KeyError: If index is outside the valid range
        """
        if not (0 <= index < self._size):
            raise KeyError(
                f"Index {index} out of bounds for array of size {self._size}"
            )

        return Array(
            _size=self._size, _fill=self._fill, _pmap=self._pmap.put(index, value)
        )

    def resize(self, new_size: int) -> Array[T]:
        """Resize the array to a new size.

        If the new size is larger, new indices will return the fill element.
        If the new size is smaller, indices >= new_size will be removed.

        Args:
            new_size: The new size for the array (must be >= 0)

        Returns:
            A new Array with the specified size

        Raises:
            ValueError: If new_size is negative
        """
        if new_size < 0:
            raise ValueError("Array size must be non-negative")

        # Copy existing values that are still within bounds
        pmap = self._pmap
        if new_size > 0:
            for index, value in self._pmap.items():
                if 0 <= index < new_size:
                    pmap = pmap.put(index, value)

        return Array(_size=new_size, _fill=self._fill, _pmap=pmap)

    def __getitem__(self, index: int) -> T:
        """Get element at index using array[index] syntax."""
        return self.get(index)

    def __len__(self) -> int:
        """Get the size of the array using len(array) syntax."""
        return self._size

    def __iter__(self) -> Iterator[T]:
        """Support iteration using for loops."""
        return self.iter()
