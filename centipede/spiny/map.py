"""Persistent map implementation based on weight-balanced trees"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Tuple, Type, override

from centipede.spiny.common import (
    Box,
    Impossible,
    LexComparable,
    Ordering,
    Sized,
    compare,
)

__all__ = ["PMap"]


# sealed
class PMap[K, V](Sized, LexComparable[Tuple[K, V], "PMap[K, V]"]):
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
        return _PMAP_EMPTY

    @staticmethod
    def singleton(key: K, value: V) -> PMap[K, V]:
        """Create a map containing a single key-value pair.

        Args:
            key: The key for the entry.
            value: The value for the entry.

        Returns:
            A map containing only the given key-value pair.
        """
        return PMapBranch(1, _PMAP_EMPTY, key, value, _PMAP_EMPTY)

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
        match self:
            case PMapEmpty():
                return True
            case PMapBranch():
                return False
            case _:
                raise Impossible

    @override
    def size(self) -> int:
        """Get the number of key-value pairs in the map."""
        match self:
            case PMapEmpty():
                return 0
            case PMapBranch(_size, _, _, _, _):
                return _size
            case _:
                raise Impossible

    @override
    def iter(self) -> Generator[Tuple[K, V]]:
        """Iterate over all key-value pairs in the map in key order."""
        match self:
            case PMapEmpty():
                return
            case PMapBranch(_, left, key, value, right):
                yield from left.iter()
                yield (key, value)
                yield from right.iter()

    def keys(self) -> Generator[K]:
        """Iterate over all keys in the map in sorted order."""
        for key, _ in self.iter():
            yield key

    def values(self) -> Generator[V]:
        """Iterate over all values in the map in key order."""
        for _, value in self.iter():
            yield value

    def items(self) -> Generator[Tuple[K, V]]:
        """Iterate over all key-value pairs in the map in key order."""
        yield from self.iter()

    def get(self, key: K) -> Optional[V]:
        """Get the value associated with a key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if the key is not found.
        """
        return _pmap_get(self, key)

    def contains(self, key: K) -> bool:
        """Check if the map contains the given key.

        Args:
            key: The key to check for.

        Returns:
            True if the key exists in the map, False otherwise.
        """
        return _pmap_contains(self, key)

    def put(self, key: K, value: V) -> PMap[K, V]:
        """Insert or update a key-value pair in the map.

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.

        Returns:
            A new map with the key-value pair inserted or updated.
        """
        return _pmap_put(self, key, value)

    def remove(self, key: K) -> PMap[K, V]:
        """Remove a key-value pair from the map.

        Args:
            key: The key to remove.

        Returns:
            A new map with the key-value pair removed.
        """
        return _pmap_remove(self, key)

    def merge(self, other: PMap[K, V]) -> PMap[K, V]:
        """Merge this map with another map.

        If both maps contain the same key, the value from this map takes precedence.

        Args:
            other: The map to merge with this one.

        Returns:
            A new map containing entries from both maps.
        """
        return _pmap_merge(self, other)

    def find_min(self) -> Optional[Tuple[K, V, PMap[K, V]]]:
        """Find the minimum key-value pair in the map.

        Returns:
            None if the map is empty, otherwise a tuple containing:
            - The minimum key
            - The value associated with the minimum key
            - A new map with the minimum entry removed
        """
        result = _pmap_find_min(self)
        if result is None:
            return None
        key, value, remaining = result
        return (key, value, remaining)

    def find_max(self) -> Optional[Tuple[PMap[K, V], K, V]]:
        """Find the maximum key-value pair in the map.

        Returns:
            None if the map is empty, otherwise a tuple containing:
            - A new map with the maximum entry removed
            - The maximum key
            - The value associated with the maximum key
        """
        result = _pmap_find_max(self)
        if result is None:
            return None
        remaining, key, value = result
        return (remaining, key, value)

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

    def split(self, pivot_key: K) -> Tuple[PMap[K, V], PMap[K, V]]:
        """Split a map into entries with keys smaller and larger than the pivot.

        Returns:
            A tuple (smaller, larger) where smaller contains entries with keys < pivot_key
            and larger contains entries with keys > pivot_key. Entries with the pivot key
            are excluded.
        """
        return _pmap_split(self, pivot_key)

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


@dataclass(frozen=True, eq=False)
class PMapEmpty[K, V](PMap[K, V]):
    pass


_PMAP_EMPTY: PMap[Any, Any] = PMapEmpty()


@dataclass(frozen=True, eq=False)
class PMapBranch[K, V](PMap[K, V]):
    _size: int
    _left: PMap[K, V]
    _key: K
    _value: V
    _right: PMap[K, V]


def _pmap_get[K, V](pmap: PMap[K, V], key: K) -> Optional[V]:
    match pmap:
        case PMapEmpty():
            return None
        case PMapBranch(_, left, branch_key, branch_value, right):
            cmp = compare(key, branch_key)
            if cmp == Ordering.Eq:
                return branch_value
            elif cmp == Ordering.Lt:
                return _pmap_get(left, key)
            else:
                return _pmap_get(right, key)
        case _:
            raise Impossible


def _pmap_contains[K, V](pmap: PMap[K, V], key: K) -> bool:
    match pmap:
        case PMapEmpty():
            return False
        case PMapBranch(_, left, branch_key, _, right):
            cmp = compare(key, branch_key)
            if cmp == Ordering.Eq:
                return True
            elif cmp == Ordering.Lt:
                return _pmap_contains(left, key)
            else:
                return _pmap_contains(right, key)
        case _:
            raise Impossible


def _pmap_put[K, V](pmap: PMap[K, V], key: K, value: V) -> PMap[K, V]:
    match pmap:
        case PMapEmpty():
            return PMapBranch(1, _PMAP_EMPTY, key, value, _PMAP_EMPTY)
        case PMapBranch(_, left, branch_key, branch_value, right):
            cmp = compare(key, branch_key)
            if cmp == Ordering.Lt:
                new_left = _pmap_put(left, key, value)
                return _pmap_balance(new_left, branch_key, branch_value, right)
            elif cmp == Ordering.Gt:
                new_right = _pmap_put(right, key, value)
                return _pmap_balance(left, branch_key, branch_value, new_right)
            else:
                # Key exists, update value
                return PMapBranch(pmap._size, left, key, value, right)
        case _:
            raise Impossible


def _pmap_remove[K, V](pmap: PMap[K, V], key: K) -> PMap[K, V]:
    match pmap:
        case PMapEmpty():
            return pmap
        case PMapBranch(_, left, branch_key, branch_value, right):
            cmp = compare(key, branch_key)
            if cmp == Ordering.Lt:
                new_left = _pmap_remove(left, key)
                return _pmap_balance(new_left, branch_key, branch_value, right)
            elif cmp == Ordering.Gt:
                new_right = _pmap_remove(right, key)
                return _pmap_balance(left, branch_key, branch_value, new_right)
            else:
                # Found the key to remove
                return _pmap_join(left, right)
        case _:
            raise Impossible


def _pmap_join[K, V](left: PMap[K, V], right: PMap[K, V]) -> PMap[K, V]:
    """Join two maps where all keys in left are smaller than all keys in right."""
    match (left, right):
        case (PMapEmpty(), _):
            return right
        case (_, PMapEmpty()):
            return left
        case (PMapBranch(), PMapBranch()):
            # Find the minimum in the right subtree to use as the new root
            min_result = _pmap_find_min(right)
            if min_result is None:
                raise Impossible
            min_key, min_value, new_right = min_result
            return _pmap_balance(left, min_key, min_value, new_right)
        case _:
            raise Impossible


def _pmap_merge[K, V](left_map: PMap[K, V], right_map: PMap[K, V]) -> PMap[K, V]:
    match (left_map, right_map):
        case (PMapEmpty(), _):
            return right_map
        case (_, PMapEmpty()):
            return left_map
        case (PMapBranch(_, left_left, left_key, left_value, left_right), _):
            # Split right_map around left_key and merge recursively
            smaller, larger = _pmap_split(right_map, left_key)
            merged_left = _pmap_merge(left_left, smaller)
            merged_right = _pmap_merge(left_right, larger)
            return _pmap_balance(merged_left, left_key, left_value, merged_right)
        case _:
            raise Impossible


def _pmap_split[K, V](pmap: PMap[K, V], pivot_key: K) -> Tuple[PMap[K, V], PMap[K, V]]:
    match pmap:
        case PMapEmpty():
            return (_PMAP_EMPTY, _PMAP_EMPTY)
        case PMapBranch(_, left, key, value, right):
            cmp = compare(pivot_key, key)
            if cmp == Ordering.Lt:
                # pivot_key < key, so this entry goes to larger side
                left_smaller, left_larger = _pmap_split(left, pivot_key)
                larger = _pmap_balance(left_larger, key, value, right)
                return (left_smaller, larger)
            elif cmp == Ordering.Gt:
                # pivot_key > key, so this entry goes to smaller side
                right_smaller, right_larger = _pmap_split(right, pivot_key)
                smaller = _pmap_balance(left, key, value, right_smaller)
                return (smaller, right_larger)
            else:
                # pivot_key == key, exclude this entry
                return (left, right)
        case _:
            raise Impossible


def _pmap_find_min[K, V](pmap: PMap[K, V]) -> Optional[Tuple[K, V, PMap[K, V]]]:
    match pmap:
        case PMapEmpty():
            return None
        case PMapBranch(_, left, key, value, right):
            if left.null():
                # This node contains the minimum key
                return (key, value, right)
            else:
                # Minimum is in the left subtree
                min_result = _pmap_find_min(left)
                if min_result is None:
                    raise Impossible
                min_key, min_value, new_left = min_result
                new_tree = _pmap_balance(new_left, key, value, right)
                return (min_key, min_value, new_tree)
        case _:
            raise Impossible


def _pmap_find_max[K, V](pmap: PMap[K, V]) -> Optional[Tuple[PMap[K, V], K, V]]:
    match pmap:
        case PMapEmpty():
            return None
        case PMapBranch(_, left, key, value, right):
            if right.null():
                # This node contains the maximum key
                return (left, key, value)
            else:
                # Maximum is in the right subtree
                max_result = _pmap_find_max(right)
                if max_result is None:
                    raise Impossible
                new_right, max_key, max_value = max_result
                new_tree = _pmap_balance(left, key, value, new_right)
                return (new_tree, max_key, max_value)
        case _:
            raise Impossible


def _pmap_balance[K, V](
    left: PMap[K, V], key: K, value: V, right: PMap[K, V]
) -> PMap[K, V]:
    left_size = left.size()
    right_size = right.size()
    total_size = left_size + 1 + right_size

    # Weight-balanced tree invariant: neither subtree should be more than
    # 3 times larger than the other
    if left_size > 3 * right_size:
        # Left is too heavy, need to rotate right
        match left:
            case PMapBranch(_, left_left, left_key, left_value, left_right):
                left_left_size = left_left.size()
                left_right_size = left_right.size()
                if left_left_size >= left_right_size:
                    # Single rotation right
                    return PMapBranch(
                        total_size,
                        left_left,
                        left_key,
                        left_value,
                        PMapBranch(
                            1 + left_right_size + right_size,
                            left_right,
                            key,
                            value,
                            right,
                        ),
                    )
                else:
                    # Double rotation left-right
                    match left_right:
                        case PMapBranch(
                            _,
                            left_right_left,
                            left_right_key,
                            left_right_value,
                            left_right_right,
                        ):
                            return PMapBranch(
                                total_size,
                                PMapBranch(
                                    1 + left_left_size + left_right_left.size(),
                                    left_left,
                                    left_key,
                                    left_value,
                                    left_right_left,
                                ),
                                left_right_key,
                                left_right_value,
                                PMapBranch(
                                    1 + left_right_right.size() + right_size,
                                    left_right_right,
                                    key,
                                    value,
                                    right,
                                ),
                            )
    elif right_size > 3 * left_size:
        # Right is too heavy, need to rotate left
        match right:
            case PMapBranch(_, right_left, right_key, right_value, right_right):
                right_left_size = right_left.size()
                right_right_size = right_right.size()
                if right_right_size >= right_left_size:
                    # Single rotation left
                    return PMapBranch(
                        total_size,
                        PMapBranch(
                            1 + left_size + right_left_size,
                            left,
                            key,
                            value,
                            right_left,
                        ),
                        right_key,
                        right_value,
                        right_right,
                    )
                else:
                    # Double rotation right-left
                    match right_left:
                        case PMapBranch(
                            _,
                            right_left_left,
                            right_left_key,
                            right_left_value,
                            right_left_right,
                        ):
                            return PMapBranch(
                                total_size,
                                PMapBranch(
                                    1 + left_size + right_left_left.size(),
                                    left,
                                    key,
                                    value,
                                    right_left_left,
                                ),
                                right_left_key,
                                right_left_value,
                                PMapBranch(
                                    1 + right_left_right.size() + right_right_size,
                                    right_left_right,
                                    right_key,
                                    right_value,
                                    right_right,
                                ),
                            )

    # Tree is balanced or no rotation needed
    return PMapBranch(total_size, left, key, value, right)
