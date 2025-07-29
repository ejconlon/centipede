"""Persistent map implementation based on weight-balanced trees"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
    override,
)

from centipede.spiny.common import (
    Box,
    Impossible,
    LexComparable,
    Ordering,
    Sized,
    compare,
)
from centipede.spiny.set import PSet, PSetBranch, PSetEmpty

__all__ = ["PMap"]


@dataclass(frozen=True)
class Missing:
    pass


_MISSING = Missing()


# sealed
class PMap[K, V](Sized, LexComparable[Tuple[K, V], "PMap[K, V]"]):
    @staticmethod
    def empty(
        _kty: Optional[Type[K]] = None, _vty: Optional[Type[V]] = None
    ) -> PMap[K, V]:
        """Create an empty map.

        Time Complexity: O(1)
        Space Complexity: O(1)

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

        Time Complexity: O(1)
        Space Complexity: O(1)

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

        Time Complexity: O(n log n) where n is the number of pairs
        Space Complexity: O(n) for the resulting tree

        Args:
            pairs: Iterable of (key, value) tuples.

        Returns:
            A map containing all the given key-value pairs.
        """
        box: Box[PMap[K, V]] = Box(PMap.empty())
        for key, value in pairs:
            box.value = box.value.put(key, value)
        return box.value

    @staticmethod
    def assoc(pset: PSet[K], value: V) -> PMap[K, V]:
        """Create a map by associating the same value with each key from a set.

        Time Complexity: O(n) for tree transformation
        Space Complexity: O(n) for new tree structure

        Args:
            pset: The set of keys to use.
            value: The value to associate with each key.

        Returns:
            A map containing all keys from the set, each associated with the given value.
        """
        return _pset_to_pmap_with_value(pset, value)

    @override
    def null(self) -> bool:
        """Check if the map is empty.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        match self:
            case PMapEmpty():
                return True
            case PMapBranch():
                return False
            case _:
                raise Impossible

    @override
    def size(self) -> int:
        """Get the number of key-value pairs in the map.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        match self:
            case PMapEmpty():
                return 0
            case PMapBranch(_size, _, _, _, _):
                return _size
            case _:
                raise Impossible

    @override
    def iter(self) -> Iterator[Tuple[K, V]]:
        """Iterate over all key-value pairs in the map in key order.

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        match self:
            case PMapEmpty():
                return
            case PMapBranch(_, left, key, value, right):
                yield from left.iter()
                yield (key, value)
                yield from right.iter()

    def keys(self) -> Iterator[K]:
        """Iterate over all keys in the map in sorted order.

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        for key, _ in self.iter():
            yield key

    def keys_set(self) -> PSet[K]:
        """Return a set containing all keys from the map.

        Time Complexity: O(n) for tree transformation
        Space Complexity: O(n) for new tree structure

        Returns:
            A PSet containing all keys from this map.
        """

        return _pmap_keys_to_pset(self)

    def values(self) -> Iterator[V]:
        """Iterate over all values in the map in key order.

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        for _, value in self.iter():
            yield value

    def items(self) -> Iterator[Tuple[K, V]]:
        """Iterate over all key-value pairs in the map in key order.

        Time Complexity: O(n) for complete iteration
        Space Complexity: O(log n) for recursion stack
        """
        yield from self.iter()

    def get(self, key: K, default: Union[V, Missing] = _MISSING) -> V:
        """Get the value associated with a key.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for recursion stack

        Args:
            key: The key to look up.
            default: Value to return if key is not found. If not provided and key
                    is not found, raises KeyError.

        Returns:
            The value associated with the key, or default if key is not found
            and default is provided.

        Raises:
            KeyError: If key is not found and no default is provided.
        """
        return _pmap_get_with_default(self, key, default)

    def lookup(self, key: K) -> Optional[V]:
        """Get the value associated with a key, returning None if not found.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for recursion stack

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if the key is not found.
        """
        try:
            return self.get(key)
        except KeyError:
            return None

    def contains(self, key: K) -> bool:
        """Check if the map contains the given key.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for recursion stack

        Args:
            key: The key to check for.

        Returns:
            True if the key exists in the map, False otherwise.
        """
        return _pmap_contains(self, key)

    def put(self, key: K, value: V) -> PMap[K, V]:
        """Insert or update a key-value pair in the map.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.

        Returns:
            A new map with the key-value pair inserted or updated.
        """
        return _pmap_put(self, key, value)

    def remove(self, key: K) -> PMap[K, V]:
        """Remove a key-value pair from the map.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Args:
            key: The key to remove.

        Returns:
            A new map with the key-value pair removed.
        """
        return _pmap_remove(self, key)

    def merge(self, other: PMap[K, V]) -> PMap[K, V]:
        """Merge this map with another map.

        Time Complexity: O(m log(n/m+1)) where m ≤ n are sizes of the maps
        Space Complexity: O(log(m + n)) for recursion and path copying

        If both maps contain the same key, the value from this map takes precedence.

        Args:
            other: The map to merge with this one.

        Returns:
            A new map containing entries from both maps.
        """
        return _pmap_merge(self, other)

    def find_min(self) -> Optional[Tuple[K, V, PMap[K, V]]]:
        """Find the minimum key-value pair in the map.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

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

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

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

    def split(self, pivot_key: K) -> Tuple[PMap[K, V], Optional[V], PMap[K, V]]:
        """Split a map into entries with keys smaller and larger than the pivot.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for path copying

        Returns:
            A tuple (smaller, pivot_value, larger) where:
            - smaller contains entries with keys < pivot_key
            - pivot_value is the value associated with pivot_key if it existed, None otherwise
            - larger contains entries with keys > pivot_key
            Entries with the pivot key are excluded from both smaller and larger.
        """
        return _pmap_split(self, pivot_key)

    def filter_keys(self, predicate: Callable[[K], bool]) -> PMap[K, V]:
        """Filter entries based on their keys.

        Args:
            predicate: A function that returns True for keys to keep.

        Returns:
            A new map containing only entries whose keys satisfy the predicate.
        """
        return _pmap_filter_keys(self, predicate)

    def map_values[W](self, fn: Callable[[V], W]) -> PMap[K, W]:
        """Transform each value using the given function while preserving structure.

        Args:
            fn: A function that transforms values from type V to type W.

        Returns:
            A new map with each value transformed by fn, preserving the tree structure.
        """
        return _pmap_map_values(self, fn)

    def fold_with_key[Z](self, fn: Callable[[Z, K, V], Z], acc: Z) -> Z:
        """Fold the map from left to right with an accumulator, key, and value.

        Time Complexity: O(n) for iteration plus cost of fn
        Space Complexity: O(log n) for recursion stack during iteration

        Args:
            fn: A function that takes an accumulator, key, and value, returns new accumulator.
            acc: The initial accumulator value.

        Returns:
            The final accumulator value after processing all key-value pairs.
        """
        result = acc
        for key, value in self.items():
            result = fn(result, key, value)
        return result

    def __rshift__(self, pair: Tuple[K, V]) -> PMap[K, V]:
        """Alias for put()."""
        key, value = pair
        return self.put(key, value)

    def __rlshift__(self, pair: Tuple[K, V]) -> PMap[K, V]:
        """Alias for put()."""
        key, value = pair
        return self.put(key, value)

    def __add__(self, other: PMap[K, V]) -> PMap[K, V]:
        """Alias for merge()."""
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


def _pmap_get_with_default[K, V](
    pmap: PMap[K, V], key: K, default: Union[V, Missing]
) -> V:
    match pmap:
        case PMapEmpty():
            if isinstance(default, Missing):
                raise KeyError(key)
            else:
                return default
        case PMapBranch(_, left, branch_key, branch_value, right):
            cmp = compare(key, branch_key)
            if cmp == Ordering.Eq:
                return branch_value
            elif cmp == Ordering.Lt:
                return _pmap_get_with_default(left, key, default)
            else:
                return _pmap_get_with_default(right, key, default)
        case _:
            raise Impossible


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
        case (PMapBranch(left_size, _, _, _, _), PMapBranch(right_size, _, _, _, _)):
            # Use the smaller map to split the larger one for better complexity
            if left_size <= right_size:
                return _pmap_merge_smaller_into_larger(
                    left_map, right_map, prefer_smaller=True
                )
            else:
                return _pmap_merge_smaller_into_larger(
                    right_map, left_map, prefer_smaller=False
                )
        case _:
            raise Impossible


def _pmap_merge_smaller_into_larger[K, V](
    smaller_map: PMap[K, V], larger_map: PMap[K, V], prefer_smaller: bool
) -> PMap[K, V]:
    """Merge by processing each entry of the smaller map against the larger map.

    This achieves O(m log(n/m+1)) complexity where m <= n are the sizes.

    Args:
        smaller_map: The smaller map to process
        larger_map: The larger map to split
        prefer_smaller: If True, prefer values from smaller_map on conflicts,
                       if False, prefer values from larger_map on conflicts
    """
    match smaller_map:
        case PMapEmpty():
            return larger_map
        case PMapBranch(_, left, key, value, right):
            # Split larger_map around key and merge recursively
            smaller_part, existing_value, larger_part = _pmap_split(larger_map, key)
            merged_left = _pmap_merge_smaller_into_larger(
                left, smaller_part, prefer_smaller
            )
            merged_right = _pmap_merge_smaller_into_larger(
                right, larger_part, prefer_smaller
            )

            # Choose which value to use based on preference
            final_value = (
                value if prefer_smaller or existing_value is None else existing_value
            )
            return _pmap_balance(merged_left, key, final_value, merged_right)
        case _:
            raise Impossible


def _pmap_split[K, V](
    pmap: PMap[K, V], pivot_key: K
) -> Tuple[PMap[K, V], Optional[V], PMap[K, V]]:
    match pmap:
        case PMapEmpty():
            return (_PMAP_EMPTY, None, _PMAP_EMPTY)
        case PMapBranch(_, left, key, value, right):
            cmp = compare(pivot_key, key)
            if cmp == Ordering.Lt:
                # pivot_key < key, so this entry goes to larger side
                left_smaller, pivot_value, left_larger = _pmap_split(left, pivot_key)
                larger = _pmap_balance(left_larger, key, value, right)
                return (left_smaller, pivot_value, larger)
            elif cmp == Ordering.Gt:
                # pivot_key > key, so this entry goes to smaller side
                right_smaller, pivot_value, right_larger = _pmap_split(right, pivot_key)
                smaller = _pmap_balance(left, key, value, right_smaller)
                return (smaller, pivot_value, right_larger)
            else:
                # pivot_key == key, found it and exclude this entry
                return (left, value, right)
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


def _pmap_keys_to_pset[K, V](pmap: PMap[K, V]) -> PSet[K]:
    """Convert a PMap to a PSet containing all its keys.

    This is implemented efficiently by transforming the tree structure directly
    rather than iterating through elements.
    """

    match pmap:
        case PMapEmpty():
            return PSetEmpty()
        case PMapBranch(_size, left, key, _, right):
            left_set = _pmap_keys_to_pset(left)
            right_set = _pmap_keys_to_pset(right)
            return PSetBranch(_size, left_set, key, right_set)
        case _:
            raise Impossible


def _pset_to_pmap_with_value[K, V](pset: PSet[K], value: V) -> PMap[K, V]:
    """Convert a PSet to a PMap by associating the same value with each key.

    This is implemented efficiently by transforming the tree structure directly
    rather than iterating through elements and inserting them one by one.
    """
    match pset:
        case PSetEmpty():
            return _PMAP_EMPTY
        case PSetBranch(_size, left, key, right):
            left_map = _pset_to_pmap_with_value(left, value)
            right_map = _pset_to_pmap_with_value(right, value)
            return PMapBranch(_size, left_map, key, value, right_map)
        case _:
            raise Impossible


def _pmap_filter_keys[K, V](
    pmap: PMap[K, V], predicate: Callable[[K], bool]
) -> PMap[K, V]:
    result: PMap[K, V] = PMap.empty()
    for key, value in pmap.iter():
        if predicate(key):
            result = result.put(key, value)
    return result


def _pmap_map_values[K, V, W](pmap: PMap[K, V], fn: Callable[[V], W]) -> PMap[K, W]:
    match pmap:
        case PMapEmpty():
            return PMap.empty()
        case PMapBranch(size, left, key, value, right):
            new_left = _pmap_map_values(left, fn)
            new_value = fn(value)
            new_right = _pmap_map_values(right, fn)
            return PMapBranch(size, new_left, key, new_value, new_right)
        case _:
            raise Impossible
