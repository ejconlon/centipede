"""Persistent sequence implementation using Hinze-Patterson finger trees.

This module provides a functional sequence data structure that supports
efficient access to both ends, concatenation, splitting, and random access
while maintaining persistence (immutability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    override,
)

from centipede.spiny.common import Box, Impossible, LexComparable, Sized

__all__ = ["PSeq"]


type OuterNode[T] = Union[Tuple[T], Tuple[T, T], Tuple[T, T, T], Tuple[T, T, T, T]]
"""Type alias for nodes at the outer edges of the finger tree (1-4 elements)."""

type InnerNode[T] = Union[Tuple[T, T], Tuple[T, T, T]]
"""Type alias for internal nodes in the finger tree (2-3 elements)."""


# sealed
class PSeq[T](Sized, LexComparable[T, "PSeq[T]"]):
    """A Hinze-Patterson finger tree as persistent catenable sequence"""

    @staticmethod
    def empty(_ty: Optional[Type[T]] = None) -> PSeq[T]:
        """Create an empty sequence.

        Args:
            _ty: Optional type hint (unused).

        Returns:
            An empty sequence instance.
        """
        return _PSEQ_EMPTY

    @staticmethod
    def singleton(value: T) -> PSeq[T]:
        """Create a sequence containing a single element.

        Args:
            value: The single element for the sequence.

        Returns:
            A sequence containing only the given element.
        """
        return PSeqSingle(value)

    @staticmethod
    def mk(values: Iterable[T]) -> PSeq[T]:
        """Create a sequence from an iterable of values.

        Args:
            values: Iterable of values to include in the sequence.

        Returns:
            A sequence containing all the given values in order.
        """
        box: Box[PSeq[T]] = Box(PSeq.empty())
        for value in values:
            box.value = box.value.snoc(value)
        return box.value

    @override
    def null(self) -> bool:
        """Check if the sequence is empty.

        Returns:
            True if the sequence contains no elements, False otherwise.
        """
        match self:
            case PSeqEmpty():
                return True
            case _:
                return False

    @override
    def size(self) -> int:
        """Get the number of elements in the sequence.

        Returns:
            The number of elements in the sequence.
        """
        match self:
            case PSeqEmpty():
                return 0
            case PSeqSingle(_):
                return 1
            case PSeqDeep(size, _, _, _):
                return size
            case _:
                raise Impossible

    def uncons(self) -> Optional[Tuple[T, PSeq[T]]]:
        """Remove and return the first element and remaining sequence.

        Returns:
            None if sequence is empty, otherwise (first_element, rest_of_sequence).
        """
        return _seq_uncons(self)

    def cons(self, value: T) -> PSeq[T]:
        """Add an element to the front of the sequence.

        Args:
            value: The element to add.

        Returns:
            A new sequence with the element prepended.
        """
        return _seq_cons(value, self)

    def unsnoc(self) -> Optional[Tuple[PSeq[T], T]]:
        """Remove and return the last element and remaining sequence.

        Returns:
            None if sequence is empty, otherwise (rest_of_sequence, last_element).
        """
        return _seq_unsnoc(self)

    def snoc(self, value: T) -> PSeq[T]:
        """Add an element to the end of the sequence.

        Args:
            value: The element to add.

        Returns:
            A new sequence with the element appended.
        """
        return _seq_snoc(self, value)

    def concat(self, other: PSeq[T]) -> PSeq[T]:
        """Concatenate this sequence with another sequence.

        Args:
            other: The sequence to concatenate with this one.

        Returns:
            A new sequence containing elements from both sequences.
        """
        return _seq_concat(self, other)

    def get(self, ix: int) -> T:
        """Get the element at the specified index.

        Args:
            ix: The index of the element to retrieve.

        Returns:
            The element at the given index.

        Raises:
            KeyError: If the index is out of bounds.
        """
        return _seq_get(self, ix)

    def lookup(self, ix: int) -> Optional[T]:
        """Get the element at the specified index, returning None if out of bounds.

        Args:
            ix: The index of the element to retrieve.

        Returns:
            The element at the given index, or None if index is out of bounds.
        """
        try:
            return _seq_get(self, ix)
        except KeyError:
            return None

    def update(self, ix: int, value: T) -> PSeq[T]:
        """Return a new sequence with the element at the given index updated.

        Args:
            ix: The index of the element to update.
            value: The new value for the element.

        Returns:
            A new sequence with the specified element updated.
        """
        return _seq_update(self, ix, value)

    @override
    def iter(self) -> Generator[T]:
        """Return a generator that yields elements in order.

        Returns:
            A generator yielding elements from first to last.
        """
        return _seq_iter(self)

    def reversed(self) -> Generator[T]:
        """Return a generator that yields elements in reverse order.

        Returns:
            A generator yielding elements from last to first.
        """
        return _seq_reversed(self)

    def __getitem__(self, ix: int) -> T:
        """Alias for get()."""
        return self.get(ix)

    def __rshift__(self, value: T) -> PSeq[T]:
        """Alias for snoc()."""
        return self.snoc(value)

    def __rlshift__(self, value: T) -> PSeq[T]:
        """Alias for cons()."""
        return self.cons(value)

    def __add__(self, other: PSeq[T]) -> PSeq[T]:
        """Alias for concat()."""
        return self.concat(other)

    def __reversed__(self) -> Generator[T]:
        """Alias for reversed()."""
        return self.reversed()


@dataclass(frozen=True, eq=False)
class PSeqEmpty[T](PSeq[T]):
    """Internal representation of an empty sequence.

    This class represents the base case in the finger tree structure.
    """

    pass


_PSEQ_EMPTY: PSeq[Any] = PSeqEmpty()


@dataclass(frozen=True, eq=False)
class PSeqSingle[T](PSeq[T]):
    """Internal representation of a single-element sequence.

    Attributes:
        _value: The single element contained in this sequence.
    """

    _value: T


@dataclass(frozen=True, eq=False)
class PSeqDeep[T](PSeq[T]):
    """Internal representation of a multi-element sequence using finger trees.

    This represents the main structure with front and back digits and
    a recursive sequence of internal nodes in between.

    Attributes:
        _size: Total number of elements in the sequence.
        _front: Elements at the front (1-4 items).
        _between: Recursive sequence of internal nodes.
        _back: Elements at the back (1-4 items).
    """

    _size: int
    _front: OuterNode[T]
    _between: PSeq[InnerNode[T]]
    _back: OuterNode[T]


def _seq_uncons[T](seq: PSeq[T]) -> Optional[Tuple[T, PSeq[T]]]:
    match seq:
        case PSeqEmpty():
            return None
        case PSeqSingle(value):
            return (value, PSeq.empty())
        case PSeqDeep(size, front, between, back):
            match front:
                case (a,):
                    if between.null():
                        match back:
                            case (b,):
                                return (a, PSeqSingle(b))
                            case (b, c):
                                return (
                                    a,
                                    PSeqDeep(size - 1, (b,), PSeq.empty(), (c,)),
                                )
                            case (b, c, d):
                                return (
                                    a,
                                    PSeqDeep(size - 1, (b, c), PSeq.empty(), (d,)),
                                )
                            case (b, c, d, e):
                                return (
                                    a,
                                    PSeqDeep(size - 1, (b, c, d), PSeq.empty(), (e,)),
                                )
                            case _:
                                raise Impossible
                    else:
                        between_uncons = _seq_uncons(between)
                        if between_uncons is None:
                            match back:
                                case (b,):
                                    return (a, PSeqSingle(b))
                                case (b, c):
                                    return (
                                        a,
                                        PSeqDeep(size - 1, (b,), PSeq.empty(), (c,)),
                                    )
                                case (b, c, d):
                                    return (
                                        a,
                                        PSeqDeep(size - 1, (b, c), PSeq.empty(), (d,)),
                                    )
                                case (b, c, d, e):
                                    return (
                                        a,
                                        PSeqDeep(
                                            size - 1, (b, c, d), PSeq.empty(), (e,)
                                        ),
                                    )
                                case _:
                                    raise Impossible
                        else:
                            inner_head, between_tail = between_uncons
                            return (
                                a,
                                PSeqDeep(size - 1, inner_head, between_tail, back),
                            )
                case (a, b):
                    return (a, PSeqDeep(size - 1, (b,), between, back))
                case (a, b, c):
                    return (a, PSeqDeep(size - 1, (b, c), between, back))
                case (a, b, c, d):
                    return (a, PSeqDeep(size - 1, (b, c, d), between, back))
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_cons[T](value: T, seq: PSeq[T]) -> PSeq[T]:
    match seq:
        case PSeqEmpty():
            return PSeqSingle(value)
        case PSeqSingle(existing_value):
            return PSeqDeep(2, (value,), PSeq.empty(), (existing_value,))
        case PSeqDeep(size, front, between, back):
            match front:
                case (a,):
                    return PSeqDeep(size + 1, (value, a), between, back)
                case (a, b):
                    return PSeqDeep(size + 1, (value, a, b), between, back)
                case (a, b, c):
                    return PSeqDeep(size + 1, (value, a, b, c), between, back)
                case (a, b, c, d):
                    new_inner = (a, b)
                    new_between = _seq_cons(new_inner, between)
                    return PSeqDeep(size + 1, (value, c, d), new_between, back)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_snoc[T](seq: PSeq[T], value: T) -> PSeq[T]:
    match seq:
        case PSeqEmpty():
            return PSeqSingle(value)
        case PSeqSingle(existing_value):
            return PSeqDeep(2, (existing_value,), PSeq.empty(), (value,))
        case PSeqDeep(size, front, between, back):
            match back:
                case (a,):
                    return PSeqDeep(size + 1, front, between, (a, value))
                case (a, b):
                    return PSeqDeep(size + 1, front, between, (a, b, value))
                case (a, b, c):
                    return PSeqDeep(size + 1, front, between, (a, b, c, value))
                case (a, b, c, d):
                    new_inner = (a, b)
                    new_between = _seq_snoc(between, new_inner)
                    return PSeqDeep(size + 1, front, new_between, (c, d, value))
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_unsnoc[T](seq: PSeq[T]) -> Optional[Tuple[PSeq[T], T]]:
    match seq:
        case PSeqEmpty():
            return None
        case PSeqSingle(value):
            return (PSeq.empty(), value)
        case PSeqDeep(size, front, between, back):
            match back:
                case (a,):
                    if between.null():
                        match front:
                            case (b,):
                                return (PSeqSingle(b), a)
                            case (b, c):
                                return (
                                    PSeqDeep(size - 1, (b,), PSeq.empty(), (c,)),
                                    a,
                                )
                            case (b, c, d):
                                return (
                                    PSeqDeep(size - 1, (b,), PSeq.empty(), (c, d)),
                                    a,
                                )
                            case (b, c, d, e):
                                return (
                                    PSeqDeep(size - 1, (b,), PSeq.empty(), (c, d, e)),
                                    a,
                                )
                            case _:
                                raise Impossible
                    else:
                        between_unsnoc = _seq_unsnoc(between)
                        if between_unsnoc is None:
                            match front:
                                case (b,):
                                    return (PSeqSingle(b), a)
                                case (b, c):
                                    return (
                                        PSeqDeep(size - 1, (b,), PSeq.empty(), (c,)),
                                        a,
                                    )
                                case (b, c, d):
                                    return (
                                        PSeqDeep(size - 1, (b,), PSeq.empty(), (c, d)),
                                        a,
                                    )
                                case (b, c, d, e):
                                    return (
                                        PSeqDeep(
                                            size - 1, (b,), PSeq.empty(), (c, d, e)
                                        ),
                                        a,
                                    )
                                case _:
                                    raise Impossible
                        else:
                            between_init, inner_last = between_unsnoc
                            return (
                                PSeqDeep(size - 1, front, between_init, inner_last),
                                a,
                            )
                case (a, b):
                    return (PSeqDeep(size - 1, front, between, (a,)), b)
                case (a, b, c):
                    return (PSeqDeep(size - 1, front, between, (a, b)), c)
                case (a, b, c, d):
                    return (PSeqDeep(size - 1, front, between, (a, b, c)), d)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_concat[T](seq: PSeq[T], other: PSeq[T]) -> PSeq[T]:
    match seq:
        case PSeqEmpty():
            return other
        case PSeqSingle(value):
            match other:
                case PSeqEmpty():
                    return seq
                case PSeqSingle(other_value):
                    return PSeqDeep(2, (value,), PSeq.empty(), (other_value,))
                case PSeqDeep(_, _, _, _):
                    return _seq_cons(value, other)
                case _:
                    raise Impossible
        case PSeqDeep(_, _, _, _):
            match other:
                case PSeqEmpty():
                    return seq
                case PSeqSingle(other_value):
                    return _seq_snoc(seq, other_value)
                case PSeqDeep(_, _, _, _):
                    return _seq_concat_deep(seq, other)
                case _:
                    raise Impossible
        case _:
            raise Impossible


def _seq_concat_deep[T](left: PSeq[T], right: PSeq[T]) -> PSeq[T]:
    match (left, right):
        case (
            PSeqDeep(left_size, left_front, left_between, left_back),
            PSeqDeep(right_size, right_front, right_between, right_back),
        ):
            middle_nodes = _nodes_from_touching_ends(left_back, right_front)
            new_between = _seq_concat_middle(left_between, middle_nodes, right_between)
            return PSeqDeep(left_size + right_size, left_front, new_between, right_back)
        case _:
            raise Impossible


def _nodes_from_touching_ends[T](
    left_back: OuterNode[T], right_front: OuterNode[T]
) -> List[InnerNode[T]]:
    combined = list(left_back)
    combined.extend(right_front)
    nodes: List[InnerNode[T]] = []

    i = 0
    while i < len(combined):
        remaining = len(combined) - i
        if remaining == 1:
            if nodes and len(nodes[-1]) == 2:
                prev_node = nodes.pop()
                a, b = prev_node  # type: ignore
                nodes.append((a, b, combined[i]))
            else:
                raise Impossible
            i += 1
        elif remaining == 2:
            nodes.append((combined[i], combined[i + 1]))
            i += 2
        elif remaining == 3:
            nodes.append((combined[i], combined[i + 1], combined[i + 2]))
            i += 3
        elif remaining == 4:
            nodes.append((combined[i], combined[i + 1]))
            nodes.append((combined[i + 2], combined[i + 3]))
            i += 4
        else:
            nodes.append((combined[i], combined[i + 1], combined[i + 2]))
            i += 3

    return nodes


def _seq_concat_middle[T](
    left_between: PSeq[InnerNode[T]],
    middle_nodes: List[InnerNode[T]],
    right_between: PSeq[InnerNode[T]],
) -> PSeq[InnerNode[T]]:
    result = left_between
    for node in middle_nodes:
        result = _seq_snoc(result, node)
    return _seq_concat(result, right_between)


def _seq_get[T](seq: PSeq[T], ix: int) -> T:
    if ix < 0:
        raise KeyError(ix)
    match seq:
        case PSeqEmpty():
            raise KeyError(ix)
        case PSeqSingle(value):
            if ix == 0:
                return value
            else:
                raise KeyError(ix)
        case PSeqDeep(size, front, between, back):
            if ix >= size:
                raise KeyError(ix)
            front_size = len(front)
            if ix < front_size:
                return front[ix]
            ix -= front_size
            back_size = len(back)
            between_total_size = size - front_size - back_size
            if ix < between_total_size:
                return _seq_get_between(between, ix)
            ix -= between_total_size
            if ix < back_size:
                return back[ix]
            raise KeyError(ix)
        case _:
            raise Impossible


def _seq_get_between[T](between: PSeq[InnerNode[T]], ix: int) -> T:
    current_offset = 0
    for inner_node in _seq_iter(between):
        node_size = len(inner_node)
        if ix < current_offset + node_size:
            node_ix = ix - current_offset
            if node_ix < len(inner_node):
                return inner_node[node_ix]
        current_offset += node_size
    raise Impossible


def _seq_update[T](seq: PSeq[T], ix: int, value: T) -> PSeq[T]:
    if ix < 0:
        return seq
    match seq:
        case PSeqEmpty():
            return seq
        case PSeqSingle(_):
            if ix == 0:
                return PSeqSingle(value)
            else:
                return seq
        case PSeqDeep(size, front, between, back):
            if ix >= size:
                return seq
            front_size = len(front)
            if ix < front_size:
                new_front = _update_outer_node(front, ix, value)
                return PSeqDeep(size, new_front, between, back)
            ix -= front_size
            back_size = len(back)
            between_total_size = size - front_size - back_size
            if ix < between_total_size:
                new_between = _seq_update_between(between, ix, value)
                return PSeqDeep(size, front, new_between, back)
            ix -= between_total_size
            if ix < back_size:
                new_back = _update_outer_node(back, ix, value)
                return PSeqDeep(size, front, between, new_back)
            return seq
        case _:
            raise Impossible


def _update_outer_node[T](node: OuterNode[T], ix: int, value: T) -> OuterNode[T]:
    match node:
        case (a,):
            if ix == 0:
                return (value,)
            else:
                return node
        case (a, b):
            if ix == 0:
                return (value, b)
            elif ix == 1:
                return (a, value)
            else:
                return node
        case (a, b, c):
            if ix == 0:
                return (value, b, c)
            elif ix == 1:
                return (a, value, c)
            elif ix == 2:
                return (a, b, value)
            else:
                return node
        case (a, b, c, d):
            if ix == 0:
                return (value, b, c, d)
            elif ix == 1:
                return (a, value, c, d)
            elif ix == 2:
                return (a, b, value, d)
            elif ix == 3:
                return (a, b, c, value)
            else:
                return node
        case _:
            raise Impossible


def _update_inner_node[T](node: InnerNode[T], ix: int, value: T) -> InnerNode[T]:
    match node:
        case (a, b):
            if ix == 0:
                return (value, b)
            elif ix == 1:
                return (a, value)
            else:
                return node
        case (a, b, c):
            if ix == 0:
                return (value, b, c)
            elif ix == 1:
                return (a, value, c)
            elif ix == 2:
                return (a, b, value)
            else:
                return node
        case _:
            raise Impossible


def _seq_update_between[T](
    between: PSeq[InnerNode[T]], ix: int, value: T
) -> PSeq[InnerNode[T]]:
    current_offset = 0
    between_list = list(_seq_iter(between))

    for i, inner_node in enumerate(between_list):
        node_size = len(inner_node)
        if ix < current_offset + node_size:
            node_ix = ix - current_offset
            if node_ix < len(inner_node):
                new_inner_node = _update_inner_node(inner_node, node_ix, value)
                new_between_list = between_list.copy()
                new_between_list[i] = new_inner_node
                return PSeq.mk(new_between_list)
        current_offset += node_size

    return between


def _seq_iter[T](seq: PSeq[T]) -> Generator[T]:
    match seq:
        case PSeqEmpty():
            pass
        case PSeqSingle(value):
            yield value
        case PSeqDeep(_, front, between, back):
            yield from front
            for inner_node in _seq_iter(between):
                yield from inner_node
            yield from back
        case _:
            raise Impossible


def _seq_reversed[T](seq: PSeq[T]) -> Generator[T]:
    match seq:
        case PSeqEmpty():
            pass
        case PSeqSingle(value):
            yield value
        case PSeqDeep(_, front, between, back):
            yield from reversed(back)
            for inner_node in _seq_reversed(between):
                yield from reversed(inner_node)
            yield from reversed(front)
        case _:
            raise Impossible
